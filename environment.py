#!/usr/bin/env python3
"""
environment.py — 沙箱执行环境

将 learn-claude-code 的 agent_loop 封装为一个可重置、可评分的
强化学习环境（Gym 风格接口）。

架构：
    ┌──────────────────────────────────────────────────────┐
    │  AgentEnvironment                                    │
    │                                                      │
    │  reset(task)  →  初始化 workdir + 消息历史           │
    │  step(action) →  执行工具调用，返回 (obs, done)       │
    │  score()      →  调用 reward.py 打分                 │
    │                                                      │
    │  内部持有：                                           │
    │    - _workdir: 临时隔离目录 (类似 s12 worktree)       │
    │    - _messages: 完整对话历史                          │
    │    - _tool_calls: 本轮所有工具调用记录                │
    └──────────────────────────────────────────────────────┘

工具调用格式（与 learn-claude-code 保持一致）：
    {
        "name": "bash",
        "input": {"command": "pytest tests/test_solution.py"},
        "output": "5 passed in 0.3s",
    }

Trajectory 格式（用于 reward 计算和 SFT 数据生成）：
    {
        "task_id": "t001",
        "prompt": "实现一个冒泡排序函数...",
        "tool_calls": [...],
        "final_answer": "...",
        "status": "success" | "failure" | "truncated",
    }
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """单次工具调用的完整记录。"""
    name: str
    input: dict
    output: str
    timestamp: float = field(default_factory=time.time)
    error: bool = False


@dataclass
class Trajectory:
    """一个完整 episode 的轨迹。"""
    task_id: str
    prompt: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    final_answer: str = ""
    status: str = "pending"   # pending | success | failure | truncated
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "tool_calls": [
                {"name": tc.name, "input": tc.input, "output": tc.output,
                 "error": tc.error, "timestamp": tc.timestamp}
                for tc in self.tool_calls
            ],
            "messages": self.messages,
            "final_answer": self.final_answer,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        traj = cls(task_id=d["task_id"], prompt=d["prompt"])
        traj.tool_calls = [
            ToolCall(name=tc["name"], input=tc["input"], output=tc["output"],
                     error=tc.get("error", False), timestamp=tc.get("timestamp", 0.0))
            for tc in d.get("tool_calls", [])
        ]
        traj.messages = d.get("messages", [])
        traj.final_answer = d.get("final_answer", "")
        traj.status = d.get("status", "pending")
        traj.metadata = d.get("metadata", {})
        return traj


# ---------------------------------------------------------------------------
# 工具实现（与 s02_tool_use.py / s07_task_system.py 保持一致的接口）
# ---------------------------------------------------------------------------

DANGEROUS_PATTERNS = [
    "rm -rf /", "sudo", "shutdown", "reboot", "> /dev/",
    ":(){ :|:& };:", "dd if=/dev/zero",
]


def _safe_bash(command: str, cwd: Path, timeout: int = 30) -> tuple[str, bool]:
    """
    在沙箱目录中执行 bash 命令。
    返回 (output, is_error)。
    timeout 比 learn-claude-code 更严格（30s vs 120s），
    防止 RL 训练时 rollout 卡死。
    """
    if any(p in command for p in DANGEROUS_PATTERNS):
        return "Error: Dangerous command blocked", True
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True,
            timeout=timeout,
        )
        out = (result.stdout + result.stderr).strip()
        return (out[:10000] if out else "(no output)"), False
    except subprocess.TimeoutExpired:
        return f"Error: Timeout ({timeout}s)", True
    except Exception as e:
        return f"Error: {e}", True


def _safe_read(path: str, workdir: Path) -> tuple[str, bool]:
    try:
        fp = (workdir / path).resolve()
        if not str(fp).startswith(str(workdir.resolve())):
            return "Error: Path escapes workspace", True
        return fp.read_text()[:10000], False
    except Exception as e:
        return f"Error: {e}", True


def _safe_write(path: str, content: str, workdir: Path) -> tuple[str, bool]:
    try:
        fp = (workdir / path).resolve()
        if not str(fp).startswith(str(workdir.resolve())):
            return "Error: Path escapes workspace", True
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}", False
    except Exception as e:
        return f"Error: {e}", True


def _safe_edit(path: str, old_text: str, new_text: str, workdir: Path) -> tuple[str, bool]:
    try:
        fp = (workdir / path).resolve()
        if not str(fp).startswith(str(workdir.resolve())):
            return "Error: Path escapes workspace", True
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}", True
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}", False
    except Exception as e:
        return f"Error: {e}", True


def _safe_search(query: str, max_results: int = 3) -> tuple[str, bool]:
    """
    网络搜索工具。优先用 DuckDuckGo（免费无 key），
    降级到返回提示信息（离线环境）。

    Search-R1 的核心洞察：让小模型通过搜索补充知识，
    可以大幅提升 RL 训练中的成功率，从而让奖励信号更密集。
    """
    query = query.strip()[:200]  # 截断过长的 query
    if not query:
        return "Error: Empty search query", True
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found.", False
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', '')}")
            lines.append(f"    {r.get('href', '')}")
            lines.append(f"    {r.get('body', '')[:300]}")
            lines.append("")
        return "\n".join(lines)[:3000], False
    except ImportError:
        return (
            "Search unavailable (install: pip install duckduckgo-search). "
            "Try solving without search."
        ), False
    except Exception as e:
        return f"Search error: {e}", True


# ---------------------------------------------------------------------------
# 核心环境类
# ---------------------------------------------------------------------------

class AgentEnvironment:
    """
    Gym 风格的 agent 执行环境。

    每个 task 对应一个隔离的临时目录，工具调用在其中执行，
    episode 结束后清理。支持 context manager 用法。

    用法示例::

        task = {
            "id": "t001",
            "prompt": "写一个 Python 函数计算斐波那契数列，并通过 test_fib.py 的测试",
            "setup_files": {"test_fib.py": "def test_fib(): assert fib(10)==55"},
            "success_criteria": {"type": "pytest", "pattern": "test_fib.py"},
        }

        env = AgentEnvironment()
        traj = env.reset(task)

        # 模拟 agent 执行一步工具调用
        obs = env.step({"name": "bash", "input": {"command": "ls"}})
        traj = env.finish("这是我的最终答案")
        print(traj.status)  # success / failure
    """

    # 默认工具定义（与 learn-claude-code 格式一致，供 LLM 推理时使用）
    TOOL_SCHEMAS = [
        {
            "name": "bash",
            "description": "Run a shell command in the sandbox workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file (creates parent dirs).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "edit_file",
            "description": "Replace exact text in a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
        {
            "name": "search",
            "description": (
                "Search the web for information. Use this when you need to look up "
                "APIs, algorithms, or examples before writing code."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (1-5)", "default": 3},
                },
                "required": ["query"],
            },
        },
    ]

    def __init__(
        self,
        max_steps: int = 20,
        bash_timeout: int = 30,
        workdir_base: str | None = None,
        keep_workdir: bool = False,
    ):
        """
        Args:
            max_steps: 最大工具调用步数，超过则 truncated。
            bash_timeout: bash 命令超时秒数。
            workdir_base: 临时目录父路径，None 则用系统 tmp。
            keep_workdir: 调试用，保留工作目录不清理。
        """
        self.max_steps = max_steps
        self.bash_timeout = bash_timeout
        self.workdir_base = workdir_base
        self.keep_workdir = keep_workdir

        self._workdir: Path | None = None
        self._trajectory: Trajectory | None = None
        self._task: dict | None = None

    # ---- 生命周期 ----

    def reset(self, task: dict) -> Trajectory:
        """
        初始化一个新 episode。

        Args:
            task: {
                "id": str,
                "prompt": str,
                "setup_files": {相对路径: 内容},  # 预置文件
                "success_criteria": {...},         # 由 reward.py 使用
            }
        Returns:
            初始化后的 Trajectory 对象。
        """
        # 清理上一轮
        self._cleanup()

        # 创建隔离工作目录
        tmp_parent = Path(self.workdir_base) if self.workdir_base else None
        self._workdir = Path(tempfile.mkdtemp(
            prefix=f"agent_env_{task.get('id', 'task')}_",
            dir=tmp_parent,
        ))

        # 写入预置文件
        for rel_path, content in task.get("setup_files", {}).items():
            fp = self._workdir / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)

        self._task = task
        self._trajectory = Trajectory(
            task_id=task.get("id", "unknown"),
            prompt=task.get("prompt", ""),
            metadata={"workdir": str(self._workdir), "max_steps": self.max_steps},
        )
        return self._trajectory

    def step(self, action: dict) -> dict:
        """
        执行一步工具调用。

        Args:
            action: {"name": tool_name, "input": {...}}

        Returns:
            observation dict: {"output": str, "done": bool, "step": int}
        """
        if self._trajectory is None:
            raise RuntimeError("Must call reset() before step()")

        step_count = len(self._trajectory.tool_calls)
        if step_count >= self.max_steps:
            self._trajectory.status = "truncated"
            return {"output": "Max steps reached", "done": True, "step": step_count}

        name = action.get("name", "")
        inp = action.get("input", {})

        output, is_error = self._dispatch(name, inp)

        tc = ToolCall(name=name, input=inp, output=output, error=is_error)
        self._trajectory.tool_calls.append(tc)

        done = False
        return {"output": output, "done": done, "step": step_count + 1}

    def finish(self, final_answer: str) -> Trajectory:
        """
        结束 episode，评估任务是否成功。

        Args:
            final_answer: agent 输出的最终文本答案。

        Returns:
            完成的 Trajectory，status 已填写。
        """
        if self._trajectory is None:
            raise RuntimeError("Must call reset() before finish()")

        self._trajectory.final_answer = final_answer

        # 根据 success_criteria 判断是否成功
        criteria = self._task.get("success_criteria", {})
        success = self._evaluate_criteria(criteria)
        self._trajectory.status = "success" if success else "failure"

        return self._trajectory

    def get_trajectory(self) -> Trajectory | None:
        return self._trajectory

    # ---- 工具分发 ----

    def _dispatch(self, name: str, inp: dict) -> tuple[str, bool]:
        if self._workdir is None:
            return "Error: Environment not initialized", True

        if name == "bash":
            return _safe_bash(inp.get("command", ""), self._workdir, self.bash_timeout)
        elif name == "read_file":
            return _safe_read(inp.get("path", ""), self._workdir)
        elif name == "write_file":
            return _safe_write(inp.get("path", ""), inp.get("content", ""), self._workdir)
        elif name == "edit_file":
            return _safe_edit(
                inp.get("path", ""), inp.get("old_text", ""),
                inp.get("new_text", ""), self._workdir,
            )
        elif name == "search":
            return _safe_search(inp.get("query", ""), inp.get("max_results", 3))
        else:
            return f"Unknown tool: {name}", True

    # ---- 成功标准评估 ----

    def _evaluate_criteria(self, criteria: dict) -> bool:
        """
        评估任务是否完成。支持多种类型：
            - pytest: 运行 pytest，检查通过率
            - file_exists: 检查文件是否存在
            - file_contains: 检查文件内容
            - bash_output: 运行命令，检查输出
        """
        if not criteria or not self._workdir:
            return True  # 无 criteria 视为成功

        ctype = criteria.get("type", "")

        if ctype == "pytest":
            pattern = criteria.get("pattern", "")
            cmd = f"python -m pytest {pattern} -q --tb=no 2>&1"
            out, err = _safe_bash(cmd, self._workdir, timeout=60)
            return not err and "passed" in out and "error" not in out.lower()

        elif ctype == "file_exists":
            path = criteria.get("path", "")
            return (self._workdir / path).exists()

        elif ctype == "file_contains":
            path = criteria.get("path", "")
            expected = criteria.get("expected", "")
            try:
                content = (self._workdir / path).read_text()
                return expected in content
            except Exception:
                return False

        elif ctype == "bash_output":
            cmd = criteria.get("command", "")
            expected = criteria.get("expected", "")
            out, err = _safe_bash(cmd, self._workdir, timeout=30)
            return not err and expected in out

        elif ctype == "any":
            # 多个标准任意一个满足
            return any(self._evaluate_criteria(c) for c in criteria.get("criteria", []))

        elif ctype == "all":
            # 多个标准全部满足
            return all(self._evaluate_criteria(c) for c in criteria.get("criteria", []))

        return False

    # ---- 清理 ----

    def _cleanup(self):
        if self._workdir and self._workdir.exists() and not self.keep_workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
        self._trajectory = None
        self._task = None

    def __del__(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cleanup()


# ---------------------------------------------------------------------------
# 任务加载器
# ---------------------------------------------------------------------------

class TaskLoader:
    """
    从 data/raw/ 加载任务定义（JSONL 格式）。
    每行一个 task dict。
    """

    def __init__(self, raw_dir: str | Path = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load(self, filename: str = "tasks.jsonl") -> list[dict]:
        path = self.raw_dir / filename
        if not path.exists():
            return []
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def load_all(self) -> list[dict]:
        tasks = []
        for fp in self.raw_dir.glob("*.jsonl"):
            tasks.extend(self.load(fp.name))
        return tasks


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = {
        "id": "demo_001",
        "prompt": "在当前目录写一个 hello.py，内容是 print('hello world')，然后运行它",
        "setup_files": {},
        "success_criteria": {
            "type": "bash_output",
            "command": "python hello.py",
            "expected": "hello world",
        },
    }

    with AgentEnvironment(max_steps=10, keep_workdir=False) as env:
        traj = env.reset(task)
        print(f"Workdir: {env._workdir}")

        # 模拟 agent 行为
        r1 = env.step({"name": "write_file", "input": {
            "path": "hello.py", "content": "print('hello world')\n"
        }})
        print(f"Step 1: {r1}")

        r2 = env.step({"name": "bash", "input": {"command": "python hello.py"}})
        print(f"Step 2: {r2}")

        traj = env.finish("我已经创建了 hello.py 并成功运行。")
        print(f"Status: {traj.status}")
        print(f"Tool calls: {len(traj.tool_calls)}")
        print(json.dumps(traj.to_dict(), indent=2, ensure_ascii=False))
