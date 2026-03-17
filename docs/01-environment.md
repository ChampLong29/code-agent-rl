# 01: Environment — 沙箱执行环境

**文件**: `environment.py`

## 设计目标

将 learn-claude-code 的 `agent_loop` 改造为 **Gym 风格的 RL 环境**：

```
原来（learn-claude-code）:          现在（rl_training）:
─────────────────────────          ─────────────────────
agent_loop(messages)               env = AgentEnvironment()
  while True:                      traj = env.reset(task)
    response = LLM(messages)       for step in ...:
    execute_tools()                    obs = env.step(action)
    append_results()               traj = env.finish(answer)
                                   reward = reward_fn(traj)
```

## 核心接口

```python
class AgentEnvironment:
    def reset(task: dict) -> Trajectory     # 初始化 episode
    def step(action: dict) -> dict          # 执行工具调用
    def finish(answer: str) -> Trajectory   # 结束 episode，评分
```

### `reset(task)`

- 创建隔离的临时工作目录（`/tmp/agent_env_<id>_xxx/`）
- 写入 `setup_files`（预置文件，如测试用例）
- 初始化 `Trajectory` 对象

```python
task = {
    "id": "task_001",
    "prompt": "实现 fib(n)，通过 test_fib.py 的测试",
    "setup_files": {
        "test_fib.py": "from solution import fib\ndef test_fib(): assert fib(10)==55"
    },
    "success_criteria": {"type": "pytest", "pattern": "test_fib.py"},
}
env.reset(task)
```

### `step(action)`

执行一步工具调用，返回 observation：

```python
obs = env.step({"name": "bash", "input": {"command": "pytest test_fib.py"}})
# → {"output": "1 passed", "done": False, "step": 1}
```

支持的工具与 learn-claude-code 完全一致：
| 工具 | 功能 |
|------|------|
| `bash` | 执行 shell 命令（30s 超时）|
| `read_file` | 读取文件 |
| `write_file` | 写入文件 |
| `edit_file` | 原地替换文本 |

### `finish(answer)`

结束 episode，评估 `success_criteria`：

| criteria 类型 | 说明 |
|---|---|
| `pytest` | 运行 pytest，检查 `passed` |
| `file_exists` | 检查文件是否存在 |
| `file_contains` | 检查文件内容 |
| `bash_output` | 运行命令，检查输出 |
| `any` / `all` | 组合多个条件 |

## 安全设计

```python
DANGEROUS_PATTERNS = [
    "rm -rf /", "sudo", "shutdown", "reboot", "> /dev/",
    ":(){ :|:& };:", "dd if=/dev/zero",
]
```

- 路径检查：不允许访问工作目录之外的文件
- 命令白名单过滤：拒绝危险命令
- 超时限制：bash 30s，pytest 60s

## 数据结构

### `Trajectory`

```python
@dataclass
class Trajectory:
    task_id: str
    prompt: str
    tool_calls: list[ToolCall]    # 所有工具调用记录
    messages: list[dict]          # 完整对话历史
    final_answer: str             # 最终文本回答
    status: str                   # success | failure | truncated
    metadata: dict
```

### `ToolCall`

```python
@dataclass
class ToolCall:
    name: str        # bash | read_file | write_file | edit_file
    input: dict      # 工具输入参数
    output: str      # 工具执行结果
    timestamp: float
    error: bool      # 是否发生错误
```

## 与 learn-claude-code 的对应关系

| learn-claude-code | environment.py |
|---|---|
| `run_bash(command)` | `_safe_bash(cmd, workdir, timeout=30)` |
| `run_read(path)` | `_safe_read(path, workdir)` |
| `run_write(path, content)` | `_safe_write(path, content, workdir)` |
| `run_edit(path, old, new)` | `_safe_edit(path, old, new, workdir)` |
| `TOOL_HANDLERS` dict | `_dispatch(name, inp)` |
| `messages.append(tool_result)` | `trajectory.tool_calls.append(tc)` |
| s12 worktree 隔离 | `tempfile.mkdtemp()` |

## 使用示例

```python
with AgentEnvironment(max_steps=10) as env:
    traj = env.reset(task)

    # 模拟 agent 执行
    env.step({"name": "write_file", "input": {
        "path": "solution.py",
        "content": "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)"
    }})
    env.step({"name": "bash", "input": {"command": "pytest test_fib.py -q"}})

    traj = env.finish("已实现 fib 函数，测试通过。")
    print(traj.status)  # "success"
```
