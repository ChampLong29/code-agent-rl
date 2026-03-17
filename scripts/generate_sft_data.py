#!/usr/bin/env python3
"""
scripts/generate_sft_data.py — SFT 数据生成器

用强教师模型（Claude / GPT-4o）对 data/raw/ 中的任务做 rollout，
生成 SFT 格式的训练数据。

输出格式（Alpaca/ShareGPT 风格，兼容 LLaMA-Factory / TRL）：
    {
        "id": "task_001_0",
        "conversations": [
            {"role": "system",  "content": "..."},
            {"role": "user",    "content": "任务描述"},
            {"role": "assistant","content": "<think>推理</think>\n工具调用 or 答案"},
            {"role": "tool",    "content": "工具返回结果"},
            ...
            {"role": "assistant","content": "最终回答"},
        ],
        "metadata": {"task_id": ..., "status": ..., "reward": ...}
    }

只保留 status == "success" 的轨迹作为 SFT 数据。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from environment import AgentEnvironment, TaskLoader, Trajectory
from reward import RewardFn
from rollout import RolloutSampler, make_anthropic_model_fn

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# 默认任务集（内嵌的种子任务，无需外部文件也能运行）
# ---------------------------------------------------------------------------

SEED_TASKS = [
    {
        "id": "task_hello_001",
        "prompt": "在当前目录创建一个 hello.py 文件，内容是打印 'Hello, World!'，然后运行它确认输出正确。",
        "setup_files": {},
        "success_criteria": {
            "type": "bash_output",
            "command": "python hello.py",
            "expected": "Hello, World!",
        },
    },
    {
        "id": "task_fib_001",
        "prompt": "实现一个斐波那契函数 fib(n)，写入 solution.py，通过下面的测试：\ntest: assert fib(0)==0, fib(1)==1, fib(10)==55",
        "setup_files": {
            "test_fib.py": (
                "from solution import fib\n"
                "def test_fib_base(): assert fib(0)==0 and fib(1)==1\n"
                "def test_fib_10(): assert fib(10)==55\n"
                "def test_fib_neg(): assert fib(-1)==0\n"
            )
        },
        "success_criteria": {
            "type": "pytest",
            "pattern": "test_fib.py",
        },
    },
    {
        "id": "task_sort_001",
        "prompt": "用冒泡排序实现 bubble_sort(lst) 函数，写入 sort.py，通过测试。",
        "setup_files": {
            "test_sort.py": (
                "from sort import bubble_sort\n"
                "def test_empty(): assert bubble_sort([])==[]\n"
                "def test_sorted(): assert bubble_sort([1,2,3])==[1,2,3]\n"
                "def test_reverse(): assert bubble_sort([3,2,1])==[1,2,3]\n"
                "def test_duplicates(): assert bubble_sort([3,1,2,1])==[1,1,2,3]\n"
            )
        },
        "success_criteria": {"type": "pytest", "pattern": "test_sort.py"},
    },
    {
        "id": "task_counter_001",
        "prompt": "实现一个 Counter 类，支持 count(item)、get(item)、most_common(n) 方法，写入 counter.py，通过测试。",
        "setup_files": {
            "test_counter.py": (
                "from counter import Counter\n"
                "def test_basic():\n"
                "    c = Counter()\n"
                "    c.count('a'); c.count('b'); c.count('a')\n"
                "    assert c.get('a')==2 and c.get('b')==1\n"
                "def test_most_common():\n"
                "    c = Counter()\n"
                "    for x in ['a','b','a','c','a','b']:\n"
                "        c.count(x)\n"
                "    top = c.most_common(2)\n"
                "    assert top[0][0]=='a' and top[1][0]=='b'\n"
            )
        },
        "success_criteria": {"type": "pytest", "pattern": "test_counter.py"},
    },
    {
        "id": "task_readme_001",
        "prompt": "读取当前目录，为项目生成一个 README.md，包含：项目名（my-project）、安装说明（pip install .）、使用示例（import my_project）。",
        "setup_files": {},
        "success_criteria": {
            "type": "all",
            "criteria": [
                {"type": "file_exists", "path": "README.md"},
                {"type": "file_contains", "path": "README.md", "expected": "my-project"},
                {"type": "file_contains", "path": "README.md", "expected": "pip install"},
            ],
        },
    },
    {
        "id": "task_refactor_001",
        "prompt": "下面的代码有重复逻辑，重构它以消除重复，并确保功能不变（通过测试）。",
        "setup_files": {
            "original.py": (
                "def area_circle(r):\n"
                "    import math\n"
                "    return math.pi * r * r\n"
                "\n"
                "def area_square(s):\n"
                "    return s * s\n"
                "\n"
                "def area_rectangle(w, h):\n"
                "    return w * h\n"
            ),
            "test_refactor.py": (
                "import math\n"
                "from original import area_circle, area_square, area_rectangle\n"
                "def test_circle(): assert abs(area_circle(1)-math.pi)<1e-9\n"
                "def test_square(): assert area_square(4)==16\n"
                "def test_rect(): assert area_rectangle(3,4)==12\n"
            ),
        },
        "success_criteria": {"type": "pytest", "pattern": "test_refactor.py"},
    },
    {
        "id": "task_debug_001",
        "prompt": "debug.py 中有一个 bug，运行测试找出并修复它。",
        "setup_files": {
            "debug.py": (
                "def find_max(lst):\n"
                "    if not lst:\n"
                "        return None\n"
                "    max_val = lst[0]\n"
                "    for i in range(len(lst)):  # bug: should be range(1, len(lst))\n"
                "        if lst[i] > max_val:\n"
                "            max_val = lst[i]\n"
                "    return max_val\n"
            ),
            "test_debug.py": (
                "from debug import find_max\n"
                "def test_normal(): assert find_max([3,1,4,1,5,9,2,6])==9\n"
                "def test_empty(): assert find_max([])==None\n"
                "def test_single(): assert find_max([42])==42\n"
                "def test_negatives(): assert find_max([-1,-5,-2])==-1\n"
            ),
        },
        "success_criteria": {"type": "pytest", "pattern": "test_debug.py"},
    },
    {
        "id": "task_api_001",
        "prompt": "实现一个简单的 JSON 文件数据库类 JsonDB，支持 set(key, val)、get(key)、delete(key)、keys() 操作，数据持久化到 db.json，通过测试。",
        "setup_files": {
            "test_jsondb.py": (
                "import os, json\n"
                "from jsondb import JsonDB\n"
                "def test_crud():\n"
                "    if os.path.exists('db.json'): os.remove('db.json')\n"
                "    db = JsonDB('db.json')\n"
                "    db.set('name', 'Alice')\n"
                "    db.set('age', 30)\n"
                "    assert db.get('name')=='Alice'\n"
                "    assert db.get('age')==30\n"
                "    assert 'name' in db.keys()\n"
                "    db.delete('name')\n"
                "    assert db.get('name') is None\n"
                "    # 持久化测试\n"
                "    db2 = JsonDB('db.json')\n"
                "    assert db2.get('age')==30\n"
            )
        },
        "success_criteria": {"type": "pytest", "pattern": "test_jsondb.py"},
    },
]


# ---------------------------------------------------------------------------
# 轨迹到 SFT 格式的转换
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding agent. Solve programming tasks step by step.

Before taking any action, reason inside <think>...</think> tags.
Use the available tools (bash, read_file, write_file, edit_file) to complete the task.
When done, provide a clear final answer.

Format:
<think>
[Your reasoning here]
</think>
[Tool call or final answer]"""


def trajectory_to_sft(trajectory: Trajectory, sample_id: str) -> dict | None:
    """
    将 Trajectory 转换为 SFT 训练格式。
    只转换 success 的轨迹。

    输出的 conversations 格式兼容 LLaMA-Factory / TRL SFTTrainer。
    """
    if trajectory.status != "success":
        return None

    conversations = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 重建对话历史
    for msg in trajectory.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            # 第一条 user 消息是 task prompt
            if isinstance(content, str):
                conversations.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # tool_result 格式
                tool_results = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_results.append(f"[Tool Result]\n{item.get('content', '')}")
                if tool_results:
                    conversations.append({
                        "role": "tool",
                        "content": "\n---\n".join(tool_results),
                    })

        elif role == "assistant":
            if isinstance(content, str):
                conversations.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # 提取文本和工具调用
                text_parts = []
                tool_calls = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append(
                                f'<tool_call>{{"name": "{block.name}", '
                                f'"arguments": {json.dumps(block.input, ensure_ascii=False)}}}'
                                f'</tool_call>'
                            )
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                f'<tool_call>{{"name": "{block.get("name")}", '
                                f'"arguments": {json.dumps(block.get("input", {}), ensure_ascii=False)}}}'
                                f'</tool_call>'
                            )
                full_content = "\n".join(text_parts + tool_calls).strip()
                if full_content:
                    conversations.append({"role": "assistant", "content": full_content})

    # 追加最终回答（如果最后一条不是 assistant）
    if conversations and conversations[-1]["role"] != "assistant" and trajectory.final_answer:
        conversations.append({"role": "assistant", "content": trajectory.final_answer})

    return {
        "id": sample_id,
        "conversations": conversations,
        "metadata": {
            "task_id": trajectory.task_id,
            "status": trajectory.status,
            "tool_calls": len(trajectory.tool_calls),
        },
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="生成 SFT 训练数据")
    parser.add_argument("--model", default=None, help="teacher model ID (默认用环境变量 MODEL_ID)")
    parser.add_argument("--tasks-file", default=None, help="自定义任务文件路径 (JSONL)")
    parser.add_argument("--output", default="data/sft/train.jsonl", help="输出文件路径")
    parser.add_argument("--n-per-task", type=int, default=3, help="每个任务生成多少条成功轨迹")
    parser.add_argument("--max-attempts", type=int, default=10, help="每个任务最多尝试次数")
    parser.add_argument("--max-steps", type=int, default=15, help="每条轨迹最大步数")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务列表，不实际运行")
    args = parser.parse_args()

    # 加载任务
    if args.tasks_file:
        loader = TaskLoader(Path(args.tasks_file).parent)
        tasks = loader.load(Path(args.tasks_file).name)
    else:
        tasks = SEED_TASKS

    if args.dry_run:
        print(f"任务数量: {len(tasks)}")
        for t in tasks:
            print(f"  - {t['id']}: {t['prompt'][:60]}...")
        return

    # 初始化
    model_id = args.model or os.environ.get("MODEL_ID", "claude-3-5-haiku-20241022")
    print(f"Teacher model: {model_id}")
    print(f"任务数: {len(tasks)}, 目标每任务 {args.n_per_task} 条成功轨迹")

    model_fn = make_anthropic_model_fn(model=model_id)
    sampler = RolloutSampler(max_steps=args.max_steps)
    reward_fn = RewardFn()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_attempts = 0

    with open(output_path, "w") as out_f:
        for task in tasks:
            success_count = 0
            for attempt in range(args.max_attempts):
                if success_count >= args.n_per_task:
                    break

                total_attempts += 1
                print(f"\n[{task['id']}] 尝试 {attempt+1}/{args.max_attempts} "
                      f"(已收集 {success_count}/{args.n_per_task})")

                traj = sampler.sample_one(task, model_fn)
                scores = reward_fn(traj)

                print(f"  status={traj.status}, steps={len(traj.tool_calls)}, "
                      f"reward={scores['total']:.3f}")

                if traj.status == "success":
                    sample_id = f"{task['id']}_{success_count}"
                    sft_sample = trajectory_to_sft(traj, sample_id)
                    if sft_sample:
                        out_f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                        out_f.flush()
                        success_count += 1
                        total_success += 1

            print(f"  {task['id']}: 收集 {success_count}/{args.n_per_task} 条")

    print(f"\n完成！成功 {total_success} 条 / 总尝试 {total_attempts} 次")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
