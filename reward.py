#!/usr/bin/env python3
"""
reward.py — 奖励函数族

Search-R1 的核心洞察：用"可验证奖励"代替人工标注。
本模块实现四类奖励信号，可组合使用。

奖励层次（从粗到细）：
    ┌────────────────────────────────────────────────────┐
    │  R_total = w1·R_outcome                            │
    │          + w2·R_process                            │
    │          + w3·R_format                             │
    │          + w4·R_efficiency                         │
    └────────────────────────────────────────────────────┘

各奖励详解：

  R_outcome  (结果奖励，0 或 1，最重要)
    - pytest 通过率：passed / total
    - 文件内容匹配：0/0.5/1.0
    - bash 输出包含期望内容

  R_process  (过程奖励，0~1，鼓励正确推理链)
    - 有 <think> 标签：+0.1
    - 工具调用顺序合理（先读后写）：+0.1
    - 中间步骤有进展（lint 错误减少）：+0.2

  R_format   (格式奖励，0~1，鼓励结构化输出)
    - 输出包含 <think>...</think>：+0.3
    - 工具调用格式正确：+0.2
    - 最终回答非空：+0.2

  R_efficiency (效率奖励，0~1，鼓励用更少步骤完成)
    - 步数越少奖励越高：max(0, 1 - steps/max_steps)
"""

import re
from dataclasses import dataclass
from typing import Callable

from environment import Trajectory


# ---------------------------------------------------------------------------
# 奖励配置
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """各奖励分量的权重，默认遵循 Search-R1 精神：结果优先。"""
    outcome: float = 1.0
    process: float = 0.3
    format_: float = 0.2
    efficiency: float = 0.1


DEFAULT_WEIGHTS = RewardWeights()


# ---------------------------------------------------------------------------
# R_outcome：结果奖励（最关键）
# ---------------------------------------------------------------------------

def reward_outcome(trajectory: Trajectory) -> float:
    """
    基于任务完成状态的二值奖励。
    - success   → 1.0
    - failure   → 0.0
    - truncated → 0.0（被截断也算失败，鼓励在步数内完成）
    """
    return 1.0 if trajectory.status == "success" else 0.0


def reward_pytest_partial(trajectory: Trajectory) -> float:
    """
    部分奖励：解析 pytest 输出，用通过率代替二值奖励。
    适用于测试驱动任务，鼓励"接近正确"的中间状态。

    解析示例:
        "3 passed, 2 failed" → 0.6
        "5 passed"            → 1.0
        "ERROR"               → 0.0
    """
    pytest_outputs = [
        tc.output for tc in trajectory.tool_calls
        if tc.name == "bash" and ("passed" in tc.output or "failed" in tc.output)
    ]
    if not pytest_outputs:
        return 0.0

    # 取最后一次 pytest 输出（最终状态）
    last_output = pytest_outputs[-1]

    passed_match = re.search(r"(\d+) passed", last_output)
    failed_match = re.search(r"(\d+) failed", last_output)
    error_match = re.search(r"(\d+) error", last_output)

    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0

    total = passed + failed + errors
    if total == 0:
        return 0.0
    return passed / total


# ---------------------------------------------------------------------------
# R_process：过程奖励
# ---------------------------------------------------------------------------

def reward_process(trajectory: Trajectory) -> float:
    """
    奖励合理的推理和工具使用过程。
    检查 messages 中的 assistant 消息是否包含 <think> 推理链，
    以及工具调用序列是否体现出合理的问题解决策略。
    """
    score = 0.0

    # 检查是否有推理链（<think> 标签）
    has_think = any(
        "<think>" in str(msg.get("content", ""))
        for msg in trajectory.messages
        if msg.get("role") == "assistant"
    )
    if has_think:
        score += 0.3

    # 检查工具调用多样性（不只是重复 bash）
    tool_names = [tc.name for tc in trajectory.tool_calls]
    unique_tools = len(set(tool_names))
    if unique_tools >= 2:
        score += 0.2
    if unique_tools >= 3:
        score += 0.1

    # 检查是否有先读后写的合理顺序（read_file 在 write_file 之前出现）
    has_read = "read_file" in tool_names
    has_write = "write_file" in tool_names
    if has_read and has_write:
        read_idx = tool_names.index("read_file")
        write_idx = tool_names.index("write_file")
        if read_idx < write_idx:
            score += 0.2

    # 检查是否有进展（bash 中错误数量随时间减少）
    bash_outputs = [tc.output for tc in trajectory.tool_calls if tc.name == "bash"]
    if len(bash_outputs) >= 2:
        first_errors = bash_outputs[0].lower().count("error")
        last_errors = bash_outputs[-1].lower().count("error")
        if last_errors < first_errors:
            score += 0.2

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# R_format：格式奖励（鼓励 Search-R1 风格的结构化输出）
# ---------------------------------------------------------------------------

# 期望的 assistant 消息格式：
#   <think>
#   ... 推理过程 ...
#   </think>
#   ... 工具调用 or 最终回答 ...

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_CALL_INDICATOR = re.compile(r'"name"\s*:\s*"(bash|read_file|write_file|edit_file)"')


def reward_format(trajectory: Trajectory) -> float:
    """
    检查输出是否符合 Search-R1 风格的格式规范。
    """
    score = 0.0
    assistant_messages = [
        str(msg.get("content", ""))
        for msg in trajectory.messages
        if msg.get("role") == "assistant"
    ]

    if not assistant_messages:
        return 0.0

    # 检查是否有 <think> 块
    any_think = any(THINK_PATTERN.search(m) for m in assistant_messages)
    if any_think:
        score += 0.4

    # 最终回答非空
    if trajectory.final_answer and len(trajectory.final_answer.strip()) > 10:
        score += 0.3

    # 工具调用有正确的格式（不是乱猜的名字）
    valid_tool_calls = sum(
        1 for tc in trajectory.tool_calls
        if tc.name in {"bash", "read_file", "write_file", "edit_file"}
    )
    total_calls = len(trajectory.tool_calls)
    if total_calls > 0:
        score += 0.3 * (valid_tool_calls / total_calls)

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# R_efficiency：效率奖励
# ---------------------------------------------------------------------------

def reward_efficiency(trajectory: Trajectory, max_steps: int = 20) -> float:
    """
    鼓励用更少的工具调用步骤完成任务。
    只有在任务成功时才给效率奖励，避免鼓励 agent 提前放弃。

    公式：max(0, 1 - steps / max_steps) * success
    """
    if trajectory.status != "success":
        return 0.0
    steps = len(trajectory.tool_calls)
    return max(0.0, 1.0 - steps / max_steps)


# ---------------------------------------------------------------------------
# 组合奖励
# ---------------------------------------------------------------------------

class RewardFn:
    """
    组合奖励函数。支持自定义权重和自定义子奖励。

    用法::

        rf = RewardFn()
        score = rf(trajectory)

        # 自定义权重
        rf = RewardFn(weights=RewardWeights(outcome=2.0, process=0.5))
        score = rf(trajectory)

        # 注册自定义子奖励
        def my_reward(traj):
            return 1.0 if "import" in traj.final_answer else 0.0
        rf.register("custom", my_reward, weight=0.5)
    """

    def __init__(
        self,
        weights: RewardWeights = DEFAULT_WEIGHTS,
        use_partial_pytest: bool = True,
        max_steps: int = 20,
    ):
        self.weights = weights
        self.use_partial_pytest = use_partial_pytest
        self.max_steps = max_steps
        self._custom: list[tuple[str, Callable, float]] = []

    def register(self, name: str, fn: Callable[[Trajectory], float], weight: float = 1.0):
        """注册自定义子奖励函数。"""
        self._custom.append((name, fn, weight))

    def __call__(self, trajectory: Trajectory) -> dict[str, float]:
        """
        计算所有子奖励，返回详细分数 dict。

        Returns:
            {
                "total": 加权总分,
                "outcome": ...,
                "process": ...,
                "format": ...,
                "efficiency": ...,
                "custom_*": ...,
            }
        """
        scores: dict[str, float] = {}

        # 结果奖励（优先用部分 pytest 奖励）
        if self.use_partial_pytest:
            partial = reward_pytest_partial(trajectory)
            outcome = max(reward_outcome(trajectory), partial)
        else:
            outcome = reward_outcome(trajectory)
        scores["outcome"] = outcome

        scores["process"] = reward_process(trajectory)
        scores["format"] = reward_format(trajectory)
        scores["efficiency"] = reward_efficiency(trajectory, self.max_steps)

        # 自定义奖励
        for name, fn, _ in self._custom:
            try:
                scores[f"custom_{name}"] = float(fn(trajectory))
            except Exception:
                scores[f"custom_{name}"] = 0.0

        # 加权求和
        total = (
            self.weights.outcome * scores["outcome"]
            + self.weights.process * scores["process"]
            + self.weights.format_ * scores["format"]
            + self.weights.efficiency * scores["efficiency"]
        )
        for name, _, weight in self._custom:
            total += weight * scores.get(f"custom_{name}", 0.0)

        scores["total"] = total
        return scores


# ---------------------------------------------------------------------------
# GRPO 用的批量奖励计算（组内相对分数）
# ---------------------------------------------------------------------------

def compute_group_advantages(
    trajectories: list[Trajectory],
    reward_fn: RewardFn,
    eps: float = 1e-6,
) -> list[dict]:
    """
    GRPO（组相对策略优化）的核心：
    对同一个 prompt 采样 G 条轨迹，计算组内相对优势。

    公式（来自 DeepSeek-R1 / Search-R1）：
        A_i = (r_i - mean(r)) / (std(r) + eps)

    Args:
        trajectories: 同一 prompt 的 G 条轨迹
        reward_fn: 奖励函数
        eps: 数值稳定项

    Returns:
        list of {"trajectory": ..., "reward_details": ..., "advantage": float}
    """
    if not trajectories:
        return []

    reward_details = [reward_fn(t) for t in trajectories]
    raw_rewards = [rd["total"] for rd in reward_details]

    n = len(raw_rewards)
    mean_r = sum(raw_rewards) / n
    std_r = (sum((r - mean_r) ** 2 for r in raw_rewards) / n) ** 0.5

    results = []
    for traj, rd, r in zip(trajectories, reward_details, raw_rewards):
        advantage = (r - mean_r) / (std_r + eps)
        results.append({
            "trajectory": traj,
            "reward_details": rd,
            "raw_reward": r,
            "advantage": advantage,
        })
    return results


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from environment import Trajectory, ToolCall

    # 构造一个模拟的成功轨迹
    traj = Trajectory(task_id="test_001", prompt="写一个求和函数并通过测试")
    traj.tool_calls = [
        ToolCall("read_file", {"path": "test_sum.py"}, "def test_sum(): assert add(1,2)==3"),
        ToolCall("write_file", {"path": "solution.py"}, "def add(a,b): return a+b"),
        ToolCall("bash", {"command": "pytest test_sum.py -q"}, "1 passed in 0.1s"),
    ]
    traj.messages = [
        {"role": "assistant", "content": "<think>先读测试文件，再写实现</think>"},
    ]
    traj.final_answer = "已实现 add 函数，测试通过。"
    traj.status = "success"

    rf = RewardFn()
    scores = rf(traj)
    print("奖励分数：")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    # 测试 GRPO 组内优势
    traj2 = Trajectory(task_id="test_001", prompt="写一个求和函数并通过测试")
    traj2.status = "failure"
    traj2.tool_calls = [ToolCall("bash", {"command": "ls"}, "test_sum.py")]

    groups = compute_group_advantages([traj, traj2], rf)
    print("\nGRPO 组内优势：")
    for g in groups:
        print(f"  status={g['trajectory'].status}, reward={g['raw_reward']:.3f}, "
              f"advantage={g['advantage']:.3f}")
