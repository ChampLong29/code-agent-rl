#!/usr/bin/env python3
"""
simulate_training.py — 模拟完整训练流程（无需 GPU）

用 mock 模型替代真实 LLM，完整跑通：
  Phase 0: 数据结构初始化
  Phase 1: SFT 数据生成（teacher rollout）
  Phase 2: GRPO 一次迭代（采样 → 奖励 → 优势 → 损失）

运行：
  uv run python simulate_training.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment import AgentEnvironment, TaskLoader, Trajectory, ToolCall
from reward import RewardFn, RewardWeights, compute_group_advantages

SEP = "─" * 60

# ──────────────────────────────────────────────────────────
# 辅助：打印分隔块
# ──────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def show(label: str, obj):
    if isinstance(obj, dict):
        print(f"\n  [{label}]")
        for k, v in obj.items():
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            print(f"    {k}: {v}")
    elif isinstance(obj, list):
        print(f"\n  [{label}] ({len(obj)} items)")
        for i, item in enumerate(obj[:3]):
            print(f"    [{i}] {str(item)[:100]}")
        if len(obj) > 3:
            print(f"    ... ({len(obj)-3} more)")
    else:
        print(f"\n  [{label}] {obj}")


# ──────────────────────────────────────────────────────────
# Phase 0: 定义任务（对应 data/raw/tasks.jsonl）
# ──────────────────────────────────────────────────────────

section("Phase 0: 任务定义")

TASKS = [
    {
        "id": "fib_001",
        "prompt": "实现 fib(n) 函数，放在 solution.py，要通过 test_fib.py 的测试",
        "setup_files": {
            "test_fib.py": (
                "from solution import fib\n"
                "def test_base(): assert fib(0)==0 and fib(1)==1\n"
                "def test_10():   assert fib(10)==55\n"
            )
        },
        "success_criteria": {"type": "pytest", "pattern": "test_fib.py"},
    },
    {
        "id": "sort_001",
        "prompt": "实现 bubble_sort(lst) 函数，放在 solution.py，要通过 test_sort.py 的测试",
        "setup_files": {
            "test_sort.py": (
                "from solution import bubble_sort\n"
                "def test_empty():   assert bubble_sort([])==[]\n"
                "def test_reverse(): assert bubble_sort([3,2,1])==[1,2,3]\n"
            )
        },
        "success_criteria": {"type": "pytest", "pattern": "test_sort.py"},
    },
]

print(f"\n  任务数: {len(TASKS)}")
for t in TASKS:
    print(f"  · {t['id']}: {t['prompt'][:50]}")


# ──────────────────────────────────────────────────────────
# Phase 1: SFT 数据生成 —— teacher rollout
# ──────────────────────────────────────────────────────────

section("Phase 1: SFT 数据生成（teacher rollout）")

print("""
  真实训练中：用 Claude 等强模型生成轨迹数据
  这里用 mock_teacher 模拟一个"总是写出正确代码"的教师模型
""")

def mock_teacher_fn(messages: list[dict], tools: list[dict]) -> dict:
    """
    模拟教师模型：总是做出正确的工具调用序列。
    真实训练里这里是 Claude claude-3-5-haiku-20241022。
    """
    # 分析对话历史，决定下一步
    user_msg = messages[0]["content"] if messages else ""
    tool_results = [m for m in messages if m["role"] == "user" and
                    isinstance(m["content"], list)]
    step = len(tool_results)

    if "fib" in user_msg:
        steps = [
            # 第 0 步：先读测试文件
            ("tool_use", "read_file", {"path": "test_fib.py"}),
            # 第 1 步：写实现
            ("tool_use", "write_file", {
                "path": "solution.py",
                "content": "def fib(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    return fib(n-1) + fib(n-2)\n"
            }),
            # 第 2 步：跑测试
            ("tool_use", "bash", {"command": "pytest test_fib.py -q"}),
        ]
    else:
        steps = [
            ("tool_use", "read_file", {"path": "test_sort.py"}),
            ("tool_use", "write_file", {
                "path": "solution.py",
                "content": "def bubble_sort(lst):\n    lst = list(lst)\n    for i in range(len(lst)):\n        for j in range(len(lst)-i-1):\n            if lst[j] > lst[j+1]: lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst\n"
            }),
            ("tool_use", "bash", {"command": "pytest test_sort.py -q"}),
        ]

    if step < len(steps):
        stop_reason, tool_name, tool_input = steps[step]
        return {
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "id": f"call_{step}",
                "name": tool_name,
                "input": tool_input,
            }],
            "text": f"<think>步骤{step+1}：执行 {tool_name}</think>",
        }
    else:
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "任务完成，所有测试通过。"}],
            "text": "任务完成，所有测试通过。",
        }


# 用 teacher 跑一条 rollout
from rollout import RolloutSampler

sampler = RolloutSampler(max_steps=10)
teacher_traj = sampler.sample_one(TASKS[0], mock_teacher_fn)

print(f"  任务: {teacher_traj.task_id}")
print(f"  状态: {teacher_traj.status}")
print(f"  工具调用步数: {len(teacher_traj.tool_calls)}")
print()
for i, tc in enumerate(teacher_traj.tool_calls):
    inp_str = str(tc.input)[:60]
    out_str = tc.output[:60].replace("\n", " ")
    print(f"  step {i+1}  {tc.name}({inp_str})")
    print(f"          → {out_str}")

# SFT 数据格式转换
print("\n  ── 转换为 SFT 训练格式（ShareGPT）──")
sft_sample = {
    "id": teacher_traj.task_id,
    "conversations": [
        {"role": "user", "content": teacher_traj.prompt},
        *[
            {"role": "assistant" if i % 2 == 0 else "tool",
             "content": f"<think>执行 {tc.name}</think>\n工具调用: {tc.name}"}
            for i, tc in enumerate(teacher_traj.tool_calls)
        ],
        {"role": "assistant", "content": teacher_traj.final_answer},
    ]
}
print(f"  对话轮数: {len(sft_sample['conversations'])}")
print(f"  示例 (assistant turn 0):")
print(f"    {sft_sample['conversations'][1]['content'][:80]}")
print()
print("  → 这些数据喂给 train/sft_lora.py 做冷启动")
print("  → 目标：让模型学会 <think>+工具调用 的格式，而不是学会解题")


# ──────────────────────────────────────────────────────────
# Phase 2: RL 训练 —— GRPO 一次完整迭代
# ──────────────────────────────────────────────────────────

section("Phase 2: GRPO 训练迭代（iteration=1）")

print("""
  GRPO 每次迭代：
    1. 对每个 task，用当前策略模型采样 G 条轨迹
    2. 对每条轨迹用 reward_fn 打分
    3. 组内 Z-score 归一化得到优势 A_i
    4. 计算 GRPO loss（PPO-clip + KL）并更新模型
""")

# ── Step 1: 采样 G 条轨迹 ──────────────────────────────────

print("  ── Step 1: 采样轨迹（G=4，模拟4种不同水平的学生模型）──\n")

GROUP_SIZE = 4

def make_student_fn(skill_level: float):
    """
    模拟不同水平的学生模型（0.0=很差, 1.0=完美）。
    真实训练里这里是 SFT 后的 Qwen2.5-Coder。
    """
    def student_fn(messages: list[dict], tools: list[dict]) -> dict:
        tool_results = [m for m in messages if m["role"] == "user" and
                        isinstance(m["content"], list)]
        step = len(tool_results)
        user_msg = messages[0]["content"] if messages else ""

        if random.random() > skill_level:
            # 低水平：乱猜或格式错误
            if step == 0:
                return {
                    "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": f"call_{step}",
                                 "name": "bash",
                                 "input": {"command": "echo 'I give up'"}}],
                    "text": "不知道怎么做",
                }
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "我不会。"}],
                "text": "我不会。",
            }

        # 高水平：正确步骤
        is_fib = "fib" in user_msg
        steps = [
            ("read_file", {"path": "test_fib.py" if is_fib else "test_sort.py"}),
            ("write_file", {
                "path": "solution.py",
                "content": (
                    "def fib(n):\n    if n<=0: return 0\n    if n==1: return 1\n    return fib(n-1)+fib(n-2)\n"
                    if is_fib else
                    "def bubble_sort(lst):\n    lst=list(lst)\n    for i in range(len(lst)):\n        for j in range(len(lst)-i-1):\n            if lst[j]>lst[j+1]: lst[j],lst[j+1]=lst[j+1],lst[j]\n    return lst\n"
                )
            }),
            ("bash", {"command": f"pytest {'test_fib.py' if is_fib else 'test_sort.py'} -q"}),
        ]
        if step < len(steps):
            name, inp = steps[step]
            think = "<think>按照正确步骤执行</think>" if random.random() < skill_level else ""
            return {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": f"call_{step}",
                             "name": name, "input": inp}],
                "text": f"{think}",
            }
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "完成。"}],
            "text": "完成。",
        }
    return student_fn


# 模拟 4 种水平（代表同一个模型的4次采样，temperature>0 导致多样性）
SKILL_LEVELS = [0.95, 0.7, 0.3, 0.1]   # 高→低
group_trajectories = []

task = TASKS[0]
for i, skill in enumerate(SKILL_LEVELS):
    model_fn = make_student_fn(skill)
    traj = sampler.sample_one(task, model_fn)
    group_trajectories.append(traj)
    status_icon = "✓" if traj.status == "success" else "✗"
    think_icon = "💭" if any("<think>" in str(m.get("content","")) for m in traj.messages) else "  "
    print(f"  {status_icon} {think_icon} rollout[{i}]  status={traj.status:<10} steps={len(traj.tool_calls)}")


# ── Step 2: 奖励计算 ──────────────────────────────────────

print("\n  ── Step 2: 奖励计算 reward_fn(trajectory) ──\n")

reward_fn = RewardFn(
    weights=RewardWeights(outcome=1.0, process=0.3, format_=0.2, efficiency=0.1)
)

print(f"  {'rollout':<10} {'outcome':>8} {'process':>8} {'format':>8} {'effic':>8} {'TOTAL':>8}")
print(f"  {'-'*52}")

all_scores = []
for i, traj in enumerate(group_trajectories):
    scores = reward_fn(traj)
    all_scores.append(scores)
    print(f"  rollout[{i}]  "
          f"{scores['outcome']:>8.3f} "
          f"{scores['process']:>8.3f} "
          f"{scores['format']:>8.3f} "
          f"{scores['efficiency']:>8.3f} "
          f"{scores['total']:>8.3f}")

print(f"\n  解读：")
print(f"  · outcome=1.0  → pytest 通过（最重要）")
print(f"  · process>0    → 有 <think> 标签 / 先读后写")
print(f"  · format>0     → 输出格式符合规范")
print(f"  · efficiency>0 → 步数少（仅 success 时给）")


# ── Step 3: 组内优势归一化（GRPO 核心）──────────────────────

print("\n  ── Step 3: 组内优势归一化 compute_group_advantages() ──\n")

adv_items = compute_group_advantages(group_trajectories, reward_fn)

raw_rewards = [item["raw_reward"] for item in adv_items]
mean_r = sum(raw_rewards) / len(raw_rewards)
std_r = (sum((r - mean_r)**2 for r in raw_rewards) / len(raw_rewards))**0.5

print(f"  原始奖励: {[f'{r:.3f}' for r in raw_rewards]}")
print(f"  mean(r) = {mean_r:.3f}")
print(f"  std(r)  = {std_r:.3f}")
print()
print(f"  {'rollout':<10} {'raw_reward':>12} {'advantage A_i':>14}  {'含义'}")
print(f"  {'-'*60}")
for i, item in enumerate(adv_items):
    r = item["raw_reward"]
    a = item["advantage"]
    meaning = "↑ 鼓励（增大概率）" if a > 0 else "↓ 惩罚（降低概率）"
    print(f"  rollout[{i}]  {r:>12.3f}  {a:>14.3f}  {meaning}")

print(f"""
  公式：A_i = (r_i - {mean_r:.3f}) / ({std_r:.3f} + ε)

  关键：绝对奖励值不重要，重要的是相对排名。
  即使全部成功（奖励都是 1.0），std≈0，优势≈0，不更新。
  → GRPO 天然处理了"奖励过于稀疏"的问题
""")


# ── Step 4: GRPO Loss 计算（不跑真实模型，展示数学过程）──────

print("  ── Step 4: GRPO Loss 计算 ──\n")

import math

print("  对每条轨迹，计算：")
print("    policy_loss = -min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)")
print("    kl_loss     = log(π_θ) - log(π_ref)  （近似）")
print("    total_loss  = policy_loss + β·kl_loss")
print()
print(f"  超参: ε={0.2}, β={0.04}")
print()

# 模拟 ratio（真实中来自 log_prob 差）
mock_ratios = [1.05, 0.92, 1.15, 0.85]   # π_θ / π_ref 的近似值
eps = 0.2
beta = 0.04

print(f"  {'rollout':<10} {'A_i':>8} {'ratio ρ':>10} {'clip(ρ)':>10} {'policy_loss':>13} {'kl':>8}")
print(f"  {'-'*65}")

total_policy_loss = 0
total_kl = 0
for i, (item, ratio) in enumerate(zip(adv_items, mock_ratios)):
    a = item["advantage"]
    ratio_clipped = max(1 - eps, min(1 + eps, ratio))
    policy_loss = -min(ratio * a, ratio_clipped * a)
    kl = math.log(ratio)   # log(π_θ/π_ref) 近似 KL
    total_policy_loss += policy_loss
    total_kl += kl
    print(f"  rollout[{i}]  {a:>8.3f}  {ratio:>10.3f}  {ratio_clipped:>10.3f}  {policy_loss:>13.4f}  {kl:>8.4f}")

mean_policy_loss = total_policy_loss / len(adv_items)
mean_kl = total_kl / len(adv_items)
final_loss = mean_policy_loss + beta * mean_kl

print(f"\n  mean_policy_loss = {mean_policy_loss:.4f}")
print(f"  mean_kl          = {mean_kl:.4f}")
print(f"  final_loss       = {mean_policy_loss:.4f} + {beta} × {mean_kl:.4f} = {final_loss:.4f}")
print()
print("  → final_loss.backward() + optimizer.step()")
print("  → 权重更新，strategy 概率向高奖励轨迹靠拢")


# ── 完整训练循环预期曲线 ─────────────────────────────────────

section("训练过程预期曲线（200 次迭代）")

print("""
  iteration    mean_reward    success_rate    format_rate    kl_div
  ─────────────────────────────────────────────────────────────────
      0           0.30            30%            20%           0.00
     20           0.38            38%            45%           0.08
     50           0.48            50%            65%           0.18
    100           0.58            62%            78%           0.32
    150           0.64            68%            85%           0.41
    200           0.70+           72%+           90%+          0.45

  关键指标说明：
  · mean_reward   持续上升 → RL 在起效
  · success_rate  跟随上升 → 模型真的学会了解任务
  · format_rate   先快速上升（格式从 SFT 学到）后趋于稳定
  · kl_div        缓慢增大但不超过 1.0（超过说明偏离 SFT 太远）

  异常情况：
  · mean_reward 不上升 → 检查 rollout 是否正常跑通
  · kl_div > 1.0       → 降低 learning_rate 或提高 kl_coeff
  · format_rate 下降   → SFT 数据不足，重新冷启动
""")


section("总结：每个文件的职责")
print("""
  environment.py   → 沙箱：给模型一个可以真实执行代码的隔离空间
                     等价于 RL 中的 Environment（提供 obs 和 reward signal）

  reward.py        → 奖励函数：pytest 通过率 + 格式 + 过程 + 效率
                     等价于 RL 中的 R(s,a)，但这里是规则计算，不是神经网络

  rollout.py       → Agent Loop：驱动模型在环境中执行，收集 Trajectory
                     等价于 RL 中的 experience collection

  train/rl_grpo.py → 训练循环：组内归一化 → PPO-clip loss → 梯度更新
                     等价于 RL 中的 policy optimization step

  train/sft_lora.py → 冷启动：先让模型学会格式，再做 RL
                      没有这步，模型根本不会用工具，RL 无法起效
""")

print(f"{SEP}\n  模拟完成 ✓  （所有步骤无需 GPU）\n{SEP}\n")
