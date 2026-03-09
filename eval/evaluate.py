#!/usr/bin/env python3
"""
eval/evaluate.py — 评估脚本

评估训练后的模型在 agent 任务上的表现。
指标：
    - 任务成功率（success_rate）
    - 平均奖励（mean_reward）
    - 平均步数（mean_steps）
    - 工具使用准确率（tool_accuracy）
    - 格式遵守率（format_rate，是否包含 <think> 标签）

支持模式：
    1. 单模型评估
    2. 多模型对比（SFT vs PPO vs DPO vs GRPO）
    3. 与 teacher 模型的 gap 分析
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import AgentEnvironment, TaskLoader
from reward import RewardFn
from rollout import RolloutSampler, make_hf_model_fn
from scripts.generate_sft_data import SEED_TASKS


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    tasks: list[dict] | None = None,
    n_per_task: int = 3,
    max_steps: int = 15,
    temperature: float = 0.3,   # 评估时用较低温度（更确定性）
    output_dir: str | None = None,
) -> dict:
    """
    评估单个模型在任务集上的表现。

    Args:
        model_path: 模型路径
        tasks: 评估任务列表（默认使用 SEED_TASKS）
        n_per_task: 每个任务运行多少次（取平均）
        max_steps: 最大步数
        temperature: 采样温度（评估时建议低一些）
        output_dir: 结果保存目录

    Returns:
        评估指标 dict
    """
    tasks = tasks or SEED_TASKS
    model_fn = make_hf_model_fn(model_path, temperature=temperature, max_new_tokens=512)
    sampler = RolloutSampler(max_steps=max_steps)
    reward_fn = RewardFn()

    all_results = []
    print(f"\n评估模型: {model_path}")
    print(f"任务数: {len(tasks)}, 每任务 {n_per_task} 次")

    for task in tasks:
        task_results = []
        for run_idx in range(n_per_task):
            traj = sampler.sample_one(task, model_fn)
            scores = reward_fn(traj)

            result = {
                "task_id": task["id"],
                "run_idx": run_idx,
                "status": traj.status,
                "reward_total": scores["total"],
                "reward_outcome": scores["outcome"],
                "reward_process": scores["process"],
                "reward_format": scores["format"],
                "n_steps": len(traj.tool_calls),
                "has_think_tag": any(
                    "<think>" in str(msg.get("content", ""))
                    for msg in traj.messages
                    if msg.get("role") == "assistant"
                ),
                "tool_names": [tc.name for tc in traj.tool_calls],
            }
            task_results.append(result)
            status_icon = "✓" if traj.status == "success" else "✗"
            print(f"  {status_icon} {task['id']} [{run_idx}] "
                  f"reward={scores['total']:.3f}, steps={len(traj.tool_calls)}")

        all_results.extend(task_results)

    # 汇总指标
    metrics = _compute_metrics(all_results)
    metrics["model_path"] = model_path
    metrics["n_tasks"] = len(tasks)
    metrics["n_runs_per_task"] = n_per_task
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 保存结果
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_name = Path(model_path).name
        result_file = out / f"eval_{model_name}_{int(time.time())}.json"
        with open(result_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "details": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {result_file}")

    _print_metrics(metrics)
    return metrics


def _compute_metrics(results: list[dict]) -> dict:
    """计算汇总指标。"""
    n = len(results)
    if n == 0:
        return {}

    success_count = sum(1 for r in results if r["status"] == "success")
    think_count = sum(1 for r in results if r["has_think_tag"])

    # 按任务分组，计算每任务的成功率
    task_groups: dict[str, list] = {}
    for r in results:
        task_groups.setdefault(r["task_id"], []).append(r)

    task_success_rates = {
        tid: sum(1 for r in runs if r["status"] == "success") / len(runs)
        for tid, runs in task_groups.items()
    }

    return {
        "success_rate": success_count / n,
        "mean_reward": sum(r["reward_total"] for r in results) / n,
        "mean_outcome_reward": sum(r["reward_outcome"] for r in results) / n,
        "mean_process_reward": sum(r["reward_process"] for r in results) / n,
        "format_rate": think_count / n,
        "mean_steps": sum(r["n_steps"] for r in results) / n,
        "task_success_rates": task_success_rates,
        "n_total": n,
        "n_success": success_count,
    }


def _print_metrics(metrics: dict):
    print("\n" + "=" * 50)
    print("评估结果汇总")
    print("=" * 50)
    print(f"成功率:       {metrics['success_rate']:.1%} ({metrics['n_success']}/{metrics['n_total']})")
    print(f"平均奖励:     {metrics['mean_reward']:.3f}")
    print(f"  └ 结果奖励: {metrics['mean_outcome_reward']:.3f}")
    print(f"  └ 过程奖励: {metrics['mean_process_reward']:.3f}")
    print(f"格式遵守率:   {metrics['format_rate']:.1%}  (有 <think> 标签)")
    print(f"平均步数:     {metrics['mean_steps']:.1f}")
    print("\n各任务成功率:")
    for tid, rate in sorted(metrics.get("task_success_rates", {}).items()):
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {tid:30s} {bar} {rate:.0%}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# 多模型对比
# ---------------------------------------------------------------------------

def compare_models(
    model_configs: list[dict],
    tasks: list[dict] | None = None,
    n_per_task: int = 2,
    output_dir: str = "eval/results",
) -> dict:
    """
    对比多个模型的性能。

    model_configs 示例：
        [
            {"name": "base",  "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
            {"name": "sft",   "path": "checkpoints/sft_merged"},
            {"name": "grpo",  "path": "checkpoints/grpo/final"},
            {"name": "ppo",   "path": "checkpoints/ppo/final"},
            {"name": "dpo",   "path": "checkpoints/dpo/final"},
        ]
    """
    tasks = tasks or SEED_TASKS
    comparison: dict[str, dict] = {}

    for cfg in model_configs:
        name = cfg["name"]
        path = cfg["path"]
        print(f"\n{'='*60}")
        print(f"评估: {name} ({path})")
        print("=" * 60)
        metrics = evaluate_model(
            path, tasks=tasks, n_per_task=n_per_task, output_dir=output_dir
        )
        comparison[name] = metrics

    # 打印对比表格
    print("\n" + "=" * 70)
    print("模型对比")
    print("=" * 70)
    headers = ["模型", "成功率", "平均奖励", "格式率", "平均步数"]
    print(f"{'模型':<15} {'成功率':>8} {'平均奖励':>10} {'格式率':>8} {'平均步数':>8}")
    print("-" * 55)
    for name, m in comparison.items():
        print(
            f"{name:<15} "
            f"{m['success_rate']:>7.1%}  "
            f"{m['mean_reward']:>9.3f}  "
            f"{m['format_rate']:>7.1%}  "
            f"{m['mean_steps']:>7.1f}"
        )
    print("=" * 55)

    # 保存对比结果
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison_file = out / f"comparison_{int(time.time())}.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果已保存: {comparison_file}")

    return comparison


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--model", default=None, help="单个模型路径")
    parser.add_argument("--compare", nargs="+", help="多个模型路径（格式：name:path）")
    parser.add_argument("--n-per-task", type=int, default=3)
    parser.add_argument("--output", default="eval/results")
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    if args.model:
        evaluate_model(
            args.model,
            n_per_task=args.n_per_task,
            temperature=args.temperature,
            output_dir=args.output,
        )
    elif args.compare:
        configs = []
        for spec in args.compare:
            if ":" in spec:
                name, path = spec.split(":", 1)
            else:
                name = Path(spec).name
                path = spec
            configs.append({"name": name, "path": path})
        compare_models(configs, n_per_task=args.n_per_task, output_dir=args.output)
    else:
        # 默认：评估所有可用的检查点
        checkpoints = {
            "sft": "checkpoints/sft_merged",
            "grpo": "checkpoints/grpo/final",
            "ppo": "checkpoints/ppo/final",
            "dpo": "checkpoints/dpo/final",
        }
        available = [
            {"name": name, "path": path}
            for name, path in checkpoints.items()
            if Path(path).exists()
        ]
        if available:
            compare_models(available, output_dir=args.output)
        else:
            print("没有找到可用的检查点，请先训练模型。")
            print("运行: make sft && make grpo")
