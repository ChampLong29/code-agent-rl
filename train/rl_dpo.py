#!/usr/bin/env python3
"""
train/rl_dpo.py — Phase 2b: DPO（Direct Preference Optimization）

DPO 是离线偏好学习方法，不需要在线 rollout 和 Critic 网络，
直接从"好-坏"轨迹对中学习。

核心思想：
    给定同一个 prompt，提供两条轨迹：
        - chosen (c)：好的轨迹（status=success）
        - rejected (r)：差的轨迹（status=failure）
    优化目标：最大化 π_θ(c) / π_ref(c) 相对于 π_θ(r) / π_ref(r) 的比值

DPO 损失（来自论文 "Direct Preference Optimization"）：
    L_DPO = -log σ( β · log(π_θ(c)/π_ref(c)) - β · log(π_θ(r)/π_ref(r)) )

优势：
    - 无需在线 rollout（离线训练，可以跑 batch）
    - 无需 Critic 网络（比 PPO 省显存）
    - 实现简单，稳定性好

劣势：
    - 需要预先准备偏好对数据
    - 不能从失败中在线改进（需要先跑 rollout 再更新）
    - 泛化能力略弱于在线 RL

数据准备（两种来源）：
    1. teacher rollout 生成：对同一 task 跑多次，按 reward 排序取 top/bottom
    2. 直接从 SFT 数据 + 噪声版本构造偏好对
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import Trajectory
from reward import RewardFn
from rollout import RolloutSampler, make_anthropic_model_fn
from scripts.generate_sft_data import SEED_TASKS, SYSTEM_PROMPT
from hub_utils import resolve_model_path


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "checkpoints/sft_merged")
OUTPUT_DIR = os.environ.get("DPO_OUTPUT_DIR", "checkpoints/dpo")
DPO_DATA_PATH = os.environ.get("DPO_DATA_PATH", "data/rl/dpo_pairs.jsonl")

DPO_CONFIG = {
    "beta": 0.1,                   # DPO 温度参数（越小越激进）
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,         # DPO 学习率要非常小
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "max_length": 2048,
    "max_prompt_length": 512,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "report_to": "none",
}


# ---------------------------------------------------------------------------
# 偏好对数据生成
# ---------------------------------------------------------------------------

def generate_dpo_pairs(
    tasks: list[dict] | None = None,
    model_fn=None,
    n_attempts: int = 6,
    output_path: str = DPO_DATA_PATH,
) -> list[dict]:
    """
    为每个 task 生成 chosen/rejected 偏好对。

    策略：
        1. 对每个 task 跑 n_attempts 次 rollout
        2. 按 reward 排序
        3. 最高 reward 的轨迹 → chosen
        4. 最低 reward 的轨迹 → rejected
        5. 只保留 reward 差距 > 0.3 的对（确保信号足够强）
    """
    tasks = tasks or SEED_TASKS
    if model_fn is None:
        model_fn = make_anthropic_model_fn()

    sampler = RolloutSampler(max_steps=15)
    reward_fn = RewardFn()
    pairs = []

    for task in tasks:
        print(f"\n生成 DPO 对: {task['id']}")
        trajs = []
        for i in range(n_attempts):
            traj = sampler.sample_one(task, model_fn)
            scores = reward_fn(traj)
            trajs.append((scores["total"], traj))
            print(f"  尝试 {i+1}: reward={scores['total']:.3f}, status={traj.status}")

        trajs.sort(key=lambda x: x[0], reverse=True)

        if len(trajs) < 2:
            continue

        best_reward, best_traj = trajs[0]
        worst_reward, worst_traj = trajs[-1]

        if best_reward - worst_reward < 0.3:
            print(f"  跳过：奖励差距太小 ({best_reward:.3f} - {worst_reward:.3f})")
            continue

        # 转换为文本格式
        chosen_text = _trajectory_to_full_text(best_traj)
        rejected_text = _trajectory_to_full_text(worst_traj)

        if chosen_text and rejected_text:
            pair = {
                "prompt": task["prompt"],
                "chosen": chosen_text,
                "rejected": rejected_text,
                "task_id": task["id"],
                "chosen_reward": best_reward,
                "rejected_reward": worst_reward,
            }
            pairs.append(pair)
            print(f"  ✓ 偏好对: chosen={best_reward:.3f}, rejected={worst_reward:.3f}")

    # 保存
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n生成 {len(pairs)} 个偏好对 → {output}")
    return pairs


def _trajectory_to_full_text(traj: Trajectory) -> str:
    """将轨迹转换为完整的对话文本。"""
    parts = []
    for msg in traj.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get("content", str(item)))
                elif hasattr(item, "text"):
                    texts.append(item.text)
                else:
                    texts.append(str(item))
            content = "\n".join(texts)
        if role == "assistant" and content:
            parts.append(str(content))
    return "\n\n".join(parts) or traj.final_answer


def load_dpo_dataset(data_path: str = DPO_DATA_PATH, val_ratio: float = 0.05):
    """加载 DPO 偏好对数据集。"""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"DPO 数据不存在: {path}\n"
            "请先运行: python train/rl_dpo.py --generate-pairs"
        )
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"加载 {len(samples)} 个 DPO 偏好对")
    split = max(1, int(len(samples) * (1 - val_ratio)))
    return Dataset.from_list(samples[:split]), Dataset.from_list(samples[split:])


# ---------------------------------------------------------------------------
# DPO 训练（TRL DPOTrainer）
# ---------------------------------------------------------------------------

def train_dpo(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    data_path: str = DPO_DATA_PATH,
    config: dict | None = None,
):
    """
    使用 TRL DPOTrainer 进行 DPO 训练。

    TRL 的 DPOTrainer 已经实现了：
        - 参考模型 log prob 计算
        - DPO loss 实现
        - 混合精度训练
        - 梯度检查点

    显存估算（1.5B, DPO, bf16）：
        - Policy model: ~3GB
        - Reference model: ~3GB（冻结，但占显存）
        - 优化器状态: ~6GB
        - 激活值: ~2GB
        - 总计: ~14GB → 单卡 A100 40G 足够
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise ImportError("请安装 trl: pip install trl")

    cfg = {**DPO_CONFIG, **(config or {})}

    # 加载数据
    train_ds, eval_ds = load_dpo_dataset(data_path)
    print(f"训练集: {len(train_ds)} 对, 验证集: {len(eval_ds)} 对")

    # 加载模型
    resolved_model_path = resolve_model_path(model_path)
    print(f"加载模型: {model_path} -> {resolved_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 参考模型（自动由 DPOTrainer 管理）
    ref_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=cfg["beta"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=cfg["bf16"],
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        report_to=cfg["report_to"],
        loss_type="sigmoid",   # 标准 DPO loss
        label_smoothing=0.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print(f"\n开始 DPO 训练...")
    print(f"  β = {cfg['beta']}")
    print(f"  数据对数: {len(train_ds)}")
    print(f"  轮数: {cfg['num_train_epochs']}")

    trainer.train()

    model.save_pretrained(Path(output_dir) / "final")
    tokenizer.save_pretrained(Path(output_dir) / "final")
    print(f"\nDPO 训练完成: {output_dir}")
    return trainer


# ---------------------------------------------------------------------------
# 朴素 DPO 实现（理解算法原理）
# ---------------------------------------------------------------------------

def dpo_loss_manual(
    policy_model,
    ref_model,
    tokenizer,
    chosen_text: str,
    rejected_text: str,
    prompt: str,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    手动实现 DPO loss，用于理解算法原理。
    生产训练请使用 TRL DPOTrainer。

    DPO 损失推导：
        设 r(x,y) = β · log(π_θ(y|x)/π_ref(y|x))
        L = -log σ( r(x,y_c) - r(x,y_r) )
          = -log σ( β·[log π_θ(y_c|x) - log π_ref(y_c|x)]
                   - β·[log π_θ(y_r|x) - log π_ref(y_r|x)] )
    """
    def get_log_prob(model, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=2048).to(next(model.parameters()).device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(**inputs, labels=inputs["input_ids"])
        return -out.loss  # log prob

    # π_θ(y_c|x) 和 π_θ(y_r|x)
    chosen_full = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant: {chosen_text}"
    rejected_full = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant: {rejected_text}"

    policy_lp_c = get_log_prob(policy_model, chosen_full)
    policy_lp_r = get_log_prob(policy_model, rejected_full)

    # π_ref(y_c|x) 和 π_ref(y_r|x)
    with torch.no_grad():
        ref_lp_c = get_log_prob(ref_model, chosen_full)
        ref_lp_r = get_log_prob(ref_model, rejected_full)

    # DPO loss
    logits = beta * ((policy_lp_c - ref_lp_c) - (policy_lp_r - ref_lp_r))
    loss = -torch.nn.functional.logsigmoid(logits)

    return loss


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2b: DPO")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--data", default=DPO_DATA_PATH)
    parser.add_argument("--generate-pairs", action="store_true", help="先生成偏好对数据")
    parser.add_argument("--teacher-model", default=None, help="生成数据的 teacher model")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO β 参数")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.generate_pairs:
        model_fn = make_anthropic_model_fn(
            model=args.teacher_model or os.environ.get("MODEL_ID", "claude-3-5-haiku-20241022")
        )
        generate_dpo_pairs(model_fn=model_fn, output_path=args.data)

    train_dpo(
        model_path=args.model,
        output_dir=args.output,
        data_path=args.data,
        config={"beta": args.beta, "num_train_epochs": args.epochs},
    )
