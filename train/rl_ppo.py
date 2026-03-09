#!/usr/bin/env python3
"""
train/rl_ppo.py — Phase 2a: PPO（Proximal Policy Optimization）

PPO 是 RLHF 的经典算法（InstructGPT、ChatGPT 都用它）。
相比 GRPO，PPO 需要一个 Critic（价值网络），训练更稳定但成本更高。

架构：
    ┌──────────────┐    rollout    ┌─────────────────────┐
    │  Actor (π_θ) │ ──────────►  │  AgentEnvironment   │
    │  Qwen 1.5B   │              │  工具调用 + 奖励     │
    └──────┬───────┘              └──────────┬──────────┘
           │                                 │
           │ ◄──── experiences (s,a,r,v) ────┘
           │
    ┌──────▼───────────────────┐
    │  Critic (V_φ)            │
    │  共享 backbone，额外      │
    │  value head               │
    └──────────────────────────┘

PPO 损失：
    L_policy = -min( r_t·A_t, clip(r_t, 1-ε, 1+ε)·A_t )
    L_value  = (V_φ(s) - V_target)²
    L_entropy = -H(π_θ)
    L_total   = L_policy + c1·L_value - c2·L_entropy

GAE（广义优势估计）：
    δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    A_t = Σ (γλ)^k · δ_{t+k}
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import AgentEnvironment
from reward import RewardFn
from rollout import RolloutSampler, make_hf_model_fn
from scripts.generate_sft_data import SEED_TASKS


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "checkpoints/sft_merged")
OUTPUT_DIR = os.environ.get("PPO_OUTPUT_DIR", "checkpoints/ppo")

PPO_CONFIG = {
    # 采样参数
    "max_steps_per_episode": 15,
    "rollout_temperature": 0.7,
    "episodes_per_update": 16,   # 每次更新收集多少条轨迹
    # 训练参数
    "num_iterations": 300,
    "learning_rate": 1e-5,
    "ppo_epochs": 4,             # 每批数据做几遍 PPO 更新
    "clip_epsilon": 0.2,
    "value_coeff": 0.5,          # Critic loss 系数 c1
    "entropy_coeff": 0.01,       # 熵正则系数 c2
    "gamma": 0.99,               # 折扣因子
    "gae_lambda": 0.95,          # GAE lambda
    "max_grad_norm": 1.0,
    "kl_target": 0.02,           # 自适应 KL 目标
    "save_steps": 30,
}


# ---------------------------------------------------------------------------
# Critic（价值网络）
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """
    在语言模型 backbone 之上添加价值头。
    输入：hidden states（最后一层）
    输出：标量价值估计 V(s)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取序列最后一个 token 的 hidden state 作为 state 表示
        last_hidden = hidden_states[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic 模型：共享 backbone，分开 policy head 和 value head。
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
            output_hidden_states=True,
        )
        hidden_size = self.base.config.hidden_size
        self.value_head = ValueHead(hidden_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        value = self.value_head(last_hidden)
        return outputs, value

    def get_log_prob(self, input_ids, attention_mask=None):
        """计算序列的 log probability。"""
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return -outputs.loss  # NLL → log prob


# ---------------------------------------------------------------------------
# GAE 优势估计
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    """
    计算 GAE（广义优势估计）和 Returns。

    Args:
        rewards: 每步的奖励
        values: Critic 估计的价值
        dones: 是否 episode 结束
        gamma: 折扣因子
        gae_lambda: GAE lambda

    Returns:
        (advantages, returns)
    """
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n

    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        if dones[t]:
            next_value = 0.0
            gae = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    # 标准化优势
    adv_mean = sum(advantages) / len(advantages)
    adv_std = (sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)) ** 0.5
    advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in advantages]

    return advantages, returns


# ---------------------------------------------------------------------------
# PPO 经验缓冲
# ---------------------------------------------------------------------------

class PPOBuffer:
    """存储一批 rollout 经验。"""

    def __init__(self):
        self.trajectories = []
        self.advantages = []
        self.returns = []
        self.log_probs_old = []

    def add(self, trajectory, advantage: float, return_: float, log_prob_old: float):
        self.trajectories.append(trajectory)
        self.advantages.append(advantage)
        self.returns.append(return_)
        self.log_probs_old.append(log_prob_old)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.trajectories)


# ---------------------------------------------------------------------------
# PPO 训练循环
# ---------------------------------------------------------------------------

def train_ppo(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
):
    """
    PPO 训练主函数。

    注意：PPO 需要 Actor-Critic 架构，显存需求比 GRPO 高约 1.5 倍。
    1.5B 模型建议至少 2×A100 40G。
    """
    cfg = {**PPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"加载 Actor-Critic 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_critic = ActorCritic(model_path)

    # 参考模型（冻结，用于 KL 约束）
    from transformers import AutoModelForCausalLM as LM
    ref_model = LM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=cfg["learning_rate"])
    reward_fn = RewardFn()
    sampler = RolloutSampler(max_steps=cfg["max_steps_per_episode"])
    model_fn = make_hf_model_fn(model_path, temperature=cfg["rollout_temperature"])

    print(f"\n开始 PPO 训练")
    print(f"  迭代: {cfg['num_iterations']}")
    print(f"  每次更新轨迹数: {cfg['episodes_per_update']}")
    log_path = output_path / "training_log.jsonl"

    import random
    buffer = PPOBuffer()

    for iteration in range(cfg["num_iterations"]):
        buffer.clear()
        actor_critic.eval()

        # Phase 1: 收集轨迹
        batch_tasks = random.choices(tasks, k=cfg["episodes_per_update"])
        rewards_this_iter = []

        for task in batch_tasks:
            traj = sampler.sample_one(task, model_fn)
            scores = reward_fn(traj)
            r = scores["total"]
            rewards_this_iter.append(r)

            # 简化：单步 episode，advantage = reward（无 bootstrap）
            buffer.add(traj, advantage=r, return_=r, log_prob_old=0.0)

        mean_reward = sum(rewards_this_iter) / len(rewards_this_iter)

        # Phase 2: PPO 更新
        actor_critic.train()
        total_loss = 0.0

        for _ in range(cfg["ppo_epochs"]):
            for item_idx in range(len(buffer)):
                traj = buffer.trajectories[item_idx]
                advantage = buffer.advantages[item_idx]
                return_ = buffer.returns[item_idx]

                if not traj.messages:
                    continue

                text = _traj_to_text(traj, tokenizer)
                if not text:
                    continue

                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048
                ).to(next(actor_critic.parameters()).device)

                outputs, value = actor_critic(
                    inputs["input_ids"], inputs.get("attention_mask"),
                    labels=inputs["input_ids"]
                )

                # Policy loss
                log_prob = -outputs.loss
                ref_outputs = ref_model(**inputs, labels=inputs["input_ids"])
                ref_log_prob = -ref_outputs.loss.detach()
                ratio = torch.exp(log_prob - ref_log_prob)
                ratio_clipped = torch.clamp(ratio, 1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"])
                adv = torch.tensor(advantage).to(value.device)
                policy_loss = -torch.min(ratio * adv, ratio_clipped * adv)

                # Value loss
                return_tensor = torch.tensor(return_, dtype=torch.float32).to(value.device)
                value_loss = cfg["value_coeff"] * (value.mean() - return_tensor) ** 2

                # Entropy（近似）
                entropy_loss = -cfg["entropy_coeff"] * log_prob

                loss = policy_loss + value_loss + entropy_loss
                optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg["max_grad_norm"])
                    optimizer.step()
                total_loss += loss.item() if not torch.isnan(loss) else 0.0

        log_entry = {
            "iteration": iteration,
            "loss": total_loss / max(1, len(buffer) * cfg["ppo_epochs"]),
            "mean_reward": mean_reward,
            "success_rate": sum(1 for t in buffer.trajectories if t.status == "success") / len(buffer),
        }
        print(f"  [iter {iteration:3d}] loss={log_entry['loss']:.4f}, "
              f"reward={mean_reward:.3f}, success={log_entry['success_rate']:.1%}")

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if (iteration + 1) % cfg["save_steps"] == 0:
            ckpt = output_path / f"checkpoint-{iteration+1}"
            actor_critic.base.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  检查点: {ckpt}")

    final = output_path / "final"
    actor_critic.base.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"\nPPO 训练完成: {final}")


def _traj_to_text(traj, tokenizer) -> str:
    conversations = []
    for msg in traj.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [
                item.get("content", "") if isinstance(item, dict) else str(item)
                for item in content
            ]
            content = "\n".join(parts)
        if role in ("user", "assistant", "system") and content:
            conversations.append({"role": role, "content": str(content)})
    if not conversations:
        return ""
    try:
        return tokenizer.apply_chat_template(conversations, tokenize=False)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# TRL PPOTrainer 备选
# ---------------------------------------------------------------------------

def train_ppo_trl(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
):
    """
    使用 TRL PPOTrainer（更成熟的实现，推荐生产使用）。
    """
    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except ImportError:
        raise ImportError("请安装 trl: pip install trl")

    cfg = {**PPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    ppo_config = PPOConfig(
        output_dir=output_dir,
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["episodes_per_update"],
        mini_batch_size=4,
        ppo_epochs=cfg["ppo_epochs"],
        gamma=cfg["gamma"],
        lam=cfg["gae_lambda"],
        cliprange=cfg["clip_epsilon"],
        vf_coef=cfg["value_coeff"],
        report_to="none",
    )

    reward_fn = RewardFn()
    sampler = RolloutSampler(max_steps=cfg["max_steps_per_episode"])

    ppo_trainer = PPOTrainer(
        config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer
    )

    import random
    print(f"\n开始 TRL PPO 训练 ({cfg['num_iterations']} 迭代)...")

    for iteration in range(cfg["num_iterations"]):
        batch_tasks = random.choices(tasks, k=cfg["episodes_per_update"])

        queries, responses, rewards = [], [], []
        model_fn = make_hf_model_fn(model_path)

        for task in batch_tasks:
            traj = sampler.sample_one(task, model_fn)
            scores = reward_fn(traj)

            prompt_enc = tokenizer(task["prompt"], return_tensors="pt")["input_ids"][0]
            response_enc = tokenizer(traj.final_answer or "(empty)", return_tensors="pt")["input_ids"][0]
            reward_tensor = torch.tensor(scores["total"])

            queries.append(prompt_enc)
            responses.append(response_enc)
            rewards.append(reward_tensor)

        stats = ppo_trainer.step(queries, responses, rewards)
        mean_reward = float(torch.stack(rewards).mean())
        print(f"  [iter {iteration:3d}] reward={mean_reward:.3f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"PPO 训练完成: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2a: PPO")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=["custom", "trl"], default="trl")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    extra = {"num_iterations": args.iterations}
    if args.mode == "trl":
        train_ppo_trl(args.model, args.output, config=extra)
    else:
        train_ppo(args.model, args.output, config=extra)
