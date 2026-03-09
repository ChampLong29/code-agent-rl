#!/usr/bin/env python3
"""
train/rl_grpo.py — Phase 2c: GRPO（Search-R1 核心方法）

GRPO（Group Relative Policy Optimization）是 Search-R1 和 DeepSeek-R1 使用的
强化学习算法，相比 PPO 更简单高效：不需要 Critic 网络，只用组内相对奖励估算优势。

核心思想：
    对同一个 prompt，采样 G 条轨迹（"组"），
    用组内平均奖励做 baseline，计算相对优势：
        A_i = (r_i - mean(r)) / (std(r) + ε)
    然后用 PPO-clip 损失更新策略。

优势（相比 PPO）：
    - 无 Critic：节省一半显存
    - 无 GAE：实现更简单
    - 天然适合可验证奖励（结果 0/1 奖励）

SLIME 框架集成：
    SLIME (https://github.com/PRIME-RL/SLIME) 是一个轻量级
    多轮 RL 框架，原生支持 function calling rollout + GRPO。
    本脚本提供两种模式：
        1. slime_mode=True  — 使用 SLIME 框架（推荐，GPU 利用率高）
        2. slime_mode=False — 使用 TRL GRPOTrainer（更易安装，作为备选）

算法流程：
    for each iteration:
        1. 采样：对每个 prompt，用当前策略模型采样 G 条轨迹
        2. 评分：用 reward_fn 计算每条轨迹的奖励
        3. 归一化：组内 Z-score 得到优势 A_i
        4. 优化：最大化 clip(π_θ/π_ref · A_i, 1±ε) 的期望
        5. KL 惩罚：避免策略偏离参考模型太远
"""

import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import AgentEnvironment, TaskLoader
from monitor import Monitor
from reward import RewardFn, compute_group_advantages
from rollout import RolloutSampler, make_vllm_model_fn
from scripts.generate_sft_data import SEED_TASKS


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "checkpoints/sft_merged")
OUTPUT_DIR = os.environ.get("GRPO_OUTPUT_DIR", "checkpoints/grpo")
DATA_PATH = os.environ.get("RL_DATA_PATH", "data/rl")

GRPO_CONFIG = {
    # 采样参数
    "group_size": 8,           # G：每个 prompt 采样多少条轨迹
    "max_steps_per_episode": 15,
    "rollout_temperature": 0.8,
    # 训练参数
    "num_iterations": 200,     # RL 训练迭代次数
    "prompts_per_batch": 4,    # 每次迭代使用多少个 prompt
    "learning_rate": 5e-6,     # RL 阶段学习率要小于 SFT
    "kl_coeff": 0.04,          # KL 散度惩罚系数（Search-R1 用 0.04）
    "clip_epsilon": 0.2,       # PPO clip 范围
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
    "save_steps": 20,
    # vLLM 参数
    "gpu_memory_utilization": 0.7,  # 留出空间给训练
    "rollout_workers": 2,       # 并发 rollout 线程数
}


# ---------------------------------------------------------------------------
# GRPO 损失函数
# ---------------------------------------------------------------------------

def grpo_loss(
    model,
    ref_model,
    tokenizer,
    trajectories_with_advantages: list[dict],
    kl_coeff: float = 0.04,
    clip_epsilon: float = 0.2,
) -> dict[str, torch.Tensor]:
    """
    计算 GRPO 损失。

    公式（来自 DeepSeek-R1 论文）：
        L_GRPO = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ] + β·KL(π_θ || π_ref)

        其中 r = π_θ(a|s) / π_old(a|s)（重要性采样比率）

    Args:
        model: 当前策略模型（被更新的模型）
        ref_model: 参考模型（SFT 后的模型，冻结）
        tokenizer: tokenizer
        trajectories_with_advantages: compute_group_advantages 的输出
        kl_coeff: KL 散度惩罚系数 β
        clip_epsilon: PPO clip 范围 ε

    Returns:
        {"loss": total_loss, "policy_loss": ..., "kl_loss": ..., "mean_reward": ...}
    """
    total_policy_loss = torch.tensor(0.0, requires_grad=True)
    total_kl_loss = torch.tensor(0.0)
    count = 0

    for item in trajectories_with_advantages:
        traj = item["trajectory"]
        advantage = item["advantage"]

        if not traj.messages:
            continue

        # 将轨迹的 messages 格式化为 token 序列
        # 只对 assistant 的生成部分计算损失（标准 causal LM 做法）
        text = _trajectory_to_training_text(traj, tokenizer)
        if not text:
            continue

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        # 策略模型 log prob
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**inputs, labels=inputs["input_ids"])
            log_prob = -outputs.loss  # outputs.loss 是负 log prob 的均值

        # 参考模型 log prob（用于 KL 惩罚）
        with torch.no_grad():
            ref_outputs = ref_model(**inputs, labels=inputs["input_ids"])
            ref_log_prob = -ref_outputs.loss

        # 重要性采样比率（近似，用序列级别的 log prob 差）
        ratio = torch.exp(log_prob - ref_log_prob.detach())
        ratio_clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        adv_tensor = torch.tensor(advantage, dtype=torch.float32).to(model.device)

        # Policy loss（PPO-clip）
        policy_loss = -torch.min(ratio * adv_tensor, ratio_clipped * adv_tensor)

        # KL 散度（近似：log(π_θ/π_ref)）
        kl = log_prob - ref_log_prob.detach()

        total_policy_loss = total_policy_loss + policy_loss
        total_kl_loss = total_kl_loss + kl.detach()
        count += 1

    if count == 0:
        return {"loss": torch.tensor(0.0, requires_grad=True),
                "policy_loss": torch.tensor(0.0),
                "kl_loss": torch.tensor(0.0),
                "mean_reward": 0.0}

    mean_policy_loss = total_policy_loss / count
    mean_kl = total_kl_loss / count
    total_loss = mean_policy_loss + kl_coeff * mean_kl

    mean_reward = sum(item["raw_reward"] for item in trajectories_with_advantages) / len(trajectories_with_advantages)

    return {
        "loss": total_loss,
        "policy_loss": mean_policy_loss.detach(),
        "kl_loss": mean_kl,
        "mean_reward": mean_reward,
    }


def _trajectory_to_training_text(traj, tokenizer) -> str:
    """将轨迹转换为训练文本。"""
    conversations = []
    for msg in traj.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # tool_result 格式
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    parts.append(f"[Tool Result]\n{item.get('content', '')}")
                elif hasattr(item, "text"):
                    parts.append(item.text)
            content = "\n".join(parts)
        elif not isinstance(content, str):
            content = str(content)

        if role in ("user", "assistant", "system"):
            conversations.append({"role": role, "content": content})

    if not conversations:
        return ""

    try:
        return tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# TRL GRPOTrainer 模式（备选，无需 SLIME）
# ---------------------------------------------------------------------------

def train_grpo_trl(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
):
    """
    使用 TRL GRPOTrainer 进行训练。
    适合快速实验，不需要 SLIME 环境。

    TRL >= 0.12.0 开始支持 GRPOTrainer。
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError("请安装 trl >= 0.12.0: pip install 'trl>=0.12.0'")

    cfg = {**GRPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS

    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 将 tasks 转换为 TRL 期望的 Dataset 格式
    # TRL GRPO 期望: {"prompt": str}
    prompts = [{"prompt": task["prompt"], "_task": json.dumps(task, ensure_ascii=False)}
               for task in tasks]
    dataset = Dataset.from_list(prompts * (cfg["num_iterations"] // len(tasks) + 1))

    reward_fn = RewardFn()
    sampler = RolloutSampler(max_steps=cfg["max_steps_per_episode"])

    # TRL GRPO 的奖励函数签名：reward_fn(prompts, completions) -> list[float]
    def trl_reward_fn(prompts_batch: list[str], completions_batch: list[str]) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts_batch, completions_batch):
            # 找到对应任务
            task = next((t for t in tasks if t["prompt"] == prompt), tasks[0])
            # 用 completion 作为最终回答构造 trajectory
            from environment import Trajectory, ToolCall
            traj = Trajectory(task_id=task["id"], prompt=prompt)
            traj.final_answer = completion
            # 简单的格式奖励（TRL 模式下无法运行真实工具）
            scores = reward_fn(traj)
            rewards.append(scores["format"])  # TRL 模式只用格式奖励
        return rewards

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=cfg["prompts_per_batch"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_generations=cfg["group_size"],
        max_new_tokens=512,
        bf16=True,
        logging_steps=5,
        save_steps=cfg["save_steps"],
        report_to="none",
        kl_coef=cfg["kl_coeff"],
        cliprange=cfg["clip_epsilon"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trl_reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print(f"\n开始 TRL GRPO 训练（{cfg['num_iterations']} 次迭代）...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存: {output_dir}")


# ---------------------------------------------------------------------------
# 自定义 GRPO 训练循环（完整工具调用支持）
# ---------------------------------------------------------------------------

def train_grpo_custom(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
):
    """
    自定义 GRPO 训练循环。
    完整支持工具调用 rollout + 可验证奖励。

    相比 TRL 模式：
        - 真实执行工具调用，奖励信号更准确
        - 实现更透明，方便 debug
        - 性能稍差（无 vLLM 批处理优化）

    流程：
        for each iteration:
            1. 随机采样 prompts_per_batch 个任务
            2. 每个任务用 vLLM 采样 group_size 条轨迹
            3. 计算组内优势
            4. 用 GRPO loss 更新模型
    """
    cfg = {**GRPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"加载策略模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 策略模型（被训练）
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    policy_model.train()

    # 参考模型（冻结，用于 KL 惩罚）
    print("加载参考模型（冻结）...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )

    reward_fn = RewardFn()
    sampler = RolloutSampler(
        max_steps=cfg["max_steps_per_episode"],
        save_dir=DATA_PATH,
    )

    # rollout 使用 HF 模型（避免 vLLM 显存冲突，生产环境建议分离）
    from rollout import make_hf_model_fn
    model_fn = make_hf_model_fn(model_path, temperature=cfg["rollout_temperature"])

    print(f"\n开始自定义 GRPO 训练")
    print(f"  迭代次数: {cfg['num_iterations']}")
    print(f"  组大小 G: {cfg['group_size']}")
    print(f"  每批 prompts: {cfg['prompts_per_batch']}")
    print(f"  KL 系数 β: {cfg['kl_coeff']}")

    import random
    log_path = output_path / "training_log.jsonl"

    monitor = Monitor(
        project=os.environ.get("WANDB_PROJECT", "code-agent-rl"),
        run_name=f"grpo-{Path(output_dir).name}",
        config=cfg,
    )

    for iteration in range(cfg["num_iterations"]):
        # 随机采样任务
        batch_tasks = random.sample(tasks, min(cfg["prompts_per_batch"], len(tasks)))

        # 采样轨迹
        batch_rollouts = sampler.sample_batch(
            batch_tasks, model_fn,
            group_size=cfg["group_size"],
            max_workers=cfg["rollout_workers"],
        )

        # 计算组内优势
        all_items = []
        for task in batch_tasks:
            trajs = batch_rollouts.get(task["id"], [])
            if not trajs:
                continue
            items = compute_group_advantages(trajs, reward_fn)
            all_items.extend(items)

        if not all_items:
            print(f"  [iter {iteration}] 无有效轨迹，跳过")
            continue

        # 累积梯度
        optimizer.zero_grad()
        grad_acc = cfg["gradient_accumulation_steps"]
        chunk_size = max(1, len(all_items) // grad_acc)

        iter_loss = 0.0
        iter_reward = 0.0

        for chunk_start in range(0, len(all_items), chunk_size):
            chunk = all_items[chunk_start:chunk_start + chunk_size]
            loss_dict = grpo_loss(
                policy_model, ref_model, tokenizer, chunk,
                kl_coeff=cfg["kl_coeff"],
                clip_epsilon=cfg["clip_epsilon"],
            )
            loss = loss_dict["loss"] / grad_acc
            if loss.requires_grad:
                loss.backward()
            iter_loss += loss_dict["loss"].item()
            iter_reward += loss_dict["mean_reward"]

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg["max_grad_norm"])
        optimizer.step()

        iter_reward /= max(1, len(all_items) // chunk_size)

        # 日志
        log_entry = {
            "iteration": iteration,
            "loss": iter_loss,
            "mean_reward": iter_reward,
            "n_trajectories": len(all_items),
            "success_rate": sum(1 for item in all_items
                                if item["trajectory"].status == "success") / len(all_items),
        }
        print(f"  [iter {iteration:3d}] loss={iter_loss:.4f}, "
              f"reward={iter_reward:.3f}, "
              f"success={log_entry['success_rate']:.1%}")

        monitor.log(log_entry, step=iteration)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # 保存检查点
        if (iteration + 1) % cfg["save_steps"] == 0:
            ckpt_path = output_path / f"checkpoint-{iteration+1}"
            policy_model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  检查点已保存: {ckpt_path}")

    # 最终保存
    final_path = output_path / "final"
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    monitor.summary("final_model_path", str(final_path))
    monitor.finish()
    print(f"\nGRPO 训练完成，最终模型: {final_path}")


# ---------------------------------------------------------------------------
# SLIME 框架集成（推荐，生产级）
# ---------------------------------------------------------------------------

def train_grpo_slime(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
):
    """
    使用 SLIME 框架进行 GRPO 训练。
    SLIME 提供高效的异步 rollout + vLLM 推理，
    GPU 利用率远高于朴素实现。

    安装 SLIME：
        git clone https://github.com/PRIME-RL/SLIME
        pip install -e SLIME/

    SLIME 的核心优势：
        - Actor（推理）和 Trainer（训练）分离
        - vLLM 做批量 rollout，利用率接近 100%
        - 支持多机多卡
    """
    try:
        import slime
    except ImportError:
        print("SLIME 未安装，回退到自定义 GRPO 模式")
        print("安装 SLIME: git clone https://github.com/PRIME-RL/SLIME && pip install -e SLIME/")
        return train_grpo_custom(model_path, output_dir, tasks, config)

    cfg = {**GRPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS
    reward_fn = RewardFn()

    # SLIME 期望的 rollout 函数签名
    def slime_rollout_fn(prompt: str, model_output: str, metadata: dict) -> float:
        """
        SLIME 回调：给定 prompt 和模型输出，返回奖励分数。
        在 SLIME 的 Actor 进程中被调用。
        """
        task = next((t for t in tasks if t["prompt"] == prompt), None)
        if task is None:
            return 0.0

        from environment import Trajectory, ToolCall
        traj = Trajectory(task_id=task.get("id", "unknown"), prompt=prompt)
        traj.final_answer = model_output
        traj.status = "success" if "passed" in model_output else "failure"

        scores = reward_fn(traj)
        return scores["total"]

    # SLIME 配置（参考 SLIME 文档）
    slime_config = slime.GRPOConfig(
        model_path=model_path,
        output_dir=output_dir,
        num_iterations=cfg["num_iterations"],
        group_size=cfg["group_size"],
        learning_rate=cfg["learning_rate"],
        kl_coeff=cfg["kl_coeff"],
        clip_epsilon=cfg["clip_epsilon"],
        rollout_temperature=cfg["rollout_temperature"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        reward_fn=slime_rollout_fn,
        prompts=[task["prompt"] for task in tasks],
    )

    trainer = slime.GRPOTrainer(slime_config)
    trainer.train()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2c: GRPO (Search-R1 风格)")
    parser.add_argument("--model", default=BASE_MODEL, help="模型路径（SFT 后的）")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--mode", choices=["trl", "custom", "slime"], default="custom",
                        help="训练模式")
    parser.add_argument("--iterations", type=int, default=200, help="训练迭代次数")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO 组大小 G")
    parser.add_argument("--kl-coeff", type=float, default=0.04, help="KL 惩罚系数")
    args = parser.parse_args()

    extra_config = {
        "num_iterations": args.iterations,
        "group_size": args.group_size,
        "kl_coeff": args.kl_coeff,
    }

    if args.mode == "trl":
        train_grpo_trl(args.model, args.output, config=extra_config)
    elif args.mode == "slime":
        train_grpo_slime(args.model, args.output, config=extra_config)
    else:
        train_grpo_custom(args.model, args.output, config=extra_config)
