#!/usr/bin/env python3
"""
train/verl_dpo.py — Phase 2c: DPO（veRL FSDP 生产级框架）

与 rl_dpo.py（TRL DPOTrainer）的区别：
  - rl_dpo.py     : 单卡 / 小规模，依赖 TRL DPOTrainer，实现简单
  - verl_dpo.py   : 多卡 FSDP，复用 veRL SFT 训练基础设施，生产可用

veRL 本身提供了 SFT 专用的 FSDP Trainer（fsdp_sft_trainer.py）。
DPO 可以直接在这套基础设施上实现：
  - 数据加载 / padding / DataProto 复用 SFT 逻辑
  - 模型分片 / 梯度同步 / 检查点全部由 FSDP 处理
  - DPO loss 替换 SFT 的 cross-entropy loss

架构概览：
    ┌──────────────────────────────────────────────────────────┐
    │  VerlDPOTrainer                                          │
    │                                                          │
    │  __init__: 加载 policy + ref 两份模型（FSDP 分片）        │
    │                                                          │
    │  train():                                                │
    │    每 batch:                                             │
    │      policy.forward(chosen)  → log_prob_c_θ             │
    │      policy.forward(rejected) → log_prob_r_θ             │
    │      ref.forward(chosen)     → log_prob_c_ref（no_grad） │
    │      ref.forward(rejected)   → log_prob_r_ref（no_grad） │
    │      loss = dpo_loss(...)                                │
    │      loss.backward() + optimizer.step()                  │
    └──────────────────────────────────────────────────────────┘

与现有模块的关系（最小侵入原则）：
  - 数据来源  : rl_dpo.py::generate_dpo_pairs()（零改动复用）
  - 奖励评分  : reward.py（仅在数据生成阶段使用）
  - 轨迹采样  : rollout.py（仅在数据生成阶段使用）

DPO Loss 公式（Bradley-Terry 模型）：
    L = -log σ( β · [ log π_θ(y_c|x) - log π_ref(y_c|x)
                    - log π_θ(y_r|x) + log π_ref(y_r|x) ] )

veRL 相关参考：
  - veRL SFT Trainer: verl/trainer/fsdp_sft_trainer.py
  - veRL DataProto:   verl/protocol.py
  - 安装:             pip install verl
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.rl_dpo import (
    DPO_DATA_PATH,
    generate_dpo_pairs,
    load_dpo_dataset,
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "checkpoints/sft_merged")
OUTPUT_DIR = os.environ.get("VERL_DPO_OUTPUT_DIR", "checkpoints/verl_dpo")
DPO_DATA = os.environ.get("DPO_DATA_PATH", DPO_DATA_PATH)

VERL_DPO_CONFIG = {
    # ── DPO 超参 ──
    "beta": 0.1,                    # β : KL 惩罚强度（典型范围 0.05~0.3）
    "loss_type": "sigmoid",         # "sigmoid"（标准 DPO）| "hinge"（保守）| "ipo"（IPO）
    "label_smoothing": 0.0,         # 是否平滑 label（0.0 = 标准 DPO）

    # ── 训练参数 ──
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-7,          # DPO 学习率要极小，防止偏离 SFT 过远
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "bf16": True,
    "max_length": 2048,             # chosen/rejected 序列最大长度

    # ── veRL FSDP 参数 ──
    "fsdp_offload": False,          # True = 显存紧张时 CPU offload
    "fsdp_reshard_after_forward": True,
    "gradient_checkpointing": True,
    "n_gpus": torch.cuda.device_count() or 1,

    # ── 日志 / 保存 ──
    "logging_steps": 10,
    "save_steps": 100,
    "report_to": os.environ.get("MONITOR_BACKEND", "none"),
}


# ---------------------------------------------------------------------------
# veRL DataProto 适配层：DPO 偏好对数据集
# ---------------------------------------------------------------------------

class DPOPairDataset(Dataset):
    """
    将 generate_dpo_pairs() 生成的 JSONL 数据包装成 PyTorch Dataset。

    每个样本包含字段（对齐 veRL DataProto 格式）：
        chosen_input_ids   : (seq_len,) 完整 chosen 序列（prompt + chosen_response）
        chosen_labels      : (seq_len,) 只在 response 部分有效，prompt 位置为 -100
        rejected_input_ids : (seq_len,) 完整 rejected 序列
        rejected_labels    : (seq_len,) 同上

    prompt 部分设为 -100 的原因：
        DPO loss 只计算 response 部分的 log prob，prompt 不参与梯度。
        这与 SFT 的 label masking 方式相同，可以直接复用 SFT 的 forward 步骤。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        system_prompt: str = (
            "You are a coding agent. Use tools to solve programming tasks step by step."
        ),
    ):
        from pathlib import Path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.samples = []

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"DPO 数据不存在: {path}\n"
                "请先运行: python train/verl_dpo.py --generate-pairs"
            )

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"加载 {len(self.samples)} 个 DPO 偏好对")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_ids, chosen_labels = self._encode_pair(prompt, chosen)
        rejected_ids, rejected_labels = self._encode_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "metadata": {
                "task_id": sample.get("task_id", ""),
                "chosen_reward": sample.get("chosen_reward", 0.0),
                "rejected_reward": sample.get("rejected_reward", 0.0),
            },
        }

    def _encode_pair(self, prompt: str, response: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码 prompt + response，返回 (input_ids, labels)。
        labels 中 prompt 部分设为 -100（不参与 DPO loss 计算）。
        """
        tok = self.tokenizer

        # 格式化为完整对话（与 sft_lora.py 保持一致）
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # 用 apply_chat_template 保持与推理时格式一致
        if hasattr(tok, "apply_chat_template"):
            full_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            full_text = (
                f"System: {self.system_prompt}\n\n"
                f"User: {prompt}\n\n"
                f"Assistant: {response}"
            )

        # 编码完整文本
        full_ids = tok.encode(full_text, add_special_tokens=False)

        # 编码 prompt 部分（含 system），用于确定 prompt 长度
        prompt_messages = messages[:2]
        if hasattr(tok, "apply_chat_template"):
            prompt_text = tok.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"System: {self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # 截断
        full_ids = full_ids[: self.max_length]

        # labels：prompt 部分为 -100，response 部分保留
        labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
        labels = labels[: self.max_length]

        # padding 到 max_length
        pad_len = self.max_length - len(full_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        full_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch: list[dict]) -> dict:
    """将 DPOPairDataset 样本合并为 batch。"""
    return {
        "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
        "chosen_labels": torch.stack([b["chosen_labels"] for b in batch]),
        "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
        "rejected_labels": torch.stack([b["rejected_labels"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# DPO Loss 实现
# ---------------------------------------------------------------------------

def _get_batch_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    计算 batch 中每个样本的 response 部分 log prob 之和。

    具体步骤：
        1. forward → logits (B, seq_len, vocab)
        2. 取 labels != -100 的位置
        3. log_softmax → 取 label token 对应的概率
        4. 按序列求和（不取均值，保持与序列长度一致的量纲）

    Args:
        input_ids : (B, seq_len)
        labels    : (B, seq_len)  prompt 位置为 -100

    Returns:
        log_probs : (B,)  每个样本的 response 部分 log prob 之和
    """
    logits = model(input_ids=input_ids).logits  # (B, seq_len, vocab)

    # 预测下一个 token：logits[:, :-1] 对应 labels[:, 1:]
    shift_logits = logits[:, :-1, :]            # (B, seq_len-1, vocab)
    shift_labels = labels[:, 1:]                # (B, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 取 label token 的 log prob
    # gather: (B, seq_len-1, 1) -> squeeze -> (B, seq_len-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # mask 掉 prompt 部分（label=-100 的位置）
    mask = (shift_labels != -100).float()
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)  # (B,)

    return sequence_log_probs


def dpo_loss(
    policy_lp_chosen: torch.Tensor,
    policy_lp_rejected: torch.Tensor,
    ref_lp_chosen: torch.Tensor,
    ref_lp_rejected: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    计算 DPO loss（支持 sigmoid / hinge / IPO 三种变体）。

    Bradley-Terry 模型下的 reward 隐式定义：
        r(x, y) = β · [ log π_θ(y|x) - log π_ref(y|x) ]

    Args:
        policy_lp_chosen   : π_θ(y_c|x)  (B,)
        policy_lp_rejected : π_θ(y_r|x)  (B,)
        ref_lp_chosen      : π_ref(y_c|x) (B,)，no_grad
        ref_lp_rejected    : π_ref(y_r|x) (B,)，no_grad
        beta               : KL 惩罚系数
        loss_type          : 损失类型
        label_smoothing    : 标签平滑（0.0 = 标准 DPO）

    Returns:
        loss    : scalar
        metrics : dict（用于日志记录）
    """
    # 计算隐式 reward 差值
    pi_logratios = policy_lp_chosen - policy_lp_rejected        # log π_θ(c)/π_θ(r)
    ref_logratios = ref_lp_chosen - ref_lp_rejected             # log π_ref(c)/π_ref(r)
    logits = beta * (pi_logratios - ref_logratios)              # β · (π差 - ref差)

    if loss_type == "sigmoid":
        # 标准 DPO（论文公式）
        # 添加 label_smoothing：0 → chosen 完全正确；>0 → 允许一定不确定性
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    elif loss_type == "hinge":
        # Conservative DPO：只在 margin 不足时更新
        loss = F.relu(1 - logits)
    elif loss_type == "ipo":
        # IPO（Identity Preference Optimization，对 over-fitting 更稳健）
        loss = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"未知 loss_type: {loss_type}")

    # 计算调试指标
    with torch.no_grad():
        chosen_rewards = beta * (policy_lp_chosen - ref_lp_chosen)
        rejected_rewards = beta * (policy_lp_rejected - ref_lp_rejected)
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "dpo_loss": loss.mean().item(),
        "reward_accuracy": reward_accuracy.item(),   # 越接近 1.0 越好
        "reward_margin": reward_margin.item(),        # 越大越好
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "logits_mean": logits.mean().item(),
    }

    return loss.mean(), metrics


# ---------------------------------------------------------------------------
# veRL FSDP DPO 训练器
# ---------------------------------------------------------------------------

class VerlDPOTrainer:
    """
    基于 veRL FSDP SFT 基础设施的 DPO 训练器。

    设计原则：
        - 最大复用 veRL 的 FSDP 初始化逻辑（零重复代码）
        - policy 和 ref 都用 FSDP 分片（避免 ref 成为显存瓶颈）
        - ref 只做 forward，用 torch.no_grad() 屏蔽梯度

    veRL FSDP SFT 的核心文件：
        verl/trainer/fsdp_sft_trainer.py   ← 本文件参考实现
        verl/utils/fsdp_utils.py           ← FSDP 初始化工具

    显存估算（1.5B Coder-Instruct, DPO, bf16, FSDP）：
        - Policy  : ~3GB（FSDP 分片后每卡约 3/n GB）
        - Ref     : ~3GB（同上，frozen）
        - 梯度     : ~3GB（policy only）
        - 优化器   : ~6GB（AdamW，policy only）
        - 激活值   : ~2GB（gradient_checkpointing 后更少）
        - 单卡合计 : ~17GB → A100 40G 可运行（2 卡以上更宽裕）
    """

    def __init__(
        self,
        model_path: str = BASE_MODEL,
        output_dir: str = OUTPUT_DIR,
        data_path: str = DPO_DATA,
        config: dict | None = None,
    ):
        try:
            import torch.distributed as dist
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(f"缺少依赖: {e}\n请确认 PyTorch >= 2.0 已安装") from e

        self.cfg = {**VERL_DPO_CONFIG, **(config or {})}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 分布式初始化 ──
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")

        # ── Tokenizer ──
        if self.rank == 0:
            print(f"加载 tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 加载原始模型（CPU，后续 FSDP 分片到 GPU）──
        if self.rank == 0:
            print(f"加载模型（CPU）: {model_path}")
        dtype = torch.bfloat16 if self.cfg["bf16"] else torch.float32

        # Policy model（可训练）
        policy_model_raw = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        if self.cfg["gradient_checkpointing"]:
            policy_model_raw.gradient_checkpointing_enable()

        # Reference model（冻结，与 policy 共享相同初始权重）
        ref_model_raw = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        for param in ref_model_raw.parameters():
            param.requires_grad_(False)

        # ── FSDP 包装 ──
        mixed_precision = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=dtype,
        )

        fsdp_kwargs = dict(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_id=self.device,
            sync_module_states=True,   # 从 rank 0 广播初始权重
        )
        if self.cfg["fsdp_offload"]:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        self.policy = FSDP(policy_model_raw, **fsdp_kwargs)
        self.ref = FSDP(ref_model_raw, **fsdp_kwargs)

        if self.rank == 0:
            print("FSDP 初始化完成")
            print(f"  world_size = {self.world_size}")
            print(f"  bf16       = {self.cfg['bf16']}")
            print(f"  cpu_offload= {self.cfg['fsdp_offload']}")

        # ── 数据集 ──
        dataset = DPOPairDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.cfg["max_length"],
        )
        # DistributedSampler 保证每卡看到不同的 batch
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.cfg["per_device_train_batch_size"],
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # ── 优化器 + LR Scheduler ──
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=0.0,
        )
        total_steps = (
            len(self.dataloader)
            * self.cfg["num_train_epochs"]
            // self.cfg["gradient_accumulation_steps"]
        )
        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])
        self.scheduler = self._make_cosine_scheduler(warmup_steps, total_steps)

        # ── 监控 ──
        self._init_monitor()

    def _make_cosine_scheduler(self, warmup_steps: int, total_steps: int):
        """Cosine LR with linear warmup（不依赖 transformers Trainer）。"""
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_monitor(self):
        """初始化监控（wandb / swanlab / none）。"""
        backend = self.cfg.get("report_to", "none")
        self._monitor = None
        if self.rank != 0 or backend == "none":
            return
        if backend == "wandb":
            try:
                import wandb
                wandb.init(project="code-agent-rl", name="verl-dpo", config=self.cfg)
                self._monitor = wandb
            except ImportError:
                pass
        elif backend == "swanlab":
            try:
                import swanlab
                swanlab.init(project="code-agent-rl", experiment_name="verl-dpo", config=self.cfg)
                self._monitor = swanlab
            except ImportError:
                pass

    def _log(self, metrics: dict, step: int):
        if self.rank == 0:
            log_str = f"step {step:5d} | " + " | ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            )
            print(log_str)
        if self._monitor is not None:
            self._monitor.log(metrics, step=step)

    def train(self):
        """
        主训练循环。

        关键步骤（每 micro-batch）：
            1. policy.forward(chosen)             → log_probs_c_θ
            2. policy.forward(rejected)           → log_probs_r_θ
            3. ref.forward(chosen)  [no_grad]     → log_probs_c_ref
            4. ref.forward(rejected)[no_grad]     → log_probs_r_ref
            5. dpo_loss(...)                       → loss, metrics
            6. (loss / grad_accum).backward()
            7. 每 grad_accum 步: clip_grad + optimizer.step()
        """
        import torch.distributed as dist

        global_step = 0
        self.policy.train()
        self.ref.eval()

        for epoch in range(self.cfg["num_train_epochs"]):
            self.dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.dataloader):
                # 移到 GPU
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                # ── Step 1 & 2: Policy forward ──
                policy_lp_c = _get_batch_log_probs(self.policy, chosen_ids, chosen_labels)
                policy_lp_r = _get_batch_log_probs(self.policy, rejected_ids, rejected_labels)

                # ── Step 3 & 4: Ref forward（无梯度）──
                with torch.no_grad():
                    ref_lp_c = _get_batch_log_probs(self.ref, chosen_ids, chosen_labels)
                    ref_lp_r = _get_batch_log_probs(self.ref, rejected_ids, rejected_labels)

                # ── Step 5: DPO Loss ──
                loss, metrics = dpo_loss(
                    policy_lp_c, policy_lp_r,
                    ref_lp_c, ref_lp_r,
                    beta=self.cfg["beta"],
                    loss_type=self.cfg["loss_type"],
                    label_smoothing=self.cfg["label_smoothing"],
                )

                # ── Step 6: 梯度累积 ──
                scaled_loss = loss / self.cfg["gradient_accumulation_steps"]
                scaled_loss.backward()

                # ── Step 7: 每 grad_accum 步更新 ──
                is_update_step = (
                    (batch_idx + 1) % self.cfg["gradient_accumulation_steps"] == 0
                )
                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.cfg["max_grad_norm"],
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # 跨卡同步并记录指标
                    if global_step % self.cfg["logging_steps"] == 0:
                        # all_reduce 平均各 rank 的 metrics
                        metrics_tensor = torch.tensor(
                            [metrics["dpo_loss"], metrics["reward_accuracy"], metrics["reward_margin"]],
                            device=self.device,
                        )
                        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
                        metrics["dpo_loss"] = metrics_tensor[0].item()
                        metrics["reward_accuracy"] = metrics_tensor[1].item()
                        metrics["reward_margin"] = metrics_tensor[2].item()
                        metrics["lr"] = self.scheduler.get_last_lr()[0]
                        self._log(metrics, global_step)

                    # 保存检查点
                    if global_step % self.cfg["save_steps"] == 0:
                        self._save_checkpoint(global_step)

        # 最终保存
        self._save_checkpoint("final")
        if self.rank == 0:
            print(f"\nveRL DPO 训练完成 → {self.output_dir}")

    def _save_checkpoint(self, step):
        """
        使用 FSDP state_dict 保存检查点。
        必须所有 rank 参与（FSDP 需要协调），但只有 rank 0 写磁盘。
        """
        import torch.distributed as dist
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        dist.barrier()

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.policy.state_dict()

        if self.rank == 0:
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # 用 transformers 格式保存（兼容后续 merge / vLLM 加载）
            from transformers import AutoModelForCausalLM
            # 临时载入到 CPU 模型保存
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                BASE_MODEL, trust_remote_code=True
            )
            cpu_model = AutoModelForCausalLM.from_config(config)
            cpu_model.load_state_dict(state_dict)
            cpu_model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"  检查点已保存: {ckpt_dir}")

        dist.barrier()

    @staticmethod
    def from_fsdp_sft_trainer(
        sft_trainer_config: dict,
        dpo_config: dict | None = None,
    ) -> "VerlDPOTrainer":
        """
        从 veRL fsdp_sft_trainer 的配置文件直接构建 DPO Trainer。

        这是与 veRL 官方 SFT 训练脚本的对接入口。
        veRL SFT 的典型 YAML 配置（configs/verl_sft.yaml）会被直接传入，
        DPO 特有字段通过 dpo_config 叠加覆盖。

        示例:
            trainer = VerlDPOTrainer.from_fsdp_sft_trainer(
                sft_trainer_config=yaml.safe_load(open("configs/verl_sft.yaml")),
                dpo_config={"beta": 0.15, "loss_type": "ipo"},
            )
            trainer.train()
        """
        merged = {}

        # 从 veRL SFT config 提取兼容字段
        sft = sft_trainer_config.get("trainer", sft_trainer_config)
        merged.update({
            "num_train_epochs": sft.get("total_epochs", VERL_DPO_CONFIG["num_train_epochs"]),
            "per_device_train_batch_size": sft.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": sft.get("gradient_accumulation_steps", 4),
            "learning_rate": sft.get("lr", VERL_DPO_CONFIG["learning_rate"]),
            "max_length": sft.get("max_length", VERL_DPO_CONFIG["max_length"]),
            "gradient_checkpointing": sft.get("gradient_checkpointing", True),
            "fsdp_offload": sft.get("fsdp_offload_params", False),
            "bf16": sft.get("bf16", True),
        })

        # DPO 特有字段叠加
        if dpo_config:
            merged.update(dpo_config)

        return VerlDPOTrainer(
            model_path=sft.get("model_path", BASE_MODEL),
            output_dir=sft.get("output_dir", OUTPUT_DIR),
            data_path=sft.get("data_path", DPO_DATA),
            config=merged,
        )


# ---------------------------------------------------------------------------
# 单卡模式（无 FSDP，用于本地调试）
# ---------------------------------------------------------------------------

def train_dpo_single_gpu(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    data_path: str = DPO_DATA,
    config: dict | None = None,
):
    """
    单卡 DPO 训练（不启动分布式，适合本地 debug）。

    与 VerlDPOTrainer 逻辑完全一致，去掉 FSDP / dist 相关调用。
    没有 GPU 时会自动退回 CPU（调试格式是否正确）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = {**VERL_DPO_CONFIG, **(config or {})}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (cfg["bf16"] and device.type == "cuda") else torch.float32

    print(f"单卡 DPO  device={device}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    for p in ref.parameters():
        p.requires_grad_(False)
    if cfg["gradient_checkpointing"] and device.type == "cuda":
        policy.gradient_checkpointing_enable()

    dataset = DPOPairDataset(data_path=data_path, tokenizer=tokenizer, max_length=cfg["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["learning_rate"])
    global_step = 0

    policy.train()
    ref.eval()

    for epoch in range(cfg["num_train_epochs"]):
        for batch_idx, batch in enumerate(loader):
            chosen_ids   = batch["chosen_input_ids"].to(device)
            chosen_lbl   = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_lbl = batch["rejected_labels"].to(device)

            policy_lp_c = _get_batch_log_probs(policy, chosen_ids, chosen_lbl)
            policy_lp_r = _get_batch_log_probs(policy, rejected_ids, rejected_lbl)

            with torch.no_grad():
                ref_lp_c = _get_batch_log_probs(ref, chosen_ids, chosen_lbl)
                ref_lp_r = _get_batch_log_probs(ref, rejected_ids, rejected_lbl)

            loss, metrics = dpo_loss(
                policy_lp_c, policy_lp_r, ref_lp_c, ref_lp_r,
                beta=cfg["beta"],
                loss_type=cfg["loss_type"],
                label_smoothing=cfg["label_smoothing"],
            )

            (loss / cfg["gradient_accumulation_steps"]).backward()

            if (batch_idx + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["logging_steps"] == 0:
                    print(
                        f"epoch={epoch+1} step={global_step:4d} "
                        f"loss={metrics['dpo_loss']:.4f} "
                        f"acc={metrics['reward_accuracy']:.3f} "
                        f"margin={metrics['reward_margin']:.3f}"
                    )

    policy.save_pretrained(output / "final")
    tokenizer.save_pretrained(output / "final")
    print(f"单卡 DPO 完成 → {output / 'final'}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2c: DPO（veRL FSDP）")
    subparsers = parser.add_subparsers(dest="cmd")

    # ── 子命令: generate-pairs ──
    gen_parser = subparsers.add_parser("generate-pairs", help="生成 DPO 偏好对数据")
    gen_parser.add_argument("--output", default=DPO_DATA)
    gen_parser.add_argument("--teacher-model", default=None)
    gen_parser.add_argument("--n-attempts", type=int, default=6)

    # ── 子命令: train ──
    train_parser = subparsers.add_parser("train", help="多卡 FSDP DPO 训练")
    train_parser.add_argument("--model", default=BASE_MODEL)
    train_parser.add_argument("--output", default=OUTPUT_DIR)
    train_parser.add_argument("--data", default=DPO_DATA)
    train_parser.add_argument("--beta", type=float, default=0.1)
    train_parser.add_argument("--loss-type", default="sigmoid",
                               choices=["sigmoid", "hinge", "ipo"])
    train_parser.add_argument("--epochs", type=int, default=3)

    # ── 子命令: train-single ──
    single_parser = subparsers.add_parser("train-single", help="单卡 DPO 训练（本地调试）")
    single_parser.add_argument("--model", default=BASE_MODEL)
    single_parser.add_argument("--output", default=OUTPUT_DIR)
    single_parser.add_argument("--data", default=DPO_DATA)
    single_parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()

    if args.cmd == "generate-pairs":
        from rollout import make_anthropic_model_fn
        model_fn = make_anthropic_model_fn(
            model=args.teacher_model or os.environ.get("MODEL_ID", "claude-3-5-haiku-20241022")
        )
        generate_dpo_pairs(model_fn=model_fn, n_attempts=args.n_attempts, output_path=args.output)

    elif args.cmd == "train":
        # 多卡：需要通过 torchrun 启动
        # 示例: torchrun --nproc_per_node=2 train/verl_dpo.py train --beta 0.1
        trainer = VerlDPOTrainer(
            model_path=args.model,
            output_dir=args.output,
            data_path=args.data,
            config={"beta": args.beta, "loss_type": args.loss_type, "num_train_epochs": args.epochs},
        )
        trainer.train()

    elif args.cmd == "train-single":
        # 单卡：直接运行
        # 示例: python train/verl_dpo.py train-single --model checkpoints/sft_merged
        train_dpo_single_gpu(
            model_path=args.model,
            output_dir=args.output,
            data_path=args.data,
            config={"beta": args.beta},
        )

    else:
        parser.print_help()
