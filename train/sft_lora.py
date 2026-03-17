#!/usr/bin/env python3
"""
train/sft_lora.py — Phase 1: LoRA SFT 冷启动

在 Qwen2.5-Coder-1.5B / 3B 上用 LoRA 做 Supervised Fine-Tuning，
让模型学会 agent 的基本行为格式（工具调用、<think> 推理链）。

训练策略：
    - 基础模型：Qwen/Qwen2.5-Coder-1.5B-Instruct 或 3B 版本
    - 方法：LoRA (r=16, alpha=32)，只训练 q_proj / v_proj / k_proj / o_proj
    - 数据：data/sft/train.jsonl（teacher rollout 生成）
    - Trainer：HuggingFace TRL SFTTrainer
    - 序列长度：2048 tokens（覆盖大部分 agent 轨迹）
    - 批次：per_device=1, gradient_accumulation=8 → 等效 bs=8（单卡 A100 40G）

显存估算（1.5B LoRA）：
    - 模型权重 bf16：~3GB
    - LoRA 参数：~50MB
    - 激活值（seq=2048, bs=1）：~4GB
    - 优化器状态（AdamW，LoRA only）：~200MB
    - 总计：~8-10GB，单卡 A100 40G 完全够用
"""

import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from hub_utils import resolve_model_path

# ---------------------------------------------------------------------------
# 配置（可通过命令行或 configs/sft.yaml 覆盖）
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "checkpoints/sft_lora")
DATA_PATH = os.environ.get("SFT_DATA_PATH", "data/sft/train.jsonl")

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # 对 Qwen 的 attention 层做 LoRA
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

TRAIN_CONFIG = {
    "max_seq_length": 2048,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "report_to": "none",  # 改为 "wandb" 开启可视化
    "dataloader_num_workers": 0,
}

# 是否使用 4-bit 量化（显存更紧张时开启，会略微降低效果）
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sft_dataset(data_path: str, val_ratio: float = 0.05):
    """
    加载 JSONL 格式的 SFT 数据。
    格式：每行是 {"id": ..., "conversations": [...], "metadata": {...}}

    返回 (train_dataset, eval_dataset)。
    """
    import json
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"SFT 数据不存在: {data_path}\n"
            "请先运行: python scripts/generate_sft_data.py"
        )

    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"加载 {len(samples)} 条 SFT 样本")

    # 分割 train/eval
    split_idx = max(1, int(len(samples) * (1 - val_ratio)))
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    return Dataset.from_list(train_samples), Dataset.from_list(eval_samples)


def format_conversations(sample: dict, tokenizer) -> str:
    """
    将 conversations 列表格式化为模型的 chat template 格式。
    Qwen2.5 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>
    """
    conversations = sample.get("conversations", [])
    # 过滤掉 tool role（合并到相邻的 user/assistant 消息）
    filtered = []
    for conv in conversations:
        role = conv.get("role", "")
        content = conv.get("content", "")
        if role == "tool":
            # 将工具结果附加到下一条 user 消息，或单独作为 user 消息
            filtered.append({"role": "user", "content": f"[Tool Result]\n{content}"})
        elif role in ("system", "user", "assistant"):
            filtered.append({"role": role, "content": content})

    return tokenizer.apply_chat_template(
        filtered, tokenize=False, add_generation_prompt=False
    )


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, use_4bit: bool = False):
    """加载基础模型和 tokenizer，应用 LoRA。"""
    resolved_model = resolve_model_path(model_name)
    print(f"加载模型: {model_name} -> {resolved_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置（可选）
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("使用 4-bit 量化（QLoRA 模式）")

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 如果安装了 flash-attn
    )

    # 应用 LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(
    model_name: str = BASE_MODEL,
    data_path: str = DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    use_4bit: bool = USE_4BIT,
    extra_train_config: dict | None = None,
):
    """
    主训练函数。

    Args:
        model_name: 基础模型路径或 HuggingFace Hub ID
        data_path: SFT 数据集路径
        output_dir: 检查点输出目录
        use_4bit: 是否使用 QLoRA
        extra_train_config: 覆盖默认 TRAIN_CONFIG 的参数
    """
    # 加载数据
    train_ds, eval_ds = load_sft_dataset(data_path)
    print(f"训练集: {len(train_ds)} 条, 验证集: {len(eval_ds)} 条")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_name, use_4bit)

    # 训练配置
    train_config = {**TRAIN_CONFIG}
    if extra_train_config:
        train_config.update(extra_train_config)
    train_config["output_dir"] = output_dir

    sft_config = SFTConfig(
        **train_config,
        max_seq_length=train_config.pop("max_seq_length", 2048),
        dataset_text_field=None,  # 使用 formatting_func
        packing=False,  # 不 packing，保持轨迹完整性
    )

    # 格式化函数
    def formatting_func(samples):
        texts = []
        for i in range(len(samples["conversations"])):
            sample = {"conversations": samples["conversations"][i]}
            texts.append(format_conversations(sample, tokenizer))
        return texts

    # 创建 Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
        formatting_func=formatting_func,
    )

    print(f"\n开始 SFT 训练...")
    print(f"  模型: {model_name}")
    print(f"  输出: {output_dir}")
    print(f"  训练轮数: {train_config['num_train_epochs']}")
    print(f"  等效批次大小: "
          f"{train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")

    trainer.train()

    # 保存 LoRA adapter
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nLoRA adapter 已保存到: {adapter_path}")

    return trainer


# ---------------------------------------------------------------------------
# 合并 LoRA 权重（可选，用于后续 RL 训练）
# ---------------------------------------------------------------------------

def merge_lora_weights(adapter_path: str, output_path: str):
    """
    将 LoRA adapter 合并回基础模型，生成完整模型。
    合并后的模型可以加载到 vLLM 中用于 RL rollout。
    """
    from peft import PeftModel

    print(f"合并 LoRA 权重: {adapter_path} → {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model_path = resolve_model_path(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print(f"合并完成: {output_path}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: LoRA SFT 冷启动")
    parser.add_argument("--model", default=BASE_MODEL, help="基础模型路径")
    parser.add_argument("--data", default=DATA_PATH, help="SFT 数据路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit", help="使用 QLoRA")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--merge", action="store_true", help="训练后合并权重")
    parser.add_argument("--merge-output", default="checkpoints/sft_merged", help="合并输出路径")
    args = parser.parse_args()

    trainer = train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        use_4bit=args.use_4bit,
        extra_train_config={"num_train_epochs": args.epochs, "learning_rate": args.lr},
    )

    if args.merge:
        final_adapter = str(Path(args.output) / "final_adapter")
        merge_lora_weights(final_adapter, args.merge_output)
