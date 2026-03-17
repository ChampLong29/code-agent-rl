# Makefile — 一键运行各训练阶段
# 用法: make <target>

.PHONY: help install install-dev install-verl install-verl-wsl install-monitor data sft sft-merge grpo grpo-trl grpo-slime grpo-verl grpo-verl-singleturn ppo dpo verl-dpo verl-dpo-single agent-cli agent-cli-hf eval compare clean

PYTHON := uv run python
MODEL ?= Qwen/Qwen2.5-Coder-1.5B-Instruct
SFT_OUTPUT ?= checkpoints/sft_lora
SFT_MERGED ?= checkpoints/sft_merged

help:
	@echo ""
	@echo "  Agentic RL Training — 基于 learn-claude-code 的 Search-R1 管线"
	@echo ""
	@echo "  阶段 0: 环境 & 数据"
	@echo "    make install          安装基础依赖（uv）"
	@echo "    make install-dev      安装开发工具（ruff, pyright）"
	@echo "    make install-monitor  安装监控工具（wandb, swanlab）"
	@echo "    make install-verl     从 git 安装 veRL 框架"
	@echo "    make install-verl-wsl 在 WSL 中安装 veRL（推荐）"
	@echo "    make data             生成 SFT 数据 (teacher rollout)"
	@echo "    make dpo-pairs        生成 DPO 偏好对数据"
	@echo ""
	@echo "  阶段 1: SFT 冷启动"
	@echo "    make sft              LoRA SFT 训练"
	@echo "    make sft-merge        合并 LoRA 权重"
	@echo ""
	@echo "  阶段 2: RL 微调"
	@echo "    ── 教学模式（零额外依赖）──"
	@echo "    make grpo             GRPO custom 模式（代码透明，教学首选）"
	@echo "    make grpo-trl         GRPO TRL 模式"
	@echo "    make grpo-slime       GRPO SLIME 框架"
	@echo "    make ppo              PPO"
	@echo "    make dpo              DPO（离线偏好）"
	@echo "    ── 生产模式（veRL 框架）──"
	@echo "    make grpo-verl        GRPO veRL multi-turn（真实工具调用，推荐）"
	@echo "    make grpo-verl-singleturn  GRPO veRL single-turn（兼容性好）"	@echo "    make verl-dpo         DPO veRL FSDP（多卡，SFT 基础设施 + DPO loss）"
	@echo "    make verl-dpo-single  DPO veRL 单卡（本地调试，无需 torchrun）"	@echo ""
	@echo "  全流程"
	@echo "    make pipeline         data→sft→sft-merge→grpo→eval（教学管线）"
	@echo "    make pipeline-verl    data→sft→sft-merge→grpo-verl→eval（生产管线）"
	@echo ""
	@echo "  评估"
	@echo "    make eval             评估所有可用检查点"
	@echo "    make compare          对比 SFT vs GRPO vs PPO vs DPO"
	@echo ""
	@echo "  工具"
	@echo "    make agent-cli        本地模型 Agent 工具调用 CLI 测试（vLLM）"
	@echo "    make agent-cli-hf     本地模型 Agent 工具调用 CLI 测试（HF）"
	@echo "    make clean            清理所有检查点"
	@echo "    make test             运行单元测试"
	@echo "    make test-verl        测试 veRL 奖励适配层（无需 GPU）"
	@echo ""

# ============================================================
# 安装
# ============================================================

install:
	uv pip install -e "..[core]" 2>/dev/null || uv pip install -r requirements.txt
	@echo "✓ 基础依赖安装完成（uv）"
	@echo ""
	@echo "Flash Attention 2（可选，显著提速）："
	@echo "  uv pip install flash-attn --no-build-isolation"
	@echo ""

install-dev:
	uv pip install -e "..[dev]" 2>/dev/null || uv pip install ruff pyright ipykernel
	@echo "✓ 开发工具已安装"

install-monitor:
	uv pip install wandb swanlab
	@echo "✓ 监控工具已安装（wandb + swanlab）"
	@echo "请在 .env 中配置 WANDB_API_KEY 和/或 SWANLAB_API_KEY"
	@echo "vLLM（可选，RL rollout 加速）："
	@echo "  pip install vllm"
	@echo ""
	@echo "veRL（可选，生产级 GRPO 框架）："
	@echo "  make install-verl"

install-verl:
	@echo "==> 安装 veRL 框架（GitHub 最新版）..."
	uv pip install git+https://github.com/verl-project/verl.git
	uv pip install ray omegaconf tensordict hydra-core codetiming pyzmq
	@echo "✓ veRL 安装完成"
	@echo ""
	@echo "可选：SGLang（multi-turn rollout 加速，推荐）"
	@echo "  uv pip install sglang"
	@echo ""
	@echo "验证安装（无需 GPU）："
	@echo "  make test-verl"

install-verl-wsl:
	@echo "==> 在 WSL 中使用 uv 安装 veRL（GitHub 最新版）..."
	wsl -e bash -lc "cd /mnt/d/project/code-agent-rl && uv venv .venv-wsl --python 3.12 --clear && uv pip install --python .venv-wsl/bin/python -U pip setuptools wheel && uv pip install --python .venv-wsl/bin/python git+https://github.com/verl-project/verl.git"
	@echo "✓ WSL uv 安装命令已执行"

# ============================================================
# 阶段 0: 数据准备
# ============================================================

data:
	@echo "==> 生成 SFT 训练数据（teacher rollout）..."
	$(PYTHON) scripts/generate_sft_data.py \
		--output data/sft/train.jsonl \
		--n-per-task 3 \
		--max-steps 15
	@echo "✓ SFT 数据已保存到 data/sft/train.jsonl"

data-dry:
	$(PYTHON) scripts/generate_sft_data.py --dry-run

dpo-pairs:
	@echo "==> 生成 DPO 偏好对数据..."
	$(PYTHON) train/rl_dpo.py \
		--generate-pairs \
		--data data/rl/dpo_pairs.jsonl
	@echo "✓ DPO 偏好对已保存到 data/rl/dpo_pairs.jsonl"

# ============================================================
# 阶段 1: SFT LoRA
# ============================================================

sft:
	@echo "==> Phase 1: LoRA SFT 冷启动..."
	BASE_MODEL=$(MODEL) $(PYTHON) train/sft_lora.py \
		--model $(MODEL) \
		--data data/sft/train.jsonl \
		--output $(SFT_OUTPUT) \
		--epochs 3
	@echo "✓ SFT LoRA adapter 已保存到 $(SFT_OUTPUT)"

sft-merge:
	@echo "==> 合并 LoRA 权重..."
	BASE_MODEL=$(MODEL) $(PYTHON) train/sft_lora.py \
		--model $(MODEL) \
		--data data/sft/train.jsonl \
		--output $(SFT_OUTPUT) \
		--merge \
		--merge-output $(SFT_MERGED)
	@echo "✓ 合并模型已保存到 $(SFT_MERGED)"

sft-4bit:
	@echo "==> Phase 1: QLoRA SFT（4-bit 量化，节省显存）..."
	BASE_MODEL=$(MODEL) $(PYTHON) train/sft_lora.py \
		--model $(MODEL) \
		--data data/sft/train.jsonl \
		--output $(SFT_OUTPUT)-4bit \
		--4bit

# ============================================================
# 阶段 2: RL 微调
# ============================================================

grpo:
	@echo "==> Phase 2c: GRPO (Search-R1 风格)..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/grpo \
		--mode custom \
		--iterations 200 \
		--group-size 8 \
		--kl-coeff 0.04
	@echo "✓ GRPO 训练完成"

grpo-trl:
	@echo "==> Phase 2c: GRPO (TRL 模式)..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/grpo-trl \
		--mode trl

grpo-slime:
	@echo "==> Phase 2c: GRPO (SLIME 框架)..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/grpo-slime \
		--mode slime

# ── veRL 生产级 GRPO ──

grpo-verl:
	@echo "==> Phase 2d: GRPO veRL multi-turn（真实工具调用）..."
	@echo "     rollout: vLLM  训练: FSDP  奖励: 可验证（pytest/bash）"
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_verl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/verl_grpo \
		--multiturn \
		--rollout-backend vllm \
		--train-backend fsdp \
		--iterations 200 \
		--group-size 8 \
		--kl-coeff 0.04
	@echo "✓ veRL GRPO 训练完成"

grpo-verl-singleturn:
	@echo "==> Phase 2d: GRPO veRL single-turn + replay（兼容模式）..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_verl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/verl_grpo_st \
		--no-multiturn \
		--iterations 200 \
		--group-size 8
	@echo "✓ veRL GRPO single-turn 训练完成"

grpo-verl-lora:
	@echo "==> Phase 2d: GRPO veRL + LoRA（显存优化）..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_verl_grpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/verl_grpo_lora \
		--multiturn \
		--lora \
		--iterations 200
	@echo "✓ veRL LoRA GRPO 训练完成"

ppo:
	@echo "==> Phase 2a: PPO..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_ppo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/ppo \
		--mode trl \
		--iterations 300
	@echo "✓ PPO 训练完成"

dpo:
	@echo "==> Phase 2b: DPO（TRL DPOTrainer，单卡）..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/rl_dpo.py \
		--model $(SFT_MERGED) \
		--output checkpoints/dpo \
		--data data/rl/dpo_pairs.jsonl \
		--beta 0.1
	@echo "✓ DPO 训练完成"

# ── veRL FSDP DPO ──

verl-dpo:
	@echo "==> Phase 2c: DPO veRL FSDP 多卡..."
	@echo "     trainer: FSDP   loss: DPO (sigmoid)   ref: frozen FSDP"
	@echo "     启动方式: torchrun --nproc_per_node=$(N_GPUS) train/verl_dpo.py train ..."
	torchrun \
		--nproc_per_node=$(or $(N_GPUS),$(shell python -c 'import torch; print(torch.cuda.device_count() or 1)')) \
		train/verl_dpo.py train \
		--model $(SFT_MERGED) \
		--output checkpoints/verl_dpo \
		--data data/rl/dpo_pairs.jsonl \
		--beta 0.1 \
		--loss-type sigmoid
	@echo "✓ veRL DPO 训练完成"

verl-dpo-single:
	@echo "==> Phase 2c: DPO veRL 单卡（本地 debug）..."
	BASE_MODEL=$(SFT_MERGED) $(PYTHON) train/verl_dpo.py train-single \
		--model $(SFT_MERGED) \
		--output checkpoints/verl_dpo_single \
		--data data/rl/dpo_pairs.jsonl \
		--beta 0.1
	@echo "✓ veRL DPO 单卡训练完成"

# ============================================================
# 全流程（GRPO 管线）
# ============================================================

pipeline:
	@echo "==> 运行完整管线（教学模式）: data → sft → sft-merge → grpo → eval"
	$(MAKE) data
	$(MAKE) sft
	$(MAKE) sft-merge
	$(MAKE) grpo
	$(MAKE) eval

pipeline-verl:
	@echo "==> 运行完整管线（veRL 生产模式）: data → sft → sft-merge → grpo-verl → eval"
	$(MAKE) data
	$(MAKE) sft
	$(MAKE) sft-merge
	$(MAKE) grpo-verl
	$(MAKE) eval

# ============================================================
# 评估
# ============================================================

eval:
	@echo "==> 评估所有可用检查点..."
	$(PYTHON) eval/evaluate.py \
		--output eval/results \
		--n-per-task 3

compare:
	@echo "==> 对比所有模型..."
	$(PYTHON) eval/evaluate.py \
		--compare \
			sft:$(SFT_MERGED) \
			grpo:checkpoints/grpo/final \
			ppo:checkpoints/ppo/final \
			dpo:checkpoints/dpo/final \
			verl-dpo:checkpoints/verl_dpo/checkpoint-final \
		--n-per-task 2 \
		--output eval/results

eval-model:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "用法: make eval-model MODEL_PATH=<路径>"; exit 1; \
	fi
	$(PYTHON) eval/evaluate.py \
		--model $(MODEL_PATH) \
		--n-per-task 3 \
		--output eval/results

# ============================================================
# 开发工具
# ============================================================

test:
	$(PYTHON) -m pytest tests/ -v 2>/dev/null || \
	$(PYTHON) environment.py && \
	$(PYTHON) reward.py && \
	$(PYTHON) rollout.py
	@echo "✓ 基础模块测试通过"

test-verl:
	@echo "==> 测试 veRL 奖励适配层（无需 GPU）..."
	$(PYTHON) train/rl_verl_grpo.py --test
	@echo "✓ veRL 适配层测试通过"

agent-cli:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "用法: make agent-cli MODEL_PATH=<本地模型路径>"; exit 1; \
	fi
	$(PYTHON) scripts/cli_agent_local.py --model $(MODEL_PATH) --backend vllm

agent-cli-hf:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "用法: make agent-cli-hf MODEL_PATH=<本地模型路径>"; exit 1; \
	fi
	$(PYTHON) scripts/cli_agent_local.py --model $(MODEL_PATH) --backend hf

test-env:
	$(PYTHON) environment.py

test-reward:
	$(PYTHON) reward.py

test-rollout:
	$(PYTHON) rollout.py

clean:
	@echo "清理所有检查点和数据..."
	rm -rf checkpoints/
	rm -rf data/sft/*.jsonl
	rm -rf data/rl/*.jsonl
	rm -rf eval/results/
	@echo "✓ 清理完成"

clean-checkpoints:
	rm -rf checkpoints/
	@echo "✓ 检查点已清理"
