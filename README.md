# Agentic RL Training — 基于 learn-claude-code 的 Search-R1 风格强化学习管线

> **目标**：用 Qwen2.5-Coder-1.5B / 3B，在 nano Claude Code agent 的行为轨迹上，
> 先做 LoRA SFT 冷启动，再通过 PPO / DPO / GRPO 三条 RL 管线进行 Agentic RL 训练，
> 复现 Search-R1 在工具调用 agent 场景下的核心思想。

```
learn-claude-code/
└── rl_training/                ← 本项目根目录
    ├── README.md               ← 你在这里
    ├── requirements.txt        ← 依赖
    ├── Makefile                ← 一键运行各阶段
    ├── environment.py          ← 沙箱环境 (agent 执行器)
    ├── reward.py               ← 奖励函数族
    ├── rollout.py              ← Rollout 采样器
    ├── configs/                ← YAML 配置文件
    │   ├── base.yaml
    │   ├── sft.yaml
    │   ├── ppo.yaml
    │   ├── dpo.yaml
    │   └── grpo.yaml
    ├── data/                   ← 数据集
    │   ├── raw/                ← 原始 prompt 种子
    │   ├── sft/                ← SFT 训练集 (teacher rollout)
    │   └── rl/                 ← RL 在线采样缓存
    ├── train/                  ← 训练脚本
    │   ├── sft_lora.py         ← Phase 1: LoRA SFT
    │   ├── rl_ppo.py           ← Phase 2a: PPO
    │   ├── rl_dpo.py           ← Phase 2b: DPO (offline)
    │   └── rl_grpo.py          ← Phase 2c: GRPO (Search-R1 核心)
    ├── eval/                   ← 评估脚本
    │   └── evaluate.py
    ├── scripts/                ← 辅助脚本
    │   └── generate_sft_data.py
    └── docs/                   ← 详细模块文档
        ├── 00-overview.md
        ├── 01-environment.md
        ├── 02-reward.md
        ├── 03-rollout.md
        ├── 04-sft-lora.md
        ├── 05-ppo.md
        ├── 06-dpo.md
        └── 07-grpo.md
```

---

## 核心思想

### 为什么用这个项目做 Agentic RL？

`learn-claude-code` 的每个课程（s01–s12）都是一个**结构化行为轨迹**：

```
User prompt
  → <think> 推理链 </think>
  → tool_call: bash("pytest tests/")
  → tool_result: "5 passed"
  → tool_call: write_file("solution.py", ...)
  → Final answer
```

这和 Search-R1 的 `<think>→<search>→<answer>` 格式完全同构，只是把搜索换成了代码执行工具。

### 三阶段训练管线

```
阶段 0: 数据准备
  Claude/GPT-4o 做 teacher rollout → SFT 数据集 (5k–20k 条)

阶段 1: SFT 冷启动 (LoRA)
  Qwen2.5-Coder-1.5B/3B + LoRA
  让模型学会 agent loop 的基本格式

阶段 2: RL 微调 (三选一或联合)
  ├── PPO  — 在线策略，稳定但慢
  ├── DPO  — 离线偏好对，快但需要对比数据
  └── GRPO — 组相对优化，Search-R1 原版方法，推荐首选
```

---

## 快速开始

### 环境要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU  | 1×A100 40G (1.5B LoRA) | 2×A100 80G (3B 全参) |
| CPU RAM | 32G | 64G |
| 磁盘 | 50G | 200G |
| Python | 3.10+ | 3.11 |

### 安装

```bash
cd learn-claude-code/rl_training
pip install -r requirements.txt
```

### 一键运行

```bash
# 生成 SFT 数据
make data

# Phase 1: SFT LoRA
make sft

# Phase 2: GRPO (推荐首选)
make grpo

# 评估
make eval
```

---

## 模块速览

| 模块 | 职责 | 文档 |
|------|------|------|
| `environment.py` | 沙箱执行环境，隔离工具调用，记录轨迹 | [docs/01-environment.md](docs/01-environment.md) |
| `reward.py` | 可验证奖励函数（格式、工具调用、任务完成度） | [docs/02-reward.md](docs/02-reward.md) |
| `rollout.py` | 批量采样 rollout，支持 vLLM 推理 | [docs/03-rollout.md](docs/03-rollout.md) |
| `train/sft_lora.py` | LoRA SFT 冷启动，HuggingFace Trainer | [docs/04-sft-lora.md](docs/04-sft-lora.md) |
| `train/rl_ppo.py` | PPO 在线强化学习，基于 TRL | [docs/05-ppo.md](docs/05-ppo.md) |
| `train/rl_dpo.py` | DPO 离线偏好学习，基于 TRL | [docs/06-dpo.md](docs/06-dpo.md) |
| `train/rl_grpo.py` | GRPO Search-R1 风格，custom / TRL / SLIME | [docs/07-grpo.md](docs/07-grpo.md) |
| `train/rl_verl_grpo.py` | **GRPO veRL 生产级**，multi-turn 真实工具调用 | — |

---

## 参考论文

- [Search-R1: Training LLMs to Reason and Leverage Search Engines with RL](https://arxiv.org/abs/2503.09516)
- [veRL: HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
- [SLIME: Scalable Lightweight Infrastructure for Multi-turn RL](https://github.com/PRIME-RL/SLIME)
- [DeepSeek-R1: Incentivizing Reasoning via RL](https://arxiv.org/abs/2501.12948)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
