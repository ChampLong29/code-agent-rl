# 00: 总览 — Agentic RL 训练管线

## 项目定位

本项目在 `learn-claude-code`（nano Claude Code agent）的基础上，
构建一条完整的 **Agentic RL 训练管线**，复现 Search-R1 在代码 agent 场景下的核心思想。

目标：用 **Qwen2.5-Coder-1.5B / 3B** 训练出一个能自主使用工具解决编程任务的 agent，
比 base 模型的任务成功率提升 **30%+**。

---

## 为什么 learn-claude-code 适合做 Agentic RL？

### 结构同构

Search-R1 的推理格式：
```
<think> 我需要搜索这个问题 </think>
<search> query </search>
<result> ... </result>
<answer> 最终答案 </answer>
```

learn-claude-code 的 agent loop 格式：
```
<think> 先读测试文件，再写实现 </think>
tool_call: read_file("test_sort.py")
tool_result: "def test_sort(): ..."
tool_call: write_file("sort.py", "def bubble_sort...")
tool_result: "Wrote 120 bytes"
Final answer: "已实现 bubble_sort，测试通过"
```

**两者完全同构**。把"搜索"换成"代码工具调用"，算法完全一样。

### 可验证奖励天然存在

```python
# 任务：实现 fib(n) 并通过测试
success_criteria = {"type": "pytest", "pattern": "test_fib.py"}

# 奖励：运行 pytest，解析通过率
reward = passed_tests / total_tests  # 完全可验证，无需人工标注
```

这正是 Search-R1 的精髓：**用自动可验证奖励替代人工反馈**。

### s12 worktree 的天然并行性

learn-claude-code 的 s12 课程展示了 worktree 隔离，
这和 RL 的**并行 rollout worker** 需求完美对应：
每个任务在独立目录运行，互不干扰。

---

## 三阶段训练管线

```
┌─────────────────────────────────────────────────────────┐
│                    完整训练管线                          │
│                                                         │
│  阶段 0: 数据准备                                        │
│  ─────────────────                                      │
│  Claude/GPT-4o（teacher）                               │
│    └→ 对 8 个种子任务做 rollout                         │
│    └→ 只保留成功轨迹                                     │
│    └→ 输出 SFT 数据集（~24 条）                         │
│                                                         │
│  阶段 1: SFT 冷启动                                      │
│  ─────────────────                                      │
│  Qwen2.5-Coder-1.5B + LoRA                             │
│    └→ 学会 agent loop 基本格式                          │
│    └→ 学会 <think> 推理链结构                           │
│    └→ 学会工具调用语法                                   │
│    └→ 输出：SFT adapter → merged 模型                   │
│                                                         │
│  阶段 2: RL 微调（三选一）                               │
│  ─────────────────                                      │
│  基于 SFT 模型，在线/离线强化                            │
│                                                         │
│  ┌─ PPO  ────────────────────────────────────────┐     │
│  │ 在线策略 + Critic                             │     │
│  │ 稳定 + 样本高效 + 需要 2x 显存               │     │
│  └───────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ DPO  ────────────────────────────────────────┐     │
│  │ 离线偏好对 + 无 Critic                        │     │
│  │ 快 + 省显存 + 需要预采样数据                 │     │
│  └───────────────────────────────────────────────┘     │
│                                                         │
│  ┌─ GRPO ─────────────── ⭐ 推荐 ──────────────────┐  │
│  │ 组相对优化 + 无 Critic                        │     │
│  │ 简单 + 效果好 + Search-R1 原版方法           │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 为什么先做 SFT 再做 RL？

直接从 base 模型做 RL 通常失败，因为：
- base 模型不知道工具调用的格式
- 初始成功率接近 0，无奖励信号，梯度为 0，无法学习

SFT 冷启动后：
- 模型知道基本格式
- 初始成功率 ~10-30%，有足够的奖励差异
- RL 可以在此基础上进一步优化

这就是 DeepSeek-R1 和 Search-R1 的两阶段策略。

### 2. 为什么推荐 GRPO 而不是 PPO？

| | PPO | GRPO |
|---|---|---|
| Critic 网络 | 需要 | **不需要** |
| 显存 | ~2x | **1x** |
| 实现复杂度 | 高 | **低** |
| 样本效率 | 高 | 中 |
| 适用场景 | 任意奖励 | **可验证奖励** ✓ |

在代码执行任务中，奖励是完全可验证的（pytest 通过/失败），
不需要 Critic 估计价值，GRPO 是最自然的选择。

### 3. 为什么用 1.5B 而不是更大的模型？

| 模型 | 最小显存需求 | 训练时间（100 iter） |
|---|---|---|
| 1.5B LoRA | 1×A100 40G | ~2h |
| 3B LoRA | 1×A100 80G | ~4h |
| 7B LoRA | 2×A100 80G | ~8h |
| 7B 全参 | 4×A100 80G | ~20h |

1.5B 可以在**单卡 A100 40G**上完成完整管线实验，
是验证方法可行性的最小成本起点。
效果验证后再扩展到 3B 或 7B。

---

## 算力需求详细估算

### 最小配置（1.5B LoRA）

```
GPU:  1 × NVIDIA A100 40G
RAM:  32G
磁盘: 50G（模型 ~6G + 数据 ~1G + 检查点 ~30G）

阶段 0 (数据生成):  ~30min（Claude API，不需要 GPU）
阶段 1 (SFT):      ~2h
阶段 2 (GRPO):     ~4h（200 迭代 × 8 rollout）

总计: ~7h，云端约 $15-20（A100 40G @$2/h）
```

### 推荐配置（3B LoRA，更好效果）

```
GPU:  2 × NVIDIA A100 80G
RAM:  64G
磁盘: 200G

阶段 1 (SFT):  ~3h
阶段 2 (GRPO): ~8h

总计: ~12h，云端约 $50-80（2×A100 80G @$4/h）
```

---

## 快速开始

```bash
cd learn-claude-code/rl_training

# 1. 安装依赖
make install

# 2. 设置环境变量（用于 teacher rollout）
export ANTHROPIC_API_KEY=sk-...
export MODEL_ID=claude-3-5-haiku-20241022

# 3. 生成 SFT 数据（约 30 分钟，调用 Claude API）
make data

# 4. SFT 冷启动（需要 GPU）
make sft && make sft-merge

# 5. GRPO 强化学习（需要 GPU）
make grpo

# 6. 评估
make eval
```

---

## 模块文档索引

| 文档 | 内容 |
|------|------|
| [01-environment.md](01-environment.md) | 沙箱执行环境详解 |
| [02-reward.md](02-reward.md) | 奖励函数设计原理 |
| [03-rollout.md](03-rollout.md) | Rollout 采样器实现 |
| [04-sft-lora.md](04-sft-lora.md) | LoRA SFT 冷启动 |
| [05-ppo.md](05-ppo.md) | PPO 算法实现 |
| [06-dpo.md](06-dpo.md) | DPO 偏好学习 |
| [07-grpo.md](07-grpo.md) | GRPO Search-R1 核心 |
