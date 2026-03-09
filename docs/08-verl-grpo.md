# 08: veRL GRPO — 生产级框架

**文件**: `train/rl_verl_grpo.py` · `configs/verl_grpo.yaml`

veRL（Volcano Engine RL）是字节跳动开源的生产级 RL 训练框架，
相比本项目的 custom GRPO，核心优势是 **Actor/Rollout/Trainer 三角色分离**，
GPU 利用率可接近 100%。

---

## 与现有代码的关系

`rl_verl_grpo.py` 采用**最小侵入原则**，现有三个核心模块**零改动**：

```
不改动（教学核心，完整保留）：
  environment.py  →  沙箱执行 + 轨迹记录
  reward.py       →  可验证奖励函数族
  rollout.py      →  rollout 采样器

本文件新增（veRL 适配层）：
  AgentEnvRolloutWorker  →  veRL Worker 接口包装
  make_verl_reward_fn    →  veRL reward 回调适配
  train_grpo_verl        →  veRL 训练入口
```

---

## 架构：HybridFlow 三角色

```
prompts ──►  ┌──────────────────────────────────┐
             │  RolloutWorker (vLLM / SGLang)    │
             │  多轮工具调用 rollout               │
             │  调用 AgentEnvironment 执行工具    │
             └────────────┬─────────────────────┘
                          │ trajectories
             ┌────────────▼─────────────────────┐
             │  RewardWorker                     │
             │  调用 reward.py::RewardFn          │
             │  compute_group_advantages (GRPO)  │
             └────────────┬─────────────────────┘
                          │ (trajectory, advantage)
             ┌────────────▼─────────────────────┐
             │  TrainerWorker (FSDP2)            │
             │  GRPO loss + KL 惩罚更新           │
             └──────────────────────────────────┘
```

**关键优势**：Rollout（推理）和 Trainer（梯度更新）在不同进程中并行，
rollout 等待训练时 GPU 不空闲，训练等待 rollout 时也在处理上一批数据。

---

## 两种 Rollout 模式

### Multi-turn（推荐）

模型每次只生成一个工具调用，然后真实执行，得到结果后再继续推理：

```
用户 prompt
  → <think>...</think><tool_call>write_file...</tool_call>
  → [Tool Result] Wrote 80 bytes
  → <tool_call>bash...pytest</tool_call>
  → [Tool Result] 3 passed
  → "任务完成"  ← stop
```

- 奖励信号最准确（每步都有真实工具输出）
- 需要 SGLang（`pip install sglang`）
- 通过 `AgentEnvRolloutWorker` 接入 `AgentEnvironment`

### Single-turn + Replay（兼容模式）

模型一次性生成包含所有工具调用的完整文本，
然后由 `_replay_trajectory_from_text()` 在沙箱中重放：

```python
# 解析文本中所有 <tool_call>...</tool_call>
actions = parse_all_tool_calls(model_output)
# 在沙箱中逐步执行，得到真实的 Trajectory
traj = replay_in_sandbox(task, actions)
# 打分
reward = reward_fn(traj)
```

- 与标准 TRL GRPOTrainer 完全兼容
- 不需要 SGLang，部署更简单
- 弱点：模型生成时看不到工具结果，学习信号弱

---

## 安装

```bash
# 方式 1：稳定版（推荐）
pip install verl

# 方式 2：从源码（获取最新 multi-turn 支持）
pip install git+https://github.com/verl-project/verl.git

# 可选：SGLang（multi-turn rollout 加速，强烈推荐）
pip install sglang

# 或使用 Makefile
make install-verl
```

---

## 运行

```bash
# Multi-turn（真实工具调用，推荐）
make grpo-verl

# Single-turn + replay（兼容模式，更易部署）
make grpo-verl-singleturn

# LoRA 节省显存（单卡 < 24G 时推荐）
make grpo-verl-lora

# 仅测试适配层（无需 GPU，本地可跑）
make test-verl
```

直接运行：

```bash
python train/rl_verl_grpo.py \
    --model checkpoints/sft_merged \
    --multiturn \
    --group-size 8 \
    --rollout-backend vllm \
    --train-backend fsdp
```

---

## 配置详解

```yaml
# configs/verl_grpo.yaml
verl_grpo:
  rollout_backend: "vllm"    # vllm | sglang | hf
  train_backend:   "fsdp"    # fsdp | fsdp2 | megatron
  multiturn:       true      # 推荐 true
  gpu_memory_utilization: 0.7  # rollout 占 70%，留 30% 给训练
  group_size: 8
  kl_coeff:   0.04
  use_lora:   false          # 1.5B 通常不需要
```

**rollout_backend 选择**：

| 后端 | 速度 | Multi-turn | 安装难度 |
|------|------|-----------|---------|
| `sglang` | 最快 | ✅ 原生支持 | 中 |
| `vllm` | 快 | ✅ 支持 | 易 |
| `hf` | 慢 | ❌ 仅 single-turn | 无需安装 |

**train_backend 选择**：

| 后端 | 适合场景 |
|------|---------|
| `fsdp` | 单机多卡（推荐，≤8 GPU）|
| `fsdp2` | 单机多卡（更省显存，需 torch≥2.4）|
| `megatron` | 多机多卡（>8 GPU，设置复杂）|

---

## 与其他模式的对比

| 维度 | custom GRPO | TRL GRPO | **veRL GRPO** |
|------|------------|----------|---------------|
| 工具调用 | ✅ 真实 | ❌ 文本奖励 | ✅ 真实 multi-turn |
| GPU 利用率 | ~50% | ~60% | **~90%** |
| 扩展性 | 单卡 | 多卡 | **多机多卡** |
| 实现复杂度 | 低 | 低 | 中 |
| 适合场景 | 学习/调试 | 快速实验 | **生产部署** |

---

## 显存估算（1.5B 模型，单卡 A100 40G）

```
Actor 模型 (bf16):              ~3GB
Ref 模型 (bf16, 冻结):          ~3GB
vLLM KV Cache (rollout):        ~8GB   (gpu_memory_utilization=0.7 × 40 × 0.7)
FSDP 优化器状态:                 ~6GB
激活值:                          ~4GB
总计:                            ~24GB  ← 40G 卡可以跑
```

双卡（2×A100）可跑 3B 模型，`gpu_memory_utilization` 适当降到 `0.6`。

---

## 全流程

```bash
# 生产级完整管线
make pipeline-verl
# 等价于：
make data          # teacher rollout 生成 SFT 数据
make sft           # LoRA SFT 冷启动
make sft-merge     # 合并权重
make grpo-verl     # veRL multi-turn GRPO
make eval          # 评估结果
```
