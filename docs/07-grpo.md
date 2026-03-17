# 07: GRPO — Search-R1 核心算法

**文件**: `train/rl_grpo.py`

GRPO 是本项目的**首选 RL 算法**，也是 Search-R1 论文的核心方法。

---

## 从 REINFORCE 到 GRPO

### 经典 REINFORCE

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)\right]$$

问题：
- 方差极大（整条轨迹的奖励作为梯度权重）
- 需要减去 baseline（通常是价值函数 $V(s)$）

### PPO 改进

引入重要性采样和裁剪，解决了 off-policy 问题，但需要 critic（价值头）。

### GRPO（Group Relative Policy Optimization）

**核心思想**：不训练 critic，改用**组内相对排名**做 baseline。

对同一个任务 prompt $q$，采样 $G$ 条轨迹 $\{o_1, o_2, ..., o_G\}$，
每条轨迹的优势估计为：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

这就是 `reward.py` 中的 `compute_group_advantages()`。

### GRPO 的损失函数

$$\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}\left[\min\left(\rho_t \hat{A}, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}\right)\right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

其中：
- $\rho_t = \frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}$：重要性采样比
- $\varepsilon = 0.2$：裁剪系数（取自 PPO 经验值）
- $\beta = 0.04$：KL 惩罚系数（取自 Search-R1 论文）
- $\pi_{ref}$：参考模型（SFT 后的冻结模型）

**实现细节**（`grpo_loss()` 函数）：

本实现用**序列级 log prob** 近似 token 级，减少计算复杂度：

```python
log_prob = sum(token_log_probs) / len(token_log_probs)  # 平均
ratio = exp(log_prob - old_log_prob)                     # ρ
clipped = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped * A).mean()
```

这与 Search-R1 的原始实现一致。

---

## GRPO vs PPO vs DPO 对比

| 维度 | GRPO | PPO | DPO |
|------|------|-----|-----|
| 是否需要 critic | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| 是否在线 | ✅ 在线 | ✅ 在线 | ❌ 离线 |
| 显存开销 | 低（1个模型） | 高（actor+critic+ref） | 中（model+ref） |
| 实现复杂度 | 中 | 高 | 低 |
| 适合场景 | 可验证任务 | 连续控制/稀疏奖励 | 人类偏好数据 |
| Search-R1 使用 | ✅ 是 | ❌ 否 | ❌ 否 |

---

## 三种实现模式

### 模式 1：`train_grpo_custom()`（首选）

```python
for iteration in range(config["iterations"]):
    # Step 1: 采样一批任务
    tasks = task_loader.sample(config["batch_size"])
    
    # Step 2: 每个任务采样 G 条轨迹（GRPO 关键）
    all_trajectories = rollout_sampler.sample_batch(
        tasks, model_fn, group_size=config["group_size"]
    )
    
    # Step 3: 计算组内相对优势
    adv_data = compute_group_advantages(all_trajectories, reward_fn)
    
    # Step 4: 计算 GRPO 损失并反向传播
    loss = grpo_loss(policy_model, ref_model, adv_data)
    loss.backward()
    optimizer.step()
```

支持完整工具调用（bash、read_file、write_file），可验证真实代码任务。

### 模式 2：`train_grpo_trl()`（备选）

使用 TRL 的 `GRPOTrainer`，代码更简洁，但奖励函数受限（不支持实际工具调用）：

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_fn],   # 只能用基于文本的格式奖励
    ...
)
```

适合快速验证格式学习效果。

### 模式 3：`train_grpo_slime()`（生产级）

```python
from slime import SLIMEActor, SLIMETrainer

actor = SLIMEActor(model_path, ...)    # 负责 rollout（可单独部署在推理节点）
trainer = SLIMETrainer(actor, ...)     # 负责梯度更新（运行在训练节点）
```

SLIME 架构特点：
- **Actor-Trainer 分离**：rollout 和梯度更新解耦，支持异步
- **vLLM 加速**：rollout 使用 vLLM，速度比 HF 快 3-5x
- **自动扩展**：支持多 GPU / 多节点
- 若 SLIME 未安装，自动 fallback 到 `train_grpo_custom()`

---

## 配置详解

```yaml
# configs/grpo.yaml
grpo:
  group_size: 8         # 每个 prompt 采样 G=8 条轨迹
  iterations: 200       # 训练迭代次数
  batch_size: 4         # 每次采样 4 个 task
  learning_rate: 5.0e-6 # 比 SFT 小 40x，精细调整
  kl_coeff: 0.04        # KL 惩罚，来自 Search-R1 论文
  clip_epsilon: 0.2     # PPO 裁剪系数
  rollout_temperature: 0.7   # 生成多样性
  max_steps: 10         # 每条轨迹最大工具调用步数
  max_tokens: 1024      # 每次生成最大 token
```

**关键超参调优建议**：

| 超参 | 建议范围 | 影响 |
|------|----------|------|
| `group_size` | 4-16 | 越大方差越小，但越慢 |
| `kl_coeff` | 0.01-0.1 | 太小会遗忘 SFT，太大限制探索 |
| `rollout_temperature` | 0.5-1.0 | 太低缺乏多样性，太高格式混乱 |
| `learning_rate` | 1e-6 - 1e-5 | 建议从 5e-6 开始 |
| `clip_epsilon` | 0.1-0.3 | 标准 PPO 值 0.2 通常最优 |

---

## 运行

```bash
# 推荐：自定义循环（支持真实工具调用）
make grpo

# 备选：TRL GRPOTrainer（快速实验）
make grpo-trl

# 生产级：SLIME 框架
make grpo-slime
```

---

## 训练监控

运行后应观察以下指标趋势（iteration=0→200）：

```
mean_reward:     0.3 → 0.65+   ← 核心指标，持续上升
success_rate:    30% → 70%+    ← 任务完成率
format_rate:     60% → 90%+    ← 格式遵守率（<think> 标签）
kl_divergence:   ~0 → ~0.5    ← 不应超过 1.0，否则降低 lr
mean_steps:      8 → 4-6       ← 效率提升（步数应减少）
```

**异常情况处理**：

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| reward 不上升 | KL 太大，被惩罚压制 | 降低 `kl_coeff` 到 0.01 |
| reward 振荡剧烈 | lr 太高 | 降低到 1e-6 |
| 格式退化（无 `<think>`）| SFT 效果弱 | 增加 SFT 数据后重新冷启动 |
| OOM | group_size 太大 | 降到 4，增加 grad_acc |
| reward 崩溃到 0 | KL 太小，模式崩溃 | 增加 `kl_coeff` 到 0.1 |
