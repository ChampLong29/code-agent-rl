# 05: PPO — Actor-Critic 强化学习

**文件**: `train/rl_ppo.py`

---

## 架构概览

PPO 需要两个模型组件：

```
┌─────────────────────────────────────┐
│           Qwen2.5-Coder             │  ← 共享 backbone（冻结 LoRA 基础层）
├──────────────────┬──────────────────┤
│  Policy Head     │   Value Head     │
│ (语言模型原有的)  │ (我们新增的)      │
│ → token 概率分布 │ → 标量 V(s)      │
└──────────────────┴──────────────────┘
           ActorCritic 类
```

**Value Head 设计**（`ValueHead` 类）：

```python
self.value_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.Tanh(),
    nn.Linear(hidden_size // 2, 1)
)
```

使用 `Tanh` 而非 `ReLU`，避免价值输出无上界。

---

## 广义优势估计（GAE）

GAE 是 PPO 中估计优势函数的标准方法：

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

**参数含义**：
- $\gamma = 0.99$：折扣因子（接近 1 = 重视长期回报）
- $\lambda = 0.95$：GAE 平滑系数（接近 1 = 低偏差高方差，接近 0 = 高偏差低方差）

**代码实现**（`compute_gae()`）：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages
```

---

## PPO 损失函数

PPO 同时优化 3 个目标：

$$\mathcal{L}_{PPO} = \mathcal{L}_{policy} + c_1 \mathcal{L}_{value} - c_2 \mathcal{L}_{entropy}$$

**1. 策略损失（带裁剪）**：
$$\mathcal{L}_{policy} = -\mathbb{E}_t\left[\min\left(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

**2. 价值函数损失**：
$$\mathcal{L}_{value} = \mathbb{E}_t\left[(V_\theta(s_t) - V_{target})^2\right]$$

**3. 熵奖励**（防止过早收敛）：
$$\mathcal{L}_{entropy} = \mathbb{E}_t\left[\mathcal{H}[\pi_\theta(\cdot|s_t)]\right]$$

---

## 两种实现方式

### 方式 1：`train_ppo()`（自定义）

完整支持工具调用，但实现复杂度高：
- 需要管理 `PPOBuffer`（存储 states, actions, rewards, values）
- 需要维护两个模型版本（actor + ref）
- 手动计算 GAE 和 PPO 损失

适合**学习算法细节**或需要完全控制训练循环。

### 方式 2：`train_ppo_trl()`（推荐）

使用 TRL 的 `PPOTrainer`：

```python
trainer = PPOTrainer(
    config=PPOConfig(
        learning_rate=1e-5,
        mini_batch_size=1,
        ppo_epochs=4,
    ),
    model=model,          # actor
    ref_model=ref_model,  # 参考模型（KL 约束）
    tokenizer=tokenizer,
    ...
)
```

**生产推荐**，TRL 处理了大量工程细节（并行化、混合精度等）。

---

## 显存需求（与 GRPO 对比）

| 组件 | PPO | GRPO |
|------|-----|------|
| Actor 模型 | ~3GB | ~3GB |
| Critic（Value Head） | ~0.1GB | ❌ 不需要 |
| Ref 模型 | ~3GB | ~3GB |
| 优化器状态 | ~6GB | ~6GB |
| Rollout 缓存 | ~2GB | ~2GB |
| **总计** | **~14GB** | **~11GB** |

GRPO 省去了 critic，显存节省约 15-20%。

---

## 运行

```bash
# 使用 Makefile
make ppo

# 直接运行（自定义 PPO 循环）
python train/rl_ppo.py \
    --model checkpoints/sft_merged \
    --tasks data/raw/tasks.jsonl \
    --output checkpoints/ppo

# TRL PPOTrainer（推荐）
python train/rl_ppo.py --use-trl
```

---

## 何时选 PPO vs GRPO？

| 场景 | 推荐 |
|------|------|
| 标准代码任务（本项目） | **GRPO**（更简单，效果相当） |
| 奖励极稀疏（>10步才有回报） | **PPO**（价值函数引导更好） |
| 连续状态空间 | PPO |
| 显存紧张（单卡 <24G） | **GRPO** |
| 需要精确信用分配 | PPO |

对于本项目的代码生成场景，**GRPO 通常优于或等于 PPO**，且实现更简单。
