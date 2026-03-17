# 06: DPO — 直接偏好优化

**文件**: `train/rl_dpo.py`

---

## 算法原理

### 从 RLHF 到 DPO

经典 RLHF 分三步：
1. 训练奖励模型 $r_\phi$
2. 用 PPO 优化 $\pi_\theta$ 使 $r_\phi$ 最大
3. 加 KL 约束防止偏离参考策略

这是一个**有约束的优化问题**：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta}\left[r(x, y)\right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

DPO 的洞见：这个问题有**解析解**！

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

反解出奖励函数：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

将这个关系代入 Bradley-Terry 偏好模型，$Z(x)$ 消去，得到 **DPO 损失**：

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 $y_w$ 是 preferred（胜者），$y_l$ 是 rejected（败者）。

**直觉理解**：增大胜者的相对 log prob，降低败者的相对 log prob，相对 ref 模型而言。

---

## 偏好对的生成

`generate_dpo_pairs()` 自动生成训练数据：

```python
for task in tasks:
    # 每个 task 运行 n_rollouts=6 次
    trajectories = [rollout(task) for _ in range(6)]
    
    # 计算每条轨迹的奖励
    rewards = [reward_fn(traj) for traj in trajectories]
    
    # 按奖励排序
    sorted_trajs = sorted(zip(rewards, trajectories))
    
    # 过滤奖励差距太小的对
    if max_reward - min_reward >= min_reward_diff:  # 默认 0.3
        pairs.append({
            "prompt": task.prompt,
            "chosen": best_trajectory_text,    # 高奖励轨迹
            "rejected": worst_trajectory_text, # 低奖励轨迹
        })
```

**为什么设 `min_reward_diff=0.3`？**

奖励差距太小时，偏好信号噪声过大，模型无法学到有意义的区分：
- diff < 0.1：接近随机，不应作为训练数据
- diff 0.3-0.7：明确偏好，最有价值
- diff > 0.8：通常是 success vs failure，信号最强

---

## β 参数的意义

$\beta$ 控制 DPO 损失对参考模型的偏离程度：

| β 值 | 效果 |
|------|------|
| **小**（0.05） | 允许大幅偏离 ref，快速学习偏好，但可能遗忘 SFT 格式 |
| **中**（0.1，默认） | 平衡偏好学习与格式保持 |
| **大**（0.5） | 紧靠 ref 模型，变化保守，适合数据质量不确定时 |

本项目推荐 `β=0.1`，与原始 DPO 论文保持一致。

---

## 与 GRPO/PPO 的对比

| 维度 | DPO | GRPO | PPO |
|------|-----|------|-----|
| **在线 vs 离线** | 离线 | 在线 | 在线 |
| **奖励模型** | 不需要 | 不需要 | 不需要（直接奖励） |
| **数据效率** | 低（需要大量偏好对） | 高（实时反馈） | 高 |
| **实现复杂度** | 低 | 中 | 高 |
| **适合场景** | 有高质量偏好数据 | 可验证任务 | 任何任务 |
| **Search-R1 使用** | ❌ 否 | ✅ 是 | ❌ 否 |

**本项目中 DPO 的定位**：
- **不是主要方法**（主要是 GRPO）
- 可用于**数据增强**：先 GRPO 训练，生成高质量轨迹对，再 DPO 精调
- 可作为 GRPO 的**替代方案**（当算力不足以做在线 RL 时）

---

## 运行

```bash
# Step 1：生成偏好对数据
make dpo-pairs

# Step 2：DPO 训练
make dpo

# 或者直接运行
python train/rl_dpo.py \
    --model checkpoints/sft_merged \
    --tasks data/raw/tasks.jsonl \
    --output checkpoints/dpo
```

---

## 常见问题

**Q：DPO 训练后格式退化？**
A：β 太小，增大到 0.2-0.5。DPO 的 ref 模型是 SFT 模型，β 小时偏离太远会忘记 SFT 格式。

**Q：偏好对不够怎么办？**
A：提高 `n_rollouts`（从 6 增到 10），或降低 `min_reward_diff`（从 0.3 降到 0.2）。

**Q：DPO 损失不下降？**
A：检查偏好对质量，确认 chosen 的实际奖励确实高于 rejected。打印 `reward_diff` 的分布。
