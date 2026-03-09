# 02: Reward — 奖励函数设计

**文件**: `reward.py`

## Search-R1 的核心洞察

> *"用可验证奖励替代人工反馈"*

传统 RLHF 需要人类标注"哪个回答更好"，成本高且主观。
Search-R1 / DeepSeek-R1 的关键创新：在**可验证任务**上，
奖励函数是确定性的，完全不需要人工。

对于代码任务：
```
pytest 通过 → reward = 1.0
pytest 失败 → reward = 0.0
```
这比人类标注更准确、更廉价、可以大规模并行。

---

## 奖励分层设计

### 总奖励公式

$$R_{total} = w_1 \cdot R_{outcome} + w_2 \cdot R_{process} + w_3 \cdot R_{format} + w_4 \cdot R_{efficiency}$$

默认权重：$w_1=1.0, w_2=0.3, w_3=0.2, w_4=0.1$

### R_outcome：结果奖励（最重要）

```python
def reward_outcome(trajectory) -> float:
    return 1.0 if trajectory.status == "success" else 0.0
```

**增强版（部分 pytest 奖励）**：

```python
def reward_pytest_partial(trajectory) -> float:
    # 解析 "3 passed, 2 failed" → 0.6
    return passed / (passed + failed + errors)
```

部分奖励解决了**稀疏奖励问题**：
- 纯二值奖励：大多数轨迹 reward=0，梯度稀疏，学习慢
- 部分奖励：即使只通过 3/5 个测试，也能得到 0.6 的信号

### R_process：过程奖励

鼓励合理的工具使用顺序：

| 行为 | 奖励 |
|------|------|
| 有 `<think>` 推理链 | +0.3 |
| 使用 ≥2 种不同工具 | +0.2 |
| 先 read 后 write（合理顺序） | +0.2 |
| 错误数随时间减少 | +0.2 |

### R_format：格式奖励

鼓励 Search-R1 风格的结构化输出：

```
<think>
先读测试文件，了解期望的函数签名
然后实现函数，再运行测试验证
</think>
<tool_call>{"name": "read_file", "arguments": {"path": "test_fib.py"}}</tool_call>
```

### R_efficiency：效率奖励

$$R_{eff} = \max(0, 1 - \frac{steps}{max\_steps}) \cdot \mathbb{1}[success]$$

只在任务成功时给效率奖励，避免鼓励 agent 提前放弃。

---

## GRPO 组内优势计算

GRPO 的核心：对同一个 prompt，采样 G 条轨迹，用**组内相对奖励**估算优势。

```python
def compute_group_advantages(trajectories, reward_fn, eps=1e-6):
    rewards = [reward_fn(t)["total"] for t in trajectories]
    mean_r = sum(rewards) / len(rewards)
    std_r  = std(rewards)
    advantages = [(r - mean_r) / (std_r + eps) for r in rewards]
    return advantages
```

**为什么这样做？**

假设 G=8，8条轨迹的 reward 为：
```
[1.0, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.0]
mean = 0.4375
```

归一化后的优势：
```
reward=1.0 → advantage ≈ +1.7  (明显优于平均，应该被强化)
reward=0.0 → advantage ≈ -1.2  (明显差于平均，应该被抑制)
```

这样做的好处：
1. **自适应 baseline**：不需要 Critic 网络
2. **尺度不变**：无论绝对奖励范围如何，优势都是标准化的
3. **稳定性**：避免某个 prompt 特别容易导致奖励爆炸

---

## 奖励函数使用示例

```python
from reward import RewardFn, RewardWeights, compute_group_advantages

# 默认配置
rf = RewardFn()
scores = rf(trajectory)
# {"total": 1.3, "outcome": 1.0, "process": 0.6, "format": 0.4, "efficiency": 0.8}

# 自定义权重（更重视过程）
rf = RewardFn(weights=RewardWeights(outcome=1.0, process=0.8))

# 注册自定义奖励
def complexity_reward(traj):
    return 1.0 if len(traj.tool_calls) <= 5 else 0.5
rf.register("simplicity", complexity_reward, weight=0.2)

# GRPO 组内优势
advantages = compute_group_advantages(group_trajectories, rf)
```

---

## 奖励稀疏性问题与解决方案

### 问题

对于复杂任务（如实现 JsonDB），初期成功率可能只有 5%：
- 95% 的轨迹 outcome reward = 0
- 梯度信号极弱，学习极慢

### 解决方案

1. **部分 pytest 奖励**（`reward_pytest_partial`）：通过率代替二值
2. **过程奖励**：即使最终失败，合理的推理链也有奖励
3. **课程学习**：先训练简单任务（hello.py），再训练复杂任务（JsonDB）
4. **奖励塑形**：在格式奖励中加入工具调用的中间步骤奖励
