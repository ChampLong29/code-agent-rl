# 04: SFT LoRA — 冷启动监督微调

**文件**: `train/sft_lora.py`

## 为什么需要 SFT 冷启动？

直接对 base 模型做 RL 几乎不可行：

```
base 模型对 "implement fib(n) with pytest" 的输出：
  "当然可以！斐波那契数列是..."  ← 普通对话格式，没有工具调用

期望格式：
  <think>先读测试文件...</think>
  <tool_call>{"name": "read_file", ...}</tool_call>
  ...
  "测试通过，任务完成"
```

SFT 阶段的目标：**格式迁移**，不是学会解题，而是学会 agent 格式。

---

## LoRA 原理

LoRA（Low-Rank Adaptation）冻结原始权重 $W$，只训练低秩矩阵：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

| 参数 | 含义 | 本项目设置 |
|------|------|------|
| `r` | 秩（控制参数量） | 16 |
| `alpha` | 缩放系数（通常 = 2r） | 32 |
| `dropout` | 正则化 | 0.05 |
| target modules | 训练哪些层 | q,k,v,o,gate,up |

**1.5B 模型的可训练参数量**：

```
全参训练: 1.5B 参数
LoRA r=16: ~13M 参数 (~0.9%)
```

显存对比：
- 全参 AdamW: 1.5B × 16byte = 24GB（仅优化器）
- LoRA AdamW: 13M × 16byte = ~200MB

---

## 数据格式

SFT 数据由 `scripts/generate_sft_data.py` 生成，
格式为 ShareGPT 风格（兼容 LLaMA-Factory 和 TRL）：

```json
{
  "id": "task_fib_001_0",
  "conversations": [
    {
      "role": "system",
      "content": "You are a coding agent. Think step by step using <think> tags..."
    },
    {
      "role": "user",
      "content": "实现 fib(n) 函数，通过 test_fib.py 的测试"
    },
    {
      "role": "assistant",
      "content": "<think>\n先读取测试文件了解期望...\n</think>\n<tool_call>{...}</tool_call>"
    },
    {
      "role": "tool",
      "content": "[Tool Result]\nfrom solution import fib\ndef test_fib(): assert fib(10)==55"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"write_file\", ...}</tool_call>"
    },
    {
      "role": "tool",
      "content": "[Tool Result]\nWrote 80 bytes to solution.py"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"pytest\"}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "[Tool Result]\n3 passed in 0.1s"
    },
    {
      "role": "assistant",
      "content": "已实现 fib 函数，所有测试通过。"
    }
  ]
}
```

---

## 训练配置

```yaml
# configs/sft.yaml
sft:
  training:
    max_seq_length:               2048   # 覆盖约 90% 的 agent 轨迹
    num_train_epochs:             3
    per_device_train_batch_size:  1
    gradient_accumulation_steps:  8      # 等效 bs=8
    learning_rate:                2.0e-4 # LoRA 可以用较大学习率
    lr_scheduler_type:            "cosine"
```

**显存使用分析（1.5B，单卡 A100 40G）**：

```
模型权重 (bf16):      ~3GB
LoRA 参数:            ~50MB
激活值 (seq=2048):    ~4GB
优化器状态 (LoRA only): ~200MB
梯度缓冲:             ~50MB
总计:                 ~8GB   ← 远低于 40G 上限
```

---

## 运行

```bash
# 方式 1：使用 Makefile
make sft

# 方式 2：直接运行
python train/sft_lora.py \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --data data/sft/train.jsonl \
    --output checkpoints/sft_lora \
    --epochs 3

# 方式 3：QLoRA（4-bit，节省约 30% 显存）
python train/sft_lora.py --4bit

# 合并权重（RL 阶段需要完整模型）
python train/sft_lora.py --merge --merge-output checkpoints/sft_merged
```

---

## 验证 SFT 效果

SFT 后应该能看到：
1. 输出中出现 `<think>` 标签
2. 输出中出现 `<tool_call>` 格式
3. 格式遵守率（format_rate）从 ~5% 提升到 ~60%+
4. 任务成功率从 ~10% 提升到 ~30%（格式正确是基础）

```bash
make eval-model MODEL_PATH=checkpoints/sft_merged
```

预期结果：
```
成功率:      30-50%
格式遵守率:  60-80%  (有 <think> 标签)
平均步数:    4-8
```

如果格式遵守率 < 50%，说明 SFT 数据量不足，增加 `--n-per-task 5`。
