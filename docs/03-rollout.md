# 03: Rollout — 轨迹采样器

**文件**: `rollout.py`

## 核心职责

Rollout 采样器负责**驱动 LLM 在环境中执行 agent loop**，生成轨迹数据。

```
RolloutSampler
    │
    ├── sample_one(task, model_fn)
    │     └→ 执行完整 agent loop
    │     └→ 返回 Trajectory
    │
    └── sample_batch(tasks, model_fn, G=8)
          └→ 并发采样 N×G 条轨迹
          └→ 返回 {task_id: [Trajectory × G]}
```

---

## Agent Loop（与 learn-claude-code 完全一致）

`sample_one` 内部实现的就是 s01_agent_loop.py 的核心循环：

```python
messages = [{"role": "user", "content": task["prompt"]}]

while True:
    response = model_fn(messages, tools)      # 调用模型
    messages.append(assistant_turn)

    if response["stop_reason"] != "tool_use":
        final_answer = response["text"]
        break                                  # 模型说完了

    # 执行工具调用
    results = []
    for block in response["content"]:
        if block.type == "tool_use":
            obs = env.step({"name": block.name, "input": block.input})
            results.append(tool_result(block.id, obs["output"]))

    messages.append({"role": "user", "content": results})  # 反馈结果

traj = env.finish(final_answer)
```

这与 `s01_agent_loop.py` 的 `agent_loop()` 函数**完全同构**，
只是把 Anthropic SDK 直接调用换成了可替换的 `model_fn` 接口。

---

## model_fn 接口规范

`model_fn` 是可替换的模型适配器，统一接口：

```python
def model_fn(messages: list[dict], tools: list[dict]) -> dict:
    return {
        "stop_reason": "tool_use" | "end_turn",
        "content": [...],     # 工具调用 blocks
        "text": str,          # 最终文本回答
    }
```

### 三种实现

| 函数 | 适用场景 | 性能 |
|------|------|------|
| `make_anthropic_model_fn()` | teacher rollout（数据生成） | Claude/GPT-4o，质量最高 |
| `make_vllm_model_fn()` | student rollout（RL 训练） | vLLM 批处理，速度最快 |
| `make_hf_model_fn()` | 调试 / 小规模实验 | HF Transformers，最易安装 |

### Qwen 工具调用格式

vLLM 和 HF 模式下，Qwen2.5 使用 ChatML + tool_call 格式：

```xml
<|im_start|>assistant
<think>
先读取测试文件，了解期望行为
</think>
<tool_call>{"name": "read_file", "arguments": {"path": "test_fib.py"}}</tool_call>
<|im_end|>
```

解析正则：`<tool_call>(.*?)</tool_call>`

---

## 批量采样（GRPO 用）

GRPO 需要对同一个 prompt 采样 G=8 条轨迹：

```python
batch = sampler.sample_batch(
    tasks=tasks,
    model_fn=model_fn,
    group_size=8,        # G
    max_workers=4,       # 并发线程数
)
# 返回: {"task_001": [Trajectory×8], "task_002": [Trajectory×8], ...}
```

**并发策略**：使用 `ThreadPoolExecutor`，
注意 `model_fn` 要线程安全（HF model 默认不安全，建议用 vLLM）。

---

## 数据保存

轨迹自动保存为 JSONL 格式到 `data/rl/`：

```json
{
  "task_id": "task_fib_001",
  "prompt": "实现 fib(n)...",
  "tool_calls": [
    {"name": "read_file", "input": {"path": "test_fib.py"}, "output": "..."},
    {"name": "write_file", "input": {"path": "solution.py", "content": "..."}, "output": "Wrote 80 bytes"},
    {"name": "bash", "input": {"command": "pytest test_fib.py"}, "output": "3 passed"}
  ],
  "final_answer": "已实现斐波那契函数，所有测试通过。",
  "status": "success",
  "metadata": {"workdir": "/tmp/agent_env_xxx", "max_steps": 15}
}
```

---

## 性能优化建议

### 单卡 A100（LoRA 训练中）

由于 GPU 同时用于训练，rollout 推理要节省显存：

```python
# 用 CPU 或量化模型做 rollout
model_fn = make_hf_model_fn(model_path, device="cpu")  # 慢但省 GPU 显存

# 或者：训练和推理分开（生产环境推荐）
# 训练进程：GPU 0
# 推理进程：GPU 1（vLLM）
```

### 双卡（推荐生产配置）

```
GPU 0: 训练（GRPO 梯度更新）
GPU 1: 推理（vLLM rollout）
```

这就是 SLIME 框架的核心设计：Actor 和 Trainer 解耦。
