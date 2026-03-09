#!/usr/bin/env python3
"""
rollout.py — Rollout 采样器

负责驱动模型（本地 vLLM 或 API）在环境中执行 agent loop，
生成用于 RL 训练的轨迹数据。

两种采样模式：
    1. teacher_rollout  — 用强模型（Claude/GPT-4o）生成 SFT 数据
    2. student_rollout  — 用训练中的学生模型（Qwen）生成 RL 数据

架构：
    ┌──────────────────────────────────────────────────────────┐
    │  RolloutSampler                                          │
    │                                                          │
    │  sample_one(task, model_fn)                              │
    │    → AgentEnvironment.reset(task)                        │
    │    → 循环：model_fn(messages) → action → env.step()      │
    │    → env.finish(answer)                                  │
    │    → Trajectory                                          │
    │                                                          │
    │  sample_batch(tasks, model_fn, G=8)                      │
    │    → 每个 task 采样 G 条轨迹（用于 GRPO）               │
    │    → 返回 {task_id: [Trajectory x G]}                    │
    └──────────────────────────────────────────────────────────┘

model_fn 接口（可替换）：
    def model_fn(messages: list[dict], tools: list[dict]) -> dict:
        \"\"\"
        Returns:
            {
                "stop_reason": "tool_use" | "end_turn",
                "content": [...],    # Anthropic 格式
                "text": str,         # 最终文本（stop_reason=end_turn 时）
            }
        \"\"\"
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from environment import AgentEnvironment, Trajectory


# ---------------------------------------------------------------------------
# Anthropic API 模型函数（用于 teacher rollout）
# ---------------------------------------------------------------------------

def make_anthropic_model_fn(
    model: str = "claude-3-5-haiku-20241022",
    system: str | None = None,
    max_tokens: int = 4096,
) -> Callable:
    """
    创建一个基于 Anthropic API 的 model_fn。
    用于 teacher rollout（生成 SFT 数据）。

    要求环境变量：ANTHROPIC_API_KEY 或 ANTHROPIC_BASE_URL
    """
    from anthropic import Anthropic
    from dotenv import load_dotenv
    load_dotenv(override=True)

    client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

    _system = system or (
        "You are a coding agent. Use tools to solve programming tasks step by step.\n"
        "Always think before acting: wrap your reasoning in <think>...</think> tags.\n"
        "Use tools when needed. Give a final answer when done."
    )

    def model_fn(messages: list[dict], tools: list[dict]) -> dict:
        response = client.messages.create(
            model=model,
            system=_system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        result = {
            "stop_reason": response.stop_reason,
            "content": response.content,
            "text": "",
        }
        if response.stop_reason == "end_turn":
            result["text"] = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
        return result

    return model_fn


# ---------------------------------------------------------------------------
# vLLM 本地模型函数（用于 student rollout / RL 训练）
# ---------------------------------------------------------------------------

def make_vllm_model_fn(
    model_path: str,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.8,
    gpu_memory_utilization: float = 0.85,
) -> Callable:
    """
    创建一个基于 vLLM 的本地 model_fn。
    用于 student rollout（RL 训练时在线采样）。

    Qwen2.5 支持 function calling，工具调用格式自动处理。
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("请安装 vllm: pip install vllm")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["</tool_call>", "<|im_end|>"],
    )

    _system = system or (
        "You are a coding agent. Use tools to solve programming tasks.\n"
        "Think step by step using <think>...</think> tags before calling tools."
    )

    def _parse_tool_call(text: str) -> dict | None:
        """
        解析 Qwen 格式的工具调用：
            <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>
        """
        import re
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        match = pattern.search(text)
        if not match:
            return None
        try:
            call_data = json.loads(match.group(1).strip())
            return {
                "name": call_data.get("name", ""),
                "input": call_data.get("arguments", {}),
            }
        except json.JSONDecodeError:
            return None

    def model_fn(messages: list[dict], tools: list[dict]) -> dict:
        # 格式化为 Qwen chat 格式
        prompt_messages = [{"role": "system", "content": _system}] + messages
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = llm.generate([text], sampling_params)
        generated = outputs[0].outputs[0].text

        # 解析是工具调用还是最终回答
        tool_call = _parse_tool_call(generated)
        if tool_call:
            return {
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", **tool_call}],
                "text": "",
                "raw": generated,
            }
        else:
            return {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": generated}],
                "text": generated,
                "raw": generated,
            }

    return model_fn


# ---------------------------------------------------------------------------
# HuggingFace Transformers 模型函数（轻量，调试用）
# ---------------------------------------------------------------------------

def make_hf_model_fn(
    model_path: str,
    system: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "auto",
) -> Callable:
    """
    基于 HuggingFace Transformers 的轻量 model_fn。
    适合在没有 vLLM 的情况下调试。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        raise ImportError("请安装 transformers: pip install transformers torch")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )

    _system = system or "You are a coding agent. Think step by step using <think> tags."

    def model_fn(messages: list[dict], tools: list[dict]) -> dict:
        prompt_messages = [{"role": "system", "content": _system}] + messages
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return {"stop_reason": "end_turn", "content": [], "text": generated}

    return model_fn


# ---------------------------------------------------------------------------
# 核心采样器
# ---------------------------------------------------------------------------

class RolloutSampler:
    """
    驱动 model_fn 在 AgentEnvironment 中执行 agent loop，生成 Trajectory。

    agent loop 逻辑（与 learn-claude-code 保持一致）：
        while stop_reason == "tool_use":
            response = model_fn(messages, tools)
            执行工具调用
            追加 tool_result
        finish(final_answer)
    """

    def __init__(
        self,
        max_steps: int = 20,
        bash_timeout: int = 30,
        save_dir: str | Path | None = None,
    ):
        self.max_steps = max_steps
        self.bash_timeout = bash_timeout
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def sample_one(
        self,
        task: dict,
        model_fn: Callable,
        keep_workdir: bool = False,
    ) -> Trajectory:
        """
        采样单条轨迹。

        核心 agent loop（学自 learn-claude-code/agents/s01_agent_loop.py）:
            messages = [{"role": "user", "content": prompt}]
            while True:
                response = model_fn(messages, tools)
                messages.append(assistant_turn)
                if stop_reason != "tool_use": break
                results = execute_all_tool_calls(response)
                messages.append({"role": "user", "content": results})
        """
        env = AgentEnvironment(
            max_steps=self.max_steps,
            bash_timeout=self.bash_timeout,
            keep_workdir=keep_workdir,
        )
        trajectory = env.reset(task)
        tools = AgentEnvironment.TOOL_SCHEMAS

        messages = [{"role": "user", "content": task["prompt"]}]
        trajectory.messages = messages

        final_answer = ""

        for _step in range(self.max_steps):
            try:
                response = model_fn(messages, tools)
            except Exception as e:
                trajectory.status = "failure"
                trajectory.metadata["error"] = str(e)
                break

            stop_reason = response.get("stop_reason", "end_turn")
            content = response.get("content", [])

            # 构造 assistant 消息
            # Anthropic 格式: content 是 list of blocks
            # vLLM 格式: content 是解析后的 dict list
            assistant_msg: dict
            if content and hasattr(content[0], "type"):
                # Anthropic SDK 对象
                assistant_msg = {"role": "assistant", "content": content}
            else:
                # 普通 dict 格式
                text_content = response.get("text", "")
                assistant_msg = {"role": "assistant", "content": text_content or str(content)}

            messages.append(assistant_msg)

            if stop_reason != "tool_use":
                final_answer = response.get("text", "")
                break

            # 执行工具调用，收集 results
            results = []
            for block in content:
                # 处理 Anthropic SDK block 对象
                if hasattr(block, "type"):
                    if block.type != "tool_use":
                        continue
                    action = {"name": block.name, "input": block.input}
                    tool_use_id = block.id
                else:
                    # dict 格式（vLLM 解析结果）
                    if block.get("type") != "tool_use":
                        continue
                    action = {"name": block.get("name", ""), "input": block.get("input", {})}
                    tool_use_id = block.get("id", f"call_{_step}")

                obs = env.step(action)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": obs["output"],
                })

                if obs["done"]:
                    break

            messages.append({"role": "user", "content": results})
            trajectory.messages = list(messages)

            if trajectory.status == "truncated":
                break

        trajectory = env.finish(final_answer)
        trajectory.messages = list(messages)

        if self.save_dir:
            self._save(trajectory)

        return trajectory

    def sample_batch(
        self,
        tasks: list[dict],
        model_fn: Callable,
        group_size: int = 8,
        max_workers: int = 4,
    ) -> dict[str, list[Trajectory]]:
        """
        批量采样，每个 task 采样 group_size 条轨迹。
        用于 GRPO 的组内优势计算。

        Args:
            tasks: task 列表
            model_fn: 模型函数
            group_size: 每个 task 的轨迹数量 G（GRPO 中的组大小）
            max_workers: 并发线程数（注意 model_fn 是否线程安全）

        Returns:
            {task_id: [Trajectory x group_size]}
        """
        results: dict[str, list[Trajectory]] = {
            task["id"]: [] for task in tasks
        }

        all_jobs = [
            (task, i)
            for task in tasks
            for i in range(group_size)
        ]

        print(f"采样 {len(tasks)} 个任务 × {group_size} 条轨迹 = {len(all_jobs)} 次 rollout")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(self.sample_one, task, model_fn): (task["id"], idx)
                for task, idx in all_jobs
            }
            for future in as_completed(future_to_job):
                task_id, idx = future_to_job[future]
                try:
                    traj = future.result()
                    results[task_id].append(traj)
                    status_icon = "✓" if traj.status == "success" else "✗"
                    print(f"  {status_icon} {task_id}[{idx}] ({traj.status}, "
                          f"{len(traj.tool_calls)} steps)")
                except Exception as e:
                    print(f"  ✗ {task_id}[{idx}] ERROR: {e}")

        return results

    def _save(self, trajectory: Trajectory):
        """保存轨迹到 JSONL 文件。"""
        ts = int(time.time())
        filename = self.save_dir / f"{trajectory.task_id}_{ts}.jsonl"
        with open(filename, "w") as f:
            # 序列化 messages（Anthropic SDK 对象需要特殊处理）
            traj_dict = trajectory.to_dict()
            f.write(json.dumps(traj_dict, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# 快速测试（使用 mock model_fn）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    def mock_model_fn(messages: list[dict], tools: list[dict]) -> dict:
        """模拟模型：随机调用工具或直接返回答案。"""
        call_count = len([m for m in messages if m["role"] == "user"]) - 1
        if call_count < 2 and random.random() < 0.7:
            return {
                "stop_reason": "tool_use",
                "content": [{
                    "type": "tool_use",
                    "id": f"call_{call_count}",
                    "name": "bash",
                    "input": {"command": f"echo 'step {call_count}'"},
                }],
                "text": "",
            }
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "任务完成。"}],
            "text": "任务完成。",
        }

    task = {
        "id": "demo_001",
        "prompt": "列出当前目录下的文件",
        "setup_files": {},
        "success_criteria": {"type": "bash_output", "command": "echo ok", "expected": "ok"},
    }

    sampler = RolloutSampler(max_steps=5)
    traj = sampler.sample_one(task, mock_model_fn)
    print(f"Status: {traj.status}")
    print(f"Steps: {len(traj.tool_calls)}")

    # 批量采样
    tasks = [task, {**task, "id": "demo_002"}]
    batch = sampler.sample_batch(tasks, mock_model_fn, group_size=3, max_workers=2)
    for tid, trajs in batch.items():
        success_rate = sum(1 for t in trajs if t.status == "success") / len(trajs)
        print(f"{tid}: {len(trajs)} 条, 成功率 {success_rate:.0%}")
