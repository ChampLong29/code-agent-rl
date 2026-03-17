#!/usr/bin/env python3
"""
scripts/cli_agent_local.py

本地模型 Agent CLI 测试：
  - 接入本地 Qwen 模型（目录路径）
  - 运行真实工具调用循环（bash/read_file/write_file/edit_file/search）
  - 观察模型是否原生产生 tool_use
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import AgentEnvironment
from rollout import make_hf_model_fn, make_vllm_model_fn


def _default_task(prompt: str) -> dict:
    return {
        "id": "cli_local_agent",
        "prompt": prompt,
        "setup_files": {
            "README_TASK.md": (
                "你可以通过工具完成任务。\n"
                "建议先 read_file，再 write_file，再 bash 验证。\n"
            )
        },
        "success_criteria": {
            "type": "any",
            "criteria": [
                {"type": "file_exists", "path": "result.txt"},
                {"type": "bash_output", "command": "ls", "expected": "README_TASK.md"},
            ],
        },
    }


def run_cli(
    model_path: str,
    prompt: str,
    backend: str = "vllm",
    max_steps: int = 12,
    temperature: float = 0.2,
):
    task = _default_task(prompt)
    env = AgentEnvironment(max_steps=max_steps, bash_timeout=30, keep_workdir=True)
    env.reset(task)
    tools = AgentEnvironment.TOOL_SCHEMAS

    if backend == "vllm":
        model_fn = make_vllm_model_fn(model_path=model_path, temperature=temperature)
    else:
        model_fn = make_hf_model_fn(model_path=model_path, temperature=temperature)

    messages = [{"role": "user", "content": task["prompt"]}]
    final_answer = ""

    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Backend: {backend}")
    print("=" * 70)

    for step in range(max_steps):
        response = model_fn(messages, tools)
        stop_reason = response.get("stop_reason", "end_turn")
        content = response.get("content", [])
        text = response.get("text", "")

        print(f"\n[step {step + 1}] stop_reason={stop_reason}")
        if text:
            print(f"assistant: {text[:500]}")

        messages.append({"role": "assistant", "content": text or str(content)})

        if stop_reason != "tool_use":
            final_answer = text
            break

        results = []
        for idx, block in enumerate(content):
            if isinstance(block, dict):
                if block.get("type") != "tool_use":
                    continue
                action = {"name": block.get("name", ""), "input": block.get("input", {})}
                tool_use_id = block.get("id", f"call_{step}_{idx}")
            else:
                continue

            print(f"tool_call: {action['name']}({action['input']})")
            obs = env.step(action)
            tool_output = obs.get("output", "")
            print(f"tool_result: {tool_output[:500]}")
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_output,
            })

        messages.append({"role": "user", "content": results})

    traj = env.finish(final_answer)
    print("\n" + "=" * 70)
    print(f"status: {traj.status}")
    print(f"tool_calls: {len(traj.tool_calls)}")
    print(f"workdir: {getattr(env, '_workdir', None)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地模型 Agent CLI 工具调用测试")
    parser.add_argument("--model", required=True, help="本地模型目录（例如 /mnt/d/models/Qwen3.5...）")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--prompt", default="请读取 README_TASK.md，然后写入 result.txt，最后说明你做了什么。")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    run_cli(
        model_path=args.model,
        prompt=args.prompt,
        backend=args.backend,
        max_steps=args.max_steps,
        temperature=args.temperature,
    )
