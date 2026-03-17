#!/usr/bin/env python3
"""
train/rl_verl_grpo.py — Phase 2d: GRPO（veRL 生产级框架）

veRL (Volcano Engine Reinforcement Learning) 是字节跳动开源的生产级 RL 训练框架，
相比本项目的 custom GRPO 模式，核心优势在于：

  - Actor/Rollout/Trainer 三角色分离，GPU 利用率接近 100%
  - 原生 vLLM / SGLang 推理加速，支持异步 rollout
  - FSDP2 / Megatron 分布式训练，可扩展到数百张卡
  - 多轮工具调用（multi-turn tool use）原生支持
  - 完整的 LoRA RL 支持，节省显存

本文件与现有代码的关系（最小侵入原则）：
  ┌─────────────────────────────────────────────────┐
  │  不改动的现有模块（教学核心，完整保留）           │
  │    environment.py  →  沙箱执行 + 轨迹记录        │
  │    reward.py       →  可验证奖励函数族            │
  │    rollout.py      →  rollout 采样器             │
  ├─────────────────────────────────────────────────┤
  │  本文件新增（veRL 生产级适配层）                  │
  │    AgentEnvWorker  →  veRL Worker 接口包装        │
  │    verl_reward_fn  →  veRL reward 回调适配        │
  │    train_grpo_verl →  veRL 训练入口               │
  └─────────────────────────────────────────────────┘

架构图（veRL HybridFlow）：
                   ┌──────────────────────────────────┐
    prompts ──►    │  RolloutWorker (vLLM / SGLang)    │
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

参考：
  - veRL 文档: https://verl.readthedocs.io
  - veRL multi-turn: https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html
  - veRL GRPO: https://verl.readthedocs.io/en/latest/algo/grpo.html

安装 veRL：
  pip install verl
  # 或从源码（获取最新 multi-turn 支持）：
  pip install git+https://github.com/verl-project/verl.git
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import AgentEnvironment, TaskLoader, Trajectory
from reward import RewardFn, RewardWeights, compute_group_advantages
from rollout import RolloutSampler
from scripts.generate_sft_data import SEED_TASKS
from hub_utils import resolve_model_path


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "checkpoints/sft_merged")
OUTPUT_DIR = os.environ.get("VERL_OUTPUT_DIR", "checkpoints/verl_grpo")
DATA_PATH = os.environ.get("RL_DATA_PATH", "data/rl")

VERL_GRPO_CONFIG = {
    # ── 采样参数 ──
    "group_size": 8,                   # G：每 prompt 采样多少条轨迹
    "max_steps_per_episode": 15,
    "rollout_temperature": 0.8,

    # ── 训练参数 ──
    "num_iterations": 200,
    "prompts_per_batch": 4,
    "learning_rate": 5e-6,
    "kl_coeff": 0.04,                  # β（Search-R1 默认值）
    "clip_epsilon": 0.2,               # ε
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
    "save_steps": 20,
    "max_seq_length": 2048,

    # ── veRL 特有参数 ──
    "train_backend": "fsdp",           # "fsdp" | "fsdp2" | "megatron"
    "rollout_backend": "vllm",         # "vllm" | "sglang" | "hf"
    "gpu_memory_utilization": 0.7,     # rollout vLLM 占用比（留出训练空间）
    "n_gpus_per_node": torch.cuda.device_count() or 1,

    # ── LoRA RL（显存紧张时开启）──
    "use_lora": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


# ---------------------------------------------------------------------------
# veRL Reward 适配层
# ---------------------------------------------------------------------------

def make_verl_reward_fn(
    tasks: list[dict],
    reward_weights: RewardWeights | None = None,
) -> Callable:
    """
    将 reward.py::RewardFn 包装为 veRL 期望的 reward function 接口。

    veRL reward function 签名：
        reward_fn(data_item: dict) -> float

    data_item 包含：
        {
            "prompt":      str,               # 原始 prompt
            "response":    str,               # 模型生成的完整文本
            "trajectory":  Trajectory | None, # 若已有轨迹对象直接使用
        }

    这是关键的"胶水层"：现有 reward.py 代码零改动，
    只需在这里完成格式适配。
    """
    reward_fn = RewardFn(
        weights=reward_weights or RewardWeights(),
        use_partial_pytest=True,
        max_steps=VERL_GRPO_CONFIG["max_steps_per_episode"],
    )

    task_map = {t["prompt"]: t for t in tasks}

    def verl_reward_fn(data_item: dict) -> float:
        # 情况 1：data_item 已经携带了完整 Trajectory（multi-turn rollout 后）
        traj = data_item.get("trajectory")
        if isinstance(traj, Trajectory):
            scores = reward_fn(traj)
            return scores["total"]

        # 情况 2：只有文本输出（single-turn 或 TRL 兼容模式）
        prompt = data_item.get("prompt", "")
        response = data_item.get("response", "")
        task = task_map.get(prompt)

        if task is None:
            # prompt 未匹配到已知任务，只做格式奖励
            traj = Trajectory(task_id="unknown", prompt=prompt)
            traj.final_answer = response
            traj.status = "failure"
        else:
            # 在沙箱中重放：从文本中解析工具调用序列，然后真实执行
            traj = _replay_trajectory_from_text(task, response)

        scores = reward_fn(traj)
        return scores["total"]

    return verl_reward_fn


def _replay_trajectory_from_text(task: dict, model_text: str) -> Trajectory:
    """
    从模型生成的文本中解析工具调用，并在沙箱中真实执行，
    得到有真实 status 的 Trajectory。

    这是 veRL single-turn 模式下实现可验证奖励的核心：
    即使 rollout 是一次性生成的，也能通过"重放"得到真实奖励。

    解析的工具调用格式（Qwen function calling）：
        <tool_call>{"name": "bash", "arguments": {"command": "pytest ..."}}</tool_call>
    """
    import re

    tool_call_pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
    )
    final_answer_pattern = re.compile(
        r"<\|im_end\|>|(?:^|\n)(?:Final Answer|最终回答)[：:]\s*(.*?)$",
        re.DOTALL | re.MULTILINE,
    )

    # 提取所有工具调用
    raw_calls = tool_call_pattern.findall(model_text)
    parsed_actions = []
    for raw in raw_calls:
        try:
            call_data = json.loads(raw.strip())
            name = call_data.get("name", "")
            args = call_data.get("arguments", call_data.get("input", {}))
            if name in {"bash", "read_file", "write_file", "edit_file"}:
                parsed_actions.append({"name": name, "input": args})
        except (json.JSONDecodeError, ValueError):
            continue

    # 提取最终回答
    final_answer = ""
    fa_match = final_answer_pattern.search(model_text)
    if fa_match and fa_match.group(1):
        final_answer = fa_match.group(1).strip()
    else:
        # 取最后一段非工具调用文本
        cleaned = tool_call_pattern.sub("", model_text).strip()
        final_answer = cleaned[-500:] if len(cleaned) > 500 else cleaned

    # 在沙箱中执行
    with AgentEnvironment(
        max_steps=len(parsed_actions) + 1,
        bash_timeout=30,
        keep_workdir=False,
    ) as env:
        env.reset(task)
        for action in parsed_actions:
            obs = env.step(action)
            if obs.get("done"):
                break
        traj = env.finish(final_answer)

    return traj


# ---------------------------------------------------------------------------
# veRL Multi-turn Rollout Worker（核心：真实工具调用）
# ---------------------------------------------------------------------------

class AgentEnvRolloutWorker:
    """
    veRL RolloutWorker 的包装实现。

    veRL 的 multi-turn rollout 接口（sglang_multiturn 风格）：
        每个 "turn" 是一次模型推理 + 可选的环境交互。

    本类作为 veRL 的 env_step 回调，在每次模型输出后：
        1. 解析工具调用
        2. 调用 AgentEnvironment.step() 执行
        3. 返回工具结果作为下一轮的 user 消息

    这样 veRL 的 rollout 循环就变成了真正的 agent loop，
    每一步都有真实工具执行，奖励信号完全可验证。

    用法（veRL multi-turn config 中配置）：
        env_class = AgentEnvRolloutWorker
        env_kwargs = {"tasks": SEED_TASKS, "max_steps": 15}
    """

    def __init__(self, tasks: list[dict], max_steps: int = 15, bash_timeout: int = 30):
        self.tasks = {t["prompt"]: t for t in tasks}
        self.max_steps = max_steps
        self.bash_timeout = bash_timeout
        # 每个并发 rollout 维护独立的 env 实例
        self._envs: dict[str, AgentEnvironment] = {}

    def reset(self, session_id: str, prompt: str) -> str:
        """
        veRL env reset 回调。
        创建新的沙箱环境，返回初始观测（空字符串，由 prompt 本身承担）。
        """
        task = self.tasks.get(prompt)
        if task is None:
            # 未知 prompt：构造最小 task
            task = {"id": session_id, "prompt": prompt,
                    "setup_files": {}, "success_criteria": {}}

        env = AgentEnvironment(
            max_steps=self.max_steps,
            bash_timeout=self.bash_timeout,
        )
        env.reset(task)
        self._envs[session_id] = env
        return ""   # veRL 期望返回初始 observation

    def step(self, session_id: str, model_output: str) -> tuple[str, bool, float]:
        """
        veRL env step 回调。

        Args:
            session_id: rollout 会话 ID
            model_output: 模型本轮生成的文本

        Returns:
            (observation, done, reward_signal)
            - observation: 工具执行结果（作为下一轮 user 消息内容）
            - done: 是否结束
            - reward_signal: 中间奖励（0，最终奖励由 RewardWorker 计算）
        """
        env = self._envs.get(session_id)
        if env is None:
            return "Error: session not found", True, 0.0

        # 解析工具调用
        action = _parse_single_tool_call(model_output)

        if action is None:
            # 没有工具调用 → 最终回答，结束 episode
            traj = env.finish(model_output)
            self._envs.pop(session_id, None)
            return "", True, 0.0   # 最终奖励由 RewardWorker 统一给

        # 执行工具
        obs = env.step(action)
        tool_result = f"[Tool: {action['name']}]\n{obs['output']}"
        done = obs.get("done", False)

        if done:
            env.finish("")
            self._envs.pop(session_id, None)

        return tool_result, done, 0.0

    def get_trajectory(self, session_id: str) -> Trajectory | None:
        """获取当前 session 的轨迹（供 RewardWorker 使用）。"""
        env = self._envs.get(session_id)
        return env.get_trajectory() if env else None

    def cleanup(self, session_id: str):
        """清理 session 资源。"""
        env = self._envs.pop(session_id, None)
        if env:
            env._cleanup()


def _parse_single_tool_call(text: str) -> dict | None:
    """
    从模型文本中解析第一个工具调用。
    支持 Qwen function calling 格式：
        <tool_call>{"name": "bash", "arguments": {"command": "ls"}}</tool_call>
    """
    import re
    pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
        name = data.get("name", "")
        args = data.get("arguments", data.get("input", {}))
        if name in {"bash", "read_file", "write_file", "edit_file"}:
            return {"name": name, "input": args}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# veRL 训练入口
# ---------------------------------------------------------------------------

def train_grpo_verl(
    model_path: str = BASE_MODEL,
    output_dir: str = OUTPUT_DIR,
    tasks: list[dict] | None = None,
    config: dict | None = None,
    multiturn: bool = True,
):
    """
    使用 veRL 框架进行 GRPO 训练。

    Args:
        model_path:  SFT 冷启动后的模型路径（或 HuggingFace Hub ID）
        output_dir:  检查点输出目录
        tasks:       任务列表，默认使用内置 SEED_TASKS
        config:      覆盖默认 VERL_GRPO_CONFIG 的参数
        multiturn:   True  → multi-turn rollout（真实工具调用，推荐）
                     False → single-turn + replay（兼容模式）

    安装要求：
        pip install verl
        # 完整 multi-turn 支持需要 SGLang：
        pip install sglang

    环境变量（可选）：
        BASE_MODEL         模型路径
        VERL_OUTPUT_DIR    输出目录
        VERL_N_GPUS        GPU 数量（默认自动检测）
    """
    _check_verl_installed()

    import verl
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.utils.reward_score import RewardManager

    cfg = {**VERL_GRPO_CONFIG, **(config or {})}
    tasks = tasks or SEED_TASKS

    print("=" * 60)
    print("veRL GRPO 训练")
    print("=" * 60)
    resolved_model_path = resolve_model_path(model_path)
    print(f"  模型:        {model_path}")
    print(f"  解析路径:    {resolved_model_path}")
    print(f"  输出:        {output_dir}")
    print(f"  任务数:      {len(tasks)}")
    print(f"  组大小 G:    {cfg['group_size']}")
    print(f"  迭代次数:    {cfg['num_iterations']}")
    print(f"  KL 系数 β:  {cfg['kl_coeff']}")
    print(f"  Rollout:     {cfg['rollout_backend']} ({'multi-turn' if multiturn else 'single-turn'})")
    print(f"  训练后端:    {cfg['train_backend']}")
    print()

    # ── 奖励函数 ──
    reward_fn = make_verl_reward_fn(tasks)

    # ── 数据集（veRL 期望 datasets.Dataset 格式，字段：prompt） ──
    from datasets import Dataset as HFDataset
    prompt_data = HFDataset.from_list([
        {"prompt": t["prompt"], "task_id": t["id"]}
        for t in tasks
    ] * (cfg["num_iterations"] // len(tasks) + 1))

    # ── veRL 配置（OmegaConf / dataclass 风格） ──
    verl_config = _build_verl_config(
        model_path=resolved_model_path,
        output_dir=output_dir,
        cfg=cfg,
        multiturn=multiturn,
    )

    # ── 启动 Ray 集群（单机模式） ──
    _ensure_ray_initialized(cfg["n_gpus_per_node"])

    # ── 创建并启动 Trainer ──
    if multiturn:
        _train_multiturn(verl_config, prompt_data, reward_fn, tasks, cfg)
    else:
        _train_singleturn(verl_config, prompt_data, reward_fn, cfg)


# ---------------------------------------------------------------------------
# Multi-turn 训练（完整工具调用 rollout）
# ---------------------------------------------------------------------------

def _train_multiturn(verl_config, prompt_data, reward_fn, tasks, cfg):
    """
    使用 veRL + SGLang multi-turn 接口进行训练。
    每次推理步骤后真实执行工具调用，获得准确的可验证奖励。

    参考：verl/examples/sglang_multiturn/
    """
    try:
        from verl.envs.base_env import BaseEnv
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    except ImportError:
        print("veRL multi-turn 需要最新版本，尝试安装：")
        print("  pip install git+https://github.com/verl-project/verl.git")
        print("回退到 single-turn 模式...")
        return _train_singleturn(verl_config, prompt_data, reward_fn, cfg)

    reward_fn_obj = RewardFn()

    class AgentEnvVeRL(BaseEnv):
        """
        将 AgentEnvironment 包装为 veRL BaseEnv 接口。
        veRL 在 rollout 的每个 turn 调用 step()，
        本类负责把工具调用路由到沙箱并返回观测。
        """

        def __init__(self):
            self._env_worker = AgentEnvRolloutWorker(
                tasks=tasks,
                max_steps=cfg["max_steps_per_episode"],
            )

        def reset(self, data_item: dict) -> dict:
            prompt = data_item["prompt"]
            session_id = data_item.get("task_id", prompt[:32])
            self._env_worker.reset(session_id, prompt)
            data_item["_session_id"] = session_id
            return data_item

        def step(self, data_item: dict, model_output: str) -> dict:
            session_id = data_item["_session_id"]
            obs, done, _ = self._env_worker.step(session_id, model_output)
            data_item["observation"] = obs
            data_item["done"] = done
            if done:
                traj = self._env_worker.get_trajectory(session_id)
                if traj:
                    scores = reward_fn_obj(traj)
                    data_item["reward"] = scores["total"]
                    data_item["reward_details"] = scores
                else:
                    data_item["reward"] = 0.0
            return data_item

    trainer = RayPPOTrainer(
        config=verl_config,
        tokenizer_path=verl_config.model.path,
        reward_fn=None,   # 奖励在 env.step 中给出
        env_class=AgentEnvVeRL,
        train_dataset=prompt_data,
    )
    print("开始 veRL multi-turn GRPO 训练...")
    trainer.fit()
    print(f"训练完成，模型保存至: {cfg.get('output_dir', OUTPUT_DIR)}")


# ---------------------------------------------------------------------------
# Single-turn 训练（replay 可验证奖励，兼容性更好）
# ---------------------------------------------------------------------------

def _train_singleturn(verl_config, prompt_data, reward_fn, cfg):
    """
    Single-turn 模式：
    模型一次性生成完整文本（含工具调用序列），
    然后通过 _replay_trajectory_from_text() 在沙箱中重放，
    获得真实的可验证奖励。

    优点：与 veRL 标准 GRPOTrainer 完全兼容，无需 multi-turn 支持
    缺点：模型生成时看不到工具执行结果，学习信号弱于 multi-turn
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("请安装 trl>=0.12.0: pip install 'trl>=0.12.0'")

    print("使用 single-turn + replay 模式（veRL GRPOTrainer）...")

    tokenizer = AutoTokenizer.from_pretrained(
        verl_config.model.path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        verl_config.model.path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # TRL GRPOTrainer 兼容的 reward 函数签名
    def trl_reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            data_item = {"prompt": prompt, "response": completion}
            rewards.append(reward_fn(data_item))
        return rewards

    grpo_cfg = GRPOConfig(
        output_dir=verl_config.trainer.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=cfg["prompts_per_batch"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_generations=cfg["group_size"],
        max_new_tokens=512,
        bf16=True,
        logging_steps=5,
        save_steps=cfg["save_steps"],
        report_to="none",
        kl_coef=cfg["kl_coeff"],
        cliprange=cfg["clip_epsilon"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trl_reward_fn,
        args=grpo_cfg,
        train_dataset=prompt_data,
        tokenizer=tokenizer,
    )

    print("开始 single-turn GRPO 训练（可验证奖励 via replay）...")
    trainer.train()
    model.save_pretrained(grpo_cfg.output_dir)
    tokenizer.save_pretrained(grpo_cfg.output_dir)
    print(f"训练完成，模型保存至: {grpo_cfg.output_dir}")


# ---------------------------------------------------------------------------
# veRL 配置构建
# ---------------------------------------------------------------------------

def _build_verl_config(
    model_path: str,
    output_dir: str,
    cfg: dict,
    multiturn: bool,
):
    """
    构建 veRL OmegaConf 配置对象。
    对应 veRL 的标准配置结构（参考 verl/examples/grpo_trainer/）。
    """
    try:
        from omegaconf import OmegaConf, DictConfig
    except ImportError:
        raise ImportError("请安装 omegaconf: pip install omegaconf")

    model_path = resolve_model_path(model_path)

    # veRL 标准配置结构
    raw_cfg = {
        "model": {
            "path": model_path,
            "tokenizer_path": model_path,
        },
        "actor_rollout_ref": {
            "model": {
                "path": model_path,
                "input_tokenizer": model_path,
            },
            "actor": {
                "strategy": cfg["train_backend"],          # fsdp / fsdp2 / megatron
                "optim": {
                    "lr": cfg["learning_rate"],
                    "weight_decay": 0.01,
                },
                "ppo_mini_batch_size": cfg["prompts_per_batch"] * cfg["group_size"],
                "ppo_micro_batch_size_per_gpu": cfg["prompts_per_batch"],
                "grad_clip": cfg["max_grad_norm"],
                "clip_ratio": cfg["clip_epsilon"],
                "entropy_coeff": 0.001,
                **({"lora_rank": cfg["lora_r"],
                    "lora_alpha": cfg["lora_alpha"],
                    "target_modules": cfg["lora_target_modules"]}
                   if cfg.get("use_lora") else {}),
            },
            "rollout": {
                "name": cfg["rollout_backend"],            # vllm / sglang / hf
                "temperature": cfg["rollout_temperature"],
                "n": cfg["group_size"],                    # GRPO 的 G
                "gpu_memory_utilization": cfg["gpu_memory_utilization"],
                "rollout_filter_ratio": 0.25,              # 过滤掉全对/全错的组
                "multiturn": multiturn,
            },
            "ref": {
                "strategy": cfg["train_backend"],
                "log_prob_micro_batch_size_per_gpu": cfg["prompts_per_batch"],
            },
        },
        "algorithm": {
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": cfg["kl_coeff"],
            },
            "adv_estimator": "grpo",                       # 使用 GRPO 组内优势
        },
        "trainer": {
            "total_epochs": cfg["num_iterations"],
            "project_name": "learn-claude-code-verl",
            "experiment_name": "grpo",
            "output_dir": output_dir,
            "save_freq": cfg["save_steps"],
            "test_freq": cfg["save_steps"],
            "logger": ["console"],                         # 可改为 ["console", "wandb"]
            "n_gpus_per_node": cfg["n_gpus_per_node"],
        },
        "data": {
            "max_prompt_length": 512,
            "max_response_length": 1024,
        },
    }

    return OmegaConf.create(raw_cfg)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _check_verl_installed():
    """检查 veRL 是否安装，给出清晰的安装指引。"""
    try:
        import verl
        print(f"veRL 版本: {getattr(verl, '__version__', 'unknown')}")
    except ImportError:
        print("=" * 60)
        print("veRL 未安装，请先安装：")
        print()
        print("  # 稳定版")
        print("  pip install verl")
        print()
        print("  # 最新版（含完整 multi-turn 支持）")
        print("  pip install git+https://github.com/verl-project/verl.git")
        print()
        print("  # 可选：SGLang（multi-turn rollout 加速）")
        print("  pip install sglang")
        print("=" * 60)
        sys.exit(1)


def _ensure_ray_initialized(n_gpus: int):
    """初始化 Ray 集群（单机模式）。"""
    try:
        import ray
        if not ray.is_initialized():
            ray.init(
                num_gpus=n_gpus,
                ignore_reinit_error=True,
            )
            print(f"Ray 已初始化，GPU 数量: {n_gpus}")
    except ImportError:
        raise ImportError("veRL 依赖 Ray，请安装: pip install ray[default]")


# ---------------------------------------------------------------------------
# 快速验证：不需要 GPU，仅测试奖励适配层逻辑
# ---------------------------------------------------------------------------

def _test_reward_adapter():
    """
    单元测试：验证 veRL reward 适配层能正确调用
    现有 reward.py 并返回合理分数。
    不需要 GPU，本地直接可运行。
    """
    from environment import ToolCall

    print("── 测试 veRL 奖励适配层 ──")

    tasks = SEED_TASKS[:2]
    reward_fn = make_verl_reward_fn(tasks)

    # 测试 1：携带完整 Trajectory 对象
    traj = Trajectory(task_id="test", prompt=tasks[0]["prompt"])
    traj.status = "success"
    traj.tool_calls = [
        ToolCall("bash", {"command": "pytest test_fib.py -q"}, "1 passed in 0.1s"),
    ]
    traj.messages = [{"role": "assistant", "content": "<think>先写再测</think>"}]
    traj.final_answer = "已完成实现"

    score1 = reward_fn({"trajectory": traj})
    print(f"  Trajectory 对象模式: {score1:.3f} (期望 > 0.5)")
    assert score1 > 0.5, f"分数异常: {score1}"

    # 测试 2：纯文本模式（字符串匹配奖励）
    score2 = reward_fn({
        "prompt": tasks[0]["prompt"],
        "response": "<think>思考中</think>\n最终回答：已完成",
    })
    print(f"  纯文本模式 (格式奖励): {score2:.3f}")
    assert score2 >= 0.0

    print("  ✓ 奖励适配层测试通过")


def _test_tool_parser():
    """
    单元测试：验证工具调用解析器能正确解析 Qwen function calling 格式。
    """
    print("── 测试工具调用解析器 ──")

    sample_output = """<think>
我需要先写文件，再运行测试。
</think>
<tool_call>{"name": "write_file", "arguments": {"path": "solution.py", "content": "def fib(n):\\n    if n <= 0: return 0\\n    if n == 1: return 1\\n    return fib(n-1)+fib(n-2)\\n"}}</tool_call>
<tool_call>{"name": "bash", "arguments": {"command": "pytest test_fib.py -q"}}</tool_call>
最终回答：斐波那契函数已实现并通过测试。"""

    action = _parse_single_tool_call(sample_output)
    assert action is not None, "未解析到工具调用"
    assert action["name"] == "write_file", f"工具名错误: {action['name']}"
    print(f"  解析到工具: {action['name']}({list(action['input'].keys())})")
    print("  ✓ 工具解析器测试通过")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2d: GRPO via veRL（生产级框架）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # multi-turn 模式（完整工具调用，推荐）
  python train/rl_verl_grpo.py --model checkpoints/sft_merged --multiturn

  # single-turn 模式（更易部署，仍有可验证奖励）
  python train/rl_verl_grpo.py --model checkpoints/sft_merged --no-multiturn

  # 使用 LoRA 节省显存
  python train/rl_verl_grpo.py --model checkpoints/sft_merged --lora

  # 仅测试奖励适配层（无需 GPU）
  python train/rl_verl_grpo.py --test
        """,
    )
    parser.add_argument("--model", default=BASE_MODEL, help="模型路径（SFT 后的）")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--multiturn", action="store_true", default=True,
                        help="使用 multi-turn rollout（真实工具调用，默认开启）")
    parser.add_argument("--no-multiturn", dest="multiturn", action="store_false",
                        help="使用 single-turn + replay 模式")
    parser.add_argument("--lora", action="store_true", help="使用 LoRA 节省显存")
    parser.add_argument("--iterations", type=int, default=200, help="训练迭代次数")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO 组大小 G")
    parser.add_argument("--kl-coeff", type=float, default=0.04, help="KL 惩罚系数 β")
    parser.add_argument("--rollout-backend", choices=["vllm", "sglang", "hf"],
                        default="vllm", help="Rollout 推理后端")
    parser.add_argument("--train-backend", choices=["fsdp", "fsdp2", "megatron"],
                        default="fsdp", help="训练后端")
    parser.add_argument("--test", action="store_true", help="仅运行本地单元测试（无需 GPU）")
    args = parser.parse_args()

    if args.test:
        _test_reward_adapter()
        _test_tool_parser()
        print("\n所有测试通过 ✓")
        sys.exit(0)

    extra_config = {
        "num_iterations": args.iterations,
        "group_size": args.group_size,
        "kl_coeff": args.kl_coeff,
        "rollout_backend": args.rollout_backend,
        "train_backend": args.train_backend,
        "use_lora": args.lora,
    }

    train_grpo_verl(
        model_path=args.model,
        output_dir=args.output,
        config=extra_config,
        multiturn=args.multiturn,
    )
