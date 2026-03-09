"""
monitor.py — 统一训练监控接口

支持多后端并行上报：
    - WandB  (https://wandb.ai)
    - SwanLab (https://swanlab.cn — 国内友好)
    - TensorBoard（无需账号，本地可视化）

用法:
    from monitor import Monitor
    m = Monitor(project="code-agent-rl", run_name="grpo-v1", config={...})
    m.log({"loss": 0.42, "reward": 0.7}, step=10)
    m.finish()

环境变量控制（.env 或系统环境）:
    MONITOR_BACKEND=wandb,swanlab   # 逗号分隔，默认 wandb
    WANDB_API_KEY=xxx
    WANDB_PROJECT=code-agent-rl
    WANDB_ENTITY=your-username
    SWANLAB_API_KEY=xxx
    SWANLAB_PROJECT=code-agent-rl
    SWANLAB_WORKSPACE=your-username
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _get_backends() -> list[str]:
    raw = os.environ.get("MONITOR_BACKEND", "wandb").strip()
    return [b.strip().lower() for b in raw.split(",") if b.strip()]


# ---------------------------------------------------------------------------
# 单后端封装
# ---------------------------------------------------------------------------

class _WandBBackend:
    def __init__(self, project: str, run_name: str | None, config: dict):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "code-agent-rl"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=run_name,
            config=config,
            resume="allow",
        )
        print(f"[monitor] WandB 已初始化  → {self._run.url}")

    def log(self, metrics: dict[str, Any], step: int | None = None):
        self._wandb.log(metrics, step=step)

    def summary(self, key: str, value: Any):
        self._wandb.run.summary[key] = value

    def finish(self):
        self._wandb.finish()


class _SwanLabBackend:
    def __init__(self, project: str, run_name: str | None, config: dict):
        import swanlab
        self._swanlab = swanlab
        swanlab.init(
            project=project or os.environ.get("SWANLAB_PROJECT", "code-agent-rl"),
            workspace=os.environ.get("SWANLAB_WORKSPACE", None),
            experiment_name=run_name,
            config=config,
            logdir=".swanlab",
        )
        print("[monitor] SwanLab 已初始化")

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if step is not None:
            self._swanlab.log(metrics, step=step)
        else:
            self._swanlab.log(metrics)

    def summary(self, key: str, value: Any):
        pass  # SwanLab 暂无独立 summary API

    def finish(self):
        self._swanlab.finish()


class _TensorBoardBackend:
    def __init__(self, project: str, run_name: str | None, config: dict):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("runs", run_name or project)
        self._writer = SummaryWriter(log_dir=log_dir)
        print(f"[monitor] TensorBoard 已初始化 → {log_dir}")

    def log(self, metrics: dict[str, Any], step: int | None = None):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step)

    def summary(self, key: str, value: Any):
        pass

    def finish(self):
        self._writer.close()


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------

class Monitor:
    """统一监控接口，支持 wandb / swanlab / tensorboard 多后端。"""

    _BACKENDS = {
        "wandb": _WandBBackend,
        "swanlab": _SwanLabBackend,
        "tensorboard": _TensorBoardBackend,
    }

    def __init__(
        self,
        project: str = "code-agent-rl",
        run_name: str | None = None,
        config: dict | None = None,
    ):
        config = config or {}
        backends_to_use = _get_backends()
        self._backends: list = []

        for name in backends_to_use:
            cls = self._BACKENDS.get(name)
            if cls is None:
                print(f"[monitor] 未知后端: {name}，跳过")
                continue
            try:
                self._backends.append(cls(project, run_name, config))
            except ImportError as e:
                print(f"[monitor] {name} 未安装，跳过: {e}")
            except Exception as e:
                print(f"[monitor] {name} 初始化失败，跳过: {e}")

        if not self._backends:
            print("[monitor] 警告：无可用监控后端，指标仅打印到终端")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """上报一组指标。"""
        for b in self._backends:
            try:
                b.log(metrics, step=step)
            except Exception as e:
                print(f"[monitor] log 失败: {e}")

    def summary(self, key: str, value: Any) -> None:
        """设置摘要指标（运行结束时的最终值）。"""
        for b in self._backends:
            try:
                b.summary(key, value)
            except Exception as e:
                print(f"[monitor] summary 失败: {e}")

    def finish(self) -> None:
        """结束所有后端会话。"""
        for b in self._backends:
            try:
                b.finish()
            except Exception as e:
                print(f"[monitor] finish 失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def init(
    project: str = "code-agent-rl",
    run_name: str | None = None,
    config: dict | None = None,
) -> Monitor:
    """创建并返回 Monitor 实例的快捷方式。"""
    return Monitor(project=project, run_name=run_name, config=config)
