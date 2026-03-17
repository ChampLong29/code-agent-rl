#!/usr/bin/env python3
"""
hub_utils.py — 模型仓库加载工具

目标：优先使用 ModelScope 生态下载/缓存模型，同时兼容本地路径与 HuggingFace 风格调用。
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_local_path(model_or_path: str) -> bool:
    p = Path(model_or_path)
    return p.exists() or model_or_path.startswith(".") or "/" not in model_or_path


def resolve_model_path(
    model_or_path: str,
    revision: str | None = None,
    prefer_modelscope: bool | None = None,
) -> str:
    """
    将模型标识解析为可加载路径。

    规则：
        1) 本地路径：原样返回
        2) 远端模型 ID：默认优先走 ModelScope snapshot_download
        3) 若 ModelScope 不可用/下载失败：回退原始 ID（由 transformers/huggingface 继续处理）
    """
    if _is_local_path(model_or_path):
        return model_or_path

    if prefer_modelscope is None:
        prefer_modelscope = os.environ.get("PREFER_MODELSCOPE", "1") == "1"

    if not prefer_modelscope:
        return model_or_path

    try:
        from modelscope import snapshot_download

        cache_dir = os.environ.get("MODELSCOPE_CACHE")
        local_dir = snapshot_download(
            model_id=model_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        return local_dir
    except Exception as e:
        print(f"[hub_utils] ModelScope 下载失败，回退默认加载: {model_or_path} ({e})")
        return model_or_path
