"""Shared helpers for serving Tinker LoRA adapters with SGLang.

One SGLang server hosts one base model and many adapters. These helpers keep
the on-disk layout, the in-container paths, and the ``--lora-paths`` argument in
agreement, so the string ``prepare`` prints can be pasted into a launch command
unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

DEFAULT_LORA_ROOT = "/adapters"
"""Where the adapter tree is mounted inside the SGLang container."""


def base_model_slug(base_model: str) -> str:
    """Filesystem-safe directory name for a base model.

    ``"Qwen/Qwen3.8-27B"`` becomes ``"Qwen__Qwen3.8-27B"``.
    """
    return base_model.replace("/", "__")


def adapter_dir(root: str | Path, base_model: str, name: str) -> str:
    """Directory holding one PEFT adapter, namespaced by its base model.

    One SGLang server serves exactly one base model; grouping adapters per
    base makes mixing bases in one ``--lora-paths`` hard to express.
    """
    return str(Path(root) / base_model_slug(base_model) / name)


def parse_adapter_spec(spec: str) -> tuple[str, str]:
    """Split a ``NAME=tinker://...`` command-line argument."""
    name, sep, tinker_path = spec.partition("=")
    if not sep or not name or not tinker_path:
        raise ValueError(f"Expected NAME=tinker://..., got {spec!r}")
    if not tinker_path.startswith("tinker://"):
        raise ValueError(f"Adapter {name!r}: path must start with tinker://, got {tinker_path!r}")
    return name, tinker_path


def lora_paths_arg(
    names: Iterable[str], base_model: str, lora_root: str = DEFAULT_LORA_ROOT
) -> str:
    """Build the ``--lora-paths`` value for a set of adapters on one base model."""
    return " ".join(f"{name}={adapter_dir(lora_root, base_model, name)}" for name in names)


def read_adapter_config(adapter_path: str | Path) -> tuple[int, list[str]]:
    """Return ``(rank, target_modules)`` from a converted PEFT adapter."""
    config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    return int(config["r"]), sorted(config.get("target_modules") or [])
