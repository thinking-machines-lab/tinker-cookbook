"""Shared scaffold and config for the Tinker -> Modal inference recipe.

Holds the app, images, the artifact Volume, the per-model registry, and the
helpers for mode selection and building the SGLang command.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import modal

APP_NAME = "tinker-modal-inference"
MINUTES = 60

# PREPARE writes an artifact here, SERVE reads it
ARTIFACTS_PATH = "/artifacts"
HF_CACHE_PATH = "/cache/huggingface"
artifacts = modal.Volume.from_name("tinker-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

Mode = Literal["adapter", "merge"]


class ModelConfig(NamedTuple):
    base_model: str
    gpu: str
    tp: int
    lora_serving: bool  # can the pinned SGLang LoRA serve this model family?

    @property
    def default_mode(self) -> Mode:
        return "adapter" if self.lora_serving else "merge"


# Adding support for a model we have tested
MODEL_REGISTRY: dict[str, ModelConfig] = {
    cfg.base_model: cfg
    for cfg in (
        ModelConfig("Qwen/Qwen3-8B", gpu="H100:1", tp=1, lora_serving=True),
        ModelConfig("Qwen/Qwen3.6-35B-A3B", gpu="H100:2", tp=2, lora_serving=True),
        # lora_serving=False forces merge mode for any model not supported
    )
}


def model_config(base_model: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[base_model]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise KeyError(
            f"{base_model!r} is not in MODEL_REGISTRY (common.py). Known: {known}"
        ) from None


def resolve_mode(cfg: ModelConfig, override: str | None) -> Mode:
    """Explicit override wins, else the model default. Rejects adapter mode on
    models the engine can't LoRA-serve, so the failure is early and clear."""
    mode = override or cfg.default_mode
    if mode not in ("adapter", "merge"):
        raise ValueError(f"mode must be 'adapter' or 'merge'; got {mode!r}")
    if mode == "adapter" and not cfg.lora_serving:
        raise ValueError(
            f"{cfg.base_model!r} can't be adapter-served on this SGLang version; "
            f"re-run with --mode merge."
        )
    return mode  # type: ignore[return-value]


def artifact_dir(name: str) -> str:
    return f"{ARTIFACTS_PATH}/{name}"


def sglang_command(
    *,
    model_path: str,
    served_name: str,
    tp: int,
    port: int,
    lora: tuple[str, str] | None = None,
) -> tuple[str, ...]:
    """Build the launch_server argv. lora=(name, path) attaches a PEFT adapter;
    None serves model_path directly."""
    argv = (
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--served-model-name", served_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tp", str(tp),
        "--trust-remote-code",
    )
    if lora is None:
        return argv
    name, path = lora
    return argv + ("--enable-lora", "--lora-paths", f"{name}={path}")


prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tinker-cookbook[modal]", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": HF_CACHE_PATH})
)

# Pinned SGLang image
SGLANG_TAG = "lmsysorg/sglang:v0.5.12.post1-cu130"
sglang_image = (
    modal.Image.from_registry(SGLANG_TAG)
    .entrypoint([])
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME)
