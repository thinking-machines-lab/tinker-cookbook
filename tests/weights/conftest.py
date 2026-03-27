"""Shared helpers for weights e2e tests.

Provides common utilities for creating tiny models from config, saving
synthetic LoRA adapters, and running the build_hf_model pipeline.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PretrainedConfig,
)

from tinker_cookbook.weights import build_hf_model, build_lora_adapter

FILL_A = 0.01  # LoRA fill for gate / first projection
FILL_B = 0.05  # LoRA fill for up / second projection


def save_model_to_disk(
    config: PretrainedConfig,
    path: Path,
    *,
    tokenizer_name: str,
    is_vision: bool = False,
    trust_remote_code: bool = True,
) -> None:
    """Create a tiny model from config and save to disk with tokenizer."""
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    model = auto_cls.from_config(config, trust_remote_code=trust_remote_code, dtype=torch.float32)
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    tok.save_pretrained(path)


def save_expert_adapter(
    path: Path,
    *,
    num_experts: int,
    in_dim: int,
    out_dim: int,
    gate_fill: float = FILL_A,
    up_fill: float = FILL_B,
    down_fill: float | None = None,
    layer_prefix: str = "base_model.model.model.layers.0.mlp.experts",
) -> None:
    """Save a LoRA adapter for expert projections matching real Tinker shapes.

    Real Tinker adapters use asymmetric broadcast patterns for 3D expert LoRA:
    - w1 (gate) / w3 (up): lora_A shared ``(1, rank, in_dim)``,
      lora_B per-expert ``(num_experts, out_dim, rank)``
    - w2 (down): lora_A per-expert ``(num_experts, rank, out_dim)``,
      lora_B shared ``(1, in_dim, rank)``

    Set ``down_fill`` to include w2 (down_proj) in the adapter.
    """
    rank = 1
    weights: dict[str, torch.Tensor] = {
        # w1 (gate_proj): A shared, B per-expert
        f"{layer_prefix}.w1.lora_A.weight": torch.ones(1, rank, in_dim) * gate_fill,
        f"{layer_prefix}.w1.lora_B.weight": torch.ones(num_experts, out_dim, rank),
        # w3 (up_proj): A shared, B per-expert
        f"{layer_prefix}.w3.lora_A.weight": torch.ones(1, rank, in_dim) * up_fill,
        f"{layer_prefix}.w3.lora_B.weight": torch.ones(num_experts, out_dim, rank),
    }
    if down_fill is not None:
        # w2 (down_proj): A per-expert, B shared (reversed from w1/w3)
        weights[f"{layer_prefix}.w2.lora_A.weight"] = (
            torch.ones(num_experts, rank, out_dim) * down_fill
        )
        weights[f"{layer_prefix}.w2.lora_B.weight"] = torch.ones(1, in_dim, rank)

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def save_dense_adapter(
    path: Path,
    *,
    in_dim: int,
    out_dim: int,
    fill: float = FILL_A,
    layer_prefix: str = "base_model.model.model.layers.0.mlp",
) -> None:
    """Save a LoRA adapter for a dense (non-expert) linear layer."""
    rank = 1
    weights = {
        f"{layer_prefix}.gate_proj.lora_A.weight": torch.ones(rank, in_dim) * fill,
        f"{layer_prefix}.gate_proj.lora_B.weight": torch.ones(out_dim, rank),
    }

    path.mkdir(parents=True)
    save_file(weights, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"lora_alpha": 1, "r": rank}))


def run_build_and_reload(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    *,
    is_vision: bool = False,
) -> dict[str, torch.Tensor]:
    """Run build_hf_model and return the reloaded state dict."""
    build_hf_model(
        base_model=str(model_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
    )
    auto_cls = AutoModelForImageTextToText if is_vision else AutoModelForCausalLM
    reloaded = auto_cls.from_pretrained(output_path, trust_remote_code=True, dtype=torch.float32)
    return reloaded.state_dict()


def run_build_adapter(
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Run build_lora_adapter and return (peft_weights, peft_config)."""
    build_lora_adapter(
        base_model=str(model_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
    )
    weights = load_file(str(output_path / "adapter_model.safetensors"))
    with open(output_path / "adapter_config.json") as f:
        config = json.load(f)
    return weights, config
