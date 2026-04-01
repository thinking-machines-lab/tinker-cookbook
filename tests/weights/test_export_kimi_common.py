"""Shared test helpers for Kimi K2 and K2.5 export tests.

Both models use INT4 group-quantized routed experts and bf16 dense layers.
They differ in weight key prefix (K2: ``model.*``, K2.5: ``language_model.model.*``)
and config nesting (K2: flat, K2.5: wrapped with ``text_config``).

The ``key_prefix`` parameter controls the prefix applied to all weight keys:
- K2: ``""`` (keys like ``model.layers.0.*``)
- K2.5: ``"language_model."`` (keys like ``language_model.model.layers.0.*``)
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.weights.conftest import FILL_A, FILL_B, save_minimal_tokenizer
from tinker_cookbook.weights._packed_int4 import quantize_int4_group

HIDDEN = 64
MLP_DIM = 128
EXPERT_DIM = 32  # must be divisible by GROUP_SIZE
NUM_EXPERTS = 2
GROUP_SIZE = 32
VOCAB = 128
RANK = 1


def build_synthetic_kimi_model(
    model_dir: Path,
    config_dict: dict,
    key_prefix: str,
) -> None:
    """Build a synthetic Kimi model directory with INT4 packed experts.

    Args:
        model_dir: Directory to write model files into (must exist).
        config_dict: Model config to write as config.json.
        key_prefix: Prefix for all weight keys. Empty string for K2,
            ``"language_model."`` for K2.5.
    """
    p = key_prefix
    tensors: dict[str, torch.Tensor] = {}

    # Dense layer 0 (bf16)
    for name, shape in [
        (f"{p}model.layers.0.self_attn.q_a_proj.weight", (HIDDEN, HIDDEN)),
        (f"{p}model.layers.0.self_attn.o_proj.weight", (HIDDEN, HIDDEN)),
        (f"{p}model.layers.0.mlp.gate_proj.weight", (MLP_DIM, HIDDEN)),
        (f"{p}model.layers.0.mlp.up_proj.weight", (MLP_DIM, HIDDEN)),
        (f"{p}model.layers.0.mlp.down_proj.weight", (HIDDEN, MLP_DIM)),
    ]:
        tensors[name] = torch.randn(*shape, dtype=torch.bfloat16)
    tensors[f"{p}model.layers.0.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    tensors[f"{p}model.layers.0.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — attention + layernorms (bf16)
    for name, shape in [
        (f"{p}model.layers.1.self_attn.q_a_proj.weight", (HIDDEN, HIDDEN)),
        (f"{p}model.layers.1.self_attn.o_proj.weight", (HIDDEN, HIDDEN)),
    ]:
        tensors[name] = torch.randn(*shape, dtype=torch.bfloat16)
    tensors[f"{p}model.layers.1.input_layernorm.weight"] = torch.ones(HIDDEN, dtype=torch.bfloat16)
    tensors[f"{p}model.layers.1.post_attention_layernorm.weight"] = torch.ones(
        HIDDEN, dtype=torch.bfloat16
    )

    # MoE layer 1 — shared experts (bf16)
    for proj, shape in [
        ("gate_proj", (EXPERT_DIM, HIDDEN)),
        ("up_proj", (EXPERT_DIM, HIDDEN)),
        ("down_proj", (HIDDEN, EXPERT_DIM)),
    ]:
        tensors[f"{p}model.layers.1.mlp.shared_experts.{proj}.weight"] = torch.randn(
            *shape, dtype=torch.bfloat16
        )

    # MoE layer 1 — routed experts (INT4 quantized)
    for i in range(NUM_EXPERTS):
        for proj, shape in [
            ("gate_proj", (EXPERT_DIM, HIDDEN)),
            ("up_proj", (EXPERT_DIM, HIDDEN)),
            ("down_proj", (HIDDEN, EXPERT_DIM)),
        ]:
            base = f"{p}model.layers.1.mlp.experts.{i}.{proj}"
            bf16_weight = torch.randn(*shape, dtype=torch.bfloat16)
            packed, scale = quantize_int4_group(bf16_weight, GROUP_SIZE)
            tensors[f"{base}.weight_packed"] = packed
            tensors[f"{base}.weight_scale"] = scale
            tensors[f"{base}.weight_shape"] = torch.tensor(shape, dtype=torch.int32)

    # Embeddings and lm_head (bf16)
    lm_prefix = "language_model." if p else ""
    tensors[f"{p}model.embed_tokens.weight"] = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    tensors[f"{lm_prefix}lm_head.weight"] = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)

    # Split into two shards: dense layer 0 + embeddings | MoE layer 1
    shard1_keys = [k for k in tensors if "layers.0" in k or "embed" in k or "lm_head" in k]
    shard2_keys = [k for k in tensors if k not in shard1_keys]

    save_file(
        {k: tensors[k] for k in shard1_keys},
        str(model_dir / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {k: tensors[k] for k in shard2_keys},
        str(model_dir / "model-00002-of-00002.safetensors"),
    )

    weight_map = {
        **{k: "model-00001-of-00002.safetensors" for k in shard1_keys},
        **{k: "model-00002-of-00002.safetensors" for k in shard2_keys},
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2)
    )
    (model_dir / "config.json").write_text(json.dumps(config_dict, indent=2))
    save_minimal_tokenizer(model_dir)


def save_kimi_adapter(
    adapter_dir: Path,
    key_prefix: str,
    *,
    include_attention: bool = True,
    include_experts: bool = True,
) -> None:
    """Save a synthetic LoRA adapter for Kimi models.

    Args:
        adapter_dir: Directory to create and write adapter files into.
        key_prefix: Weight key prefix (empty for K2, ``"language_model."`` for K2.5).
            The adapter uses ``base_model.model.`` + model keys without the outer prefix,
            since Tinker strips the VL prefix in adapters.
    """
    weights: dict[str, torch.Tensor] = {}

    if include_attention:
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight"] = (
            torch.ones(RANK, HIDDEN) * FILL_A
        )
        weights["base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight"] = torch.ones(
            HIDDEN, RANK
        )

    if include_experts:
        prefix = "base_model.model.model.layers.1.mlp.experts"
        weights[f"{prefix}.w1.lora_A.weight"] = torch.ones(1, RANK, HIDDEN) * FILL_A
        weights[f"{prefix}.w1.lora_B.weight"] = torch.ones(NUM_EXPERTS, EXPERT_DIM, RANK)
        weights[f"{prefix}.w3.lora_A.weight"] = torch.ones(1, RANK, HIDDEN) * FILL_B
        weights[f"{prefix}.w3.lora_B.weight"] = torch.ones(NUM_EXPERTS, EXPERT_DIM, RANK)

    adapter_dir.mkdir(parents=True)
    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": 1, "r": RANK})
    )
