"""Default merge planning for standard models (Qwen3, Kimi, etc.).

Handles models without special merge requirements. Detects fused expert
layout (concatenated, not interleaved) and vision model prefix from key
names alone.
"""

from __future__ import annotations

import torch

from tinker_cookbook.weights._merge import MergeOp, MergeProfile
from tinker_cookbook.weights._merge_utils import (
    build_name_remaps,
    extract_adapter_weight_names,
    plan_expert_ops,
    plan_fused_projection_op,
    plan_standard_op,
    remap_adapter_name,
    validate_adapter_config,
)


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile:
    """Default profile for models without special merge requirements.

    Handles Qwen, Kimi, and other standard model families. Detects fused
    expert layout (concatenated, not interleaved) and vision model prefix
    from key names alone.
    """
    has_fused = any(k.endswith(".experts.gate_up_proj") for k in model_state_keys)
    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        model_family="default",
        expert_layout="fused_concatenated" if has_fused else "separate",
        has_language_model_prefix=has_lm_prefix,
    )


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan merge ops for standard models."""
    scaling = validate_adapter_config(adapter_config, profile)
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    name_remaps = build_name_remaps(profile, model_state_keys)

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = remap_adapter_name(n, name_remaps)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            if plan_fused_projection_op(
                target_key,
                lora_A,
                lora_B,
                n,
                adapter_weights,
                profile,
                model_state_keys,
                ops,
            ):
                continue
            plan_standard_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            plan_expert_ops(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)

    return ops
