"""GPT-OSS merge planning.

GPT-OSS uses ``.attn`` instead of ``.self_attn`` for attention layers, and
an interleaved ``[g0, u0, g1, u1, ...]`` layout for fused gate/up expert
projections.
"""

from __future__ import annotations

import torch

from tinker_cookbook.weights._merge import MergeOp, MergeProfile
from tinker_cookbook.weights._merge_utils import (
    build_name_remaps,
    extract_adapter_weight_names,
    plan_expert_ops,
    plan_standard_op,
    remap_adapter_name,
    validate_adapter_config,
)


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
    """Detect GPT-OSS models.

    GPT-OSS uses ``.attn`` instead of ``.self_attn`` for attention layers, and
    an interleaved ``[g0, u0, g1, u1, ...]`` layout for fused gate/up expert
    projections.
    """
    architectures = model_config.get("architectures", [])
    if not any("GptOss" in a for a in architectures):
        return None

    has_fused = any(k.endswith(".experts.gate_up_proj") for k in model_state_keys)
    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        model_family="gpt_oss",
        expert_layout="fused_interleaved" if has_fused else "separate",
        extra_key_remaps=((".attn", ".self_attn"),),
        has_language_model_prefix=has_lm_prefix,
    )


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan merge ops for GPT-OSS models."""
    scaling = validate_adapter_config(adapter_config, profile)
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    is_fused = profile.expert_layout in ("fused_interleaved", "fused_concatenated")
    is_interleaved = profile.expert_layout == "fused_interleaved"
    name_remaps = build_name_remaps(profile, model_state_keys)

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = remap_adapter_name(n, name_remaps)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            plan_standard_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            plan_expert_ops(
                target_key, lora_A, lora_B, n, model_state_keys, ops, is_fused, is_interleaved
            )

    return ops
