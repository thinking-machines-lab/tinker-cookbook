"""DeepSeek V3/V3.1 merge planning.

DeepSeek uses separate per-expert weights (not fused) and standard key
naming. Detection is based on ``model_type`` rather than architecture
strings for reliability across versions.
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
    """Detect DeepSeek V3/V3.1 models from config.

    DeepSeek uses separate per-expert weights (not fused) and standard key
    naming. Detection is based on ``model_type`` rather than architecture
    strings for reliability across versions.

    Args:
        model_config (dict): Parsed ``config.json`` dict. Checked for
            ``model_type == "deepseek_v3"``.
        model_state_keys (set[str]): Weight key names from the model.

    Returns:
        MergeProfile | None: Profile with ``model_family="deepseek"`` if
            the model is DeepSeek, otherwise ``None``.
    """
    if model_config.get("model_type") not in ("deepseek_v3",):
        return None

    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        model_family="deepseek",
        expert_layout="separate",
        has_language_model_prefix=has_lm_prefix,
    )


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan merge ops for DeepSeek models.

    DeepSeek always uses separate (non-fused) expert layout, so expert
    weights are expanded into individual per-expert 2D merge operations.

    Args:
        adapter_weights (dict[str, torch.Tensor]): LoRA weight tensors from
            the adapter.
        adapter_config (dict): Adapter config with ``lora_alpha`` and ``r``.
        model_state_keys (set[str]): Weight key names in the base model.
        profile (MergeProfile): Model-specific merge configuration.

    Returns:
        dict[str, list[MergeOp]]: Mapping from model weight key to list of
            merge operations targeting it.
    """
    scaling = validate_adapter_config(adapter_config, profile)
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    name_remaps = build_name_remaps(profile, model_state_keys)

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = remap_adapter_name(n, name_remaps)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            plan_standard_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            # DeepSeek always uses separate expert layout
            plan_expert_ops(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)

    return ops
