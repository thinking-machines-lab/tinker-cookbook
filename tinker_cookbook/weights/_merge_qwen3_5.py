"""Qwen3.5 merge planning (dense and MoE).

Qwen3.5 uses hybrid linear + full attention. Linear attention layers store a
fused ``in_proj_qkv`` weight (Q‖K‖V, with potentially unequal dims), but
Tinker trains separate ``in_proj_q/k/v`` LoRA adapters. All Qwen3.5 models
are vision-language with the ``model.language_model.*`` key prefix.

Supported models: Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-397B-A17B,
Qwen3.6-27B, Qwen3.6-35B-A3B (same ``qwen3_5`` / ``qwen3_5_moe`` architecture family).
"""

from __future__ import annotations

import torch

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._merge import MergeOp, MergeProfile
from tinker_cookbook.weights._merge_utils import (
    extract_adapter_weight_names,
    plan_expert_ops,
    plan_standard_op,
    remap_adapter_name,
    validate_adapter_config,
)

SPLIT_QKV_SUFFIXES: dict[str, str] = {
    ".in_proj_q.weight": "q",
    ".in_proj_k.weight": "k",
    ".in_proj_v.weight": "v",
}
"""Mapping from split-QKV key suffixes to their role (q/k/v).

Public so that :mod:`tinker_cookbook.weights._adapter` can reuse it for
PEFT adapter conversion without duplicating the constant.
"""

# Backward-compatible alias (internal callers may use the old name).
_SPLIT_QKV_SUFFIXES = SPLIT_QKV_SUFFIXES


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
    """Detect Qwen3.5 models (dense and MoE).

    Qwen3.5 uses hybrid linear + full attention. Linear attention layers store
    a fused ``in_proj_qkv`` weight (Q‖K‖V, with potentially unequal dims), but
    Tinker trains separate ``in_proj_q/k/v`` LoRA adapters. All Qwen3.5 models
    are vision-language with the ``model.language_model.*`` key prefix.

    Supported models: Qwen3.5-4B, Qwen3.5-9B,
    Qwen3.5-397B-A17B, Qwen3.6-27B, and Qwen3.6-35B-A3B (same ``qwen3_5`` /
    ``qwen3_5_moe`` model_type).
    """
    if model_config.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        return None

    has_fused_experts = any(k.endswith(".experts.gate_up_proj") for k in model_state_keys)
    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        model_family="qwen3_5",
        expert_layout="fused_concatenated" if has_fused_experts else "separate",
        has_language_model_prefix=has_lm_prefix,
        split_qkv_projections=True,
    )


# ---------------------------------------------------------------------------
# Name remapping (Qwen3.5-specific)
# ---------------------------------------------------------------------------


def build_qwen3_5_name_remaps(
    profile: MergeProfile, model_state_keys: set[str]
) -> list[tuple[str, str]]:
    """Build name remaps for Qwen3.5.

    Handles the ``unembed_tokens`` remap for vision models with tied
    embeddings: when ``lm_head.weight`` is absent (``tie_word_embeddings=True``,
    e.g. Qwen3.5-4B), merges into ``embed_tokens`` instead.

    Public so that :mod:`tinker_cookbook.weights._adapter` can reuse it.
    """
    remaps: list[tuple[str, str]] = [("base_model.model.", "")]
    if profile.has_language_model_prefix:
        remaps.append(("model.", "model.language_model."))
        has_top_level_lm_head = any(
            k == "lm_head.weight" or k.startswith("lm_head.") for k in model_state_keys
        )
        if has_top_level_lm_head:
            # Non-tied (e.g. Qwen3.6-27B): lm_head stored at top level
            remaps.append(("model.language_model.unembed_tokens", "lm_head"))
        else:
            # Tied (e.g. Qwen3.5-4B): no separate lm_head; merge into embed_tokens
            remaps.append(
                ("model.language_model.unembed_tokens", "model.language_model.embed_tokens")
            )
    else:
        remaps.append(("model.unembed_tokens", "lm_head"))
    return remaps


# ---------------------------------------------------------------------------
# Split QKV → fused QKV planning
# ---------------------------------------------------------------------------


def _plan_split_qkv_op(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    adapter_weights: dict[str, torch.Tensor],
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> bool:
    """Try to plan a split-QKV → fused-QKV merge op.

    Checks whether ``target_key`` ends with a split QKV suffix (``in_proj_q``,
    ``in_proj_k``, or ``in_proj_v``) and the model has a corresponding fused
    ``in_proj_qkv`` key. If so, computes the row offset from sibling ``lora_B``
    shapes and appends a sliced :class:`MergeOp`.

    Returns:
        True if the key was handled (caller should skip normal planning),
        False if this is not a split-QKV key or the fused target doesn't exist.
    """
    qkv_match = next(
        (
            (suffix, role)
            for suffix, role in _SPLIT_QKV_SUFFIXES.items()
            if target_key.endswith(suffix)
        ),
        None,
    )
    if qkv_match is None:
        return False

    suffix, role = qkv_match
    fused_key = target_key[: -len(suffix)] + ".in_proj_qkv.weight"
    if fused_key not in model_state_keys:
        return False

    # Derive row offsets from sibling lora_B shapes. The fused layout is Q‖K‖V
    # where each section's row count equals the corresponding lora_B out_dim.
    # All three of q/k/v must be present in the adapter for offset computation.
    adapter_prefix = adapter_name[: -len(suffix)]
    q_key = adapter_prefix + ".in_proj_q.lora_B.weight"
    k_key = adapter_prefix + ".in_proj_k.lora_B.weight"
    if q_key not in adapter_weights or k_key not in adapter_weights:
        raise WeightsMergeError(
            f"Split QKV fusion requires all three in_proj_q/k/v adapters for the same "
            f"layer, but sibling weights are missing for {adapter_name!r}"
        )
    q_rows = adapter_weights[q_key].shape[0]
    k_rows = adapter_weights[k_key].shape[0]
    start = {"q": 0, "k": q_rows, "v": q_rows + k_rows}[role]

    ops.setdefault(fused_key, []).append(
        MergeOp(target_key=fused_key, lora_A=lora_A, lora_B=lora_B, slice_start=start)
    )
    return True


# ---------------------------------------------------------------------------
# Main planning function
# ---------------------------------------------------------------------------


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan merge ops for Qwen3.5 models."""
    scaling = validate_adapter_config(adapter_config, profile)
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    name_remaps = build_qwen3_5_name_remaps(profile, model_state_keys)

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = remap_adapter_name(n, name_remaps)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            if _plan_split_qkv_op(
                target_key, lora_A, lora_B, n, adapter_weights, model_state_keys, ops
            ):
                continue
            plan_standard_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            plan_expert_ops(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)

    return ops
