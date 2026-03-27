"""Shared planning building blocks for per-model merge modules.

Provides reusable functions for common merge planning patterns:
name remapping, standard (non-expert) op planning, and expert op planning.
Per-model modules compose these to build their ``plan_merge_ops`` functions.
"""

from __future__ import annotations

import logging
import re

import torch

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._merge import (
    _VALID_EXPERT_LAYOUTS,
    MergeOp,
    MergeProfile,
    expand_expert_lora_tensors,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation and adapter weight extraction
# ---------------------------------------------------------------------------


def validate_adapter_config(adapter_config: dict, profile: MergeProfile) -> float:
    """Validate adapter config and return the scaling factor.

    Args:
        adapter_config: Adapter config with ``lora_alpha`` and ``r`` keys.
        profile: Model-specific merge configuration.

    Returns:
        Scaling factor ``lora_alpha / r``.

    Raises:
        WeightsMergeError: If required config keys are missing or
            ``profile.expert_layout`` is invalid.
    """
    for key in ("lora_alpha", "r"):
        if key not in adapter_config:
            raise WeightsMergeError(f"Adapter config missing required key: {key!r}")

    if profile.expert_layout not in _VALID_EXPERT_LAYOUTS:
        raise WeightsMergeError(
            f"Invalid expert_layout {profile.expert_layout!r}. "
            f"Must be one of: {sorted(_VALID_EXPERT_LAYOUTS)}"
        )

    return adapter_config["lora_alpha"] / adapter_config["r"]


def extract_adapter_weight_names(adapter_weights: dict[str, torch.Tensor]) -> list[str]:
    """Extract adapter weight names from the adapter dict.

    Returns the base weight names (with ``.lora_A`` stripped). Logs a warning
    if no LoRA weights are found.
    """
    names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]
    if not names:
        logger.warning(
            "No LoRA weights found in adapter (no keys containing '.lora_A'). "
            "The output model will be identical to the base model. "
            "Check that the adapter path points to a valid Tinker LoRA adapter."
        )
    return names


# ---------------------------------------------------------------------------
# Name remapping
# ---------------------------------------------------------------------------


def build_name_remaps(
    profile: MergeProfile,
    model_state_keys: set[str],
) -> list[tuple[str, str]]:
    """Build the standard name remap list for adapter → model key mapping.

    Order matters: strip ``base_model.model.`` prefix first, then remap
    ``unembed_tokens``, then apply vision model prefix.

    Args:
        profile: Model-specific merge configuration.
        model_state_keys: Set of weight key names in the base model.
    """
    remaps: list[tuple[str, str]] = [
        ("base_model.model.", ""),
        ("model.unembed_tokens", "lm_head"),
    ]
    if profile.has_language_model_prefix:
        remaps.append(("model.", "model.language_model."))
    return remaps


def remap_adapter_name(name: str, remaps: list[tuple[str, str]]) -> str:
    """Apply sequential string replacements to map adapter name to model key."""
    for old, new in remaps:
        name = name.replace(old, new)
    return name


# ---------------------------------------------------------------------------
# Non-expert op planning
# ---------------------------------------------------------------------------


def plan_standard_op(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    profile: MergeProfile,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    """Plan a merge op for a standard (non-expert) linear layer.

    Applies ``profile.extra_key_remaps`` and validates the target key exists
    in the model state dict.
    """
    for old, new in profile.extra_key_remaps:
        target_key = target_key.replace(old, new)

    if target_key not in model_state_keys:
        raise WeightsMergeError(
            f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
            f"which does not exist in the model state dict"
        )
    ops.setdefault(target_key, []).append(
        MergeOp(target_key=target_key, lora_A=lora_A, lora_B=lora_B)
    )


# ---------------------------------------------------------------------------
# Fused projection op planning
# ---------------------------------------------------------------------------


def plan_fused_projection_op(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    adapter_weights: dict[str, torch.Tensor],
    profile: MergeProfile,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> bool:
    """Try to plan a fused-projection merge op.

    Checks whether ``target_key`` targets a component projection that the HF
    model fuses into a single module (e.g. Nemotron Mamba's ``gate_proj`` and
    ``x_proj`` are fused into ``in_proj``).  If so, computes the row offset
    from sibling ``lora_B`` shapes and appends a sliced :class:`MergeOp`.

    Follows the same pattern as :func:`_plan_split_qkv_op` in
    ``_merge_qwen3_5.py``.

    Returns:
        True if the key was handled (caller should skip normal planning),
        False if this is not a fused-projection component.
    """
    if not profile.fused_projection_map:
        return False

    leaf = target_key.removesuffix(".weight").rsplit(".", 1)[-1]
    fused_target: str | None = None
    comp_idx: int | None = None
    for target_name, components in profile.fused_projection_map:
        if leaf in components:
            fused_target = target_name
            comp_idx = components.index(leaf)
            break

    if fused_target is None:
        return False

    # Derive the fused target key in the model state dict.
    # The adapter namespace may differ from the model namespace (e.g.
    # adapter has "model.layers.0.mixer.gate_proj.weight" but model has
    # "backbone.layers.0.mixer.in_proj.weight").
    fused_suffix = f".{fused_target}.weight"
    layer_prefix = target_key.removesuffix(f".{leaf}.weight")

    layer_match = re.search(r"\.layers\.(\d+)\.", layer_prefix)
    if layer_match is None:
        return False

    layer_idx = layer_match.group(1)
    pattern = f".layers.{layer_idx}."
    fused_key: str | None = None
    for k in model_state_keys:
        if pattern in k and k.endswith(fused_suffix):
            fused_key = k
            break

    if fused_key is None:
        return False

    # Compute row offset from sibling lora_B shapes. The fused layout has
    # components in order, each contributing lora_B.shape[0] rows.
    _, components = next((t, c) for t, c in profile.fused_projection_map if t == fused_target)

    start = 0
    adapter_prefix = adapter_name.removesuffix(f".{leaf}.weight")
    for i, comp_name in enumerate(components):
        if i == comp_idx:
            break
        sibling_B_key = f"{adapter_prefix}.{comp_name}.lora_B.weight"
        if sibling_B_key not in adapter_weights:
            raise WeightsMergeError(
                f"Fused projection merge requires all components {components} "
                f"for the same layer, but {sibling_B_key!r} is missing"
            )
        start += adapter_weights[sibling_B_key].shape[0]

    ops.setdefault(fused_key, []).append(
        MergeOp(target_key=fused_key, lora_A=lora_A, lora_B=lora_B, slice_start=start)
    )
    return True


# ---------------------------------------------------------------------------
# Expert op planning
# ---------------------------------------------------------------------------


def plan_expert_ops(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    profile: MergeProfile,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    """Plan merge ops for expert weights (separate or fused)."""
    # Skip empty expert LoRA tensors — these are placeholders for projections
    # that don't exist in the model (e.g. Nemotron's empty w3).
    if lora_A.numel() == 0 and lora_B.numel() == 0:
        return

    if lora_A.ndim != 3 or lora_B.ndim != 3:
        raise WeightsMergeError(
            f"Expert LoRA weights must be 3D, got lora_A: {lora_A.shape}, lora_B: {lora_B.shape}"
        )
    lora_A, lora_B = expand_expert_lora_tensors(lora_A, lora_B)

    # Apply general key remaps (e.g. model. → backbone. for Nemotron)
    for old, new in profile.extra_key_remaps:
        target_key = target_key.replace(old, new)

    for old, new in profile.expert_key_remaps:
        target_key = target_key.replace(old, new)

    is_fused = profile.expert_layout in ("fused_interleaved", "fused_concatenated")
    is_interleaved = profile.expert_layout == "fused_interleaved"

    if not is_fused:
        # Separate per-expert weights: create one 2D MergeOp per expert
        for exp_idx in range(lora_A.shape[0]):
            target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
            if target_key_exp not in model_state_keys:
                raise WeightsMergeError(
                    f"Adapter weight {adapter_name!r} mapped to {target_key_exp!r} "
                    f"which does not exist in the model state dict"
                )
            ops.setdefault(target_key_exp, []).append(
                MergeOp(
                    target_key=target_key_exp,
                    lora_A=lora_A[exp_idx],
                    lora_B=lora_B[exp_idx],
                )
            )
    else:
        # Fused expert weights: create one 3D MergeOp
        fused_proj_idx: int | None = None
        if target_key.endswith(".gate_proj.weight"):
            fused_proj_idx = 0
            target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj")
        elif target_key.endswith(".up_proj.weight"):
            fused_proj_idx = 1
            target_key = target_key.replace(".up_proj.weight", ".gate_up_proj")
        else:
            target_key = target_key.replace(".down_proj.weight", ".down_proj")

        if target_key not in model_state_keys:
            raise WeightsMergeError(
                f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
                f"which does not exist in the model state dict"
            )
        ops.setdefault(target_key, []).append(
            MergeOp(
                target_key=target_key,
                lora_A=lora_A,
                lora_B=lora_B,
                is_expert_3d=True,
                fused_proj_idx=fused_proj_idx,
                fused_proj_interleaved=is_interleaved,
            )
        )
