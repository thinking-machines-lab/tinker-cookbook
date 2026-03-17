"""LoRA adapter merge logic.

Provides shared merge primitives used by all export strategies:

- ``MergeProfile`` / ``detect_merge_profile``: model-specific merge configuration
- ``MergeOp`` / ``plan_merge_ops`` / ``apply_merge_op``: plan-then-execute merge pipeline
- ``merge_lora_matrices`` / ``expand_expert_lora_tensors``: low-level math utilities
- ``merge_adapter_weights``: backward-compatible convenience wrapper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    # Profile detector: (config_dict, model_state_keys) -> MergeProfile | None
    _ProfileDetector = Callable[[dict, "set[str]"], "MergeProfile | None"]

# ---------------------------------------------------------------------------
# MergeProfile — model-specific merge configuration
# ---------------------------------------------------------------------------

_VALID_EXPERT_LAYOUTS = frozenset({"separate", "fused_interleaved", "fused_concatenated"})


@dataclass(frozen=True)
class MergeProfile:
    """Describes model-specific merge behavior.

    Captures merge-level variation between model families: how adapter weight
    names map to model weight names, and how expert weights are laid out.

    Does NOT capture export-level concerns (output format, quantization,
    shard layout) — those belong in export strategy modules.
    """

    expert_layout: str = "separate"
    """How expert weights are arranged in the model.

    - ``"separate"`` — individual weight per expert (Qwen3 MoE, DeepSeek)
    - ``"fused_interleaved"`` — gate_up_proj with [g0, u0, g1, u1, ...] (GPT-OSS)
    - ``"fused_concatenated"`` — gate_up_proj with [gate | up] (Qwen3.5, Qwen3-VL)
    """

    extra_key_remaps: tuple[tuple[str, str], ...] = ()
    """Additional key remapping rules applied after standard remaps.

    Each ``(old, new)`` pair is applied via ``str.replace`` on the target key.
    Example: ``((".attn", ".self_attn"),)`` for GPT-OSS.

    Note: these remaps are applied to non-expert keys only. Expert keys go
    through a separate remapping path (``w1→gate_proj``, etc.) that doesn't
    use ``extra_key_remaps``. If a future model needs remaps on expert keys,
    this should be extended.

    Uses tuple-of-tuples rather than dict so ``MergeProfile`` stays hashable.
    """

    has_language_model_prefix: bool = False
    """Whether model keys use ``model.language_model.`` prefix (vision models)."""


def detect_merge_profile(
    model_config: dict,
    model_state_keys: set[str],
) -> MergeProfile:
    """Detect merge profile from model config and weight key names.

    Tries each registered model-specific detector in order. The first one
    that returns a profile wins. Falls back to :func:`_detect_default_profile`
    if none match.

    To add support for a new model family, write a detector function with
    signature ``(dict, set[str]) -> MergeProfile | None`` and append it to
    :data:`_PROFILE_DETECTORS`.

    Works with both loaded models (full path) and safetensors headers
    (shard path), since both can provide a config dict and key names.

    Args:
        model_config: Parsed ``config.json`` or equivalent dict. Uses the
            ``"architectures"`` key for model family detection.
        model_state_keys: Weight key names from the model state dict
            or safetensors headers.
    """
    for detector in _PROFILE_DETECTORS:
        profile = detector(model_config, model_state_keys)
        if profile is not None:
            return profile
    return _detect_default_profile(model_config, model_state_keys)


# ---------------------------------------------------------------------------
# Per-model profile detectors
#
# Each detector returns a MergeProfile if it recognizes the model, or None
# to pass to the next detector. Add new detectors to _PROFILE_DETECTORS.
# ---------------------------------------------------------------------------


def _detect_gpt_oss_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
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
        expert_layout="fused_interleaved" if has_fused else "separate",
        extra_key_remaps=((".attn", ".self_attn"),),
        has_language_model_prefix=has_lm_prefix,
    )


def _detect_default_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile:
    """Default profile for models without special merge requirements.

    Handles Qwen, DeepSeek, and other standard model families. Detects fused
    expert layout (concatenated, not interleaved) and vision model prefix
    from key names alone.
    """
    has_fused = any(k.endswith(".experts.gate_up_proj") for k in model_state_keys)
    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        expert_layout="fused_concatenated" if has_fused else "separate",
        has_language_model_prefix=has_lm_prefix,
    )


# Detectors are tried in order. First match wins.
_PROFILE_DETECTORS: list = [
    _detect_gpt_oss_profile,
]


# ---------------------------------------------------------------------------
# MergeOp — a pending LoRA merge operation
# ---------------------------------------------------------------------------


@dataclass
class MergeOp:
    """A pending LoRA merge operation.

    Stores only the small rank-sized LoRA matrices. The model-sized delta is
    computed on-the-fly during :func:`apply_merge_op`, keeping peak memory
    proportional to LoRA rank rather than model size.
    """

    target_key: str

    lora_A: torch.Tensor
    """Shape ``(rank, in_dim)`` for 2D ops, ``(num_experts, rank, in_dim)`` for 3D."""

    lora_B: torch.Tensor
    """Shape ``(out_dim, rank)`` for 2D ops, ``(num_experts, out_dim, rank)`` for 3D.
    Pre-scaled by ``lora_alpha / r``."""

    is_expert_3d: bool = False
    """True for fused expert weights where lora_A/B are 3D."""

    fused_proj_idx: int | None = None
    """For fused gate/up projections: 0 = gate, 1 = up, None = not fused."""

    fused_proj_interleaved: bool = False
    """GPT-OSS stores fused gate/up projections interleaved rather than concatenated."""


# ---------------------------------------------------------------------------
# Low-level math utilities
# ---------------------------------------------------------------------------


def merge_lora_matrices(lora_A: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
    """Compute 2D LoRA delta: ``lora_B @ lora_A``.

    Args:
        lora_A: Shape ``(rank, in_dim)``.
        lora_B: Shape ``(out_dim, rank)``, pre-scaled by ``alpha / r``.

    Returns:
        Delta tensor of shape ``(out_dim, in_dim)``.
    """
    return lora_B @ lora_A


def expand_expert_lora_tensors(
    lora_A: torch.Tensor, lora_B: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast shared expert LoRA tensors to match num_experts.

    When one tensor has ``shape[0] == 1`` and the other has ``shape[0] > 1``,
    expands the single-expert tensor to match.

    Args:
        lora_A: Shape ``(num_experts_a, rank, in_dim)``.
        lora_B: Shape ``(num_experts_b, out_dim, rank)``.

    Returns:
        Tuple of ``(lora_A, lora_B)`` with matching ``shape[0]``.

    Raises:
        ValueError: If both tensors have ``shape[0] == 1``.
    """
    if lora_A.shape[0] == 1 and lora_B.shape[0] == 1:
        raise ValueError(
            f"Cannot broadcast expert LoRA: both A and B have 1 expert "
            f"(lora_A: {lora_A.shape}, lora_B: {lora_B.shape})"
        )
    if lora_A.shape[0] == 1:
        lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
    elif lora_B.shape[0] == 1:
        lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
    return lora_A, lora_B


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor) -> None:
    """Add a merged LoRA delta to a model weight tensor in-place."""
    if target.shape != merged_lora.shape:
        raise ValueError(
            f"Shape mismatch: target {target.shape} vs merged LoRA {merged_lora.shape}"
        )
    new_data = target.float() + merged_lora.float().to(target.device)
    target.copy_(new_data.to(target.dtype))


# ---------------------------------------------------------------------------
# Plan + apply
# ---------------------------------------------------------------------------


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan all merge operations without executing them.

    Maps adapter weight names to model weight keys using the profile's
    remapping rules, validates all target keys exist, and returns a dict
    of pending merge operations grouped by target key.

    Args:
        adapter_weights: LoRA weight tensors from the adapter.
        adapter_config: Adapter config with ``lora_alpha`` and ``r`` keys.
        model_state_keys: Set of weight key names in the base model.
        profile: Model-specific merge configuration.

    Returns:
        Mapping from model weight key to list of :class:`MergeOp` targeting it.

    Raises:
        KeyError: If adapter config is missing required keys, or adapter
            weights map to keys not found in the model.
        ValueError: If expert LoRA tensors have unexpected shapes, or
            ``profile.expert_layout`` is invalid.
    """
    for key in ("lora_alpha", "r"):
        if key not in adapter_config:
            raise KeyError(f"Adapter config missing required key: {key!r}")

    if profile.expert_layout not in _VALID_EXPERT_LAYOUTS:
        raise ValueError(
            f"Invalid expert_layout {profile.expert_layout!r}. "
            f"Must be one of: {sorted(_VALID_EXPERT_LAYOUTS)}"
        )

    scaling = adapter_config["lora_alpha"] / adapter_config["r"]
    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

    is_fused = profile.expert_layout in ("fused_interleaved", "fused_concatenated")
    is_interleaved = profile.expert_layout == "fused_interleaved"

    # Standard name remapping (order matters: strip prefix before vision remap)
    name_remaps: list[tuple[str, str]] = [
        ("base_model.model.", ""),
        ("model.unembed_tokens", "lm_head"),
    ]
    if profile.has_language_model_prefix:
        name_remaps.append(("model.", "model.language_model."))

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = n
        for old, new in name_remaps:
            target_key = target_key.replace(old, new)

        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            _plan_non_expert_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            _plan_expert_ops(
                target_key, lora_A, lora_B, n, model_state_keys, ops, is_fused, is_interleaved
            )

    return ops


def _plan_non_expert_op(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    profile: MergeProfile,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    """Plan a merge op for a standard (non-expert) linear layer."""
    for old, new in profile.extra_key_remaps:
        target_key = target_key.replace(old, new)

    if target_key not in model_state_keys:
        raise KeyError(
            f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
            f"which does not exist in the model state dict"
        )
    ops.setdefault(target_key, []).append(
        MergeOp(target_key=target_key, lora_A=lora_A, lora_B=lora_B)
    )


def _plan_expert_ops(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
    is_fused: bool,
    is_interleaved: bool,
) -> None:
    """Plan merge ops for expert weights (separate or fused)."""
    if lora_A.ndim != 3 or lora_B.ndim != 3:
        raise ValueError(
            f"Expert LoRA weights must be 3D, got lora_A: {lora_A.shape}, lora_B: {lora_B.shape}"
        )
    lora_A, lora_B = expand_expert_lora_tensors(lora_A, lora_B)

    # Expert weight name remapping
    target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
    target_key = target_key.replace(".w3.weight", ".up_proj.weight")
    target_key = target_key.replace(".w2.weight", ".down_proj.weight")

    if not is_fused:
        # Separate per-expert weights: create one 2D MergeOp per expert
        for exp_idx in range(lora_A.shape[0]):
            target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
            if target_key_exp not in model_state_keys:
                raise KeyError(
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
            raise KeyError(
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


def apply_merge_op(tensors: dict[str, torch.Tensor], op: MergeOp) -> None:
    """Apply a single merge operation to a dict of tensors.

    Computes the LoRA delta on-the-fly and merges into the target tensor.
    Works with full model state dicts or individual shard tensor dicts.

    Args:
        tensors: Mutable dict of tensors (e.g. from :func:`safetensors.torch.load_file`
            or :meth:`torch.nn.Module.state_dict`). Modified in-place.
        op: The merge operation to apply.

    Raises:
        ValueError: If tensor shapes are incompatible.
    """
    target = tensors[op.target_key]

    if op.is_expert_3d:
        # (num_experts, rank, in_dim), (num_experts, out_dim, rank)
        # → (num_experts, in_dim, out_dim) via bmm of transposed
        delta = torch.bmm(op.lora_A.transpose(-1, -2), op.lora_B.transpose(-1, -2))

        if op.fused_proj_idx is not None:
            if op.fused_proj_interleaved:
                target_view = target[:, :, op.fused_proj_idx :: 2]
            else:
                proj_width = target.shape[-1] // 2
                start = op.fused_proj_idx * proj_width
                target_view = target[:, :, start : start + proj_width]
            apply_merged_weight(target_view, delta)
        else:
            apply_merged_weight(target, delta)
    else:
        # 2D: standard linear or per-expert (already sliced during planning)
        delta = merge_lora_matrices(op.lora_A, op.lora_B)
        apply_merged_weight(tensors[op.target_key], delta)


# ---------------------------------------------------------------------------
# Backward-compatible convenience wrapper
# ---------------------------------------------------------------------------


def merge_adapter_weights(
    base_model: torch.nn.Module, adapter_weights: dict[str, torch.Tensor], config: dict
) -> None:
    """Merge LoRA adapter weights into a base model's state dict in-place.

    Backward-compatible wrapper around :func:`plan_merge_ops` and
    :func:`apply_merge_op`.

    Handles:
    - Standard (non-expert) linear layers
    - Separate per-expert weights (Qwen3 MoE, DeepSeek, Kimi)
    - Fused expert weights with interleaved layout (GPT-OSS)
    - Fused expert weights with concatenated layout (Qwen3.5, Qwen3-VL)
    - Vision model name prefix remapping
    - GPT-OSS attention name remapping

    Args:
        base_model: The HuggingFace model to merge into.
        adapter_weights: Dict of LoRA weight tensors from the adapter.
        config: Adapter config dict with ``lora_alpha`` and ``r`` keys.

    Raises:
        KeyError: If required config keys are missing or adapter weight
            names don't map to any model weight.
        ValueError: If tensor shapes are incompatible.
    """
    model_state_dict = base_model.state_dict()
    model_state_keys = set(model_state_dict.keys())

    # Build a synthetic config dict for detect_merge_profile. The full-model
    # path detects GPT-OSS from the class name (matching the original behavior).
    is_gpt_oss = "GptOss" in str(type(base_model))
    model_config: dict = {"architectures": ["GptOssForCausalLM"] if is_gpt_oss else []}
    profile = detect_merge_profile(model_config, model_state_keys)

    ops = plan_merge_ops(adapter_weights, config, model_state_keys, profile)
    for op_list in ops.values():
        for op in op_list:
            apply_merge_op(model_state_dict, op)
