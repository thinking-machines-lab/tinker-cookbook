"""LoRA adapter merge logic.

Shared infrastructure for all model families:

- ``MergeProfile`` / ``detect_merge_profile``: model-specific merge configuration
- ``MergeOp`` / ``plan_merge_ops`` / ``apply_merge_op``: plan-then-execute merge pipeline
- ``merge_lora_matrices`` / ``expand_expert_lora_tensors``: low-level math utilities
- ``merge_adapter_weights``: backward-compatible convenience wrapper

Per-model planning lives in ``_merge_<model>.py`` modules. This file provides
the shared primitives they compose and the dispatch layer that routes to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tinker_cookbook.exceptions import WeightsMergeError

if TYPE_CHECKING:
    from collections.abc import Callable

    # Profile detector: (config_dict, model_state_keys) -> MergeProfile | None
    _ProfileDetector = Callable[[dict, "set[str]"], "MergeProfile | None"]

    # Per-model planning function
    _PlanFn = Callable[
        [dict, dict, "set[str]", "MergeProfile"],
        "dict[str, list[MergeOp]]",
    ]

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

    model_family: str = "default"
    """Identifier for the model family. Used to dispatch to the correct
    per-model planning module in :func:`plan_merge_ops`."""

    expert_key_remaps: tuple[tuple[str, str], ...] = (
        (".w1.weight", ".gate_proj.weight"),
        (".w3.weight", ".up_proj.weight"),
        (".w2.weight", ".down_proj.weight"),
    )
    """Mapping from Tinker internal expert projection names (w1/w2/w3) to
    HuggingFace parameter names (gate_proj/up_proj/down_proj).

    The default mapping matches standard 3-projection MoE models (Qwen3 MoE,
    DeepSeek, Kimi). Override for models with different expert layouts, e.g.
    Nemotron which has only up_proj and down_proj (no gate_proj).

    Uses tuple-of-tuples rather than dict so ``MergeProfile`` stays hashable.
    """

    fused_projection_map: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Mapping from fused target module → component projection names that
    Tinker trains as separate LoRA but HF/vLLM fuse into one module.

    During adapter conversion, component weights are merged into the fused
    target. For example, Nemotron Mamba trains ``gate_proj``/``x_proj``
    separately, but HF fuses both into ``in_proj``. The mapping would be
    ``(("in_proj", ("gate_proj", "x_proj")),)``.

    The component projections must map to consecutive row slices in the
    fused target, in the order listed.

    Empty by default — most models don't have this mismatch.
    """

    split_qkv_projections: bool = False
    """Whether Tinker trains separate ``in_proj_q/k/v`` LoRA adapters that must
    be merged into a single fused ``in_proj_qkv`` weight in the HF model. When
    True, the per-model planner redirects split keys to the fused target with
    row-slice offsets derived from sibling ``lora_B`` shapes.

    Currently only applies to Qwen3.5 models (hybrid linear attention layers
    with Q‖K‖V layout where Q, K, V may have unequal dimensions)."""


def detect_merge_profile(
    model_config: dict,
    model_state_keys: set[str],
) -> MergeProfile:
    """Detect merge profile from model config and weight key names.

    Tries each registered model-specific detector in order. The first one
    that returns a profile wins. Falls back to the default detector if none
    match.

    To add support for a new model family, create a ``_merge_<model>.py``
    module with ``detect_profile`` and ``plan_merge_ops`` functions, then
    register it in :data:`_PROFILE_DETECTORS` and :data:`_PLAN_FUNCTIONS`.

    Works with both loaded models (full path) and safetensors headers
    (shard path), since both can provide a config dict and key names.

    Args:
        model_config: Parsed ``config.json`` or equivalent dict. Uses the
            ``"architectures"`` key for model family detection.
        model_state_keys: Weight key names from the model state dict
            or safetensors headers.
    """
    for detector in _get_profile_detectors():
        profile = detector(model_config, model_state_keys)
        if profile is not None:
            return profile
    from tinker_cookbook.weights._merge_default import detect_profile as _detect_default

    return _detect_default(model_config, model_state_keys)


# ---------------------------------------------------------------------------
# Per-model profile detectors and plan functions
#
# Each per-model module exports detect_profile() and plan_merge_ops().
# Detectors are tried in order; first match wins. The plan function is
# dispatched via model_family on the profile.
# ---------------------------------------------------------------------------


def _get_profile_detectors() -> list[_ProfileDetector]:
    """Import per-model detectors. Uses function-local imports to avoid
    circular dependencies (per-model modules import from this file)."""
    from tinker_cookbook.weights._merge_deepseek import detect_profile as _deepseek
    from tinker_cookbook.weights._merge_gpt_oss import detect_profile as _gpt_oss
    from tinker_cookbook.weights._merge_nemotron import detect_profile as _nemotron
    from tinker_cookbook.weights._merge_qwen3_5 import detect_profile as _qwen3_5

    return [_gpt_oss, _deepseek, _nemotron, _qwen3_5]


def _get_plan_functions() -> dict[str, _PlanFn]:
    """Import per-model plan functions. Uses function-local imports to avoid
    circular dependencies (per-model modules import from this file)."""
    from tinker_cookbook.weights._merge_deepseek import plan_merge_ops as _deepseek_plan
    from tinker_cookbook.weights._merge_default import plan_merge_ops as _default_plan
    from tinker_cookbook.weights._merge_gpt_oss import plan_merge_ops as _gpt_oss_plan
    from tinker_cookbook.weights._merge_qwen3_5 import plan_merge_ops as _qwen3_5_plan

    return {
        "default": _default_plan,
        "gpt_oss": _gpt_oss_plan,
        "deepseek": _deepseek_plan,
        "nemotron": _default_plan,
        "qwen3_5": _qwen3_5_plan,
    }


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

    slice_start: int | None = None
    """Row offset into a fused target weight for split projections (e.g.
    ``in_proj_qkv``). When set, :func:`apply_merge_op` writes the delta only
    to ``target[slice_start : slice_start + out_dim]``.

    Mutually exclusive with ``is_expert_3d`` — slice targeting is only
    supported for 2D ops."""

    def __post_init__(self) -> None:
        if self.slice_start is not None and self.is_expert_3d:
            raise ValueError("slice_start is not supported for 3D expert ops")


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
        raise WeightsMergeError(
            f"Cannot broadcast expert LoRA: both A and B have 1 expert "
            f"(lora_A: {lora_A.shape}, lora_B: {lora_B.shape})"
        )
    if lora_A.shape[0] == 1:
        lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
    elif lora_B.shape[0] == 1:
        lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
    elif lora_A.shape[0] != lora_B.shape[0]:
        raise WeightsMergeError(
            f"Expert count mismatch: lora_A has {lora_A.shape[0]} experts, "
            f"lora_B has {lora_B.shape[0]} experts "
            f"(lora_A: {lora_A.shape}, lora_B: {lora_B.shape})"
        )
    return lora_A, lora_B


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor) -> None:
    """Add a merged LoRA delta to a model weight tensor in-place."""
    if target.shape != merged_lora.shape:
        raise WeightsMergeError(
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

    Dispatches to the per-model planning function based on
    ``profile.model_family``. Each per-model module
    (``_merge_<model>.py``) provides its own ``plan_merge_ops`` that
    handles model-specific name remapping and op construction.

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
    plan_functions = _get_plan_functions()
    plan_fn = plan_functions.get(profile.model_family)
    if plan_fn is None:
        raise WeightsMergeError(
            f"Unknown model_family {profile.model_family!r}. "
            f"Registered families: {sorted(plan_functions)}"
        )
    return plan_fn(adapter_weights, adapter_config, model_state_keys, profile)


def _detect_fused_axis(
    target_shape: tuple[int, ...],
    delta_shape: tuple[int, ...],
    target_key: str,
) -> int:
    """Detect which axis of a 3D fused expert tensor holds the fused dimension.

    MoE models with fused ``gate_up_proj`` store the concatenated gate+up
    projections on different axes depending on the model family:

    - **Qwen3-VL MoE**: ``(n_exp, hidden, fused)`` — fused on dim 2
    - **Qwen3.5 MoE**: ``(n_exp, fused, hidden)`` — fused on dim 1

    The LoRA delta has shape ``(n_exp, in_dim, out_dim)`` where ``out_dim``
    equals *half* the fused dimension. We detect the fused axis by checking
    which target dimension is consistent with the delta.

    For dim 2 fused: ``target = (n, in_dim, 2*out_dim)``, delta matches directly.
    For dim 1 fused: ``target = (n, 2*out_dim, in_dim)``, delta must be
    transposed before applying (``apply_merge_op`` handles this).

    Returns:
        ``1`` or ``2`` — the axis index of the fused dimension.

    Raises:
        WeightsMergeError: If neither axis is consistent with the delta.
    """
    _, t1, t2 = target_shape
    _, d1, d2 = delta_shape

    # Check dim 2: target = (n, in_dim, 2*out_dim)
    # delta (n, in_dim, out_dim) where d1==t1 and t2==2*d2
    if d1 == t1 and t2 == 2 * d2:
        return 2

    # Check dim 1: target = (n, 2*out_dim, in_dim)
    # delta (n, in_dim, out_dim) is transposed to (n, out_dim, in_dim) at apply time
    if t2 == d1 and t1 == 2 * d2:
        return 1

    raise WeightsMergeError(
        f"Cannot determine fused axis for {target_key!r}: "
        f"target shape {target_shape}, delta shape {delta_shape}. "
        f"Expected one axis to be exactly 2x the corresponding delta dimension."
    )


def validate_merge_op_shapes(
    ops: dict[str, list[MergeOp]],
    model_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate all merge op output shapes against model weight shapes upfront.

    Call this after :func:`plan_merge_ops` and before processing any shards.
    Catches shape mismatches early, before expensive shard I/O begins.

    Args:
        ops: Mapping from target key to merge ops (from :func:`plan_merge_ops`).
        model_shapes: Mapping from weight key to shape (from
            :func:`~tinker_cookbook.weights._artifacts.get_model_state_shapes`).

    Raises:
        ValueError: If any merge op's delta shape doesn't match its target.
    """
    for target_key, op_list in ops.items():
        target_shape = model_shapes[target_key]
        for op in op_list:
            if op.is_expert_3d:
                # bmm(A.T, B.T) → (num_experts, in_dim, out_dim)
                n_exp, rank, in_dim = op.lora_A.shape
                _, out_dim, _ = op.lora_B.shape
                delta_shape = (n_exp, in_dim, out_dim)

                if op.fused_proj_idx is not None:
                    # Delta targets a slice of the fused tensor.  The fused
                    # dimension can be either dim 1 or dim 2 depending on the
                    # model family:
                    #   Qwen3-VL MoE:  gate_up_proj = (n, hidden, fused)  → dim 2
                    #   Qwen3.5 MoE:   gate_up_proj = (n, fused, hidden)  → dim 1
                    fused_axis = _detect_fused_axis(target_shape, delta_shape, target_key)
                    if fused_axis == 2:
                        # delta (n, in_dim, out_dim) targets slice (n, dim1, dim2//2)
                        expected = (target_shape[0], target_shape[1], target_shape[2] // 2)
                    else:
                        # delta (n, in_dim, out_dim) is transposed to (n, out_dim, in_dim)
                        # to match slice (n, dim1//2, dim2)
                        expected = (target_shape[0], target_shape[2], target_shape[1] // 2)
                else:
                    # Non-fused 3D expert weight.  Some models store expert
                    # weights transposed: (n, out_dim, in_dim) instead of
                    # (n, in_dim, out_dim).  Accept either orientation; the
                    # delta is transposed at apply time if needed.
                    transposed = (target_shape[0], target_shape[2], target_shape[1])
                    if delta_shape != target_shape and delta_shape != transposed:
                        raise WeightsMergeError(
                            f"Shape mismatch for {target_key!r}: "
                            f"merge op produces {delta_shape} but target "
                            f"expects {target_shape}"
                        )
                    # Skip the common check below — already validated.
                    continue

                if delta_shape != expected:
                    raise WeightsMergeError(
                        f"Shape mismatch for {target_key!r}: "
                        f"merge op produces {delta_shape} but target "
                        f"{'slice ' if op.fused_proj_idx is not None else ''}"
                        f"expects {expected}"
                    )
            else:
                # 2D: delta = lora_B @ lora_A → (out_dim, in_dim)
                delta_shape = (op.lora_B.shape[0], op.lora_A.shape[1])
                if op.slice_start is not None:
                    # Sliced op: check in_dim matches and slice fits within target rows
                    end = op.slice_start + delta_shape[0]
                    if delta_shape[1] != target_shape[1] or end > target_shape[0]:
                        raise WeightsMergeError(
                            f"Shape mismatch for {target_key!r} "
                            f"slice [{op.slice_start}:{end}]: "
                            f"merge op produces {delta_shape} "
                            f"but target has shape {target_shape}"
                        )
                elif delta_shape != target_shape:
                    raise WeightsMergeError(
                        f"Shape mismatch for {target_key!r}: "
                        f"merge op produces {delta_shape} but target expects {target_shape}"
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
            # Determine which target axis is fused (see _detect_fused_axis).
            fused_axis = _detect_fused_axis(target.shape, delta.shape, op.target_key)
            if op.fused_proj_interleaved:
                # Interleaved layout: always along last dim ([g0,u0,g1,u1,...])
                target_view = target[:, :, op.fused_proj_idx :: 2]
            elif fused_axis == 2:
                # Concatenated along last dim: (n, hidden, [gate|up])
                proj_width = target.shape[2] // 2
                start = op.fused_proj_idx * proj_width
                target_view = target[:, :, start : start + proj_width]
            else:
                # Concatenated along dim 1: (n, [gate|up], hidden)
                # Delta is (n, in_dim, out_dim) but target slice is
                # (n, out_dim, in_dim), so transpose the delta.
                delta = delta.transpose(-1, -2)
                proj_width = target.shape[1] // 2
                start = op.fused_proj_idx * proj_width
                target_view = target[:, start : start + proj_width, :]
            apply_merged_weight(target_view, delta)
        else:
            # Non-fused: transpose delta if target uses opposite layout.
            if delta.shape != target.shape:
                delta = delta.transpose(-1, -2)
            apply_merged_weight(target, delta)
    else:
        # 2D: standard linear or per-expert (already sliced during planning)
        delta = merge_lora_matrices(op.lora_A, op.lora_B)
        if op.slice_start is not None:
            target = tensors[op.target_key]
            apply_merged_weight(target[op.slice_start : op.slice_start + delta.shape[0]], delta)
        else:
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

    # Build a config dict for detect_merge_profile. Prefer the model's HF
    # config (which has the real architectures list) over fragile class name
    # string matching.
    config_obj = getattr(base_model, "config", None)
    if config_obj is not None and hasattr(config_obj, "to_dict"):
        model_config = config_obj.to_dict()
    else:
        # Fallback for non-HF models (e.g. test mocks)
        is_gpt_oss = "GptOss" in str(type(base_model))
        model_config = {"architectures": ["GptOssForCausalLM"] if is_gpt_oss else []}
    profile = detect_merge_profile(model_config, model_state_keys)

    ops = plan_merge_ops(adapter_weights, config, model_state_keys, profile)
    for op_list in ops.values():
        for op in op_list:
            apply_merge_op(model_state_dict, op)
