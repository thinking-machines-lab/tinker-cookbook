"""Kimi K2.5 merge planning.

Kimi K2.5 is a vision-language model (``model_type=kimi_k25``) wrapping a
Kimi-K2 / DeepSeek-style text backbone.  It differs from other VL models in
two ways that require a dedicated merge module:

1. **VL prefix**: weight keys are ``language_model.model.layers.*`` (outer
   prefix ``language_model.``), whereas other VL models use
   ``model.language_model.layers.*`` (inner prefix ``language_model``).

2. **INT4 quantized experts**: routed expert weights use the
   compressed-tensors ``pack-quantized`` format (``weight_packed`` /
   ``weight_scale`` / ``weight_shape``).  To allow merge planning against
   standard ``.weight`` key names, this module creates *virtual* ``.weight``
   entries in the model-state key set, and the shard export handles the
   actual dequant → merge → requant.
"""

from __future__ import annotations

import torch

from tinker_cookbook.weights._merge import MergeOp, MergeProfile
from tinker_cookbook.weights._merge_utils import (
    extract_adapter_weight_names,
    plan_expert_ops,
    plan_standard_op,
    remap_adapter_name,
    validate_adapter_config,
)

# ---------------------------------------------------------------------------
# Profile detection
# ---------------------------------------------------------------------------


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
    """Detect Kimi K2.5 from ``model_type``.

    K2.5 uses DeepSeek-style separate per-expert weights and a non-standard
    VL prefix (``language_model.model.*``).  We set
    ``has_language_model_prefix=False`` because the default name-remap logic
    assumes the *inner* prefix pattern — this module provides its own
    remapping in :func:`plan_merge_ops`.

    Returns ``None`` for non-K2.5 models so the detector chain continues.
    """
    if model_config.get("model_type") != "kimi_k25":
        # Also check nested text_config for wrapper configs
        text_config = model_config.get("text_config", {})
        if not isinstance(text_config, dict):
            return None
        # The outer model_type should be kimi_k25
        if model_config.get("model_type") != "kimi_k25":
            return None

    return MergeProfile(
        model_family="kimi_k25",
        expert_layout="separate",
        has_language_model_prefix=False,  # We handle the prefix ourselves
    )


# ---------------------------------------------------------------------------
# Virtual weight keys
# ---------------------------------------------------------------------------

_PACKED_SUFFIX = ".weight_packed"
_WEIGHT_SUFFIX = ".weight"


def create_virtual_weight_keys(
    model_state_keys: set[str],
) -> tuple[set[str], dict[str, str]]:
    """Add virtual ``.weight`` keys for compressed-tensors packed weights.

    Some model checkpoints (Kimi K2.5) store routed expert weights in the
    compressed-tensors ``pack-quantized`` format: ``weight_packed``,
    ``weight_scale``, ``weight_shape``.  LoRA merge planning targets plain
    ``.weight`` keys, so we create virtual entries for the planner.

    Args:
        model_state_keys: Original key set from safetensors headers.

    Returns:
        Tuple of ``(augmented_keys, packed_map)`` where:
        - ``augmented_keys``: keys with virtual ``.weight`` entries added
        - ``packed_map``: mapping from virtual ``.weight`` key →
          ``.weight_packed`` key (used during shard processing)
    """
    packed_map: dict[str, str] = {}
    augmented = set(model_state_keys)
    for k in model_state_keys:
        if k.endswith(_PACKED_SUFFIX):
            virtual = k.removesuffix(_PACKED_SUFFIX) + _WEIGHT_SUFFIX
            if virtual not in model_state_keys:
                augmented.add(virtual)
                packed_map[virtual] = k
    return augmented, packed_map


def create_virtual_weight_shapes(
    model_shapes: dict[str, tuple[int, ...]],
    packed_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    """Add virtual ``.weight`` shape entries from packed weight shapes.

    For INT4 ``pack-quantized`` format, the packed tensor shape is
    ``(out_dim, in_dim // 8)`` (8 INT4 values per int32).  We reconstruct
    the original shape as ``(out_dim, in_dim // 8 * 8)`` which equals
    ``(out_dim, in_dim)`` for the correctly-padded case.

    If a ``weight_shape`` tensor is available (stores the original
    dimensions), its shape ``(2,)`` tells us the tensor is a shape
    descriptor, but we can't read its *values* from headers alone.
    So we use the packed shape + pack ratio instead.

    Args:
        model_shapes: Original shapes from safetensors headers.
        packed_map: Virtual-key → packed-key map from
            :func:`create_virtual_weight_keys`.

    Returns:
        Augmented shape dict with virtual entries added.
    """
    augmented = dict(model_shapes)
    for virtual_key, packed_key in packed_map.items():
        packed_shape = model_shapes.get(packed_key)
        if packed_shape is not None and len(packed_shape) == 2:
            # I32 packs 8 INT4 values → multiply packed cols by 8
            augmented[virtual_key] = (packed_shape[0], packed_shape[1] * 8)
    return augmented


# ---------------------------------------------------------------------------
# Name remapping
# ---------------------------------------------------------------------------


def _build_kimi_k25_name_remaps() -> list[tuple[str, str]]:
    """Build name remaps for Kimi K2.5's ``language_model.model.*`` prefix.

    Remap order matters — applied sequentially via ``str.replace``:

    1. Strip Tinker's ``base_model.model.`` prefix.
    2. Apply VL prefix: ``model.`` → ``language_model.model.``
       (so ``model.unembed_tokens`` becomes ``language_model.model.unembed_tokens``
       and ``model.layers.*`` becomes ``language_model.model.layers.*``).
    3. Remap ``language_model.model.unembed_tokens`` → ``language_model.lm_head``
       (lm_head lives directly under ``language_model.``, not ``language_model.model.``).

    Step 2 must come before step 3 because ``str.replace("model.", ...)``
    would match inside ``language_model.`` if we did the unembed remap first.
    """
    return [
        ("base_model.model.", ""),
        ("model.", "language_model.model."),
        ("language_model.model.unembed_tokens", "language_model.lm_head"),
    ]


# ---------------------------------------------------------------------------
# Merge planning
# ---------------------------------------------------------------------------


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan merge ops for Kimi K2.5.

    Uses K2.5-specific name remapping (``language_model.model.*`` prefix)
    and separate per-expert expansion.  The ``model_state_keys`` should
    already include virtual ``.weight`` entries from
    :func:`create_virtual_weight_keys`.

    Args:
        adapter_weights: LoRA weight tensors from the adapter.
        adapter_config: Adapter config with ``lora_alpha`` and ``r``.
        model_state_keys: Weight key names in the base model, augmented
            with virtual ``.weight`` keys for packed expert weights.
        profile: Merge profile (``model_family="kimi_k25"``).

    Returns:
        Mapping from model weight key to list of merge operations.
    """
    scaling = validate_adapter_config(adapter_config, profile)
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    name_remaps = _build_kimi_k25_name_remaps()

    ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = remap_adapter_name(n, name_remaps)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            plan_standard_op(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)
        else:
            plan_expert_ops(target_key, lora_A, lora_B, n, profile, model_state_keys, ops)

    return ops
