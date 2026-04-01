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
        return None

    return MergeProfile(
        model_family="kimi_k25",
        expert_layout="separate",
        has_language_model_prefix=False,
    )


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
