"""Nemotron-3 merge profile detection.

Nemotron-3 is a hybrid Mamba+Attention+MoE architecture where MoE layers have
only two expert projections (up_proj and down_proj) — no gate_proj. Tinker maps
these as w1=up_proj, w2=down_proj with w3 as an empty placeholder.

This module provides the profile detector; the default planning function is
reused since the only merge-level difference is the expert key remapping.
"""

from __future__ import annotations

from tinker_cookbook.weights._merge import MergeProfile

# Nemotron's MoE experts have only up_proj and down_proj (no gate_proj).
# Tinker stores w1=up_proj, w2=down_proj, w3=empty.
_NEMOTRON_EXPERT_KEY_REMAPS = (
    (".w1.weight", ".up_proj.weight"),
    (".w2.weight", ".down_proj.weight"),
)

# Nemotron Mamba layers train separate gate_proj/x_proj LoRA, but the HF
# checkpoint and vLLM fuse both into a single in_proj.  During adapter
# conversion, these are merged into the fused in_proj target.
_NEMOTRON_FUSED_PROJECTION_MAP = (("in_proj", ("gate_proj", "x_proj")),)

# Nemotron HF checkpoints use "backbone.layers.*" for layer weights but
# "lm_head" (no prefix) for the output head.  Tinker adapters use
# "model.*" for everything (after stripping "base_model.model.").
# These remaps bridge the namespace so merge ops target the correct keys.
_NEMOTRON_EXTRA_KEY_REMAPS = (
    ("model.layers.", "backbone.layers."),
    ("model.lm_head", "lm_head"),
)


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
    """Detect Nemotron-3 models by architecture name."""
    archs = model_config.get("architectures", [])
    if not any("NemotronH" in a for a in archs):
        return None

    has_lm_prefix = any(k.startswith("model.language_model.") for k in model_state_keys)

    return MergeProfile(
        model_family="nemotron",
        expert_layout="separate",
        extra_key_remaps=_NEMOTRON_EXTRA_KEY_REMAPS,
        expert_key_remaps=_NEMOTRON_EXPERT_KEY_REMAPS,
        fused_projection_map=_NEMOTRON_FUSED_PROJECTION_MAP,
        has_language_model_prefix=has_lm_prefix,
    )
