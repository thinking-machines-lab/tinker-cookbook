"""Convert Tinker LoRA adapters to PEFT format for serving.

Produces standard PEFT LoRA adapters compatible with vLLM, SGLang, and
other frameworks that support ``--lora-modules`` / ``--lora-paths``.

The conversion remaps Tinker's internal adapter key names to match the
HuggingFace model's parameter names (which serving frameworks expect),
expands 3D expert LoRA tensors to per-expert 2D keys, and generates a
PEFT-compatible ``adapter_config.json``.

Unlike :func:`~tinker_cookbook.weights.build_hf_model`, this does **not**
merge LoRA weights into the base model — the output is a lightweight
adapter that the serving framework applies at inference time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from tinker_cookbook.exceptions import WeightsAdapterError
from tinker_cookbook.weights._artifacts import (
    get_model_state_keys,
    load_adapter_weights,
    resolve_model_dir,
)
from tinker_cookbook.weights._export import (
    cleanup_on_failure,
    load_config_dict,
    resolve_trust_remote_code,
)
from tinker_cookbook.weights._merge import (
    MergeProfile,
    detect_merge_profile,
    expand_expert_lora_tensors,
)
from tinker_cookbook.weights._merge_qwen3_5 import build_qwen3_5_name_remaps
from tinker_cookbook.weights._merge_utils import (
    build_name_remaps,
    extract_adapter_weight_names,
    remap_adapter_name,
)

logger = logging.getLogger(__name__)

# Model families that do not support LoRA adapter serving in any framework.
_UNSUPPORTED_FAMILIES = {
    "deepseek": (
        "DeepSeek V3/V3.1 does not support LoRA adapter serving in vLLM or SGLang. "
        "Use build_hf_model to merge the adapter into a full HF model instead."
    ),
}

# Model types that are deferred (not yet validated for adapter conversion).
_DEFERRED_MODEL_TYPES = {
    "nemotron_h": (
        "Nemotron-3 adapter conversion is not yet supported (non-standard 'backbone.*' "
        "weight prefix needs investigation). "
        "Use build_hf_model to merge the adapter into a full HF model instead."
    ),
}

# Expert remapping: Tinker internal names → HuggingFace parameter names.
_EXPERT_KEY_REMAPS = (
    (".w1.weight", ".gate_proj.weight"),
    (".w3.weight", ".up_proj.weight"),
    (".w2.weight", ".down_proj.weight"),
)


def build_lora_adapter(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool | None = None,
) -> None:
    """Convert a Tinker LoRA adapter to standard PEFT format for serving.

    The output can be loaded directly by vLLM (``--lora-modules``),
    SGLang (``--lora-paths``), or any framework supporting PEFT LoRA
    adapters. No merging into base model weights is performed.

    Args:
        base_model: HuggingFace model name (e.g. ``"Qwen/Qwen3-8B"``)
            or local path. Needed to resolve model-specific weight naming.
        adapter_path: Local path to the Tinker adapter directory
            (must contain ``adapter_model.safetensors`` and
            ``adapter_config.json``).
        output_path: Directory where the PEFT adapter will be saved.
            Must not already exist.
        trust_remote_code: Whether to trust remote code when loading HF
            model configs. If ``None`` (default), falls back to the
            ``HF_TRUST_REMOTE_CODE`` environment variable, then ``False``.

    Raises:
        FileNotFoundError: If adapter files are missing.
        FileExistsError: If output_path already exists.
        WeightsAdapterError: If the model family does not support LoRA
            adapter serving, or adapter keys cannot be remapped.
    """
    # Resolve trust_remote_code from parameter or HF_TRUST_REMOTE_CODE env var,
    # consistent with build_hf_model and get_tokenizer.
    _trust = resolve_trust_remote_code(trust_remote_code)  # reserved for future HF calls
    out = Path(output_path)
    if out.exists():
        raise FileExistsError(f"Output path already exists: {out}")

    # Load adapter weights and config (lightweight — just the LoRA matrices).
    adapter_weights, adapter_config = load_adapter_weights(Path(adapter_path))
    for key in ("lora_alpha", "r"):
        if key not in adapter_config:
            raise WeightsAdapterError(f"Adapter config missing required key: {key!r}")

    # Resolve model directory (may download from HF Hub), then load config
    # from the local directory.
    model_dir = resolve_model_dir(base_model)
    config_dict = load_config_dict(model_dir)
    model_state_keys = get_model_state_keys(model_dir)

    # Detect model family.
    profile = detect_merge_profile(config_dict, model_state_keys)
    _check_model_support(profile, config_dict)

    # Check if adapter contains expert weights (for MoE warning).
    has_expert_weights = any(".experts" in k for k in adapter_weights)
    if has_expert_weights:
        _warn_experimental_moe(profile)

    logger.info(
        "Converting adapter for %s (family=%s, expert_layout=%s)",
        base_model,
        profile.model_family,
        profile.expert_layout,
    )

    try:
        out.mkdir(parents=True)

        # Core conversion: remap keys, expand experts, produce PEFT tensors.
        peft_weights, target_modules = _convert_adapter(adapter_weights, model_state_keys, profile)

        # Write output.
        peft_config = _build_peft_config(adapter_config, base_model, target_modules)
        _write_peft_adapter(out, peft_weights, peft_config)

        logger.info(
            "PEFT adapter saved to %s (%d weight tensors, target_modules=%s)",
            out,
            len(peft_weights),
            target_modules,
        )
    except Exception:
        cleanup_on_failure(out)
        raise


def _check_model_support(profile: MergeProfile, config_dict: dict) -> None:
    """Check whether the model family supports LoRA adapter serving."""
    # Hard-blocked families.
    if profile.model_family in _UNSUPPORTED_FAMILIES:
        raise WeightsAdapterError(_UNSUPPORTED_FAMILIES[profile.model_family])

    # Deferred model types (detected from config, not profile.model_family).
    model_type = config_dict.get("model_type", "")
    if model_type in _DEFERRED_MODEL_TYPES:
        raise WeightsAdapterError(_DEFERRED_MODEL_TYPES[model_type])


def _warn_experimental_moe(profile: MergeProfile) -> None:
    """Warn if MoE expert LoRA serving support is experimental for this model family.

    Only called when the adapter actually contains expert weights.
    GPT-OSS and Kimi-K2 have stable vLLM MoE LoRA support and are excluded.
    """
    # Families with confirmed stable MoE LoRA support in vLLM.
    stable_moe_families = {"gpt_oss"}

    if profile.model_family not in stable_moe_families:
        logger.warning(
            "MoE expert LoRA serving for %s models is experimental in vLLM and "
            "not yet supported in SGLang. The adapter will be produced but may "
            "not work with all serving configurations.",
            profile.model_family,
        )


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def _convert_adapter(
    adapter_weights: dict[str, torch.Tensor],
    model_state_keys: set[str],
    profile: MergeProfile,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert Tinker adapter weights to PEFT-compatible format.

    Returns:
        Tuple of ``(peft_weights, target_modules)`` where:
        - ``peft_weights``: dict with PEFT key names → raw (unscaled) tensors
        - ``target_modules``: sorted list of short module names for the
          PEFT ``adapter_config.json``
    """
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    if profile.split_qkv_projections:
        name_remaps = build_qwen3_5_name_remaps(profile, model_state_keys)
    else:
        name_remaps = build_name_remaps(profile, model_state_keys)

    peft_weights: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()

    for n in adapter_weight_names:
        lora_A_key = n.replace(".weight", ".lora_A.weight")
        lora_B_key = n.replace(".weight", ".lora_B.weight")
        lora_A = adapter_weights[lora_A_key]
        lora_B = adapter_weights[lora_B_key]

        # Remap Tinker adapter name to HF model parameter name.
        target_key = remap_adapter_name(n, name_remaps)

        # Apply model-specific extra remaps (e.g., .attn → .self_attn for GPT-OSS).
        for old, new in profile.extra_key_remaps:
            target_key = target_key.replace(old, new)

        if ".experts" in n:
            _expand_expert_weights(target_key, lora_A, lora_B, peft_weights, target_modules)
        else:
            _add_peft_weight(target_key, lora_A, lora_B, peft_weights, target_modules)

    if not peft_weights:
        raise WeightsAdapterError(
            "No LoRA weights found in adapter. Check that the adapter path "
            "points to a valid Tinker LoRA adapter."
        )

    return peft_weights, sorted(target_modules)


def _add_peft_weight(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    peft_weights: dict[str, torch.Tensor],
    target_modules: set[str],
) -> None:
    """Add a single LoRA weight pair to the PEFT output."""
    # Strip .weight suffix, wrap in PEFT naming convention.
    module_path = target_key.removesuffix(".weight")
    peft_key_a = f"base_model.model.{module_path}.lora_A.weight"
    peft_key_b = f"base_model.model.{module_path}.lora_B.weight"

    if peft_key_a in peft_weights:
        raise WeightsAdapterError(
            f"Duplicate PEFT key: {peft_key_a!r}. Two adapter weights mapped to "
            f"the same target. This likely indicates a misconfigured adapter or "
            f"an unsupported model architecture."
        )

    peft_weights[peft_key_a] = lora_A
    peft_weights[peft_key_b] = lora_B

    # Extract leaf module name for target_modules.
    leaf = module_path.rsplit(".", 1)[-1]
    target_modules.add(leaf)


def _expand_expert_weights(
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    peft_weights: dict[str, torch.Tensor],
    target_modules: set[str],
) -> None:
    """Expand 3D expert LoRA tensors to per-expert 2D PEFT keys.

    Tinker stores expert LoRA as 3D tensors ``(num_experts, rank, dim)`` with
    a shared key like ``experts.w1``. PEFT format requires separate 2D tensors
    per expert like ``experts.0.gate_proj``.
    """
    # Apply expert key remapping (w1→gate_proj, w3→up_proj, w2→down_proj).
    for old, new in _EXPERT_KEY_REMAPS:
        target_key = target_key.replace(old, new)

    if lora_A.ndim != 3 or lora_B.ndim != 3:
        raise WeightsAdapterError(
            f"Expert LoRA weights must be 3D, got lora_A: {lora_A.shape}, "
            f"lora_B: {lora_B.shape} for key targeting {target_key!r}"
        )

    # Broadcast shared expert tensors (e.g., 1 expert in A, N in B).
    lora_A, lora_B = expand_expert_lora_tensors(lora_A, lora_B)
    num_experts = lora_A.shape[0]

    for exp_idx in range(num_experts):
        exp_key = target_key.replace(".experts", f".experts.{exp_idx}")
        _add_peft_weight(exp_key, lora_A[exp_idx], lora_B[exp_idx], peft_weights, target_modules)


# ---------------------------------------------------------------------------
# PEFT config and output writing
# ---------------------------------------------------------------------------


def _build_peft_config(
    adapter_config: dict,
    base_model: str,
    target_modules: list[str],
) -> dict:
    """Build a PEFT-compatible adapter_config.json dict."""
    return {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": adapter_config["lora_alpha"],
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": adapter_config["r"],
        "rank_pattern": {},
        "alpha_pattern": {},
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }


def _write_peft_adapter(
    output_path: Path,
    peft_weights: dict[str, torch.Tensor],
    peft_config: dict,
) -> None:
    """Write PEFT adapter files to disk."""
    save_file(peft_weights, str(output_path / "adapter_model.safetensors"))
    (output_path / "adapter_config.json").write_text(json.dumps(peft_config, indent=2) + "\n")
