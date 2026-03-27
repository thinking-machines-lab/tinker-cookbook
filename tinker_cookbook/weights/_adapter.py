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
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

from tinker_cookbook.exceptions import WeightsAdapterError
from tinker_cookbook.weights._artifacts import (
    get_model_state_keys,
    get_model_state_shapes,
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

# Model types (from config.json "model_type") that do not support adapter conversion.
# Keyed by the exact model_type string so that future models in the same family
# (e.g. a hypothetical DeepSeek V4 with LoRA support) are not accidentally blocked.
_UNSUPPORTED_MODEL_TYPES: dict[str, str] = {
    "deepseek_v3": (
        "DeepSeek V3/V3.1 (model_type='deepseek_v3') does not support LoRA adapter "
        "serving in vLLM or SGLang. "
        "Use build_hf_model to merge the adapter into a full HF model instead."
    ),
}

# Serving frameworks (vLLM, SGLang) may rename weight prefixes when loading
# HF checkpoints. For example, vLLM's NemotronH WeightsMapper converts
# "backbone.*" → "model.*". PEFT adapter keys must match the serving
# framework's internal parameter names, not the original HF checkpoint names.
#
# Map from model_type → list of (old, new) prefix replacements to apply to
# PEFT output keys.
_SERVING_PREFIX_REMAPS: dict[str, tuple[tuple[str, str], ...]] = {
    "nemotron_h": (("backbone.", "model."),),
}



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
    _check_model_support(config_dict)

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

        # Load model weight shapes if needed for fused projection merging.
        model_state_shapes: dict[str, tuple[int, ...]] | None = None
        if profile.fused_projection_map:
            model_state_shapes = get_model_state_shapes(model_dir)

        # Core conversion: remap keys, expand experts, produce PEFT tensors.
        peft_weights, target_modules, rank_overrides = _convert_adapter(
            adapter_weights, model_state_keys, profile, model_state_shapes,
        )

        # Apply serving-framework prefix remaps (e.g., backbone.* → model.* for Nemotron).
        model_type = config_dict.get("model_type", "")
        if model_type in _SERVING_PREFIX_REMAPS:
            peft_weights = _apply_serving_prefix_remaps(peft_weights, model_type)

        # Write output.
        peft_config = _build_peft_config(
            adapter_config, base_model, target_modules, rank_overrides,
        )
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


def _check_model_support(config_dict: dict) -> None:
    """Check whether the model supports LoRA adapter conversion.

    Uses the exact ``model_type`` from config.json rather than the broad
    ``profile.model_family`` so that future models in the same family
    are not accidentally blocked.
    """
    model_type = config_dict.get("model_type", "")
    if model_type in _UNSUPPORTED_MODEL_TYPES:
        raise WeightsAdapterError(_UNSUPPORTED_MODEL_TYPES[model_type])


def _apply_serving_prefix_remaps(
    peft_weights: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    """Remap PEFT key prefixes so they match the serving framework's internal names.

    Some models use non-standard prefixes in their HF checkpoints (e.g.,
    Nemotron uses ``backbone.*`` instead of ``model.*``). Serving frameworks
    like vLLM remap these at load time, so PEFT adapter keys must match the
    remapped names, not the original HF checkpoint names.
    """
    remaps = _SERVING_PREFIX_REMAPS[model_type]
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in peft_weights.items():
        new_key = key
        for old, new in remaps:
            new_key = new_key.replace(old, new)
        remapped[new_key] = tensor
    return remapped


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
    model_state_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str], dict[str, int]]:
    """Convert Tinker adapter weights to PEFT-compatible format.

    Returns:
        Tuple of ``(peft_weights, target_modules, rank_overrides)`` where:
        - ``peft_weights``: dict with PEFT key names → raw (unscaled) tensors
        - ``target_modules``: sorted list of short module names for the
          PEFT ``adapter_config.json``
        - ``rank_overrides``: mapping from module name → rank for modules
          whose LoRA rank differs from the adapter's base rank (e.g. fused
          Mamba projections that double the rank)
    """
    adapter_weight_names = extract_adapter_weight_names(adapter_weights)

    if profile.split_qkv_projections:
        name_remaps = build_qwen3_5_name_remaps(profile, model_state_keys)
    else:
        name_remaps = build_name_remaps(profile, model_state_keys)

    # Build lookup for fused projection merging: component_name → fused_target_name.
    fused_component_to_target: dict[str, str] = {}
    for fused_target, components in profile.fused_projection_map:
        for comp in components:
            fused_component_to_target[comp] = fused_target

    peft_weights: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    # Collect component LoRA weights for fused projection merging.
    # Key: (layer_prefix, fused_target) → list of (component_name, lora_A, lora_B)
    fused_pending: dict[tuple[str, str], list[tuple[str, torch.Tensor, torch.Tensor]]] = {}

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
            # Skip empty expert LoRA tensors — these are placeholders for
            # projections that don't exist in the model (e.g. Nemotron MoE
            # has no gate_proj, so w3 entries are empty).
            if lora_A.numel() == 0 and lora_B.numel() == 0:
                continue
            _expand_expert_weights(
                target_key, lora_A, lora_B, peft_weights, target_modules,
                expert_key_remaps=profile.expert_key_remaps,
            )
        else:
            # Check if this is a component of a fused projection.
            leaf_module = target_key.removesuffix(".weight").rsplit(".", 1)[-1]
            if leaf_module in fused_component_to_target:
                fused_target = fused_component_to_target[leaf_module]
                layer_prefix = target_key.removesuffix(f".{leaf_module}.weight")
                key = (layer_prefix, fused_target)
                fused_pending.setdefault(key, []).append((leaf_module, lora_A, lora_B))
            else:
                _add_peft_weight(target_key, lora_A, lora_B, peft_weights, target_modules)

    # Merge collected fused projection components.
    rank_overrides: dict[str, int] = {}
    if fused_pending:
        assert model_state_shapes is not None, (
            "model_state_shapes required for fused projection merging"
        )
        for (layer_prefix, fused_target), components in fused_pending.items():
            # The layer_prefix comes from the remapped adapter key namespace
            # (e.g. "model.layers.0.mixer") which may differ from the model
            # state dict namespace (e.g. "backbone.layers.0.mixer" for
            # Nemotron).  Find the matching fused target in model_state_keys.
            fused_suffix = f".{fused_target}.weight"
            fused_model_key = _find_model_key(
                layer_prefix, fused_suffix, model_state_keys,
            )
            fused_rank = _merge_fused_projections(
                fused_model_key, layer_prefix, components, model_state_shapes,
                peft_weights, target_modules, profile,
            )
            rank_overrides[fused_target] = fused_rank

    if not peft_weights:
        raise WeightsAdapterError(
            "No LoRA weights found in adapter. Check that the adapter path "
            "points to a valid Tinker LoRA adapter."
        )

    return peft_weights, sorted(target_modules), rank_overrides


def _find_model_key(
    layer_prefix: str,
    suffix: str,
    model_state_keys: set[str],
) -> str:
    """Find a model state key matching a layer prefix and suffix.

    The adapter's remapped layer prefix (e.g. ``model.layers.0.mixer``)
    may differ from the model's prefix (e.g. ``backbone.layers.0.mixer``).
    This function finds the matching key by extracting the layer index
    and matching on the suffix.
    """
    layer_match = re.search(r"\.layers\.(\d+)\.", layer_prefix)
    if layer_match:
        layer_idx = layer_match.group(1)
        pattern = f".layers.{layer_idx}."
        for k in model_state_keys:
            if pattern in k and k.endswith(suffix):
                return k

    raise WeightsAdapterError(
        f"Cannot find model state key matching layer prefix {layer_prefix!r} "
        f"with suffix {suffix!r}."
    )


def _merge_fused_projections(
    fused_model_key: str,
    adapter_layer_prefix: str,
    components: list[tuple[str, torch.Tensor, torch.Tensor]],
    model_state_shapes: dict[str, tuple[int, ...]],
    peft_weights: dict[str, torch.Tensor],
    target_modules: set[str],
    profile: MergeProfile,
) -> int:
    """Merge component LoRA weights into a fused projection target.

    When Tinker trains separate LoRA for projections that the HF model
    fuses into one module (e.g. Nemotron Mamba ``gate_proj``/``x_proj`` →
    ``in_proj``), this function combines them into a single LoRA with
    doubled rank.

    The component projections must correspond to consecutive row slices
    in the fused target, in the order specified by
    ``profile.fused_projection_map``.

    Args:
        fused_model_key: The model state dict key for the fused target
            (e.g. ``backbone.layers.0.mixer.in_proj.weight``).
        adapter_layer_prefix: The adapter-namespace layer prefix
            (e.g. ``model.layers.0.mixer``), used to construct PEFT keys.

    Returns the merged LoRA rank (sum of component ranks).
    """
    fused_out_dim = model_state_shapes[fused_model_key][0]

    # Look up the expected component order from the profile.
    fused_target_name = fused_model_key.removesuffix(".weight").rsplit(".", 1)[-1]
    component_order: tuple[str, ...] | None = None
    for target, comps in profile.fused_projection_map:
        if target == fused_target_name:
            component_order = comps
            break
    assert component_order is not None

    # Sort components to match the expected order.
    comp_by_name = {name: (lora_A, lora_B) for name, lora_A, lora_B in components}

    # Build merged lora_A by concatenating along rank dimension.
    # Build merged lora_B by placing each component's lora_B in the
    # correct row slice of the fused output, with zeros elsewhere.
    lora_A_parts: list[torch.Tensor] = []
    merged_rank = 0
    comp_slices: list[tuple[int, int, int]] = []  # (row_start, row_end, rank)
    row_offset = 0

    for comp_name in component_order:
        if comp_name not in comp_by_name:
            raise WeightsAdapterError(
                f"Missing component {comp_name!r} for fused target {fused_model_key!r}. "
                f"Expected components: {component_order}"
            )
        lora_A, lora_B = comp_by_name[comp_name]
        r = lora_A.shape[0]
        out_dim = lora_B.shape[0]
        lora_A_parts.append(lora_A)
        comp_slices.append((row_offset, row_offset + out_dim, r))
        row_offset += out_dim
        merged_rank += r

    # lora_A: (merged_rank, hidden_dim) — concatenation of all component A matrices
    merged_lora_A = torch.cat(lora_A_parts, dim=0)

    # lora_B: (fused_out_dim, merged_rank) — block-diagonal placement
    merged_lora_B = torch.zeros(fused_out_dim, merged_rank, dtype=lora_A_parts[0].dtype)
    rank_offset = 0
    for i, (row_start, row_end, r) in enumerate(comp_slices):
        _, lora_B = comp_by_name[component_order[i]]
        merged_lora_B[row_start:row_end, rank_offset:rank_offset + r] = lora_B
        rank_offset += r

    # Use the adapter-namespace prefix for the PEFT output key (the serving
    # prefix remap will be applied later by the caller if needed).
    peft_target_key = f"{adapter_layer_prefix}.{fused_target_name}.weight"
    _add_peft_weight(peft_target_key, merged_lora_A, merged_lora_B, peft_weights, target_modules)
    return merged_rank


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
    *,
    expert_key_remaps: tuple[tuple[str, str], ...],
) -> None:
    """Expand 3D expert LoRA tensors to per-expert 2D PEFT keys.

    Tinker stores expert LoRA as 3D tensors ``(num_experts, rank, dim)`` with
    a shared key like ``experts.w1``. PEFT format requires separate 2D tensors
    per expert like ``experts.0.gate_proj``.
    """
    # Apply expert key remapping (e.g. w1→gate_proj for standard MoE,
    # w1→up_proj for Nemotron).
    for old, new in expert_key_remaps:
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
        # Clone per-expert slices so they don't share storage after broadcast
        # expansion — safetensors requires each tensor to have its own memory.
        _add_peft_weight(
            exp_key, lora_A[exp_idx].clone(), lora_B[exp_idx].clone(),
            peft_weights, target_modules,
        )


# ---------------------------------------------------------------------------
# PEFT config and output writing
# ---------------------------------------------------------------------------


def _build_peft_config(
    adapter_config: dict,
    base_model: str,
    target_modules: list[str],
    rank_overrides: dict[str, int] | None = None,
) -> dict:
    """Build a PEFT-compatible adapter_config.json dict."""
    base_rank = adapter_config["r"]
    base_alpha = adapter_config["lora_alpha"]

    # For fused projections with doubled rank, set rank_pattern and
    # alpha_pattern so the scaling factor (alpha/rank) stays correct.
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    if rank_overrides:
        for module_name, merged_rank in rank_overrides.items():
            rank_pattern[module_name] = merged_rank
            # Scale alpha proportionally to keep alpha/rank unchanged:
            # merged_alpha / merged_rank == base_alpha / base_rank
            alpha_pattern[module_name] = int(base_alpha * merged_rank / base_rank)

    return {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": base_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": base_rank,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
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
