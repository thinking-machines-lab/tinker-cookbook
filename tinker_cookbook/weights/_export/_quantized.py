"""Quantized export strategy.

Merges LoRA adapters shard-by-shard and quantizes routed expert weights to FP8.
Produces output compatible with vLLM's compressed-tensors format.

Currently supports DeepSeek V3/V3.1 models. The infrastructure (FP8 math, vLLM
config generation, resume support) is reusable for future model families.

The :class:`FP8BlockFormat` class implements the
:class:`~._quant_format.QuantizationFormat` protocol.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from tinker_cookbook.exceptions import WeightsMergeError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek detection
# ---------------------------------------------------------------------------

_DEEPSEEK_MODEL_TYPES = frozenset({"deepseek_v3"})


def is_deepseek_config(config_dict: dict) -> bool:
    """Check if config describes a DeepSeek model family."""
    return config_dict.get("model_type") in _DEEPSEEK_MODEL_TYPES


# ---------------------------------------------------------------------------
# FP8 blockwise quantization
# ---------------------------------------------------------------------------

# DeepSeek V3/V3.1 native FP8 block size
_FP8_BLOCK_SIZE = 128


def _get_fp8_max() -> float:
    """Get max representable value in float8_e4m3fn, with fallback for older PyTorch."""
    try:
        return float(torch.finfo(torch.float8_e4m3fn).max)
    except TypeError:
        return 448.0


_FP8_MAX = _get_fp8_max()


def quantize_blockwise(
    tensor: torch.Tensor,
    block_size: tuple[int, int] = (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 using blockwise scaling.

    Divides the tensor into blocks, computes a per-block scale factor, and
    quantizes each block to float8_e4m3fn.

    Args:
        tensor: 2D float tensor to quantize.
        block_size: (row_block, col_block) sizes. Tensor is padded if dimensions
            are not evenly divisible.

    Returns:
        Tuple of (quantized_fp8, scale_inv) where:
        - quantized_fp8: float8_e4m3fn tensor, same shape as input
        - scale_inv: float32 tensor of shape (ceil(rows/row_block), ceil(cols/col_block))
    """
    assert tensor.ndim == 2, f"Expected 2D tensor, got {tensor.ndim}D"
    rows, cols = tensor.shape
    rb, cb = block_size

    # Pad to block boundaries
    pad_rows = (rb - rows % rb) % rb
    pad_cols = (cb - cols % cb) % cb
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.zeros(
            rows + pad_rows, cols + pad_cols, dtype=tensor.dtype, device=tensor.device
        )
        padded[:rows, :cols] = tensor
    else:
        padded = tensor

    # Reshape into blocks
    pr, pc = padded.shape
    blocks = padded.reshape(pr // rb, rb, pc // cb, cb).permute(0, 2, 1, 3)

    # Per-block max for scale computation
    block_max = blocks.abs().reshape(blocks.shape[0], blocks.shape[1], -1).max(dim=-1).values
    # Avoid division by zero
    block_max = block_max.clamp(min=1e-12)

    scale = block_max / _FP8_MAX
    scale_inv = scale  # scale_inv[i,j] = max_val / FP8_MAX

    # Quantize: scale each block, clamp, cast
    inv_scale = 1.0 / scale.unsqueeze(-1).unsqueeze(-1)  # broadcast over block dims
    scaled_blocks = blocks.float() * inv_scale
    clamped = scaled_blocks.clamp(-_FP8_MAX, _FP8_MAX)

    # Reshape back to padded shape
    quantized_padded = clamped.permute(0, 2, 1, 3).reshape(pr, pc)

    # Trim padding
    quantized = quantized_padded[:rows, :cols].to(torch.float8_e4m3fn)

    return quantized, scale_inv.float()


def dequantize_blockwise(
    quantized: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor back to float using blockwise scales.

    Args:
        quantized: float8_e4m3fn tensor.
        scale_inv: float32 scale tensor from :func:`quantize_blockwise`.
        block_size: Must match the block_size used during quantization.
        dtype: Output dtype.

    Returns:
        Dequantized tensor in the requested dtype.
    """
    assert quantized.ndim == 2, f"Expected 2D tensor, got {quantized.ndim}D"
    rows, cols = quantized.shape
    rb, cb = block_size

    # Pad to block boundaries
    pad_rows = (rb - rows % rb) % rb
    pad_cols = (cb - cols % cb) % cb
    if pad_rows > 0 or pad_cols > 0:
        padded = torch.zeros(
            rows + pad_rows, cols + pad_cols, dtype=torch.float32, device=quantized.device
        )
        padded[:rows, :cols] = quantized.float()
    else:
        padded = quantized.float()

    # Reshape into blocks
    pr, pc = padded.shape
    blocks = padded.reshape(pr // rb, rb, pc // cb, cb).permute(0, 2, 1, 3)

    # Multiply by scale
    blocks = blocks * scale_inv.unsqueeze(-1).unsqueeze(-1)

    # Reshape back
    result = blocks.permute(0, 2, 1, 3).reshape(pr, pc)
    return result[:rows, :cols].to(dtype)


# ---------------------------------------------------------------------------
# Weight classification
# ---------------------------------------------------------------------------

# Pattern for routed expert weights in DeepSeek models
# e.g. "model.layers.3.mlp.experts.42.gate_proj.weight"
_ROUTED_EXPERT_PATTERN = ".mlp.experts."
_SHARED_EXPERT_PATTERN = ".mlp.shared_experts."


def _is_routed_expert_weight(key: str) -> bool:
    """Check if a weight key belongs to a routed (non-shared) expert."""
    return _ROUTED_EXPERT_PATTERN in key and _SHARED_EXPERT_PATTERN not in key


# ---------------------------------------------------------------------------
# Keys to skip in DeepSeek checkpoints
# ---------------------------------------------------------------------------

# DeepSeek has some keys that should not be part of the merge:
# - Layer 61 is a placeholder/unused layer in some checkpoints
# - rotary_emb inverse frequency is derived, not a trained parameter
_SKIP_SUFFIXES = (".rotary_emb.inv_freq",)
_SKIP_LAYER_INDICES = frozenset({61})


def _should_skip_checkpoint_key(key: str) -> bool:
    """Check if a checkpoint key should be excluded from merge planning."""
    if any(key.endswith(s) for s in _SKIP_SUFFIXES):
        return True
    # Check for layer 61 (DeepSeek-specific)
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
                if layer_idx in _SKIP_LAYER_INDICES:
                    return True
            except ValueError:
                pass
    return False


# ---------------------------------------------------------------------------
# Native FP8 checkpoint handling
# ---------------------------------------------------------------------------


def _has_native_fp8_quantization(config_dict: dict) -> bool:
    """Check if the model checkpoint uses native FP8 quantization.

    DeepSeek V3.1 checkpoints can ship with native FP8 weights and
    ``quantization_config.quant_method == "fp8"``. These need to be
    dequantized before re-quantizing with our own scales.
    """
    quant_config = config_dict.get("quantization_config")
    if quant_config is None:
        return False
    if isinstance(quant_config, dict):
        return quant_config.get("quant_method", "") == "fp8"
    return False


def _get_native_block_size(config_dict: dict) -> tuple[int, int]:
    """Get the FP8 block size from the model's native quantization config.

    Falls back to the standard DeepSeek block size (128, 128) if not specified.
    """
    quant_config = config_dict.get("quantization_config", {})
    if isinstance(quant_config, dict):
        block_size = quant_config.get("weight_block_size", [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE])
        return (int(block_size[0]), int(block_size[1]))
    return (_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE)


def _make_cross_shard_tensor_loader(
    model_dir: Path,
) -> Callable[[str], torch.Tensor]:
    """Create a loader that can fetch a single tensor from any shard by key name.

    Used when a weight tensor and its scale are in different shards. Reads
    the safetensors index to find which shard contains a given key, then
    uses ``safe_open`` to load only that one tensor — no full shard loading.

    This keeps peak memory at O(single tensor) rather than O(full shard),
    which matters for DeepSeek V3 where shards are ~4-5 GB each.
    """
    # Build key → shard mapping from index
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index_weight_map: dict[str, str] = json.load(f)["weight_map"]
    else:
        # Single shard — build map from the one file
        shard_files = sorted(model_dir.glob("*.safetensors"))
        index_weight_map = {}
        for sf_path in shard_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():  # noqa: SIM118
                    index_weight_map[key] = sf_path.name

    def load_tensor(name: str) -> torch.Tensor:
        if name not in index_weight_map:
            raise KeyError(f"Tensor {name!r} not found in any shard at {model_dir}")
        shard_name = index_weight_map[name]
        with safe_open(str(model_dir / shard_name), framework="pt") as f:
            return f.get_tensor(name)

    return load_tensor


# ---------------------------------------------------------------------------
# vLLM compressed-tensors config
# ---------------------------------------------------------------------------


def _weight_scale_key(weight_key: str) -> str:
    """Map a weight key to its compressed-tensors scale key.

    Uses ``.weight_scale`` (compressed-tensors convention), NOT
    ``.weight_scale_inv`` (DeepSeek native convention).

    Handles both per-expert keys (``experts.0.down_proj.weight`` →
    ``experts.0.down_proj.weight_scale``) and fused 3D keys
    (``experts.down_proj`` → ``experts.down_proj.weight_scale``).
    """
    if weight_key.endswith(".weight"):
        return weight_key.removesuffix(".weight") + ".weight_scale"
    return weight_key + ".weight_scale"


def _build_vllm_quantization_config(output_weight_map: dict[str, str]) -> dict:
    """Build compressed-tensors quantization config for vLLM.

    Produces a config dict that tells vLLM which layers are FP8-quantized
    (routed experts) and which to ignore (everything else). No library
    imports needed — the schema is fixed and well-known.

    The decision is binary for every ``.weight`` key in the output:

    * **Has a matching ``.weight_scale``** → quantized → omitted from
      ``ignore`` so vLLM loads it as FP8.
    * **No matching ``.weight_scale``** → not quantized → added to
      ``ignore`` so vLLM leaves it alone.

    Adding non-``nn.Linear`` modules (norms, embeddings) to ``ignore`` is
    harmless — vLLM only consults the ignore list for modules that match
    ``targets: ["Linear"]``, so extra entries are silently skipped.

    Args:
        output_weight_map: Mapping of weight key -> shard filename.

    Returns:
        Dict suitable for config.json's ``compression_config`` field.
    """
    # Determine which modules have been quantized (have .weight_scale)
    quantized_prefixes = {
        key.removesuffix(".weight_scale")
        for key in output_weight_map
        if key.endswith(".weight_scale")
    }

    # Build ignore list: every .weight module that was NOT quantized.
    # This is model-agnostic — no need to enumerate projection suffixes
    # per architecture.
    ignore: list[str] = []
    for key in sorted(output_weight_map):
        if not key.endswith(".weight"):
            continue
        prefix = key.removesuffix(".weight")
        if prefix not in quantized_prefixes:
            ignore.append(prefix)

    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "block",
                    "block_structure": [_FP8_BLOCK_SIZE, _FP8_BLOCK_SIZE],
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor",
                    "dynamic": True,
                },
            },
        },
        "ignore": ignore,
    }


_VLLM_COMPAT_QUANT_CONFIG_FIELDS = {
    "config_groups",
    "format",
    "global_compression_ratio",
    "ignore",
    "kv_cache_scheme",
    "quantization_status",
}
_VLLM_COMPAT_QUANT_SCHEME_FIELDS = {
    "format",
    "input_activations",
    "output_activations",
    "targets",
    "weights",
}
_VLLM_COMPAT_QUANT_ARGS_FIELDS = {
    "actorder",
    "block_structure",
    "dynamic",
    "group_size",
    "num_bits",
    "observer",
    "observer_kwargs",
    "strategy",
    "symmetric",
    "type",
}


def _serialize_for_vllm(config: dict) -> dict:
    """Serialize only the compressed-tensors fields the current vLLM path needs.

    Uses an allowlist so new compressed-tensors fields are omitted automatically
    instead of breaking older vLLM builds.
    """
    serialized: dict = {}
    for key, value in config.items():
        if key == "config_groups" and isinstance(value, dict):
            serialized[key] = {
                group_name: _serialize_vllm_scheme(group)
                for group_name, group in value.items()
                if isinstance(group, dict)
            }
            continue
        if key in _VLLM_COMPAT_QUANT_CONFIG_FIELDS:
            serialized[key] = value
    serialized["quant_method"] = "compressed-tensors"
    return serialized


def _serialize_vllm_scheme(group: dict) -> dict:
    """Serialize a single quantization scheme for vLLM compatibility."""
    serialized: dict = {}
    for key, value in group.items():
        if key in {"weights", "input_activations", "output_activations"} and isinstance(
            value, dict
        ):
            serialized[key] = {
                field: field_value
                for field, field_value in value.items()
                if field in _VLLM_COMPAT_QUANT_ARGS_FIELDS
            }
            continue
        if key in _VLLM_COMPAT_QUANT_SCHEME_FIELDS:
            serialized[key] = value
    return serialized


# ---------------------------------------------------------------------------
# FP8BlockFormat — QuantizationFormat implementation
# ---------------------------------------------------------------------------


class FP8BlockFormat:
    """FP8 blockwise quantization for DeepSeek V3-family models.

    Implements the :class:`~._quant_format.QuantizationFormat` protocol.
    Quantizes routed expert weights to FP8 with blockwise scaling after
    LoRA merge, preserving dense/shared-expert weights in BF16.

    For models with native FP8 checkpoints (e.g. DeepSeek V3.1), handles
    dequantization before merge so LoRA math works in float precision.

    Args:
        config_dict: Parsed config.json dict.
        model_dir: Resolved local model directory.
        serving_format: Serving framework format (e.g. ``"vllm"``).
        device: Device for quantization math (``"cpu"``, ``"cuda"``, etc.).
    """

    def __init__(
        self,
        config_dict: dict,
        model_dir: Path,
        serving_format: str,
        device: str = "cpu",
    ) -> None:
        self._serving_format = serving_format
        self._device = torch.device(device)

        # Native FP8 handling
        self._is_native_fp8 = _has_native_fp8_quantization(config_dict)
        self._native_block_size = (
            _get_native_block_size(config_dict) if self._is_native_fp8 else None
        )
        self._cross_shard_loader = (
            _make_cross_shard_tensor_loader(model_dir) if self._is_native_fp8 else None
        )

        if self._is_native_fp8:
            logger.info(
                "Native FP8 checkpoint detected (block_size=%s), will dequantize before re-quantize",
                self._native_block_size,
            )

    def should_skip_output_key(self, key: str) -> bool:
        """Skip checkpoint keys and native scale tensors in output."""
        return _should_skip_checkpoint_key(key) or key.endswith(".weight_scale_inv")

    def filter_model_keys(self, keys: set[str]) -> set[str]:
        """Exclude keys that shouldn't be merge targets.

        Delegates to :meth:`should_skip_output_key` to keep skip logic
        in one place.
        """
        return {k for k in keys if not self.should_skip_output_key(k)}

    def pre_merge_transform(
        self,
        key: str,
        tensor: torch.Tensor,
        shard_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Dequantize native FP8 weights before merge.

        Native FP8 checkpoints store weights in float8_e4m3fn + scale_inv.
        We dequantize to BF16 so LoRA merge math works in float precision.
        """
        if not (
            key.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn and self._is_native_fp8
        ):
            return tensor

        scale_key = key.replace(".weight", ".weight_scale_inv")
        scale_inv = shard_tensors.get(scale_key)
        if scale_inv is None and self._cross_shard_loader is not None:
            scale_inv = self._cross_shard_loader(scale_key)
        if scale_inv is None:
            raise WeightsMergeError(
                f"Native FP8 weight {key!r} has no .weight_scale_inv tensor "
                f"in any shard. Cannot dequantize for merge."
            )
        assert self._native_block_size is not None
        return dequantize_blockwise(tensor, scale_inv, block_size=self._native_block_size)

    def post_merge_transform(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Quantize routed expert weights to FP8, pass through everything else."""
        if not _is_routed_expert_weight(key):
            return {key: tensor}

        if tensor.ndim == 2:
            return self._quantize_expert(key, tensor)
        elif tensor.ndim == 3:
            return self._quantize_fused_experts(key, tensor)
        return {key: tensor}

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._device.type != "cpu":
            return tensor.to(self._device)
        return tensor

    def _to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._device.type != "cpu":
            return tensor.cpu()
        return tensor

    def _quantize_expert(self, key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize a 2D expert weight tensor to FP8."""
        fp8_tensor, scale = quantize_blockwise(self._to_device(tensor))
        return {key: self._to_cpu(fp8_tensor), _weight_scale_key(key): self._to_cpu(scale)}

    def _quantize_fused_experts(self, key: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize a fused 3D expert tensor [num_experts, out, in] to FP8.

        Moves the entire tensor to the compute device once, quantizes each
        expert slice independently, then moves results back.
        """
        tensor = self._to_device(tensor)

        fp8_slices = []
        scale_slices = []
        for i in range(tensor.shape[0]):
            fp8_slice, scale_slice = quantize_blockwise(tensor[i])
            fp8_slices.append(fp8_slice)
            scale_slices.append(scale_slice)

        return {
            key: self._to_cpu(torch.stack(fp8_slices)),
            _weight_scale_key(key): self._to_cpu(torch.stack(scale_slices)),
        }

    def finalize_config(
        self,
        config_dict: dict,
        weight_map: dict[str, str],
    ) -> dict:
        """Patch config with compressed-tensors metadata for vLLM."""
        patched = dict(config_dict)
        if self._serving_format == "vllm":
            quant_config = _build_vllm_quantization_config(weight_map)
            patched["compression_config"] = _serialize_for_vllm(quant_config)
            patched.pop("quantization_config", None)
        return patched


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_quantized(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool,
    model_dir: Path,
    config_dict: dict,
    serving_format: str,
    device: str = "cpu",
) -> None:
    """Merge LoRA adapter and quantize routed experts to FP8.

    Delegates to :func:`~._shard_engine.run_shard_merge` with an
    :class:`FP8BlockFormat` instance for quantization.

    Args:
        base_model: Model name or path (for tokenizer loading).
        adapter_path: Path to adapter directory.
        output_path: Where to write the quantized model.
        trust_remote_code: Whether to trust remote code.
        model_dir: Resolved local model directory.
        config_dict: Parsed config.json dict.
        serving_format: Serving framework format (e.g. "vllm").
        device: Device for quantization math ("cpu", "cuda", etc.).
    """
    from tinker_cookbook.weights._export._shard_engine import run_shard_merge

    quant_format = FP8BlockFormat(config_dict, model_dir, serving_format, device)

    run_shard_merge(
        base_model=base_model,
        adapter_path=adapter_path,
        output_path=output_path,
        trust_remote_code=trust_remote_code,
        model_dir=model_dir,
        config_dict=config_dict,
        quant_format=quant_format,
        resume=True,
        preserve_shard_names=True,
    )
