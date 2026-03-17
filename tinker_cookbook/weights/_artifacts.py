"""Model artifact I/O utilities for weight export.

Provides utilities for reading safetensors metadata, writing sharded output,
loading adapters, resolving model directories, and copying non-weight files.
Used by both standard export strategies (``_export/_full.py``,
``_export/_shard.py``) and model-specific export modules.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# Custom model code files to copy (for trust_remote_code models).
_MODEL_CODE_PATTERNS = ("*.py",)

_MAX_SHARD_SIZE = 10 * (1024**3)  # 10 GB


# ---------------------------------------------------------------------------
# Reading model metadata without loading weights
# ---------------------------------------------------------------------------


def _raise_no_safetensors(model_dir: Path) -> None:
    """Raise FileNotFoundError with a helpful message for missing safetensors."""
    bin_files = sorted(model_dir.glob("*.bin"))
    if bin_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}. "
            f"Found {len(bin_files)} .bin file(s) — this model may use the older PyTorch format. "
            f"Try merge_strategy='full' which loads via from_pretrained and handles both formats."
        )
    raise FileNotFoundError(
        f"No .safetensors files found in {model_dir}. "
        f"Ensure the model has been fully downloaded, or try merge_strategy='full'."
    )


def get_model_state_keys(model_dir: Path) -> set[str]:
    """Get all weight key names from safetensors files without loading tensor data.

    Uses ``safetensors.safe_open`` to read headers only, which is fast and
    uses negligible memory regardless of model size.

    Args:
        model_dir: Directory containing ``.safetensors`` files.

    Returns:
        Set of all tensor key names across all shard files.

    Raises:
        FileNotFoundError: If no ``.safetensors`` files are found.
    """
    return set(get_model_state_shapes(model_dir).keys())


def get_model_state_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Get shape for each weight key from safetensors headers without loading tensor data.

    Uses ``safetensors.safe_open`` to read headers only. This is fast and uses
    negligible memory regardless of model size. Useful for upfront shape
    validation before loading any weight shards.

    Args:
        model_dir: Directory containing ``.safetensors`` files.

    Returns:
        Dict mapping tensor key names to their shapes.

    Raises:
        FileNotFoundError: If no ``.safetensors`` files are found.
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        _raise_no_safetensors(model_dir)

    shapes: dict[str, tuple[int, ...]] = {}
    for sf_path in shard_files:
        with safe_open(str(sf_path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 — safe_open doesn't support `in`
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def get_shard_files(model_dir: Path) -> list[str]:
    """Get sorted list of safetensors shard filenames in a model directory.

    Prefers reading ``model.safetensors.index.json`` for the canonical shard
    list. Falls back to globbing for ``.safetensors`` files.

    Args:
        model_dir: Directory containing the model shards.

    Returns:
        Sorted list of shard filenames (not full paths).

    Raises:
        FileNotFoundError: If no ``.safetensors`` files are found.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return sorted(set(weight_map.values()))

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        _raise_no_safetensors(model_dir)
    return [f.name for f in shard_files]


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(base_model: str) -> Path:
    """Resolve a HuggingFace model name or local path to a local directory.

    If ``base_model`` is already a local directory, returns it directly.
    Otherwise downloads from HuggingFace Hub via ``snapshot_download``.

    Args:
        base_model: HuggingFace model name (e.g. ``"Qwen/Qwen3-8B"``) or
            local path to a model directory.

    Returns:
        Path to local directory containing model files.
    """
    if os.path.isdir(base_model):
        logger.info("Using local model directory: %s", base_model)
        return Path(base_model)

    from huggingface_hub import snapshot_download

    logger.info("Downloading model files for %s", base_model)
    local_dir = snapshot_download(repo_id=base_model)
    return Path(local_dir)


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------


def load_adapter_weights(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load adapter weights and config from disk.

    Args:
        adapter_dir: Directory containing ``adapter_model.safetensors`` and
            ``adapter_config.json``.

    Returns:
        Tuple of ``(weights_dict, config_dict)``.

    Raises:
        FileNotFoundError: If adapter files are missing.
    """
    adapter_dir = adapter_dir.expanduser().resolve()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Adapter weights not found: {safetensors_path}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    weights = load_file(str(safetensors_path), device=device)
    with open(config_path) as f:
        config = json.load(f)
    return weights, config


# ---------------------------------------------------------------------------
# Non-weight file copying
# ---------------------------------------------------------------------------


def copy_model_code_files(model_dir: Path, output_path: Path) -> None:
    """Copy custom model code files (``*.py``) to the output directory.

    Some model architectures require ``trust_remote_code=True`` and ship
    custom Python files (e.g. ``configuration_*.py``, ``modeling_*.py``).
    This copies those files so the merged model can be loaded standalone.

    Only copies ``*.py`` files. Config and tokenizer files are handled
    separately via HF APIs (``AutoConfig.save_pretrained``, etc.) to
    avoid accidentally copying stale index files or other artifacts that
    could break downstream loaders like vLLM/SGLang.

    Args:
        model_dir: Source model directory.
        output_path: Destination directory (must exist).
    """
    for pattern in _MODEL_CODE_PATTERNS:
        for item in sorted(model_dir.glob(pattern)):
            dest = output_path / item.name
            if not dest.exists():
                shutil.copy2(item, dest)
                logger.debug("Copied %s", item.name)


# ---------------------------------------------------------------------------
# ShardWriter — accumulate and write safetensors shards
# ---------------------------------------------------------------------------


class ShardWriter:
    """Accumulates tensors and writes numbered safetensors shard files.

    Writes to temporary files during processing, then renames to final names
    in :meth:`finalize`. This ensures partial failures don't leave behind
    confusingly-named output files.

    Args:
        output_path: Directory where shard files will be written. Must exist.
        max_shard_size: Maximum size (bytes) per output shard. Default 10 GB.
    """

    def __init__(self, output_path: Path, max_shard_size: int = _MAX_SHARD_SIZE):
        self._output_path = output_path
        self._max_shard_size = max_shard_size
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_size: int = 0
        self._shard_count: int = 0
        self._shard_keys: list[list[str]] = []
        self._total_size: int = 0

    def add_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Add a tensor to the current shard buffer.

        Automatically flushes the buffer when adding this tensor would exceed
        ``max_shard_size``.
        """
        size = tensor.nelement() * tensor.element_size()
        if self._pending and self._pending_size + size > self._max_shard_size:
            self.flush()
        self._pending[key] = tensor
        self._pending_size += size
        self._total_size += size

    def flush(self) -> None:
        """Write buffered tensors to a temporary shard file."""
        if not self._pending:
            return
        # Use next shard number for temp file name, but only commit the
        # count increment after save_file succeeds — avoids inconsistent
        # state if the write fails (e.g. disk full).
        next_idx = self._shard_count + 1
        temp_name = f"shard-{next_idx:05d}.tmp.safetensors"
        save_file(self._pending, str(self._output_path / temp_name))
        # Write succeeded — commit state updates
        self._shard_count = next_idx
        self._shard_keys.append(list(self._pending.keys()))
        logger.debug("Flushed %d tensors to %s", len(self._pending), temp_name)
        self._pending = {}
        self._pending_size = 0

    def finalize(self) -> dict[str, str]:
        """Flush remaining tensors, rename temps to final names, return weight map.

        Returns:
            Dict mapping tensor key to shard filename, suitable for
            ``model.safetensors.index.json``.
        """
        self.flush()
        total = self._shard_count
        weight_map: dict[str, str] = {}

        for i in range(total):
            temp_name = f"shard-{i + 1:05d}.tmp.safetensors"
            if total == 1:
                final_name = "model.safetensors"
            else:
                final_name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            (self._output_path / temp_name).rename(self._output_path / final_name)
            for key in self._shard_keys[i]:
                weight_map[key] = final_name

        logger.info(
            "Wrote %d output shard(s), total %.1f GB",
            total,
            self._total_size / (1024**3),
        )
        return weight_map

    @property
    def total_size(self) -> int:
        """Total bytes of tensors written (including pending)."""
        return self._total_size
