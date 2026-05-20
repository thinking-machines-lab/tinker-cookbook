"""Drop-in replacements for :func:`datasets.load_dataset` and
:func:`datasets.load_from_disk` with transparent cloud storage support.

Cloud URIs (``s3://``, ``gs://``, ``az://``) are automatically detected and
routed through the appropriate HuggingFace + fsspec code paths.  For local
paths and HuggingFace Hub names the functions delegate directly to the
upstream ``datasets`` library — existing behaviour is unchanged.

Usage::

    # Just change the import — everything else stays the same.
    from tinker_cookbook.utils.dataset_loading import load_dataset, load_from_disk

    # HuggingFace Hub (unchanged behaviour)
    ds = load_dataset("openai/gsm8k", name="main", split="test")

    # Cloud data files
    ds = load_dataset("s3://my-bucket/data/train.parquet", split="train")
    ds = load_dataset("gs://my-bucket/data/*.jsonl", split="train")

    # Cloud Arrow dataset (saved with dataset.save_to_disk)
    ds = load_from_disk("s3://my-bucket/datasets/my_dataset")
"""

from __future__ import annotations

import hashlib
import logging
import os
import posixpath
from typing import Any

import datasets as _hf_datasets

logger = logging.getLogger(__name__)

__all__ = ["is_cloud_uri", "load_dataset", "load_from_disk"]

_CLOUD_SCHEMES = ("s3://", "gs://", "gcs://", "az://", "abfs://", "abfss://")

_EXT_TO_FORMAT: dict[str, str] = {
    ".parquet": "parquet",
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".arrow": "arrow",
    ".txt": "text",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_cloud_uri(path: str) -> bool:
    """Return ``True`` if *path* looks like a cloud storage URI."""
    return any(path.startswith(scheme) for scheme in _CLOUD_SCHEMES)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def load_dataset(
    path: str,
    *,
    name: str | None = None,
    split: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> (
    _hf_datasets.Dataset
    | _hf_datasets.DatasetDict
    | _hf_datasets.IterableDataset
    | _hf_datasets.IterableDatasetDict
):
    """Load a dataset from HuggingFace Hub, a local path, or cloud storage.

    This is a drop-in replacement for :func:`datasets.load_dataset`.  When
    *path* is a cloud URI the function infers the file format from the
    extension and passes the URI as ``data_files``.  When the URI has no
    recognisable file extension it is treated as a saved Arrow dataset and
    loaded via :func:`load_from_disk`.

    All keyword arguments not consumed here are forwarded to the underlying
    ``datasets`` call, so ``streaming``, ``trust_remote_code``, etc. work as
    usual.

    Args:
        path: HuggingFace Hub dataset name (e.g. ``"openai/gsm8k"``),
            local path, or cloud URI (``s3://``, ``gs://``, ``az://``).
        name: Dataset configuration name (HuggingFace Hub only).
        split: Which split to load.
        storage_options: Extra keyword arguments forwarded to the fsspec
            filesystem (e.g. ``{"anon": True}`` for public S3 buckets).
        **kwargs: Forwarded to :func:`datasets.load_dataset`.

    Returns:
        The loaded dataset.

    Raises:
        ImportError: If a cloud URI is used but ``fsspec`` (and the
            appropriate backend) is not installed.
        ValueError: If the cloud URI has an unrecognised file extension.
    """
    if not is_cloud_uri(path):
        # Standard HuggingFace Hub or local-path load.
        kw: dict[str, Any] = {**kwargs}
        if name is not None:
            kw["name"] = name
        if split is not None:
            kw["split"] = split
        return _hf_datasets.load_dataset(path, **kw)

    # --- Cloud URI ---
    _check_cloud_deps(path)

    fmt = _infer_format(path)

    if fmt is None:
        # No file extension → treat as a saved Arrow dataset directory.
        logger.info("Cloud URI without file extension — loading as Arrow dataset: %s", path)
        return load_from_disk(path, storage_options=storage_options)

    logger.info("Loading cloud dataset (%s format): %s", fmt, path)
    kw = {**kwargs}
    if split is not None:
        kw["split"] = split
    if storage_options is not None:
        kw["storage_options"] = storage_options
    return _hf_datasets.load_dataset(fmt, data_files=path, **kw)


# ---------------------------------------------------------------------------
# load_from_disk
# ---------------------------------------------------------------------------


def load_from_disk(
    path: str,
    *,
    cache_dir: str | None = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> _hf_datasets.Dataset | _hf_datasets.DatasetDict:
    """Load a saved Arrow dataset from a local path or cloud storage.

    This is a drop-in replacement for :func:`datasets.load_from_disk`.  For
    cloud URIs the dataset is first downloaded to a local cache directory so
    that Arrow memory-mapping works at local speed.

    Args:
        path: Local path or cloud URI to a dataset saved with
            :meth:`Dataset.save_to_disk`.
        cache_dir: Where to cache cloud downloads.  Defaults to
            ``$HF_DATASETS_CACHE/cloud_cache``.
        storage_options: Extra keyword arguments forwarded to the fsspec
            filesystem.
        **kwargs: Forwarded to :func:`datasets.load_from_disk`.

    Returns:
        The loaded dataset.
    """
    if not is_cloud_uri(path):
        return _hf_datasets.load_from_disk(path, **kwargs)

    _check_cloud_deps(path)

    local_path = _cached_cloud_download(path, cache_dir=cache_dir, storage_options=storage_options)
    logger.info("Loading cloud Arrow dataset from local cache: %s → %s", path, local_path)
    return _hf_datasets.load_from_disk(local_path, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_format(uri: str) -> str | None:
    """Infer the HuggingFace dataset format string from a URI's extension.

    Returns ``None`` when no recognisable extension is found (the caller
    should treat the URI as a saved Arrow dataset directory).
    """
    # Strip trailing slashes — directory-like URIs have no extension.
    clean = uri.rstrip("/")

    # Handle glob patterns: ``s3://bucket/data/*.parquet``
    # Expand the glob into a concrete filename for extension detection.
    if "*" in clean:
        # Replace glob characters so splitext can find the extension.
        # e.g. "s3://bucket/data/*.parquet" → "s3://bucket/data/X.parquet"
        clean = clean.replace("*", "X")

    ext = posixpath.splitext(clean)[1].lower()
    if not ext:
        return None

    fmt = _EXT_TO_FORMAT.get(ext)
    if fmt is not None:
        return fmt

    supported = ", ".join(sorted(_EXT_TO_FORMAT.keys()))
    raise ValueError(
        f"Cannot infer dataset format from cloud URI extension '{ext}' "
        f"(URI: {uri}). Supported extensions: {supported}. "
        f"If this is a saved Arrow dataset, remove the trailing filename "
        f"and point to the directory instead."
    )


def _check_cloud_deps(uri: str) -> None:
    """Raise a helpful ``ImportError`` if fsspec is missing."""
    try:
        import fsspec  # noqa: F401
    except ImportError:
        raise ImportError(
            f"fsspec is required for cloud storage URIs ({uri}). "
            "Install it with: uv pip install 'tinker-cookbook[cloud]'"
        ) from None


def _cached_cloud_download(
    uri: str,
    *,
    cache_dir: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> str:
    """Download a cloud Arrow dataset to a local cache and return the path.

    The local cache key is a SHA-256 hash of the URI so repeated loads of
    the same URI reuse the cached copy.  A marker file records the source
    URI for debugging.
    """
    import shutil

    import fsspec

    if cache_dir is None:
        from datasets import config as _hf_config

        hf_cache = str(_hf_config.HF_DATASETS_CACHE)
        cache_dir = os.path.join(hf_cache, "cloud_cache")

    cache_key = hashlib.sha256(uri.encode()).hexdigest()[:16]
    local_dir = os.path.join(cache_dir, cache_key)

    # If already cached, reuse.
    marker = os.path.join(local_dir, ".cloud_source")
    if os.path.isfile(marker):
        logger.debug("Using cached cloud dataset: %s → %s", uri, local_dir)
        return local_dir

    logger.info("Downloading cloud Arrow dataset to local cache: %s → %s", uri, local_dir)
    os.makedirs(local_dir, exist_ok=True)

    fs, fs_path = fsspec.core.url_to_fs(uri, **(storage_options or {}))

    try:
        # Recursively copy the remote directory (handles nested splits, shards, etc.).
        fs.get(fs_path, local_dir, recursive=True)
    except Exception:
        # Clean up partial download so the next attempt starts fresh.
        shutil.rmtree(local_dir, ignore_errors=True)
        raise

    # Write marker so we know this cache entry is complete.
    with open(marker, "w") as f:
        f.write(uri + "\n")

    return local_dir
