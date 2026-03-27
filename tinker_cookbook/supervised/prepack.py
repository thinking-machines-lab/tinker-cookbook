"""
Offline pre-packing utility for supervised fine-tuning datasets.

Pre-renders and packs a JSONL dataset of conversations into sharded files of
packed Datums. This is the offline complement to ``streaming_packing.py``:
instead of rendering on-the-fly during training, the heavy tokenization work
is done once upfront using multiple CPU cores, and the packed output can be
loaded instantly for repeated training runs.

**CLI usage**::

    python -m tinker_cookbook.supervised.prepack \\
        source_file=/data/sft.jsonl \\
        output_dir=/data/packed \\
        model_name=nvidia/Nemotron-3-8B \\
        renderer_name=nemotron3_disable_thinking \\
        max_packed_length=49152 \\
        num_workers=64

**Programmatic usage**::

    from tinker_cookbook.supervised.prepack import PrepackedSupervisedDataset

    dataset = PrepackedSupervisedDataset(
        packed_dir="/data/packed",
        batch_size=4,
    )
    # Satisfies SupervisedDataset -- use directly in train.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
import time
from pathlib import Path

import chz
import tinker
import torch

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SHARD_SIZE = 10_000
"""Number of packed Datums per shard file."""

_METADATA_FILENAME = "metadata.json"

# ---------------------------------------------------------------------------
# Worker functions for multiprocessing.Pool
# ---------------------------------------------------------------------------

_worker_renderer = None
_worker_train_on_what = None


def _worker_init(renderer_name: str, model_name: str, train_on_what_str: str) -> None:
    """Per-worker initializer: creates a renderer once per process."""
    global _worker_renderer, _worker_train_on_what
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    _worker_renderer = get_renderer(renderer_name, tokenizer)
    _worker_train_on_what = TrainOnWhat(train_on_what_str)


def _render_one(messages: list[dict]) -> tuple[list[int], list[float]] | None:
    """Render a single conversation to (token_ids, weights).

    Returns ``None`` if the conversation produces no tokens (e.g. empty).
    """
    assert _worker_renderer is not None, "Worker not initialized"
    assert _worker_train_on_what is not None, "Worker not initialized"
    try:
        model_input, weights = _worker_renderer.build_supervised_example(
            messages, train_on_what=_worker_train_on_what
        )
        tokens = list(model_input.to_ints())
        if len(tokens) == 0:
            return None
        return tokens, weights.tolist()
    except Exception:
        logger.warning("Failed to render a conversation, skipping", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Greedy bin-packing (from plain token lists)
# ---------------------------------------------------------------------------


def pack_token_lists(
    rendered: list[tuple[list[int], list[float]]],
    max_packed_length: int,
) -> list[tinker.Datum]:
    """Pack pre-rendered (tokens, weights) pairs into Datums via greedy bin-packing.

    This mirrors the logic in ``streaming_packing.pack_rendered_examples`` but
    operates on plain Python lists instead of ``ModelInput`` / ``torch.Tensor``
    objects, which makes it safe for use with data returned from worker
    processes.

    Args:
        rendered: List of ``(token_ids, weights)`` tuples.
        max_packed_length: Maximum number of tokens per packed sequence.

    Returns:
        List of packed ``Datum`` objects.
    """
    packed_datums: list[tinker.Datum] = []
    current_tokens: list[int] = []
    current_weights: list[float] = []

    for tokens, weights in rendered:
        example_len = len(tokens)
        if example_len == 0:
            continue

        # Oversized example: flush buffer, emit alone (truncated).
        if example_len > max_packed_length:
            if current_tokens:
                _flush(current_tokens, current_weights, max_packed_length, packed_datums)
                current_tokens = []
                current_weights = []
            _flush(tokens, weights, max_packed_length, packed_datums)
            continue

        # Would adding this example overflow the current buffer?
        if len(current_tokens) + example_len > max_packed_length:
            _flush(current_tokens, current_weights, max_packed_length, packed_datums)
            current_tokens = []
            current_weights = []

        current_tokens.extend(tokens)
        current_weights.extend(weights)

    if current_tokens:
        _flush(current_tokens, current_weights, max_packed_length, packed_datums)

    return packed_datums


def _flush(
    tokens: list[int],
    weights: list[float],
    max_packed_length: int,
    out: list[tinker.Datum],
) -> None:
    """Build a Datum from concatenated tokens/weights and append to *out*."""
    combined_input = tinker.ModelInput.from_ints(tokens[:max_packed_length])
    combined_weights = torch.tensor(weights[:max_packed_length], dtype=torch.float32)
    out.append(datum_from_model_input_weights(combined_input, combined_weights, max_packed_length))


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_packed_datums(
    datums: list[tinker.Datum],
    output_dir: str | Path,
    metadata: dict,
    shard_size: int = _DEFAULT_SHARD_SIZE,
) -> None:
    """Write packed Datums to sharded JSONL files plus a metadata file.

    Args:
        datums: The packed Datum objects to persist.
        output_dir: Directory to write into (created if needed).
        metadata: Provenance dict to write as ``metadata.json``.
        shard_size: Maximum number of Datums per shard file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    for start in range(0, len(datums), shard_size):
        shard_datums = datums[start : start + shard_size]
        shard_path = output_dir / f"shard_{shard_idx:06d}.jsonl"
        with open(shard_path, "w") as f:
            for datum in shard_datums:
                f.write(datum.model_dump_json() + "\n")
        shard_idx += 1

    # Write metadata last (signals successful completion).
    metadata_out = {
        **metadata,
        "num_datums": len(datums),
        "num_shards": shard_idx,
        "shard_size": shard_size,
    }
    with open(output_dir / _METADATA_FILENAME, "w") as f:
        json.dump(metadata_out, f, indent=2)

    logger.info(
        "Saved %d packed Datums in %d shards to %s", len(datums), shard_idx, output_dir
    )


def load_packed_datums(packed_dir: str | Path) -> tuple[list[tinker.Datum], dict]:
    """Load all packed Datums and metadata from a pre-packed directory.

    Args:
        packed_dir: Path to directory created by ``save_packed_datums``.

    Returns:
        Tuple of (list of Datums, metadata dict).

    Raises:
        FileNotFoundError: If the metadata file is missing.
    """
    packed_dir = Path(packed_dir)
    metadata_path = packed_dir / _METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata file found at {metadata_path}. "
            f"Is this a valid pre-packed directory?"
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    num_shards: int = metadata["num_shards"]
    datums: list[tinker.Datum] = []
    for shard_idx in range(num_shards):
        shard_path = packed_dir / f"shard_{shard_idx:06d}.jsonl"
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    datums.append(tinker.Datum.model_validate_json(line))

    expected = metadata.get("num_datums", len(datums))
    if len(datums) != expected:
        logger.warning(
            "Expected %d Datums from metadata but loaded %d", expected, len(datums)
        )

    return datums, metadata


# ---------------------------------------------------------------------------
# PrepackedSupervisedDataset
# ---------------------------------------------------------------------------


class PrepackedSupervisedDataset(SupervisedDataset):
    """A ``SupervisedDataset`` backed by pre-packed Datums on disk.

    All Datums are loaded into memory at construction time (they are already
    compressed by packing, so this is memory-efficient). The dataset supports:

    * Exact ``__len__`` (known at load time).
    * Random-access ``get_batch``.
    * Epoch shuffling via ``set_epoch`` (shuffles pack order with a seed).

    This class can be used directly with the standard training loop in
    ``tinker_cookbook.supervised.train``.

    Args:
        packed_dir: Path to a directory created by ``save_packed_datums`` or
            the CLI tool.
        batch_size: Number of packed Datums per training batch.
    """

    def __init__(self, packed_dir: str | Path, batch_size: int):
        self._datums, self._metadata = load_packed_datums(packed_dir)
        self._batch_size = batch_size
        # Maintain a shuffled index list for epoch-aware access.
        self._indices: list[int] = list(range(len(self._datums)))
        if len(self._datums) == 0:
            logger.warning("PrepackedSupervisedDataset: loaded 0 Datums from %s", packed_dir)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return the batch at the given index.

        Args:
            index: Zero-based batch index.

        Returns:
            A list of ``batch_size`` packed Datums (or fewer for the last batch).
        """
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._datums[self._indices[i]] for i in range(start, end)]

    def set_epoch(self, seed: int = 0) -> None:
        """Shuffle the pack order for a new epoch.

        The shuffle is deterministic for a given seed.

        Args:
            seed: Random seed for reproducible shuffling.
        """
        rng = random.Random(seed)
        self._indices = list(range(len(self._datums)))
        rng.shuffle(self._indices)

    def __len__(self) -> int:
        """Return the number of complete batches."""
        return len(self._datums) // self._batch_size

    @property
    def metadata(self) -> dict:
        """Return the metadata dict loaded from disk."""
        return self._metadata


# ---------------------------------------------------------------------------
# CLI: pre-pack a JSONL dataset
# ---------------------------------------------------------------------------


@chz.chz
class PrepackConfig:
    """Configuration for the pre-packing CLI."""

    source_file: str
    output_dir: str
    model_name: str
    renderer_name: str
    max_packed_length: int = 49152
    train_on_what: str = "all_assistant_messages"
    num_workers: int = 8
    shuffle_seed: int = 42
    shard_size: int = _DEFAULT_SHARD_SIZE
    imap_chunksize: int = 500


def prepack(config: PrepackConfig) -> None:
    """Run the full pre-pack pipeline: load, render in parallel, pack, save.

    Args:
        config: CLI configuration object.
    """
    from tqdm import tqdm

    # 1. Load raw conversations.
    logger.info("Loading conversations from %s", config.source_file)
    conversations: list[list[dict]] = []
    with open(config.source_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "messages" not in data:
                raise ValueError(
                    f"Each JSONL line must have a 'messages' field. Got keys: {list(data.keys())}"
                )
            conversations.append(data["messages"])

    if len(conversations) == 0:
        logger.warning("Source file is empty, nothing to pack.")
        save_packed_datums(
            [],
            config.output_dir,
            metadata=_build_metadata(config, num_raw=0, num_rendered=0),
        )
        return

    logger.info("Loaded %d conversations", len(conversations))

    # 2. Shuffle with seed.
    rng = random.Random(config.shuffle_seed)
    rng.shuffle(conversations)

    # 3. Render in parallel across CPU cores.
    logger.info(
        "Rendering %d conversations using %d workers...",
        len(conversations),
        config.num_workers,
    )
    t0 = time.monotonic()

    rendered: list[tuple[list[int], list[float]]] = []

    if config.num_workers <= 1:
        # Single-process fallback (useful for debugging).
        _worker_init(config.renderer_name, config.model_name, config.train_on_what)
        for msgs in tqdm(conversations, desc="Rendering"):
            result = _render_one(msgs)
            if result is not None:
                rendered.append(result)
    else:
        with multiprocessing.Pool(
            processes=config.num_workers,
            initializer=_worker_init,
            initargs=(config.renderer_name, config.model_name, config.train_on_what),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_render_one, conversations, chunksize=config.imap_chunksize),
                total=len(conversations),
                desc="Rendering",
            ):
                if result is not None:
                    rendered.append(result)

    elapsed = time.monotonic() - t0
    logger.info(
        "Rendered %d/%d examples in %.1fs (%.0f examples/s)",
        len(rendered),
        len(conversations),
        elapsed,
        len(rendered) / max(elapsed, 0.001),
    )

    # 4. Pack rendered examples.
    logger.info("Packing into sequences of up to %d tokens...", config.max_packed_length)
    packed_datums = pack_token_lists(rendered, config.max_packed_length)
    logger.info("Packed into %d sequences", len(packed_datums))

    # 5. Save to sharded JSONL + metadata.
    metadata = _build_metadata(
        config, num_raw=len(conversations), num_rendered=len(rendered)
    )
    save_packed_datums(
        packed_datums, config.output_dir, metadata=metadata, shard_size=config.shard_size
    )


def _build_metadata(config: PrepackConfig, num_raw: int, num_rendered: int) -> dict:
    """Build the provenance metadata dict."""
    return {
        "source_file": config.source_file,
        "model_name": config.model_name,
        "renderer_name": config.renderer_name,
        "max_packed_length": config.max_packed_length,
        "train_on_what": config.train_on_what,
        "shuffle_seed": config.shuffle_seed,
        "num_workers": config.num_workers,
        "num_raw_conversations": num_raw,
        "num_rendered_examples": num_rendered,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    chz.nested_entrypoint(prepack, allow_hyphens=True)
