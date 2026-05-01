"""Streaming dataset rendering via PyTorch DataLoader.

Renders JSONL rows on the fly inside DataLoader workers, with prefetch
hiding per-row tokenization behind the GPU train step. Workers are
spawned (not forked) so each gets its own tokenizer / renderer state
without copy-on-write inflating the parent's heap.

Replaces the earlier "render to a disk-backed pickle store, then read
back during training" pipeline because:

* Per-row CPU render cost is small relative to the trainer's
  ``forward_backward`` step, so DataLoader prefetch hides it.
* No disk persistence is needed for SFT -- multi-epoch simply
  re-renders the same JSONL rows (cheap on the CPU orchestrator pod).
* Eliminates ~200 LoC of bespoke worker pool / disk store / index code
  in favour of battle-tested DataLoader machinery.

Used today by ``sft_loop``. DPO/ORPO can reuse :class:`JsonlRenderDataset`
with a preference-pair ``render_fn``; their multi-epoch reference logprob
cache is a separate concern (kept on disk so the reference trainer can
be released after epoch 0).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Callable, Iterator, List

import torch
import torch.utils.data as torch_data

logger = logging.getLogger(__name__)

# Defaults validated end-to-end on a 110K-example, 256K-context SFT
# dataset on a 16 vCPU / 58 GiB orchestrator pod.
DEFAULT_RENDER_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2


# ---------------------------------------------------------------------------
# JSONL → render → object dataset
# ---------------------------------------------------------------------------


def _scan_jsonl_offsets(path: str, max_examples: int | None = None) -> List[int]:
    """Return the byte offset of every non-blank line in a JSONL file."""
    offsets: List[int] = []
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            if line.strip():
                offsets.append(offset)
                if max_examples is not None and len(offsets) >= max_examples:
                    break
            offset += len(line)
    return offsets


class JsonlRenderDataset(torch_data.Dataset):
    """Map-style dataset that lazily renders each JSONL row on access.

    A single linear scan at construction time builds the byte-offset
    table; each ``__getitem__(i)`` is a seek + readline + ``json.loads``
    + ``render_fn``. ``render_fn`` must be a top-level (module-level)
    function so it is picklable for spawn workers.

    ``render_fn`` may return ``None`` for rows that should be dropped
    (empty messages, over-length sequences, ...). The companion
    :func:`make_render_dataloader` wires up a collate that filters Nones.
    """

    def __init__(
        self,
        path: str,
        render_fn: Callable[[dict], Any | None],
        *,
        max_examples: int | None = None,
        indices: List[int] | None = None,
    ):
        self._path = path
        self._render_fn = render_fn
        self._offsets = _scan_jsonl_offsets(path, max_examples)
        self._index_map: List[int] = (
            list(indices)
            if indices is not None
            else list(range(len(self._offsets)))
        )

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, i: int) -> Any:
        offset = self._offsets[self._index_map[i]]
        with open(self._path) as f:
            f.seek(offset)
            line = f.readline()
        return self._render_fn(json.loads(line))

    def with_indices(self, indices: List[int]) -> "JsonlRenderDataset":
        """Return a view of this dataset restricted to ``indices``.

        Shares the underlying offset table and render_fn; only the
        index mapping differs. Used to carve out a contiguous eval slice
        from the head of the training data without rescanning the file.
        """
        view = object.__new__(type(self))
        view._path = self._path
        view._render_fn = self._render_fn
        view._offsets = self._offsets
        view._index_map = list(indices)
        return view

    @property
    def num_underlying_rows(self) -> int:
        return len(self._offsets)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _drop_none_collate(batch: List[Any]) -> List[Any]:
    return [d for d in batch if d is not None]


def make_render_dataloader(
    dataset: torch_data.Dataset,
    *,
    batch_size: int,
    num_workers: int = DEFAULT_RENDER_WORKERS,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> torch_data.DataLoader:
    """Build a DataLoader for ``dataset`` with our spawn / collate defaults.

    * ``multiprocessing_context="spawn"`` keeps each worker's RSS
      independent of the parent (no copy-on-write inflation of the
      tokenizer / renderer state).
    * ``persistent_workers=True`` keeps the per-worker tokenizer alive
      across epochs so we pay the spawn-and-init cost once.
    * ``collate_fn`` returns a python list of rendered items, dropping
      any ``None`` entries. Tinker's ``forward_backward`` accepts
      variable-size batches so dropping is safe.

    ``num_workers <= 1`` falls back to single-process rendering. Unit
    tests rely on this to monkey-patch the renderer (spawn workers can't
    see test-time monkey patches), and a single worker subprocess
    rarely earns its keep over in-process rendering anyway.
    """
    if num_workers <= 1:
        return torch_data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=_drop_none_collate,
        )
    return torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context="spawn",
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        collate_fn=_drop_none_collate,
    )


# ---------------------------------------------------------------------------
# Append-only pickle log (DPO ref-cache)
# ---------------------------------------------------------------------------


class AppendOnlyPickleLog:
    """Sequential disk-backed object log: append in epoch 0, iterate in epochs 1+.

    Designed for DPO's reference-logprob cache: we need to spool enriched
    preference pairs to disk so the (expensive) reference trainer can be
    released after epoch 0, but epochs 1+ only ever iterate the cache
    front-to-back. That eliminates everything the previous
    ``DiskBackedDatumStore`` carried (offset index, mmap, random access,
    truncation safety) and reduces the disk store to ~30 LoC of pickle.

    Lifecycle: ``append(...)`` while writing → ``close_write()`` → iterate
    via ``for x in log``. Iterating before ``close_write()`` raises.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh: Any = open(path, "wb")
        self._count = 0

    def append(self, obj: Any) -> None:
        if self._fh is None:
            raise RuntimeError("AppendOnlyPickleLog is closed; cannot append")
        pickle.dump(obj, self._fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._count += 1

    def close_write(self) -> None:
        """Flush + fsync + close the write handle so readers see all data."""
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None

    def __len__(self) -> int:
        return self._count

    def disk_size_bytes(self) -> int:
        try:
            return os.path.getsize(self._path)
        except OSError:
            return 0

    def __iter__(self) -> Iterator[Any]:
        if self._fh is not None:
            raise RuntimeError(
                "close_write() must be called before iterating an AppendOnlyPickleLog"
            )
        with open(self._path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return

    def close(self) -> None:
        if self._fh is not None:
            self.close_write()

    def __enter__(self) -> "AppendOnlyPickleLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
