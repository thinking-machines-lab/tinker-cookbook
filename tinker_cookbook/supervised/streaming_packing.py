"""
Streaming packed dataset for supervised fine-tuning.

Renders and packs examples in a background thread so that training can start
immediately instead of waiting for the entire dataset to be pre-rendered.
The packing algorithm is the same greedy bin-packing used by
``tinker_cookbook.recipes.nemotron_cascade.sft_datasets.PackedSupervisedDataset``,
but rendering happens incrementally in chunks and packed batches are fed to the
training loop through a bounded queue.

Usage::

    from tinker_cookbook.supervised.streaming_packing import (
        StreamingPackedSupervisedDataset,
        pack_rendered_examples,
    )

    dataset = StreamingPackedSupervisedDataset(
        examples=my_raw_examples,          # Iterable of raw examples
        render_fn=my_render_fn,            # Example -> (ModelInput, weights)
        batch_size=4,
        max_packed_length=49152,
        total_examples=len(my_raw_examples),
    )

    # Satisfies the SupervisedDataset interface — use directly in train.py
    for batch_idx in range(len(dataset)):
        batch = dataset.get_batch(batch_idx)
"""

import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar

import tinker
import torch

from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Pure packing algorithm
# ---------------------------------------------------------------------------


def pack_rendered_examples(
    rendered: Iterable[tuple[tinker.ModelInput, torch.Tensor]],
    max_packed_length: int,
) -> list[tinker.Datum]:
    """Pack multiple pre-shift rendered examples into Datums using greedy bin-packing.

    Each rendered example is a ``(ModelInput, weights)`` pair as returned by
    ``renderer.build_supervised_example``.  This function concatenates multiple
    examples' tokens and weights into packed sequences up to *max_packed_length*
    tokens, then applies the standard right-shift via
    ``datum_from_model_input_weights``.

    Args:
        rendered: Iterable of ``(ModelInput, weights_tensor)`` pairs.
        max_packed_length: Maximum number of tokens per packed sequence.

    Returns:
        List of packed Datums.  Each Datum may contain multiple concatenated
        examples.
    """
    packed_datums: list[tinker.Datum] = []
    current_tokens: list[int] = []
    current_weights: list[float] = []

    for model_input, weights in rendered:
        tokens = list(model_input.to_ints())
        w = weights.tolist()
        example_len = len(tokens)

        if example_len == 0:
            continue

        # If this single example exceeds the limit, pack it alone (truncated).
        if example_len > max_packed_length:
            if current_tokens:
                _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)
                current_tokens = []
                current_weights = []
            _flush_packed(tokens, w, max_packed_length, packed_datums)
            continue

        # Would adding this example exceed the limit?
        if len(current_tokens) + example_len > max_packed_length:
            _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)
            current_tokens = []
            current_weights = []

        current_tokens.extend(tokens)
        current_weights.extend(w)

    # Flush remaining.
    if current_tokens:
        _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)

    return packed_datums


def _flush_packed(
    tokens: list[int],
    weights: list[float],
    max_packed_length: int,
    out: list[tinker.Datum],
) -> None:
    """Create a Datum from concatenated tokens/weights and append to *out*."""
    combined_input = tinker.ModelInput.from_ints(tokens[:max_packed_length])
    combined_weights = torch.tensor(weights[:max_packed_length], dtype=torch.float32)
    out.append(datum_from_model_input_weights(combined_input, combined_weights, max_packed_length))


# ---------------------------------------------------------------------------
# Incremental packer (packs a stream, yields Datums one at a time)
# ---------------------------------------------------------------------------


def _incremental_pack(
    rendered_iter: Iterator[tuple[tinker.ModelInput, torch.Tensor]],
    max_packed_length: int,
) -> Iterator[tinker.Datum]:
    """Yield packed Datums one at a time from a stream of rendered examples.

    This is the core of the streaming pipeline: it greedily fills a buffer up to
    *max_packed_length* tokens, flushes a Datum, and continues.  The caller
    receives Datums as soon as they are ready rather than waiting for the whole
    dataset to be rendered.

    Any carry-over from chunk boundaries is handled naturally: leftover examples
    that did not fit in the previous pack remain in the buffer for the next one.
    """
    current_tokens: list[int] = []
    current_weights: list[float] = []

    for model_input, weights in rendered_iter:
        tokens = list(model_input.to_ints())
        w = weights.tolist()
        example_len = len(tokens)

        if example_len == 0:
            continue

        # Oversized example: flush current buffer, then emit the oversized one alone.
        if example_len > max_packed_length:
            if current_tokens:
                yield _make_datum(current_tokens, current_weights, max_packed_length)
                current_tokens = []
                current_weights = []
            yield _make_datum(tokens, w, max_packed_length)
            continue

        # Would this example overflow the current buffer?
        if len(current_tokens) + example_len > max_packed_length:
            yield _make_datum(current_tokens, current_weights, max_packed_length)
            current_tokens = []
            current_weights = []

        current_tokens.extend(tokens)
        current_weights.extend(w)

    # Flush any remaining tokens.
    if current_tokens:
        yield _make_datum(current_tokens, current_weights, max_packed_length)


def _make_datum(
    tokens: list[int],
    weights: list[float],
    max_packed_length: int,
) -> tinker.Datum:
    """Build a single Datum from concatenated tokens and weights."""
    combined_input = tinker.ModelInput.from_ints(tokens[:max_packed_length])
    combined_weights = torch.tensor(weights[:max_packed_length], dtype=torch.float32)
    return datum_from_model_input_weights(combined_input, combined_weights, max_packed_length)


# ---------------------------------------------------------------------------
# Sentinel / error wrappers for queue communication
# ---------------------------------------------------------------------------


@dataclass
class _ErrorSentinel:
    """Carries an exception from the background thread to the consumer."""

    exception: BaseException


# None is used as the end-of-stream sentinel.
_QueueItem = list[tinker.Datum] | _ErrorSentinel | None


# ---------------------------------------------------------------------------
# StreamingPackedSupervisedDataset
# ---------------------------------------------------------------------------


class StreamingPackedSupervisedDataset(SupervisedDataset):
    """A streaming, packed supervised dataset that renders in the background.

    Instead of pre-rendering every example before training begins, this class
    spawns a daemon thread that renders raw examples through a user-supplied
    ``render_fn``, packs the resulting token sequences using greedy bin-packing,
    groups packed Datums into batches, and deposits them into a bounded
    ``queue.Queue``.  The training loop calls ``get_batch(index)`` which pulls
    the next batch from the queue.

    **Key properties:**

    * Forward-only iteration (like ``StreamingSupervisedDatasetFromHFDataset``).
    * The background thread starts as soon as the dataset is constructed.
    * Errors in the background thread are propagated to the main thread on the
      next ``get_batch`` call.
    * ``set_epoch`` cleanly stops the current background pipeline and restarts
      with the new epoch's iterator.
    * ``__len__`` returns a conservative estimate based on ``total_examples``
      and average tokens-per-example (updated as rendering progresses).

    Args:
        examples: An iterable of raw examples.  Must be re-iterable (called
            again on each epoch) OR a callable returning a fresh iterator.
        render_fn: Callable that converts a raw example into a
            ``(ModelInput, weights_tensor)`` pair.
        batch_size: Number of packed Datums per batch.
        max_packed_length: Maximum number of tokens per packed sequence.
        total_examples: Approximate total number of raw examples (used to
            estimate ``__len__``).
        queue_size: Maximum number of pre-built batches to buffer.  Larger
            values decouple render speed from training speed at the cost of
            memory.
        avg_example_tokens: Initial estimate of average tokens per raw example.
            Updated as rendering progresses.  Used to estimate how many packed
            sequences the dataset will produce.
    """

    def __init__(
        self,
        examples: Iterable[T] | Callable[[], Iterable[T]],
        render_fn: Callable[[T], tuple[tinker.ModelInput, torch.Tensor]],
        batch_size: int,
        max_packed_length: int,
        total_examples: int,
        queue_size: int = 64,
        avg_example_tokens: float = 256.0,
    ):
        self._examples_source = examples
        self._render_fn = render_fn
        self._batch_size = batch_size
        self._max_packed_length = max_packed_length
        self._total_examples = total_examples
        self._queue_size = queue_size

        # Running statistics for __len__ estimation.
        self._avg_example_tokens = avg_example_tokens
        self._total_tokens_seen = 0.0
        self._total_examples_rendered = 0
        self._stats_lock = threading.Lock()

        # The batch queue and background thread state.
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_batch_index = -1
        self._exhausted = False

        # Start the background pipeline.
        self._start_background()

    # ------------------------------------------------------------------
    # SupervisedDataset interface
    # ------------------------------------------------------------------

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return the batch at *index*.

        Only forward sequential access is supported.  If *index* skips ahead,
        intermediate batches are silently consumed.  Backward seeks raise
        ``ValueError``.

        Raises ``StopIteration`` when the stream is exhausted.
        """
        if self._exhausted:
            raise StopIteration("Streaming dataset is exhausted")

        if index < self._current_batch_index + 1:
            raise ValueError(
                f"StreamingPackedSupervisedDataset only supports forward iteration. "
                f"Cannot seek backward from batch {self._current_batch_index} to {index}."
            )

        # Skip forward if needed.
        batches_to_skip = index - self._current_batch_index - 1
        for _ in range(batches_to_skip):
            self._consume_one()

        self._current_batch_index = index
        return self._consume_one()

    def set_epoch(self, seed: int = 0) -> None:
        """Stop the current pipeline and restart for a new epoch.

        The *seed* is currently unused for shuffling (the caller is responsible
        for providing a differently-ordered ``examples`` iterable if shuffling
        is desired), but the method resets internal state so the dataset can be
        iterated from the beginning again.
        """
        self._stop_background()

        # Reset consumer state.
        self._current_batch_index = -1
        self._exhausted = False

        # Create a fresh queue and restart.
        self._queue = queue.Queue(maxsize=self._queue_size)
        self._stop_event = threading.Event()
        self._start_background()

        logger.info("StreamingPackedSupervisedDataset: reset for epoch (seed=%d)", seed)

    def __len__(self) -> int:
        """Return a conservative estimate of the number of batches.

        The estimate is based on the total number of raw examples, the average
        tokens per example (updated as rendering progresses), and the packing
        length.  It may decrease as better statistics become available.
        """
        with self._stats_lock:
            avg_tokens = self._avg_example_tokens

        # Estimate how many packed sequences we will produce.
        estimated_packed = max(1, int(self._total_examples * avg_tokens / self._max_packed_length))
        return estimated_packed // self._batch_size

    # ------------------------------------------------------------------
    # Background pipeline
    # ------------------------------------------------------------------

    def _start_background(self) -> None:
        """Spawn the background rendering/packing thread."""
        self._thread = threading.Thread(
            target=self._background_worker,
            name="streaming-packer",
            daemon=True,
        )
        self._thread.start()

    def _stop_background(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            # Drain the queue so the background thread is not blocked on put().
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning(
                    "StreamingPackedSupervisedDataset: background thread did not "
                    "terminate within timeout"
                )
            self._thread = None

    def _get_examples_iter(self) -> Iterable:
        """Return a fresh iterable over raw examples."""
        if callable(self._examples_source) and not isinstance(self._examples_source, (list, tuple)):
            return self._examples_source()
        return self._examples_source

    def _background_worker(self) -> None:
        """Render, pack, and enqueue batches until the data is exhausted or stop is requested."""
        try:
            examples_iter = self._get_examples_iter()
            rendered_iter = self._rendering_iterator(examples_iter)
            packed_iter = _incremental_pack(rendered_iter, self._max_packed_length)

            batch: list[tinker.Datum] = []
            for datum in packed_iter:
                if self._stop_event.is_set():
                    return
                batch.append(datum)
                if len(batch) == self._batch_size:
                    self._put(batch)
                    batch = []

            # Flush a partial final batch (the training loop handles short batches).
            if batch and not self._stop_event.is_set():
                self._put(batch)

            # Signal end-of-stream.
            if not self._stop_event.is_set():
                self._put(None)

        except Exception as exc:
            logger.error(
                "StreamingPackedSupervisedDataset: background thread error: %s", exc
            )
            try:
                self._queue.put(_ErrorSentinel(exc), timeout=5.0)
            except queue.Full:
                pass

    def _rendering_iterator(
        self, examples: Iterable,
    ) -> Iterator[tuple[tinker.ModelInput, torch.Tensor]]:
        """Apply render_fn to each raw example, tracking statistics."""
        count = 0
        total_tokens = 0
        for example in examples:
            if self._stop_event.is_set():
                return
            model_input, weights = self._render_fn(example)
            n_tokens = model_input.length
            count += 1
            total_tokens += n_tokens

            # Update running statistics periodically.
            if count % 1000 == 0:
                with self._stats_lock:
                    self._total_examples_rendered += 1000
                    self._total_tokens_seen += total_tokens
                    self._avg_example_tokens = (
                        self._total_tokens_seen / self._total_examples_rendered
                    )
                    total_tokens = 0
                if count % 50_000 == 0:
                    logger.info(
                        "StreamingPackedSupervisedDataset: rendered %d examples "
                        "(avg %.0f tokens/example)",
                        self._total_examples_rendered,
                        self._avg_example_tokens,
                    )

            yield model_input, weights

        # Final stats update.
        remainder = count % 1000
        if remainder > 0:
            with self._stats_lock:
                self._total_examples_rendered += remainder
                self._total_tokens_seen += total_tokens
                if self._total_examples_rendered > 0:
                    self._avg_example_tokens = (
                        self._total_tokens_seen / self._total_examples_rendered
                    )

        logger.info(
            "StreamingPackedSupervisedDataset: finished rendering %d examples "
            "(avg %.0f tokens/example)",
            self._total_examples_rendered,
            self._avg_example_tokens,
        )

    def _put(self, item: _QueueItem) -> None:
        """Put an item on the queue, respecting the stop event."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def _consume_one(self) -> list[tinker.Datum]:
        """Pull the next batch from the queue.

        Raises ``StopIteration`` if the stream is exhausted, or re-raises any
        exception that occurred in the background thread.
        """
        if self._exhausted:
            raise StopIteration("Streaming dataset is exhausted")

        item = self._queue.get()
        if item is None:
            self._exhausted = True
            raise StopIteration("Streaming dataset is exhausted")
        if isinstance(item, _ErrorSentinel):
            raise RuntimeError(
                "Background rendering thread failed"
            ) from item.exception
        return item
