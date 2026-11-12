"""Utilities for composing :class:`~tinker_cookbook.rl.types.RLDataset` instances."""

from __future__ import annotations

import functools
import random
from collections.abc import Sequence

from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset


@functools.lru_cache(maxsize=64)
def _cycle_permutation_cached(size: int, seed: str, cycle: int) -> tuple[int, ...]:
    perm = list(range(size))
    random.Random(f"{seed}:{cycle}").shuffle(perm)
    return tuple(perm)


def cycle_permutation(size: int, seed: int | str, cycle: int) -> Sequence[int]:
    """Deterministic permutation of ``range(size)`` for pass ``cycle`` over a dataset.

    Depends only on ``(size, seed, cycle)``, so any index in any cycle can be
    resolved without replaying earlier cycles. Resuming from a checkpoint
    therefore sees exactly the order the original run would have seen.
    """
    return _cycle_permutation_cached(size, str(seed), cycle)


def epoch_batch_index(index: int, num_batches: int, shuffle_seed: int | None = None) -> int:
    """Map global step ``index`` to a batch index in ``range(num_batches)``.

    Epoch ``index // num_batches`` visits batches in order, or in
    ``cycle_permutation(num_batches, shuffle_seed, epoch)`` order when
    ``shuffle_seed`` is set.
    """
    epoch, pos = divmod(index, num_batches)
    if shuffle_seed is not None:
        pos = cycle_permutation(num_batches, shuffle_seed, epoch)[pos]
    return pos


class EpochCycledRLDataset(RLDataset):
    """Repeat an :class:`RLDataset` for ``num_epochs`` passes.

    Global batch ``i`` belongs to epoch ``i // len(inner)``. Without
    ``shuffle_seed``, it is ``inner.get_batch(i % len(inner))``, so every epoch
    replays the inner dataset in its own order and ``num_epochs=1`` is the
    identity. With ``shuffle_seed``, each epoch visits the inner batches in a
    fresh :func:`cycle_permutation`. Only the order of whole batches changes;
    each batch keeps the same groups.

    Args:
        inner: Dataset to repeat. Must be non-empty.
        num_epochs: Number of passes (``>= 1``).
        shuffle_seed: If set, reshuffle batch order every epoch, deterministically
            per ``(shuffle_seed, epoch)``.
    """

    def __init__(self, inner: RLDataset, num_epochs: int, shuffle_seed: int | None = None):
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
        self.inner = inner
        self.num_epochs = num_epochs
        self.shuffle_seed = shuffle_seed

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if not (0 <= index < len(self)):
            raise IndexError(f"Batch index {index} out of range (len={len(self)})")
        return self.inner.get_batch(epoch_batch_index(index, len(self.inner), self.shuffle_seed))

    def __len__(self) -> int:
        return len(self.inner) * self.num_epochs
