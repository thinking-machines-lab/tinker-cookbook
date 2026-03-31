"""Interleaved RL dataset — blends multiple sources by weighted sampling.

Enables multi-domain RL training where each batch contains
:class:`EnvGroupBuilder` instances drawn from multiple source datasets
according to configured weights. Groups are never mixed across domains —
each group comes from a single source, preserving GRPO's within-group
advantage centering.

Example::

    builder = InterleavedRLDatasetBuilder(
        sources=[mcqa_builder, workbench_builder, structured_output_builder],
        weights=[0.55, 0.30, 0.15],
        total_batches=70,
    )
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Sequence

import chz

from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

logger = logging.getLogger(__name__)


class InterleavedRLDataset(RLDataset):
    """Blends multiple :class:`RLDataset` sources by weighted sampling.

    Produces batches where each :class:`EnvGroupBuilder` is drawn from one
    source according to the configured weights. The full schedule is
    precomputed from the seed for deterministic reproducibility — the same
    ``(seed, batch_index)`` always produces the same groups, which is
    required for checkpoint-based recoverability.

    Args:
        sources: List of source RLDatasets to blend.
        weights: Relative weight for each source (auto-normalized).
        groups_per_batch: Number of EnvGroupBuilders per batch.
        total_batches: Total number of batches. If None, computed as the
            smallest source exhaustion point (no cycling).
        seed: Random seed for deterministic schedule.
    """

    def __init__(
        self,
        sources: list[RLDataset],
        weights: list[float],
        groups_per_batch: int,
        total_batches: int | None = None,
        seed: int = 0,
    ):
        if len(sources) != len(weights):
            raise ValueError(
                f"sources and weights must have the same length, "
                f"got {len(sources)} sources and {len(weights)} weights"
            )
        if not sources:
            raise ValueError("At least one source is required")
        if any(w <= 0 for w in weights):
            raise ValueError(f"All weights must be positive, got {weights}")

        self._sources = sources
        self._groups_per_batch = groups_per_batch
        self._seed = seed

        # Normalize weights
        total_weight = sum(weights)
        self._weights = [w / total_weight for w in weights]

        # Compute total batches if not specified
        if total_batches is None:
            total_batches = self._compute_natural_length(groups_per_batch)
            logger.info(
                f"InterleavedRLDataset: total_batches not set, "
                f"using smallest-source exhaustion: {total_batches} batches"
            )

        self._total_batches = total_batches

        # Precompute the full schedule for deterministic replay.
        # schedule[batch_idx] = list of (source_idx, group_index_within_source)
        self._schedule, self._source_counts = self._build_schedule()

        logger.info(
            f"InterleavedRLDataset: {len(sources)} sources, "
            f"weights={[f'{w:.2f}' for w in self._weights]}, "
            f"{total_batches} batches, {groups_per_batch} groups/batch"
        )
        for i, (src, count) in enumerate(zip(sources, self._source_counts)):
            src_len = len(src) * groups_per_batch  # approximate total groups
            logger.info(
                f"  Source {i}: {count} groups scheduled "
                f"(~{count / max(1, sum(self._source_counts)) * 100:.0f}%, "
                f"{'cycles' if count > src_len else 'partial'})"
            )

    def _compute_natural_length(self, groups_per_batch: int) -> int:
        """Compute batches until the smallest source is exhausted.

        For each source, the number of groups it contributes per batch is
        approximately ``weight * groups_per_batch``. The source is exhausted
        after ``len(source) * groups_per_batch / (weight * groups_per_batch)``
        = ``len(source) / weight`` batches.
        """
        batches_per_source = []
        for src, w in zip(self._sources, self._weights):
            total_groups = len(src) * self._groups_per_batch
            groups_per_step = w * self._groups_per_batch
            if groups_per_step > 0:
                batches_per_source.append(int(total_groups / groups_per_step))
            else:
                batches_per_source.append(0)
        return max(1, min(batches_per_source))

    def _build_schedule(
        self,
    ) -> tuple[list[list[tuple[int, int]]], list[int]]:
        """Precompute the deterministic assignment of groups to batches.

        Returns:
            schedule: List of batches, each a list of (source_idx, group_idx).
            source_counts: Total groups assigned to each source.
        """
        rng = random.Random(self._seed)
        n_sources = len(self._sources)

        # Per-source cursor and total capacity (with cycling)
        cursors = [0] * n_sources
        source_sizes = [len(src) for src in self._sources]
        source_counts = [0] * n_sources

        schedule: list[list[tuple[int, int]]] = []

        for _ in range(self._total_batches):
            batch: list[tuple[int, int]] = []
            for _ in range(self._groups_per_batch):
                # Weighted random choice of source
                src_idx = rng.choices(range(n_sources), weights=self._weights, k=1)[0]

                # Map cursor to a batch index in the source dataset.
                # Each source's get_batch returns groups_per_batch groups,
                # so we need to track which source batch to draw from.
                src_batch_idx = cursors[src_idx]

                # Cycle if we've exceeded the source's length
                if source_sizes[src_idx] > 0:
                    src_batch_idx = src_batch_idx % source_sizes[src_idx]

                batch.append((src_idx, src_batch_idx))
                cursors[src_idx] += 1
                source_counts[src_idx] += 1

            schedule.append(batch)

        return schedule, source_counts

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return a blended batch of EnvGroupBuilders for the given step.

        Deterministic — the same index always returns the same groups,
        enabling checkpoint-based recovery.

        Args:
            index: The batch index (``0 <= index < len(self)``).

        Returns:
            List of EnvGroupBuilders from multiple sources.
        """
        if index >= len(self._schedule):
            raise IndexError(
                f"Batch index {index} out of range "
                f"(total_batches={self._total_batches})"
            )

        assignments = self._schedule[index]
        builders: list[EnvGroupBuilder] = []

        for src_idx, src_batch_idx in assignments:
            # Get the source batch (which returns groups_per_batch builders)
            # and take the first one. Each assignment is one group.
            src = self._sources[src_idx]
            src_batch = src.get_batch(src_batch_idx)
            if src_batch:
                # Take the first builder from this source batch.
                # The source's get_batch may return multiple builders
                # (groups_per_batch), but we only need one per assignment.
                builders.append(src_batch[0])

        return builders

    def __len__(self) -> int:
        """Return the total number of batches in the interleaved schedule."""
        return self._total_batches


@chz.chz
class InterleavedRLDatasetBuilder(RLDatasetBuilder):
    """Builds an interleaved RL dataset from multiple sources.

    Each batch contains :class:`EnvGroupBuilder` instances drawn from
    multiple source datasets according to the configured weights. This
    enables multi-domain RL training (e.g., 55% MCQA + 30% Workbench +
    15% Structured Output from the Nemotron-Cascade-2 paper).

    Groups are sampled at the group level (not within-group), preserving
    GRPO's within-group advantage centering. The schedule is precomputed
    from the seed for deterministic reproducibility and checkpoint recovery.

    Args:
        sources: List of :class:`RLDatasetBuilder` instances to blend.
        weights: Relative weight for each source (auto-normalized).
        total_batches: Total number of batches. ``None`` = run until the
            smallest source is exhausted (no cycling).
        seed: Random seed for deterministic schedule.

    Example::

        # Nemotron-Cascade-2 multi-domain blend
        builder = InterleavedRLDatasetBuilder(
            sources=[mcqa_builder, workbench_builder, structured_output_builder],
            weights=[0.55, 0.30, 0.15],
            total_batches=70,
        )
    """

    sources: list[RLDatasetBuilder]
    weights: list[float]
    total_batches: int | None = None
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """Build the interleaved training dataset.

        Calls each source builder, collects the training datasets, and
        wraps them in an :class:`InterleavedRLDataset`.

        Returns:
            A two-element tuple of (train_dataset, test_dataset). The test
            dataset is the first non-None test dataset from the sources,
            or None if no source provides one.
        """
        if len(self.sources) != len(self.weights):
            raise ValueError(
                f"sources and weights must have the same length, "
                f"got {len(self.sources)} sources and {len(self.weights)} weights"
            )

        train_datasets: list[RLDataset] = []
        test_dataset: RLDataset | None = None

        for i, source in enumerate(self.sources):
            train_ds, test_ds = await source()
            train_datasets.append(train_ds)
            if test_ds is not None and test_dataset is None:
                test_dataset = test_ds
            logger.info(
                f"Source {i} ({type(source).__name__}): "
                f"{len(train_ds)} batches, weight={self.weights[i]}"
            )

        # Infer groups_per_batch from the first source's first batch
        first_batch = train_datasets[0].get_batch(0)
        groups_per_batch = len(first_batch)

        interleaved = InterleavedRLDataset(
            sources=train_datasets,
            weights=self.weights,
            groups_per_batch=groups_per_batch,
            total_batches=self.total_batches,
            seed=self.seed,
        )

        return interleaved, test_dataset
