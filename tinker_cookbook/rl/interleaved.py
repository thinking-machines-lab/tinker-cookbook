"""Interleaved RL dataset for multi-domain training.

Provides :class:`InterleavedRLDatasetBuilder`, which blends multiple
:class:`RLDatasetBuilder` sources into a single training dataset. Each
training batch contains :class:`EnvGroupBuilder` instances drawn from
different sources according to configured weights. Groups are never
mixed across domains — each group comes from a single source,
preserving GRPO's within-group advantage centering.

Example::

    builder = InterleavedRLDatasetBuilder(
        sources=[mcqa_builder, workbench_builder, structured_output_builder],
        weights=[0.55, 0.30, 0.15],
        groups_per_batch=8,
        total_batches=70,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence

import chz

from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


class _InterleavedRLDataset(RLDataset):
    """Blends multiple :class:`RLDataset` sources by weighted sampling.

    This is the runtime dataset created by :class:`InterleavedRLDatasetBuilder`.
    Users should not construct this directly.

    **Schedule construction.** At init, the dataset builds a complete
    schedule mapping every ``(batch_index, group_slot)`` to a specific
    ``(source_index, group_index)`` pair. The schedule is fully determined
    by the ``seed``, so the same ``(seed, batch_index)`` always produces
    the same groups — enabling checkpoint-based recovery.

    **Shuffled permutations.** Each source maintains a shuffled
    permutation of its group indices. Within each cycle (one pass through
    all of a source's groups), groups are visited in shuffled order.
    When a cycle is exhausted, a fresh permutation is generated from the
    same deterministic RNG stream, so repeated passes see problems in a
    different order.

    **Lazy resolution.** Source groups are not materialized during init.
    Only one ``get_batch(0)`` call per source is made to learn the number
    of groups per batch; flat indices are resolved to
    ``(batch_idx, within_batch_idx)`` arithmetically in ``get_batch()``.

    Args:
        sources: Built RLDatasets to blend.
        weights: Relative weight per source (auto-normalized to sum to 1).
        groups_per_batch: Number of :class:`EnvGroupBuilder` instances per
            output batch.
        total_batches: Fixed number of batches. If ``None``, computed as
            the smallest source exhaustion point (no cycling).
        seed: Random seed for deterministic schedule generation.
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
        if groups_per_batch < 1:
            raise ValueError(f"groups_per_batch must be >= 1, got {groups_per_batch}")

        self._sources = sources
        self._groups_per_batch = groups_per_batch
        self._seed = seed

        total_weight = sum(weights)
        self._weights = [w / total_weight for w in weights]

        # Probe each source once to learn groups-per-source-batch, avoiding
        # expensive get_batch calls on every index during init.
        self._source_groups_per_batch: list[int] = []
        self._source_total_groups: list[int] = []
        for i, src in enumerate(sources):
            if len(src) == 0:
                raise ValueError(f"Source {i} has 0 batches — all sources must be non-empty")
            probe = src.get_batch(0)
            gpb = len(probe)
            if gpb == 0:
                raise ValueError(
                    f"Source {i} returned 0 groups for batch 0 — "
                    f"all sources must produce at least 1 group per batch"
                )
            self._source_groups_per_batch.append(gpb)
            # Account for the last batch potentially being shorter (ragged).
            # Probe the last batch to get the actual count instead of assuming
            # all batches have the same size as batch 0.
            if len(src) == 1:
                total = gpb
            else:
                last_batch = src.get_batch(len(src) - 1)
                total = (len(src) - 1) * gpb + len(last_batch)
            self._source_total_groups.append(total)

        if total_batches is None:
            total_batches = self._compute_natural_length()
            logger.info(
                f"InterleavedRLDataset: total_batches not set, "
                f"using smallest-source exhaustion: {total_batches} batches"
            )

        self._total_batches = total_batches

        self._schedule, source_counts = self._build_schedule()

        logger.info(
            f"InterleavedRLDataset: {len(sources)} sources, "
            f"weights={[f'{w:.2f}' for w in self._weights]}, "
            f"{total_batches} batches, {groups_per_batch} groups/batch"
        )
        total_scheduled = max(1, sum(source_counts))
        for i, count in enumerate(source_counts):
            total_groups = self._source_total_groups[i]
            logger.info(
                f"  Source {i}: {total_groups} total groups, "
                f"{count} scheduled "
                f"(~{count / total_scheduled * 100:.0f}%, "
                f"{'cycles' if count > total_groups else 'partial'})"
            )

    def _compute_natural_length(self) -> int:
        """Compute batches until the smallest source is exhausted.

        For each source, the expected groups consumed per batch is
        ``weight * groups_per_batch``. The source is exhausted after
        ``total_groups / (weight * groups_per_batch)`` batches.
        """
        batches_per_source = []
        for total_groups, w in safezip(self._source_total_groups, self._weights):
            groups_per_step = w * self._groups_per_batch
            batches_per_source.append(int(total_groups / groups_per_step))
        return max(1, min(batches_per_source))

    def _build_schedule(
        self,
    ) -> tuple[list[list[tuple[int, int]]], list[int]]:
        """Precompute the deterministic assignment of groups to batches.

        Uses a single seeded RNG for both source selection and permutation
        generation, ensuring the full schedule is reproducible from ``seed``.

        Each source maintains a shuffled permutation of its group indices.
        When all groups in a cycle are consumed, a fresh permutation is
        generated so the next pass sees problems in a different order.

        Returns:
            schedule: List of batches, each a list of
                ``(source_idx, flat_group_idx)`` pairs.
            source_counts: Total groups assigned to each source.
        """
        rng = random.Random(self._seed)
        n_sources = len(self._sources)

        def new_permutation(size: int) -> list[int]:
            perm = list(range(size))
            rng.shuffle(perm)
            return perm

        permutations = [new_permutation(self._source_total_groups[i]) for i in range(n_sources)]
        positions = [0] * n_sources
        source_counts = [0] * n_sources
        schedule: list[list[tuple[int, int]]] = []

        for _ in range(self._total_batches):
            batch: list[tuple[int, int]] = []
            for _ in range(self._groups_per_batch):
                src_idx = rng.choices(range(n_sources), weights=self._weights, k=1)[0]

                if positions[src_idx] >= self._source_total_groups[src_idx]:
                    permutations[src_idx] = new_permutation(self._source_total_groups[src_idx])
                    positions[src_idx] = 0

                flat_idx = permutations[src_idx][positions[src_idx]]
                batch.append((src_idx, flat_idx))
                positions[src_idx] += 1
                source_counts[src_idx] += 1

            schedule.append(batch)

        return schedule, source_counts

    def _resolve_flat_idx(self, src_idx: int, flat_idx: int) -> EnvGroupBuilder:
        """Resolve a flat group index to the actual EnvGroupBuilder.

        Converts ``flat_idx`` to ``(batch_idx, within_batch_idx)`` via
        integer division by the source's groups-per-batch, then fetches
        the builder from the source dataset.

        Raises:
            IndexError: If a non-last source batch has fewer groups than
                expected. All batches except the last must have the same
                number of groups as batch 0.
        """
        gpb = self._source_groups_per_batch[src_idx]
        batch_idx = flat_idx // gpb
        within_idx = flat_idx % gpb
        src_batch = self._sources[src_idx].get_batch(batch_idx)
        if within_idx >= len(src_batch):
            raise IndexError(
                f"Source {src_idx} batch {batch_idx} has {len(src_batch)} groups, "
                f"expected {gpb} (based on batch 0). "
                f"InterleavedRLDataset requires all batches except the last to "
                f"have the same number of groups as batch 0."
            )
        return src_batch[within_idx]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return a blended batch of EnvGroupBuilders for the given step.

        Deterministic: the same ``(seed, index)`` always returns the same
        group configuration, enabling checkpoint-based recovery. The
        returned builders are not yet rolled out — the training loop
        runs rollouts on-policy with the current model.

        Args:
            index: The batch index (``0 <= index < len(self)``).

        Returns:
            Exactly ``groups_per_batch`` EnvGroupBuilders, each from a
            single source.
        """
        if not (0 <= index < self._total_batches):
            raise IndexError(
                f"Batch index {index} out of range (total_batches={self._total_batches})"
            )

        return [
            self._resolve_flat_idx(src_idx, flat_idx) for src_idx, flat_idx in self._schedule[index]
        ]

    def __len__(self) -> int:
        return self._total_batches


@chz.chz
class InterleavedRLDatasetBuilder(RLDatasetBuilder):
    """Blend multiple RL dataset sources for multi-domain training.

    Produces a single :class:`RLDataset` whose batches contain
    :class:`EnvGroupBuilder` instances drawn from multiple source
    datasets according to configured weights. This enables multi-domain
    RL training where different problem types (e.g., MCQA, code,
    structured output) are mixed within each training batch.

    **How it works.** Each source is an :class:`RLDatasetBuilder` that
    produces a problem set (a list of :class:`EnvGroupBuilder` factories).
    The interleaved builder flattens each source's groups into a single
    index space, then builds a precomputed schedule that assigns each
    group slot in each batch to a specific source and group. At training
    time, the training loop calls ``get_batch(i)`` which returns the
    scheduled :class:`EnvGroupBuilder` instances — rollouts happen
    on-policy with the current model, so no rollouts are wasted.

    **Deterministic schedule.** The full batch schedule is determined
    by ``seed`` alone. The same ``(seed, batch_index)`` always produces
    the same groups, enabling checkpoint-based recovery: after a crash,
    training resumes at the next batch index with exactly the same
    problem assignment.

    **Shuffled cycling.** Within each source, groups are visited in a
    shuffled permutation. When all groups have been used (one cycle),
    a fresh permutation is generated so the next pass sees problems
    in a different order. This is important because RL rollouts are
    always on-policy — the same problem produces different rollouts
    as the model improves, and reshuffling ensures diverse batches.

    **Group-level blending.** Each :class:`EnvGroupBuilder` in a batch
    comes from exactly one source. This preserves GRPO's within-group
    advantage centering, which requires all rollouts in a group to
    share the same problem. Sources can have different numbers of
    groups per batch — the interleaved dataset addresses groups
    individually, not by source batch.

    **Per-domain metrics.** The existing :meth:`EnvGroupBuilder.logging_tags`
    mechanism works automatically — if your source builders return tags
    like ``["mcqa"]`` and ``["workbench"]``, training logs will show
    per-domain metrics (e.g., ``env/mcqa/reward``, ``env/workbench/reward``)
    in Weights & Biases.

    Args:
        sources: :class:`RLDatasetBuilder` instances to blend. Each is
            built concurrently at the start of training.
        weights: Relative weight per source, controlling the expected
            fraction of group slots allocated to each source. Weights
            are auto-normalized to sum to 1 (e.g., ``[55, 30, 15]`` and
            ``[0.55, 0.30, 0.15]`` are equivalent). All weights must be
            positive.
        groups_per_batch: Number of :class:`EnvGroupBuilder` instances
            (problems) per output batch. This controls how many groups
            are in each batch, not how many rollouts per group — each
            source's ``EnvGroupBuilder`` independently determines its
            own ``group_size`` (number of rollouts per problem). Sources
            with different groups-per-batch compose correctly.
        total_batches: Fixed number of training batches. With cycling,
            sources wrap around when exhausted. If ``None``, training
            runs until the smallest source is exhausted (no cycling).
        seed: Random seed for schedule generation. Changing the seed
            changes both the source assignment and the within-source
            group ordering.

    Example::

        builder = InterleavedRLDatasetBuilder(
            sources=[mcqa_builder, workbench_builder, structured_output_builder],
            weights=[0.55, 0.30, 0.15],
            groups_per_batch=8,
            total_batches=70,
        )
    """

    sources: list[RLDatasetBuilder]
    weights: list[float]
    groups_per_batch: int
    total_batches: int | None = None
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        """Build the interleaved training dataset.

        Calls each source builder concurrently via ``asyncio.gather``,
        collects the training datasets, and wraps them in an interleaved
        dataset with the configured weights and schedule.

        Returns:
            A two-element tuple of ``(train_dataset, None)``. Test datasets
            from individual sources are discarded — set up evaluation
            separately.
        """
        results = await asyncio.gather(*(source() for source in self.sources))

        train_datasets: list[RLDataset] = []
        for i, (train_ds, _test_ds) in enumerate(results):
            train_datasets.append(train_ds)
            logger.info(
                f"Source {i} ({type(self.sources[i]).__name__}): "
                f"{len(train_ds)} batches, weight={self.weights[i]}"
            )

        interleaved = _InterleavedRLDataset(
            sources=train_datasets,
            weights=self.weights,
            groups_per_batch=self.groups_per_batch,
            total_batches=self.total_batches,
            seed=self.seed,
        )

        return interleaved, None
