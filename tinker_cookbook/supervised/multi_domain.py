"""
Multi-domain supervised learning dataset with deterministic, resumable batch access.

Loads from multiple HuggingFace datasets with configurable mixing weights.
Uses memory-mapped Arrow datasets for O(1) random access without loading into RAM.
Supports crash recovery: get_batch(i) always returns the same data for a given (seed, epoch, i).
"""

import dataclasses
import logging
from collections.abc import Callable

import datasets
import numpy as np
import tinker

from tinker_cookbook.supervised.types import SupervisedDataset

logger = logging.getLogger(__name__)

CHUNK_SIZE = 10_000


@dataclasses.dataclass
class DomainConfig:
    """Configuration for a single data domain.

    Attributes:
        name: Human-readable domain identifier (e.g. "math", "code").
        weight: Mixing weight (relative, will be normalized).
        dataset: HuggingFace memory-mapped dataset supporting O(1) random access.
    """

    name: str
    weight: float
    dataset: datasets.Dataset


class DomainMixer:
    """Generates a deterministic schedule mapping global positions to (domain, row) pairs.

    The global sequence is divided into chunks of ``CHUNK_SIZE``. Each chunk gets its
    own PRNG seeded from ``(base_seed, epoch, chunk_index)``, so any position can be
    derived in O(CHUNK_SIZE) without replaying the entire history.

    Within each domain, row indices come from a full permutation of the domain's rows
    (seeded per-epoch). Small domains cycle: when all rows are exhausted the permutation
    is re-shuffled with a new seed and the cursor wraps.
    """

    def __init__(
        self,
        domains: list[DomainConfig],
        seed: int = 42,
    ) -> None:
        self._domains = domains
        self._base_seed = seed

        # Normalize weights (exclude zero-weight domains from schedule)
        total_weight = sum(d.weight for d in domains)
        if total_weight <= 0:
            raise ValueError("Total domain weight must be positive")
        self._weights = np.array([d.weight / total_weight for d in domains], dtype=np.float64)

        # Total rows across all non-zero-weight domains
        self._total_rows = sum(
            len(d.dataset) for d, w in zip(domains, self._weights) if w > 0
        )

        # Per-domain state, rebuilt on set_epoch
        self._epoch = 0
        self._domain_permutations: list[np.ndarray] = []
        self._domain_cursors: list[int] = []
        self._domain_cycle_counts: list[int] = []
        self._build_permutations()

    def _build_permutations(self) -> None:
        """Build shuffled index permutations for each domain using the current epoch."""
        self._domain_permutations = []
        self._domain_cursors = []
        self._domain_cycle_counts = []
        for i, domain in enumerate(self._domains):
            size = len(domain.dataset)
            if size == 0:
                self._domain_permutations.append(np.array([], dtype=np.int32))
            else:
                rng = np.random.default_rng(self._make_domain_perm_seed(i, 0))
                self._domain_permutations.append(rng.permutation(size).astype(np.int32))
            self._domain_cursors.append(0)
            self._domain_cycle_counts.append(0)

    def _make_domain_perm_seed(self, domain_idx: int, cycle: int) -> int:
        """Deterministic seed for a domain's permutation at a given cycle."""
        # Combine base_seed, epoch, domain_idx, and cycle count
        rng = np.random.default_rng([self._base_seed, self._epoch, domain_idx + 1, cycle])
        return int(rng.integers(0, 2**62))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._build_permutations()

    def _get_row_for_domain(self, domain_idx: int, cursor: int, cycle: int) -> int:
        """Get the dataset row index for a domain at the given cursor position.

        Handles cycling: if cursor >= domain size, re-derive the permutation for
        the appropriate cycle.
        """
        domain = self._domains[domain_idx]
        size = len(domain.dataset)
        if size == 0:
            raise ValueError(f"Domain '{domain.name}' has no rows")

        actual_cycle = cycle + cursor // size
        pos_in_perm = cursor % size

        rng = np.random.default_rng(self._make_domain_perm_seed(domain_idx, actual_cycle))
        perm = rng.permutation(size)
        return int(perm[pos_in_perm])

    def get_positions(self, start: int, count: int) -> list[tuple[int, int]]:
        """Return ``count`` (domain_index, row_index) pairs starting from global position ``start``.

        This replays from the beginning of the chunk containing ``start`` to build
        correct per-domain cursor state, then emits the requested positions.
        """
        if count == 0:
            return []

        end = start + count
        first_chunk = start // CHUNK_SIZE
        last_chunk = (end - 1) // CHUNK_SIZE

        # We need to track per-domain cumulative cursors from the beginning of
        # first_chunk up to ``end``.  Cursors accumulate from position 0 of first_chunk.
        # To know the cursor offset at first_chunk's start, we need to count how many
        # items each domain received in all prior chunks.
        domain_cursor_offsets = self._compute_cursor_offsets_at_chunk(first_chunk)
        domain_cursors = list(domain_cursor_offsets)

        results: list[tuple[int, int]] = []

        for chunk_idx in range(first_chunk, last_chunk + 1):
            chunk_start_pos = chunk_idx * CHUNK_SIZE
            chunk_rng = np.random.default_rng(
                [self._base_seed, self._epoch, 0xDEAD, chunk_idx]
            )

            # How many positions in this chunk?
            chunk_end_pos = min(chunk_start_pos + CHUNK_SIZE, self._total_rows)
            chunk_len = chunk_end_pos - chunk_start_pos
            if chunk_len <= 0:
                break

            # Draw domain assignments for the full chunk
            domain_assignments = chunk_rng.choice(
                len(self._domains), size=chunk_len, p=self._weights
            )

            # Iterate through positions in this chunk that overlap [start, end)
            overlap_start = max(chunk_start_pos, start) - chunk_start_pos
            overlap_end = min(chunk_end_pos, end) - chunk_start_pos

            # But we must advance cursors for positions before overlap_start too
            for pos_in_chunk in range(chunk_len):
                d = int(domain_assignments[pos_in_chunk])
                if overlap_start <= pos_in_chunk < overlap_end:
                    row = self._get_row_for_domain(d, domain_cursors[d], 0)
                    results.append((d, row))
                domain_cursors[d] += 1

        return results

    def _compute_cursor_offsets_at_chunk(self, target_chunk: int) -> list[int]:
        """Count how many items each domain received in all chunks before ``target_chunk``."""
        offsets = [0] * len(self._domains)
        for chunk_idx in range(target_chunk):
            chunk_start_pos = chunk_idx * CHUNK_SIZE
            chunk_end_pos = min(chunk_start_pos + CHUNK_SIZE, self._total_rows)
            chunk_len = chunk_end_pos - chunk_start_pos
            if chunk_len <= 0:
                break

            chunk_rng = np.random.default_rng(
                [self._base_seed, self._epoch, 0xDEAD, chunk_idx]
            )
            domain_assignments = chunk_rng.choice(
                len(self._domains), size=chunk_len, p=self._weights
            )
            for d in domain_assignments:
                offsets[int(d)] += 1
        return offsets

    @property
    def total_rows(self) -> int:
        return self._total_rows


class MultiDomainSupervisedDataset(SupervisedDataset):
    """A deterministic, resumable, multi-domain supervised dataset.

    Mixes rows from multiple HuggingFace datasets according to configured weights.
    ``get_batch(i)`` always returns the same data for a given ``(seed, epoch, i)``,
    enabling exact crash recovery without sequential replay.

    Args:
        domains: List of domain configurations with datasets and mixing weights.
        batch_size: Number of datums per batch.
        render_fn: Callable that converts a raw dataset row (dict) to a ``tinker.Datum``.
        seed: Base random seed for deterministic scheduling.
    """

    def __init__(
        self,
        domains: list[DomainConfig],
        batch_size: int,
        render_fn: Callable[[dict], tinker.Datum],
        seed: int = 42,
    ) -> None:
        # Filter out zero-weight domains
        active_domains = [d for d in domains if d.weight > 0 and len(d.dataset) > 0]
        if not active_domains:
            raise ValueError("At least one domain must have positive weight and non-empty dataset")

        self._domains = active_domains
        self._batch_size = batch_size
        self._render_fn = render_fn
        self._seed = seed
        self._mixer = DomainMixer(active_domains, seed=seed)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return the batch at the given index. Deterministic for a given (seed, epoch, index)."""
        start = index * self._batch_size
        positions = self._mixer.get_positions(start, self._batch_size)

        datums: list[tinker.Datum] = []
        for domain_idx, row_idx in positions:
            domain = self._domains[domain_idx]
            row = domain.dataset[row_idx]
            datums.append(self._render_fn(row))

        return datums

    def set_epoch(self, seed: int = 0) -> None:
        """Re-shuffle domain permutations for a new epoch."""
        self._mixer.set_epoch(seed)

    def __len__(self) -> int:
        """Total number of batches (based on total rows across all active domains)."""
        return self._mixer.total_rows // self._batch_size
