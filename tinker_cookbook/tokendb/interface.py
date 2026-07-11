"""Backend protocol for token stores.

Both halves of the token DB live behind a single :class:`TokenStoreBackend`
protocol so the capture layer, viewer, and agent API never know what's
underneath. :class:`~tinker_cookbook.tokendb.writer.ParquetSegmentBackend` is
the default implementation (parquet segments through the ``Storage``
protocol); a hosted backend can implement the same protocol later with no
changes to callers.

Raw SQL is deliberately not part of the protocol — it stays a
backend-specific escape hatch, and the portable surface is the structured
methods.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from tinker_cookbook.tokendb.schema import TokenRow


@runtime_checkable
class TokenWriter(Protocol):
    """Write handle for a token store: buffered row appends with flush/close."""

    def append_rows(self, rows: Sequence[TokenRow]) -> None:
        """Buffer rows for writing. May trigger a flush when the buffer fills."""
        ...

    def flush(self) -> None:
        """Persist all buffered rows. Safe to call concurrently; no-op if empty."""
        ...

    def close(self) -> None:
        """Flush remaining rows and release resources. Idempotent."""
        ...


class TokenStoreBackend(Protocol):
    """Storage-agnostic interface to a token store (write + read halves)."""

    def open_writer(self, run_context: Mapping[str, Any]) -> TokenWriter:
        """Open a writer for this store.

        Args:
            run_context: Identity and capture config recorded with the run
                (e.g. ``model_name``, ``recipe_name``). Workers on other
                processes/hosts pass explicit ``run_id`` / ``run_attempt``
                here; the coordinator owns ``run.json``.
        """
        ...

    # --- Read half. ---

    def query(self, **filters: Any) -> Any:
        """Structured row query (split, iteration range, tags, reward range, ...)."""
        ...

    def get_rollout(
        self,
        split: str,
        iteration: int,
        group_idx: int,
        traj_idx: int,
        run_attempt: int | None = None,
    ) -> Any:
        """Fetch all rows (turns) for one trajectory (latest attempt by default)."""
        ...

    def search(self, **kwargs: Any) -> Any:
        """Regex search over text columns, and/or token-ID-subsequence match."""
        ...

    def subscribe(self, **filters: Any) -> AsyncIterator[Any]:
        """Async iterator over newly written rows matching the filters."""
        ...

    def add_label(
        self,
        key: Mapping[str, Any],
        label_key: str,
        label_value: Any,
        *,
        author: str,
        note: str | None = None,
    ) -> None:
        """Append an annotation for a rollout (last-write-wins by timestamp)."""
        ...

    def labels(self, **filters: Any) -> Any:
        """Fetch annotations matching the filters."""
        ...
