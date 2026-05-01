"""Cumulative raw-row cursor shared across SFT/DPO/RL/IGPO recipes."""

from __future__ import annotations


class RawRowCursor:
    """Cumulative count of raw source rows consumed across all runs.

    Same accounting unit ``ResumeInfo.data_consumed`` carries: raw rows
    pulled from the source dataset, including drops / sample failures.
    Set ``max_rows`` to clamp (e.g. RL's pre-multiplied row list); leave
    ``None`` for cursors that grow across epochs (SFT/DPO multi-epoch).
    """

    def __init__(self, *, max_rows: int | None = None) -> None:
        self._max = max_rows
        self._value = 0

    @property
    def value(self) -> int:
        return self._value

    def resume(self, persisted: int | None, *, fallback: int = 0) -> None:
        """Set cursor from a checkpoint's ``data_consumed``, falling back to a
        step-derived count when persisted is missing or zero on a non-fresh resume.

        ``persisted=0`` with ``fallback>0`` is the legacy-checkpoint signal
        (``dataloader.json`` not yet written): take the step-derived path.
        ``persisted=0`` with ``fallback=0`` is a fresh start: trust the 0.
        """
        if persisted is not None and (persisted > 0 or fallback == 0):
            self._value = self._clamp(persisted)
            return
        self._value = self._clamp(fallback)

    def record(self, raw_rows: int) -> None:
        """Advance cursor by ``raw_rows`` (clamped). Negative is a no-op."""
        if raw_rows <= 0:
            return
        self._value = self._clamp(self._value + raw_rows)

    def _clamp(self, v: int) -> int:
        v = max(0, v)
        if self._max is not None:
            v = min(v, self._max)
        return v
