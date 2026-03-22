"""Optional hook for completed trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tinker_cookbook.rl.types import Trajectory


@runtime_checkable
class TrajectoryHook(Protocol):
    async def __call__(self, trajectory: Trajectory) -> None: ...
