"""Shared helpers for route modules."""

from fastapi import HTTPException

from tinker_cookbook.stores import RunInfo, RunRegistry


def require_run(registry: RunRegistry, run_id: str) -> RunInfo:
    """Look up a run or raise 404."""
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run
