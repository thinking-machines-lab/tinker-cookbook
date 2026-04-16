"""Timing data API routes."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Query

from tinker_cookbook.chef.routes._helpers import require_run
from tinker_cookbook.stores import RunRegistry


def create_router(resolve_registry: Callable[..., RunRegistry]) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["timing"])

    @router.get("/{run_id}/timing")
    def get_timing(
        run_id: str,
        step_start: int | None = Query(None),
        step_end: int | None = Query(None),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        records = registry.get_training_store(run_id).read_timing()
        if step_start is not None:
            records = [r for r in records if r.get("step", 0) >= step_start]
        if step_end is not None:
            records = [r for r in records if r.get("step", 0) <= step_end]
        return {"run_id": run_id, "total_records": len(records), "records": records}

    @router.get("/{run_id}/timing/flat")
    def get_timing_flat(
        run_id: str,
        step_start: int | None = Query(None),
        step_end: int | None = Query(None),
        source: list[str] = Query(default=[]),
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        store = registry.get_training_store(run_id)
        spans = store.flatten_timing_spans(step_start=step_start, step_end=step_end)
        return {"run_id": run_id, "total_spans": len(spans), "spans": spans}

    @router.get("/{run_id}/timing/concurrency/{step}")
    def get_concurrency(
        run_id: str, step: int, source: list[str] = Query(default=[])
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        store = registry.get_training_store(run_id)
        spans = store.flatten_timing_spans(step=step)
        if not spans:
            return {"step": step, "spans": [], "max_concurrency": 0, "timeline": []}
        sorted_spans = sorted(spans, key=lambda s: s.get("wall_start", 0))
        events: list[tuple[float, int]] = []
        for s in sorted_spans:
            ws = s.get("wall_start", 0)
            we = s.get("wall_end", ws + s.get("duration", 0))
            events.append((ws, 1))
            events.append((we, -1))
        events.sort(key=lambda e: (e[0], e[1]))
        max_c = 0
        current = 0
        timeline: list[dict[str, Any]] = []
        for t, delta in events:
            current += delta
            max_c = max(max_c, current)
            timeline.append({"time": t, "concurrency": current})
        return {
            "step": step,
            "spans": sorted_spans,
            "max_concurrency": max_c,
            "timeline": timeline,
        }

    @router.get("/{run_id}/timing/tree/{step}")
    def get_timing_tree(
        run_id: str, step: int, source: list[str] = Query(default=[])
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        return registry.get_training_store(run_id).build_timing_tree(step)

    return router
