"""Metrics API routes including SSE streaming."""

import asyncio
import json
import logging
from fnmatch import fnmatch
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from tinker_cookbook.stores import RunRegistry

logger = logging.getLogger(__name__)


def create_router(registry: RunRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/runs", tags=["metrics"])

    @router.get("/{run_id}/metrics")
    async def get_metrics(
        run_id: str,
        keys: str | None = Query(None),
    ) -> dict[str, Any]:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        store = registry.get_training_store(run_id)
        records = store.read_metrics()
        if keys:
            patterns = [p.strip() for p in keys.split(",")]
            records = [_filter_record(r, patterns) for r in records]
        return {"run_id": run_id, "total_records": len(records), "records": records}

    @router.get("/{run_id}/metrics/keys")
    async def get_metric_keys(run_id: str) -> list[str]:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        store = registry.get_training_store(run_id)
        store.read_metrics()
        return sorted(store.metric_keys())

    @router.get("/{run_id}/metrics/stream")
    async def stream_metrics(run_id: str, request: Request) -> StreamingResponse:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        store = registry.get_training_store(run_id)

        async def event_generator():
            store.read_metrics()
            idle_cycles = 0
            max_idle = 40

            while True:
                if await request.is_disconnected():
                    break
                new = store.read_new_metrics()
                if new:
                    idle_cycles = 0
                    for record in new:
                        yield f"data: {json.dumps(record)}\n\n"
                else:
                    idle_cycles += 1
                    yield ": keepalive\n\n"
                    if idle_cycles >= max_idle:
                        yield "event: timeout\ndata: {}\n\n"
                        break
                await asyncio.sleep(15)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    return router


def _filter_record(record: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for key, value in record.items():
        if key == "step":
            filtered[key] = value
        elif any(fnmatch(key, p) for p in patterns):
            filtered[key] = value
    return filtered
