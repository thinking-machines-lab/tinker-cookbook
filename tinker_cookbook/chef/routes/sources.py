"""Data source management API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from tinker_cookbook.chef.routes._registry_cache import (
    get_default_sources,
    refresh_registry,
)


def create_router() -> APIRouter:
    router = APIRouter(prefix="/api/sources", tags=["sources"])

    @router.get("/defaults")
    def defaults() -> list[dict[str, str]]:
        return get_default_sources()

    @router.post("/refresh")
    def refresh(source: list[str] = Query(default=[])) -> dict[str, int]:
        runs = refresh_registry(source or None)
        return {"runs": runs}

    return router
