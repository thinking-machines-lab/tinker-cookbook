"""Data source management API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from tinker_cookbook.stores import RunRegistry, storage_from_uri


class AddSourceRequest(BaseModel):
    uri: str


def create_router(registry: RunRegistry) -> APIRouter:
    router = APIRouter(prefix="/api/sources", tags=["sources"])

    @router.get("")
    def list_sources() -> list[dict[str, Any]]:
        return registry.list_sources()

    @router.post("")
    def add_source(body: AddSourceRequest) -> dict[str, Any]:
        try:
            storage = storage_from_uri(body.uri)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=f"Path not found: {exc}") from exc
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Missing dependency for cloud storage: {exc}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid URI: {exc}"
            ) from exc

        runs = registry.add_storage(storage)
        return {"sources": registry.list_sources(), "runs_discovered": len(runs)}

    @router.delete("")
    def remove_source(url: str = Query(..., description="URL of the source to remove")) -> dict[str, Any]:
        runs = registry.remove_storage(url)
        return {"sources": registry.list_sources(), "runs_remaining": len(runs)}

    @router.post("/refresh")
    def refresh_sources() -> dict[str, int]:
        runs = registry.refresh()
        return {"runs": len(runs)}

    return router
