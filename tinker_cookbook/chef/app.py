"""FastAPI application factory for Tinker Chef."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tinker_cookbook.chef.routes import chat, eval as eval_routes
from tinker_cookbook.chef.routes import metrics, rollouts, runs, sources, timing
from tinker_cookbook.chef.routes._registry_cache import get_registry, init_default
from tinker_cookbook.stores import LocalStorage, RunRegistry
from tinker_cookbook.stores.storage import Storage

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    root: str | Path | Sequence[str | Path],
    registry: RunRegistry | None = None,
) -> FastAPI:
    """Create and configure the Tinker Chef FastAPI application.

    Args:
        root: One or more directories to scan for training runs.
        registry: Pre-built RunRegistry. If None, creates one from root paths.
    """
    if registry is not None:
        # Wrap a pre-built registry: store its storages and init the cache
        storages: list[Storage] = registry._storages
    else:
        if isinstance(root, (str, Path)):
            roots = [Path(root).resolve()]
        else:
            roots = [Path(r).resolve() for r in root]
        storages = []
        for r in roots:
            if not r.is_dir():
                raise FileNotFoundError(f"Directory does not exist: {r}")
            storages.append(LocalStorage(r))

    # Initialize the registry cache eagerly so routes work even without
    # lifespan (e.g. in TestClient without context-manager usage).
    default_reg = init_default(storages)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        run_list = default_reg.get_runs()
        logger.info(
            "Tinker Chef started -- %d run(s) from %d source(s)",
            len(run_list), default_reg.storage_count,
        )
        for run in run_list:
            logger.info("  Run '%s': %d iterations", run.run_id, run.iteration_count)
        yield

    app = FastAPI(
        title="Tinker Chef",
        description="Training visualization dashboard for tinker-cookbook",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(runs.create_router(get_registry))
    app.include_router(metrics.create_router(get_registry))
    app.include_router(rollouts.create_router(get_registry))
    app.include_router(timing.create_router(get_registry))
    app.include_router(eval_routes.create_router(get_registry))
    app.include_router(chat.create_router(get_registry))
    app.include_router(sources.create_router())

    if (_STATIC_DIR / "index.html").exists():
        assets_dir = _STATIC_DIR / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            static_file = (_STATIC_DIR / full_path).resolve()
            # Prevent path traversal -- only serve files under static dir
            if (
                static_file.is_file()
                and static_file.is_relative_to(_STATIC_DIR.resolve())
                and not full_path.startswith("api/")
            ):
                return FileResponse(str(static_file))
            return FileResponse(str(_STATIC_DIR / "index.html"))
    else:
        @app.get("/")
        async def no_frontend() -> dict[str, str]:
            return {"message": "Tinker Chef API is running.", "api_docs": "/docs"}

    return app
