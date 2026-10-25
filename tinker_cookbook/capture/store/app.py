"""aiohttp application for the capture store daemon.

Endpoints:

- ``POST /ingest/wire``: ``{"rows": [...]}``, idempotent (fully-keyed dedupe).
- ``POST /ingest/annotations``: ``{"annotations": [...]}``, deduped on ``event_id``.
- ``GET /runs``: aggregate run listing (no registration).
- ``GET /runs/{run_id}/rows``: filtered, cursor-paged wire rows.
- ``GET /stream``: SSE over the shared cursor with ``id:`` cursors,
  heartbeats, and exact resume (``?cursor=`` or ``Last-Event-ID``).
- ``GET /healthz``: liveness (does not count as activity for idle shutdown).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable

from aiohttp import web

from tinker_cookbook.capture.store.db import CaptureDB

_DB_KEY = web.AppKey("capture_db", CaptureDB)
_IDENTITY_KEY: web.AppKey[dict[str, str]] = web.AppKey("daemon_identity", dict)
_ACTIVITY_KEY: web.AppKey[list[float]] = web.AppKey("last_activity", list)  # 1-element cell
# 1-element cell set by the idle monitor the moment it commits to shutdown,
# BEFORE shutdown_cb: from then until the listener actually closes, every
# request (including /touch and /healthz) gets a 503, so ensure_daemon can
# never claim a dying daemon (it treats the 503 as gone and respawns).
_SHUTTING_DOWN_KEY: web.AppKey[list[bool]] = web.AppKey("shutting_down", list)
_IDLE_TASK_KEY: web.AppKey[asyncio.Task[None]] = web.AppKey("idle_monitor_task", asyncio.Task)
_STREAM_POLL_SEC = 0.25
_MAX_PAGE_LIMIT = 1000
# SQLite binds integers as signed 64-bit; larger parsed query values would
# raise OverflowError at bind time, after this layer's error handling.
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _query_int(raw: str, key: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key!r} must be an integer") from None
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"{key!r} exceeds the signed 64-bit integer range")
    return value


_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_HEARTBEAT_SEC = 15.0


def _touch(app: web.Application) -> None:
    app[_ACTIVITY_KEY][0] = time.monotonic()


@web.middleware
async def _activity_middleware(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    if request.app[_SHUTTING_DOWN_KEY][0]:
        return web.json_response({"error": "daemon is shutting down"}, status=503)
    if request.path != "/healthz":
        _touch(request.app)
    return await handler(request)


async def _touch_endpoint(request: web.Request) -> web.Response:
    """Activity-counting no-op: resets the idle timer (via the middleware).

    ``/healthz`` deliberately does NOT count as activity, so a client about
    to reuse a near-idle daemon claims it with this endpoint instead
    (``ensure_daemon``); a health probe alone must never keep a daemon alive.
    """
    return web.json_response({"status": "ok"})


async def _healthz(request: web.Request) -> web.Response:
    identity = request.app.get(_IDENTITY_KEY, {})
    return web.json_response({"status": "ok", "pid": os.getpid(), **identity})


async def _ingest_batch(request: web.Request, key: str) -> web.Response:
    """Shared ingest handler: validate the payload SHAPE up front (valid JSON
    with the wrong shape, e.g. a mapping instead of a list or a null entry,
    would otherwise surface as an AttributeError 500 deep in the DB), then
    translate per-row validation failures into a JSON 400. The DB rolls the
    whole batch back on failure."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return web.json_response({"error": "request body must be valid JSON"}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "request body must be a JSON object"}, status=400)
    if key not in body:
        # Defaulting a missing/misspelled key to an empty batch would report
        # a silent success while the caller loses data.
        return web.json_response({"error": f"request body must contain {key!r}"}, status=400)
    items = body[key]
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return web.json_response({"error": f"{key!r} must be a list of JSON objects"}, status=400)
    db = request.app[_DB_KEY]
    try:
        if key == "rows":
            result = await asyncio.to_thread(db.ingest_wire, items)
        else:
            result = await asyncio.to_thread(db.ingest_annotations, items)
    except (ValueError, TypeError) as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"inserted": result.inserted, "deduped": result.deduped})


async def _ingest_wire(request: web.Request) -> web.Response:
    return await _ingest_batch(request, "rows")


async def _ingest_annotations(request: web.Request) -> web.Response:
    return await _ingest_batch(request, "annotations")


async def _list_runs(request: web.Request) -> web.Response:
    runs = await asyncio.to_thread(request.app[_DB_KEY].list_runs)
    return web.json_response({"runs": [r.to_dict() for r in runs]})


async def _get_rows(request: web.Request) -> web.Response:
    run_id = request.match_info["run_id"]
    query = request.query
    filters: dict[str, str | int] = {}
    try:
        for key in ("split", "purpose"):
            if key in query:
                filters[key] = query[key]
        for key in ("iteration", "group_idx", "traj_idx"):
            if key in query:
                filters[key] = _query_int(query[key], key)
        cursor = _query_int(query.get("cursor", "0"), "cursor")
        limit = _query_int(query.get("limit", str(_MAX_PAGE_LIMIT)), "limit")
    except ValueError as e:
        return web.json_response({"error": f"invalid query parameter: {e}"}, status=400)
    if limit < 1:
        # LIMIT -1 is "no limit" in SQLite; a malformed query must not be
        # able to bypass pagination and serialize an entire run.
        return web.json_response({"error": "'limit' must be a positive integer"}, status=400)
    limit = min(limit, _MAX_PAGE_LIMIT)
    rows = await asyncio.to_thread(
        lambda: request.app[_DB_KEY].query_rows(run_id, filters=filters, cursor=cursor, limit=limit)
    )
    next_cursor = rows[-1]["cursor"] if rows else None
    return web.json_response({"rows": rows, "next_cursor": next_cursor})


async def _get_annotations(request: web.Request) -> web.Response:
    run_id = request.match_info["run_id"]
    query = request.query
    kind = query.get("kind")
    try:
        cursor = _query_int(query.get("cursor", "0"), "cursor")
        limit = _query_int(query.get("limit", str(_MAX_PAGE_LIMIT)), "limit")
    except ValueError as e:
        return web.json_response({"error": f"invalid query parameter: {e}"}, status=400)
    if limit < 1:
        return web.json_response({"error": "'limit' must be a positive integer"}, status=400)
    limit = min(limit, _MAX_PAGE_LIMIT)
    annotations = await asyncio.to_thread(
        lambda: request.app[_DB_KEY].query_annotations(
            run_id, kind=kind, cursor=cursor, limit=limit
        )
    )
    next_cursor = annotations[-1]["cursor"] if annotations else None
    return web.json_response({"annotations": annotations, "next_cursor": next_cursor})


async def _stream(request: web.Request) -> web.StreamResponse:
    run_id = request.query.get("run_id")
    cursor_param = request.query.get("cursor") or request.headers.get("Last-Event-ID")
    try:
        cursor = _query_int(cursor_param, "cursor") if cursor_param else 0
    except ValueError as e:
        # The cursor can arrive via either carrier; do not blame the wrong one.
        return web.json_response(
            {"error": f"{e} (query parameter or Last-Event-ID header)"}, status=400
        )

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
    await response.prepare(request)
    db = request.app[_DB_KEY]
    last_heartbeat = time.monotonic()
    try:
        while True:
            events = await asyncio.to_thread(db.stream_events, run_id, cursor)
            for event in events:
                payload = json.dumps(event.data, default=str)
                await response.write(
                    f"id: {event.cursor}\nevent: {event.event_type}\ndata: {payload}\n\n".encode()
                )
                cursor = event.cursor
            _touch(request.app)
            now = time.monotonic()
            if not events and now - last_heartbeat >= _HEARTBEAT_SEC:
                await response.write(b": heartbeat\n\n")
                last_heartbeat = now
            await asyncio.sleep(_STREAM_POLL_SEC)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def _idle_monitor(
    app: web.Application, idle_shutdown_sec: float, shutdown_cb: Callable[[], None]
) -> None:
    poll = min(max(idle_shutdown_sec / 10.0, 0.05), 10.0)
    while True:
        await asyncio.sleep(poll)
        if time.monotonic() - app[_ACTIVITY_KEY][0] > idle_shutdown_sec:
            # Flip the flag before signaling shutdown: both run on the app's
            # event loop, so no request handler can interleave between the
            # flag and the 503 behavior it enables.
            app[_SHUTTING_DOWN_KEY][0] = True
            shutdown_cb()
            return


def make_app(
    db: CaptureDB,
    *,
    idle_shutdown_sec: float | None = None,
    shutdown_cb: Callable[[], None] | None = None,
    identity: dict[str, str] | None = None,
) -> web.Application:
    """Build the capture store aiohttp application.

    Args:
        db: The backing :class:`CaptureDB`.
        idle_shutdown_sec: If set (with ``shutdown_cb``), call ``shutdown_cb``
            once no non-healthz request has been seen for this many seconds.
        shutdown_cb: Callback invoked by the idle monitor.
        identity: Extra fields merged into the ``/healthz`` response (the
            daemon's canonical ``data_dir`` and random ``instance_token``) so
            clients can verify they are talking to the daemon they expect
            rather than an unrelated one that reused the port.
    """
    # aiohttp's default request-body cap is 1 MiB; a single long-context
    # sample (prompt_tokens repeated per returned sequence) or a 256-record
    # batch easily exceeds that and would be 413'd before the handler runs.
    # The client also splits oversized payloads (see client._MAX_POST_BYTES),
    # but be generous here so one huge record still fits.
    app = web.Application(middlewares=[_activity_middleware], client_max_size=_MAX_REQUEST_BYTES)
    app[_DB_KEY] = db
    app[_IDENTITY_KEY] = identity or {}
    app[_ACTIVITY_KEY] = [time.monotonic()]
    app[_SHUTTING_DOWN_KEY] = [False]
    app.router.add_get("/healthz", _healthz)
    app.router.add_get("/touch", _touch_endpoint)
    app.router.add_post("/ingest/wire", _ingest_wire)
    app.router.add_post("/ingest/annotations", _ingest_annotations)
    app.router.add_get("/runs", _list_runs)
    app.router.add_get("/runs/{run_id}/rows", _get_rows)
    app.router.add_get("/runs/{run_id}/annotations", _get_annotations)
    app.router.add_get("/stream", _stream)

    if idle_shutdown_sec is not None and shutdown_cb is not None:

        async def _start_idle_monitor(app: web.Application) -> None:
            task = asyncio.create_task(_idle_monitor(app, idle_shutdown_sec, shutdown_cb))
            app[_IDLE_TASK_KEY] = task

        async def _stop_idle_monitor(app: web.Application) -> None:
            app[_IDLE_TASK_KEY].cancel()

        app.on_startup.append(_start_idle_monitor)
        app.on_cleanup.append(_stop_idle_monitor)

    return app
