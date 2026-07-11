"""Local viewer server for the token DB.

A thin aiohttp layer over :class:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader`:
the HTTP API exposes the reader's structured methods (query / get_rollout /
search / sql / labels), a websocket pushes newly written rows via
``reader.subscribe()``, and the built Vite UI is served from
``tokendb/static/`` (see ``tokendb/ui/README.md`` for the frontend dev loop).

Run against a training run's log directory::

    python -m tinker_cookbook.tokendb.serve log_path=~/runs/my-run port=7423

Binds 127.0.0.1 by default; this is a local, unauthenticated viewer.

The server best-effort loads the run's tokenizer (from ``run.json``'s
``model_name``) so the UI can render per-token spans. If the tokenizer can't
load, ``/api/tokens/decode`` returns 503 and the UI falls back to whole-turn
text plus raw token IDs (both stored, so nothing is lost).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chz
from aiohttp import WSMsgType, web

from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.tokendb.reader import (
    LABELS_PATH,
    ParquetSegmentReader,
    reconstruct_full_ob,
)
from tinker_cookbook.tokendb.writer import RUN_JSON_PATH

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7423
STATIC_DIR = Path(__file__).parent / "static"

_json_dumps = functools.partial(json.dumps, default=str)

# Typed application-state keys (aiohttp's recommended alternative to str keys).
LOG_PATH_KEY = web.AppKey("log_path", str)
STORAGE_KEY: web.AppKey[Any] = web.AppKey("storage")
READER_KEY = web.AppKey("reader", ParquetSegmentReader)
TOKENIZER_KEY: web.AppKey[Any] = web.AppKey("tokenizer")
TOKENIZER_ERROR_KEY = web.AppKey("tokenizer_error", str)
TOKENIZER_TASK_KEY: web.AppKey[Any] = web.AppKey("tokenizer_task")


def _json_response(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=_json_dumps)


def _error_response(message: str, status: int) -> web.Response:
    return _json_response({"error": message}, status=status)


# Query/filter parameter types, mirroring ParquetSegmentReader._build_where.
_ROLLOUT_FILTER_TYPES: dict[str, type] = {
    "run_id": str,
    "run_attempt": int,
    "split": str,
    "iteration": int,
    "min_iteration": int,
    "max_iteration": int,
    "group_idx": int,
    "traj_idx": int,
    "min_reward": float,
    "max_reward": float,
    "stop_reason": str,
    "source": str,
    "filtered_reason": str,
    "tag": str,
    "env_row_id": str,
    "text_regex": str,
}

_LABEL_FILTER_TYPES: dict[str, type] = {
    "run_id": str,
    "split": str,
    "iteration": int,
    "group_idx": int,
    "traj_idx": int,
    "step_idx": int,
    "label_key": str,
    "author": str,
}

# Trajectory-grain aggregation over the row-grain `rollouts` view. Matches the
# reader's `trajectories` view but keeps the browsing columns the feed needs
# (tags, ac preview, superseded).
_TRAJECTORIES_SQL = """
    WITH filtered AS (SELECT * FROM rollouts {where})
    SELECT run_id, run_attempt, split, iteration, group_idx, traj_idx,
           count(*) AS n_steps,
           sum(len(ac_tokens))::BIGINT AS n_ac_tokens,
           any_value(total_reward) AS total_reward,
           any_value(final_reward) AS final_reward,
           arg_max(stop_reason, step_idx) AS stop_reason,
           any_value(filtered_reason) AS filtered_reason,
           any_value(env_row_id) AS env_row_id,
           any_value(tags) AS tags,
           any_value(source) AS source,
           any_value(sampling_client_step) AS sampling_client_step,
           arg_max(ac_text, step_idx) AS ac_preview,
           bool_or(superseded) AS superseded,
           min(ts) AS ts
    FROM filtered
    GROUP BY run_id, run_attempt, split, iteration, group_idx, traj_idx
    ORDER BY iteration DESC, ts DESC, group_idx, traj_idx
    LIMIT {limit} OFFSET {offset}
"""

_FALLBACK_PAGE = """<!doctype html>
<html><head><title>Token DB viewer</title></head>
<body style="font-family: system-ui; max-width: 40em; margin: 4em auto; color: #ddd; background: #1a1a1e">
<h1>Token DB viewer</h1>
<p>The UI has not been built: <code>tinker_cookbook/tokendb/static/</code> is missing.</p>
<p>Build it from the frontend source (see <code>tinker_cookbook/tokendb/ui/README.md</code>):</p>
<pre>cd tinker_cookbook/tokendb/ui
npm install
npm run build</pre>
<p>The HTTP API is up regardless; try <a href="/api/run" style="color:#8bf">/api/run</a>.</p>
</body></html>"""


def _parse_filters(query: Mapping[str, str], types: Mapping[str, type]) -> dict[str, Any]:
    """Coerce known query params to their filter types; raises ValueError on bad values."""
    out: dict[str, Any] = {}
    for name, typ in types.items():
        if name in query and query[name] != "":
            try:
                out[name] = typ(query[name])
            except (TypeError, ValueError) as e:
                raise ValueError(f"Bad value for filter {name!r}: {query[name]!r}") from e
    return out


def _coerce_body_filters(filters: Any) -> dict[str, Any]:
    """Validate a JSON-body filter dict against the known rollout filters."""
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object")
    bad = set(filters) - set(_ROLLOUT_FILTER_TYPES)
    if bad:
        raise ValueError(f"Unsupported filters: {sorted(bad)}")
    return _parse_filters({k: str(v) for k, v in filters.items()}, _ROLLOUT_FILTER_TYPES)


def _flag(query: Mapping[str, str], name: str, default: bool = False) -> bool:
    if name not in query:
        return default
    return query[name].lower() in ("1", "true", "yes")


# --- Handlers ---


async def _handle_run(request: web.Request) -> web.Response:
    storage = request.app[STORAGE_KEY]
    if not storage.exists(RUN_JSON_PATH):
        return _error_response(f"{RUN_JSON_PATH} not found under this log_path", 404)
    return _json_response(json.loads(storage.read(RUN_JSON_PATH).decode()))


async def _handle_rollouts(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    q = request.rel_url.query
    try:
        filters = _parse_filters(q, _ROLLOUT_FILTER_TYPES)
        grain = q.get("grain", "trajectories")
        latest_only = _flag(q, "latest_only")
        limit = int(q.get("limit", "500"))
        offset = int(q.get("offset", "0"))
        if grain == "rollouts":
            rows = reader.query(latest_only=latest_only, limit=limit, offset=offset, **filters)
        elif grain == "trajectories":
            # Reuse the reader's filter builder so query semantics match
            # /api/rollouts?grain=rollouts exactly (internal to this package).
            where, params = reader._build_where(**filters)
            if latest_only:
                where = f"{where} AND NOT superseded" if where else "WHERE NOT superseded"
            sql = _TRAJECTORIES_SQL.format(where=where, limit=int(limit), offset=int(offset))
            rows = reader.sql(sql, params)
        else:
            raise ValueError(f"Unknown grain {grain!r} (expected 'trajectories' or 'rollouts')")
    except ValueError as e:
        return _error_response(str(e), 400)
    return _json_response({"grain": grain, "rows": rows})


async def _handle_rollout_detail(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    try:
        split = request.match_info["split"]
        iteration = int(request.match_info["iteration"])
        group_idx = int(request.match_info["group_idx"])
        traj_idx = int(request.match_info["traj_idx"])
        run_attempt_raw = request.rel_url.query.get("run_attempt")
        run_attempt = int(run_attempt_raw) if run_attempt_raw else None
    except ValueError as e:
        return _error_response(str(e), 400)
    steps = reader.get_rollout(split, iteration, group_idx, traj_idx, run_attempt)
    if not steps:
        return _error_response("rollout not found", 404)
    for step, full_ob in zip(steps, reconstruct_full_ob(steps)):
        step["ob_full_tokens"] = full_ob
    tokenizer = request.app.get(TOKENIZER_KEY)
    if tokenizer is not None:
        for step in steps:
            step["ac_token_strs"] = [tokenizer.decode([t]) for t in step["ac_tokens"]]
    labels = reader.labels(split=split, iteration=iteration, group_idx=group_idx, traj_idx=traj_idx)
    siblings = reader.sql(
        "SELECT DISTINCT traj_idx FROM rollouts"
        " WHERE split = ? AND iteration = ? AND group_idx = ? ORDER BY traj_idx",
        [split, iteration, group_idx],
    )
    return _json_response(
        {
            "steps": steps,
            "labels": labels,
            "group_traj_idxs": [row["traj_idx"] for row in siblings],
        }
    )


async def _handle_search(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    try:
        body = await request.json()
        filters = _coerce_body_filters(body.get("filters"))
        kwargs: dict[str, Any] = dict(filters)
        if body.get("regex"):
            kwargs["regex"] = str(body["regex"])
        if body.get("fields"):
            kwargs["fields"] = [str(f) for f in body["fields"]]
        if body.get("token_subsequence"):
            kwargs["token_subsequence"] = [int(t) for t in body["token_subsequence"]]
        if "regex" not in kwargs and "token_subsequence" not in kwargs:
            raise ValueError("search needs a regex and/or a token_subsequence")
        rows = reader.search(limit=int(body.get("limit", 200)), **kwargs)
        # Hit counts over the same (un-limited) search, grouped by iteration.
        hit_counts = reader.search_hit_counts(**kwargs)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _error_response(str(e), 400)
    except Exception as e:  # e.g. a bad regex surfacing from DuckDB
        return _error_response(str(e), 400)
    return _json_response(
        {"rows": rows, "hit_counts": {str(k): v for k, v in sorted(hit_counts.items())}}
    )


async def _handle_sql(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    try:
        body = await request.json()
        query = body.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("body must include a 'query' string")
        rows = reader.sql(query, body.get("params"))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _error_response(str(e), 400)
    except Exception as e:  # DuckDB parse/binder errors on user SQL
        return _error_response(str(e), 400)
    return _json_response({"rows": rows})


async def _handle_labels_get(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    try:
        filters = _parse_filters(request.rel_url.query, _LABEL_FILTER_TYPES)
        rows = reader.labels(**filters)
    except ValueError as e:
        return _error_response(str(e), 400)
    return _json_response({"labels": rows})


async def _handle_labels_post(request: web.Request) -> web.Response:
    reader: ParquetSegmentReader = request.app[READER_KEY]
    try:
        body = await request.json()
        key = body.get("key")
        label_key = body.get("label_key")
        author = body.get("author")
        if not isinstance(key, dict) or not label_key or not author:
            raise ValueError("body must include 'key' (object), 'label_key', and 'author'")
        reader.add_label(
            key,
            str(label_key),
            body.get("label_value"),
            author=str(author),
            note=body.get("note"),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _error_response(str(e), 400)
    return _json_response({"ok": True})


async def _handle_decode(request: web.Request) -> web.Response:
    tokenizer = request.app.get(TOKENIZER_KEY)
    if tokenizer is None:
        return _error_response(
            f"tokenizer unavailable: {request.app.get(TOKENIZER_ERROR_KEY, 'not loaded')}", 503
        )
    try:
        body = await request.json()
        tokens = [int(t) for t in body.get("tokens", [])]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _error_response(str(e), 400)
    return _json_response({"strs": [tokenizer.decode([t]) for t in tokens]})


# --- Websocket: per-client filter subscription + label-change pings ---


async def _push_rows(
    app: web.Application, ws: web.WebSocketResponse, filters: dict[str, Any], poll_interval_s: float
) -> None:
    # A fresh reader per subscription: the shared reader's connection is not
    # safe to tail from multiple subscribers (each tracks its own segment set).
    reader = ParquetSegmentReader(app[LOG_PATH_KEY])
    async for row in reader.subscribe(poll_interval_s=poll_interval_s, **filters):
        await ws.send_str(_json_dumps({"type": "row", "row": row}))


async def _push_label_updates(
    app: web.Application, ws: web.WebSocketResponse, poll_interval_s: float
) -> None:
    """Ping the client when labels.jsonl changes (agents write labels out-of-band)."""
    storage = app[STORAGE_KEY]
    last = storage.stat(LABELS_PATH)
    while True:
        await asyncio.sleep(poll_interval_s)
        current = storage.stat(LABELS_PATH)
        if current is not None and (last is None or current.size != last.size):
            await ws.send_str(_json_dumps({"type": "labels_changed"}))
        last = current


async def _handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    tasks: list[asyncio.Task[None]] = []

    def _cancel_tasks() -> None:
        for task in tasks:
            task.cancel()
        tasks.clear()

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
                if data.get("type") != "subscribe":
                    raise ValueError(f"Unknown message type: {data.get('type')!r}")
                filters = _coerce_body_filters(data.get("filters"))
                poll_interval_s = float(data.get("poll_interval_s", 2.0))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                await ws.send_str(_json_dumps({"type": "error", "error": str(e)}))
                continue
            _cancel_tasks()  # a new subscription replaces the previous one
            tasks = [
                asyncio.create_task(_push_rows(request.app, ws, filters, poll_interval_s)),
                asyncio.create_task(_push_label_updates(request.app, ws, poll_interval_s)),
            ]
            await ws.send_str(_json_dumps({"type": "subscribed", "filters": filters}))
    finally:
        _cancel_tasks()
    return ws


# --- Tokenizer (best-effort, background) ---


async def _load_tokenizer_task(app: web.Application) -> None:
    storage = app[STORAGE_KEY]
    try:
        if not storage.exists(RUN_JSON_PATH):
            app[TOKENIZER_ERROR_KEY] = f"{RUN_JSON_PATH} not found"
            return
        run_info = json.loads(storage.read(RUN_JSON_PATH).decode())
        model_name = (run_info.get("context") or {}).get("model_name")
        if not model_name:
            app[TOKENIZER_ERROR_KEY] = "run.json has no model_name"
            return
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        app[TOKENIZER_KEY] = await asyncio.to_thread(get_tokenizer, model_name)
        logger.info("Loaded tokenizer for %s", model_name)
    except Exception as e:
        # Best-effort: the UI falls back to whole-turn text + raw token IDs.
        app[TOKENIZER_ERROR_KEY] = str(e)
        logger.warning("Could not load tokenizer (per-token view degraded): %s", e)


async def _start_tokenizer_load(app: web.Application) -> None:
    app[TOKENIZER_TASK_KEY] = asyncio.create_task(_load_tokenizer_task(app))


# --- App assembly ---


def build_app(
    log_path: str | Path,
    *,
    static_dir: Path | None = STATIC_DIR,
    load_tokenizer: bool = True,
) -> web.Application:
    """Build the viewer application for one run's ``log_path``.

    Args:
        log_path: The training run's log directory (local path or cloud URI).
        static_dir: Directory with the built UI; a fallback page is served
            when it's missing (e.g. the frontend hasn't been built).
        load_tokenizer: Kick off the best-effort background tokenizer load on
            startup (disable in tests to avoid network access).
    """
    app = web.Application()
    app[LOG_PATH_KEY] = str(log_path)
    app[STORAGE_KEY] = storage_from_uri(str(log_path))
    app[READER_KEY] = ParquetSegmentReader(app[STORAGE_KEY])
    app[TOKENIZER_KEY] = None
    app[TOKENIZER_ERROR_KEY] = "not loaded"

    app.router.add_get("/api/run", _handle_run)
    app.router.add_get("/api/rollouts", _handle_rollouts)
    app.router.add_get(
        "/api/rollout/{split}/{iteration}/{group_idx}/{traj_idx}", _handle_rollout_detail
    )
    app.router.add_post("/api/search", _handle_search)
    app.router.add_post("/api/sql", _handle_sql)
    app.router.add_get("/api/labels", _handle_labels_get)
    app.router.add_post("/api/labels", _handle_labels_post)
    app.router.add_post("/api/tokens/decode", _handle_decode)
    app.router.add_get("/ws", _handle_ws)

    if static_dir is not None and (static_dir / "index.html").is_file():
        index_path = static_dir / "index.html"

        async def _index(request: web.Request) -> web.FileResponse:
            return web.FileResponse(index_path)

        app.router.add_get("/", _index)
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.router.add_static("/assets", assets_dir)
    else:

        async def _fallback(request: web.Request) -> web.Response:
            return web.Response(text=_FALLBACK_PAGE, content_type="text/html")

        app.router.add_get("/", _fallback)

    if load_tokenizer:
        app.on_startup.append(_start_tokenizer_load)
    return app


@chz.chz
class Config:
    """CLI config: ``python -m tinker_cookbook.tokendb.serve log_path=... port=7423``."""

    log_path: str
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"  # local viewer; not meant to be exposed


def run(config: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    app = build_app(config.log_path)
    logger.info(
        "Token DB viewer on http://%s:%d (log_path=%s)", config.host, config.port, config.log_path
    )
    web.run_app(app, host=config.host, port=config.port)


if __name__ == "__main__":
    chz.nested_entrypoint(run)
