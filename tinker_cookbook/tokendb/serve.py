"""Local viewer server for the token DB.

A thin aiohttp layer over :class:`~tinker_cookbook.tokendb.reader.ParquetSegmentReader`:
the HTTP API exposes the reader's structured methods (query / get_rollout /
search / sql / labels), a websocket pushes newly written rows via
``reader.subscribe()``, and the built Vite UI is served from
``tokendb/static/`` (see ``tokendb/ui/README.md`` for the frontend dev loop).

Two modes:

**Single-run mode** (``log_path`` given): serve one run's store, endpoints at
their original paths (``/api/rollouts``, ``/ws``, ...)::

    python -m tinker_cookbook.tokendb.serve log_path=~/runs/my-run port=7423

**Registry mode** (no ``log_path``): serve every run registered in the local
run registry (see :mod:`tinker_cookbook.tokendb.registry`), so concurrently
running experiments show up in one dashboard. ``GET /api/runs`` lists the
registered runs with liveness, ``GET /api/dashboard`` adds per-run
aggregates, and all single-run endpoints are mounted per run under
``/api/runs/{run_id}/...`` (per-run readers are constructed lazily and
LRU-cached)::

    python -m tinker_cookbook.tokendb.serve

Binds 127.0.0.1 by default; this is a local, unauthenticated viewer.

The server best-effort loads a run's tokenizer (from ``run.json``'s
``model_name``) so the UI can render per-token spans. If the tokenizer can't
load, ``/api/tokens/decode`` returns 503 and the UI falls back to whole-turn
text plus raw token IDs (both stored, so nothing is lost).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import time
from collections import OrderedDict
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
from tinker_cookbook.tokendb.registry import (
    DEFAULT_LIVE_WINDOW_S,
    list_runs,
    load_run_record,
    resolve_registry_dir,
)
from tinker_cookbook.tokendb.writer import RUN_JSON_PATH

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7423
STATIC_DIR = Path(__file__).parent / "static"

# Registry mode: cached per-run reader contexts and dashboard aggregates.
RUN_CACHE_MAX = 32
DEFAULT_DASHBOARD_TTL_S = 10.0
DASHBOARD_RECENT_ITERATIONS = 5  # K for mean_recent_reward
REWARD_SERIES_ITERATIONS = 50  # sparkline length
DEFAULT_DASHBOARD_PUSH_INTERVAL_S = 5.0

_json_dumps = functools.partial(json.dumps, default=str)


@dataclasses.dataclass
class RunContext:
    """Everything the per-run handlers need for one run's store."""

    log_path: str
    storage: Any
    reader: ParquetSegmentReader
    tokenizer: Any = None
    tokenizer_error: str = "not loaded"
    tokenizer_task: asyncio.Task[None] | None = None
    tokenizer_started: bool = False


# Typed application-state keys (aiohttp's recommended alternative to str keys).
SINGLE_CTX_KEY: web.AppKey[RunContext | None] = web.AppKey("single_ctx")
REGISTRY_DIR_KEY: web.AppKey[str | None] = web.AppKey("registry_dir")
RUN_CACHE_KEY: web.AppKey[OrderedDict[str, RunContext]] = web.AppKey("run_cache")
DASHBOARD_CACHE_KEY: web.AppKey[dict[str, tuple[float, dict[str, Any]]]] = web.AppKey(
    "dashboard_cache"
)
DASHBOARD_TTL_KEY = web.AppKey("dashboard_ttl_s", float)
LIVE_WINDOW_KEY = web.AppKey("live_window_s", float)
LOAD_TOKENIZER_KEY = web.AppKey("load_tokenizer", bool)


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

# Dashboard aggregates (registry mode); one cheap pass per run, TTL-cached.
_DASHBOARD_TOTALS_SQL = """
    SELECT count(*) AS n_rows,
           count(*) FILTER (WHERE source = 'filtered') AS n_filtered_rows,
           max(iteration) FILTER (WHERE iteration >= 0) AS latest_iteration
    FROM rollouts
"""
# Per-iteration mean of per-trajectory total_reward, excluding filtered rows
# (their placeholder reward of 0.0 would skew the mean).
_DASHBOARD_SERIES_SQL = f"""
    SELECT iteration, avg(total_reward) AS mean_total_reward
    FROM (
        SELECT iteration, any_value(total_reward) AS total_reward
        FROM rollouts
        WHERE iteration >= 0 AND source <> 'filtered'
        GROUP BY run_attempt, split, iteration, group_idx, traj_idx
    )
    GROUP BY iteration
    ORDER BY iteration DESC
    LIMIT {REWARD_SERIES_ITERATIONS}
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
<p>The HTTP API is up regardless; try <a href="/api/run" style="color:#8bf">/api/run</a>
(single-run mode) or <a href="/api/runs" style="color:#8bf">/api/runs</a> (registry mode).</p>
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


# --- Run-context resolution (single-run vs registry mode) ---


def _make_run_ctx(log_path: str) -> RunContext:
    storage = storage_from_uri(str(log_path))
    return RunContext(log_path=str(log_path), storage=storage, reader=ParquetSegmentReader(storage))


def _get_or_create_ctx(app: web.Application, run_id: str) -> RunContext:
    """Resolve *run_id* to a cached (LRU) per-run context in registry mode.

    Raises ``web.HTTPNotFound`` (JSON body) for unknown run IDs.
    """
    cache = app[RUN_CACHE_KEY]
    ctx = cache.get(run_id)
    if ctx is not None:
        cache.move_to_end(run_id)
        return ctx
    record = load_run_record(app[REGISTRY_DIR_KEY], run_id)
    if record is None or not record.get("log_path"):
        raise web.HTTPNotFound(
            text=_json_dumps({"error": f"unknown run_id {run_id!r}"}),
            content_type="application/json",
        )
    ctx = _make_run_ctx(str(record["log_path"]))
    cache[run_id] = ctx
    while len(cache) > RUN_CACHE_MAX:
        cache.popitem(last=False)
    return ctx


def _request_ctx(request: web.Request) -> RunContext:
    """The run context for this request: the single run, or match_info's run_id."""
    ctx = request.app[SINGLE_CTX_KEY]
    if ctx is not None:
        return ctx
    return _get_or_create_ctx(request.app, request.match_info["run_id"])


def _ctx_with_tokenizer(request: web.Request) -> RunContext:
    """Like :func:`_request_ctx`, but kicks off the lazy tokenizer load.

    Registry mode loads tokenizers on demand (a dashboard listing dozens of
    runs must not trigger dozens of tokenizer downloads); single-run mode
    starts the load at startup as before.
    """
    ctx = _request_ctx(request)
    if request.app[LOAD_TOKENIZER_KEY] and not ctx.tokenizer_started:
        ctx.tokenizer_started = True
        ctx.tokenizer_task = asyncio.create_task(_load_tokenizer(ctx))
    return ctx


# --- Handlers (shared between single-run paths and /api/runs/{run_id}/...) ---


async def _handle_run(request: web.Request) -> web.Response:
    ctx = _request_ctx(request)
    if not ctx.storage.exists(RUN_JSON_PATH):
        return _error_response(f"{RUN_JSON_PATH} not found under this log_path", 404)
    return _json_response(json.loads(ctx.storage.read(RUN_JSON_PATH).decode()))


async def _handle_rollouts(request: web.Request) -> web.Response:
    reader = _request_ctx(request).reader
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
    ctx = _ctx_with_tokenizer(request)
    reader = ctx.reader
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
    if ctx.tokenizer is not None:
        for step in steps:
            step["ac_token_strs"] = [ctx.tokenizer.decode([t]) for t in step["ac_tokens"]]
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
    reader = _request_ctx(request).reader
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
    reader = _request_ctx(request).reader
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
    reader = _request_ctx(request).reader
    try:
        filters = _parse_filters(request.rel_url.query, _LABEL_FILTER_TYPES)
        rows = reader.labels(**filters)
    except ValueError as e:
        return _error_response(str(e), 400)
    return _json_response({"labels": rows})


async def _handle_labels_post(request: web.Request) -> web.Response:
    reader = _request_ctx(request).reader
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
    ctx = _ctx_with_tokenizer(request)
    if ctx.tokenizer is None:
        return _error_response(f"tokenizer unavailable: {ctx.tokenizer_error}", 503)
    try:
        body = await request.json()
        tokens = [int(t) for t in body.get("tokens", [])]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return _error_response(str(e), 400)
    return _json_response({"strs": [ctx.tokenizer.decode([t]) for t in tokens]})


# --- Registry mode: run listing + dashboard aggregates ---


async def _handle_runs(request: web.Request) -> web.Response:
    """List registered runs, newest first, with a cheap liveness/status probe."""
    runs = list_runs(request.app[REGISTRY_DIR_KEY], live_window_s=request.app[LIVE_WINDOW_KEY])
    return _json_response({"runs": runs})


def _compute_dashboard_row(record: dict[str, Any], reader: ParquetSegmentReader) -> dict[str, Any]:
    """One dashboard row: registry record + status + DuckDB aggregates."""
    status = record.get("status") or {}
    totals = reader.sql(_DASHBOARD_TOTALS_SQL)[0]
    series = reader.sql(_DASHBOARD_SERIES_SQL)
    series.reverse()  # oldest -> newest for sparklines
    recent = [
        p["mean_total_reward"]
        for p in series[-DASHBOARD_RECENT_ITERATIONS:]
        if p["mean_total_reward"] is not None
    ]
    latest_iteration = totals["latest_iteration"]
    if latest_iteration is None:
        latest_iteration = status.get("latest_iteration")
    return {
        "run_id": record.get("run_id"),
        "run_attempt": record.get("run_attempt"),
        "log_path": record.get("log_path"),
        "model_name": record.get("model_name"),
        "recipe_name": record.get("recipe_name"),
        "started_at": record.get("started_at"),
        "live": bool(status.get("live", False)),
        "last_activity_ts": status.get("last_activity_ts"),
        "latest_iteration": latest_iteration,
        "n_rows": totals["n_rows"],
        "n_filtered_rows": totals["n_filtered_rows"],
        "mean_recent_reward": (sum(recent) / len(recent)) if recent else None,
        "reward_series": series,
    }


def _dashboard_rows(app: web.Application) -> list[dict[str, Any]]:
    """Dashboard rows for every registered run, TTL-cached per run.

    The TTL cache (default 10s) lets the dashboard poll without hammering
    DuckDB; within the TTL a run's row is returned as-is, so repeated polls
    are consistent and cheap.
    """
    cache = app[DASHBOARD_CACHE_KEY]
    ttl = app[DASHBOARD_TTL_KEY]
    now = time.monotonic()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in list_runs(app[REGISTRY_DIR_KEY], live_window_s=app[LIVE_WINDOW_KEY]):
        run_id = str(record["run_id"])
        seen.add(run_id)
        cached = cache.get(run_id)
        if cached is not None and now - cached[0] <= ttl:
            rows.append(cached[1])
            continue
        try:
            ctx = _get_or_create_ctx(app, run_id)
            row = _compute_dashboard_row(record, ctx.reader)
        except web.HTTPNotFound:
            continue
        except Exception as e:
            logger.warning("Dashboard aggregation failed for run %s: %s", run_id, e)
            row = {
                "run_id": record.get("run_id"),
                "run_attempt": record.get("run_attempt"),
                "log_path": record.get("log_path"),
                "model_name": record.get("model_name"),
                "recipe_name": record.get("recipe_name"),
                "started_at": record.get("started_at"),
                "live": bool((record.get("status") or {}).get("live", False)),
                "last_activity_ts": (record.get("status") or {}).get("last_activity_ts"),
                "latest_iteration": (record.get("status") or {}).get("latest_iteration"),
                "n_rows": None,
                "n_filtered_rows": None,
                "mean_recent_reward": None,
                "reward_series": [],
                "error": str(e),
            }
        cache[run_id] = (now, row)
        rows.append(row)
    # Drop cache entries for unregistered runs.
    for run_id in set(cache) - seen:
        del cache[run_id]
    return rows


async def _handle_dashboard(request: web.Request) -> web.Response:
    return _json_response({"runs": _dashboard_rows(request.app)})


# --- Websocket: per-client filter subscription + label-change pings ---


async def _push_rows(
    ctx: RunContext, ws: web.WebSocketResponse, filters: dict[str, Any], poll_interval_s: float
) -> None:
    # A fresh reader per subscription: the shared reader's connection is not
    # safe to tail from multiple subscribers (each tracks its own segment set).
    reader = ParquetSegmentReader(ctx.log_path)
    async for row in reader.subscribe(poll_interval_s=poll_interval_s, **filters):
        await ws.send_str(_json_dumps({"type": "row", "row": row}))


async def _push_label_updates(
    ctx: RunContext, ws: web.WebSocketResponse, poll_interval_s: float
) -> None:
    """Ping the client when labels.jsonl changes (agents write labels out-of-band)."""
    storage = ctx.storage
    last = storage.stat(LABELS_PATH)
    while True:
        await asyncio.sleep(poll_interval_s)
        current = storage.stat(LABELS_PATH)
        if current is not None and (last is None or current.size != last.size):
            await ws.send_str(_json_dumps({"type": "labels_changed"}))
        last = current


def _ws_ctx(request: web.Request, data: dict[str, Any]) -> RunContext:
    """Resolve the run context for a websocket subscription message.

    Single-run mode uses the app's run. Registry mode takes the run from the
    URL (``/api/runs/{run_id}/ws``) or from the message's ``run_id`` field
    (``/ws``). Raises ``ValueError`` for missing/unknown run IDs.
    """
    ctx = request.app[SINGLE_CTX_KEY]
    if ctx is not None:
        return ctx
    run_id = request.match_info.get("run_id") or data.get("run_id")
    if not run_id:
        raise ValueError("registry mode: subscribe messages must include a run_id")
    try:
        return _get_or_create_ctx(request.app, str(run_id))
    except web.HTTPNotFound as e:
        raise ValueError(f"unknown run_id {run_id!r}") from e


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
                ctx = _ws_ctx(request, data)
                filters = _coerce_body_filters(data.get("filters"))
                poll_interval_s = float(data.get("poll_interval_s", 2.0))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                await ws.send_str(_json_dumps({"type": "error", "error": str(e)}))
                continue
            _cancel_tasks()  # a new subscription replaces the previous one
            tasks = [
                asyncio.create_task(_push_rows(ctx, ws, filters, poll_interval_s)),
                asyncio.create_task(_push_label_updates(ctx, ws, poll_interval_s)),
            ]
            await ws.send_str(_json_dumps({"type": "subscribed", "filters": filters}))
    finally:
        _cancel_tasks()
    return ws


async def _handle_ws_dashboard(request: web.Request) -> web.WebSocketResponse:
    """Push dashboard rows on a poll interval (registry mode).

    Protocol: the server sends ``{"type": "dashboard", "runs": [...]}`` on
    connect and then every ``poll_interval_s`` seconds (query param, default
    5). Aggregates come from the same TTL cache as ``GET /api/dashboard``.
    """
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    try:
        poll_interval_s = float(
            request.rel_url.query.get("poll_interval_s") or DEFAULT_DASHBOARD_PUSH_INTERVAL_S
        )
    except ValueError:
        poll_interval_s = DEFAULT_DASHBOARD_PUSH_INTERVAL_S

    async def _sender() -> None:
        while not ws.closed:
            rows = _dashboard_rows(request.app)
            await ws.send_str(_json_dumps({"type": "dashboard", "runs": rows}))
            await asyncio.sleep(poll_interval_s)

    sender = asyncio.create_task(_sender())
    try:
        async for _ in ws:  # drain client messages until the socket closes
            pass
    finally:
        sender.cancel()
    return ws


# --- Tokenizer (best-effort, background) ---


async def _load_tokenizer(ctx: RunContext) -> None:
    try:
        if not ctx.storage.exists(RUN_JSON_PATH):
            ctx.tokenizer_error = f"{RUN_JSON_PATH} not found"
            return
        run_info = json.loads(ctx.storage.read(RUN_JSON_PATH).decode())
        model_name = (run_info.get("context") or {}).get("model_name")
        if not model_name:
            ctx.tokenizer_error = "run.json has no model_name"
            return
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        ctx.tokenizer = await asyncio.to_thread(get_tokenizer, model_name)
        logger.info("Loaded tokenizer for %s", model_name)
    except Exception as e:
        # Best-effort: the UI falls back to whole-turn text + raw token IDs.
        ctx.tokenizer_error = str(e)
        logger.warning("Could not load tokenizer (per-token view degraded): %s", e)


async def _start_tokenizer_load(app: web.Application) -> None:
    ctx = app[SINGLE_CTX_KEY]
    if ctx is not None and not ctx.tokenizer_started:
        ctx.tokenizer_started = True
        ctx.tokenizer_task = asyncio.create_task(_load_tokenizer(ctx))


# --- App assembly ---


def build_app(
    log_path: str | Path | None = None,
    *,
    registry_dir: str | None = None,
    static_dir: Path | None = STATIC_DIR,
    load_tokenizer: bool = True,
    dashboard_ttl_s: float = DEFAULT_DASHBOARD_TTL_S,
    live_window_s: float = DEFAULT_LIVE_WINDOW_S,
) -> web.Application:
    """Build the viewer application.

    Args:
        log_path: A training run's log directory (local path or cloud URI)
            for single-run mode, or ``None`` for registry mode (serve every
            run in the local run registry).
        registry_dir: Registry directory override for registry mode;
            ``None`` resolves via ``TINKER_TOKENDB_REGISTRY`` then the
            default (``~/.cache/tinker-cookbook/tokendb/runs``).
        static_dir: Directory with the built UI; a fallback page is served
            when it's missing (e.g. the frontend hasn't been built).
        load_tokenizer: Allow the best-effort tokenizer loads (single-run:
            at startup; registry mode: lazily per run). Disable in tests to
            avoid network access.
        dashboard_ttl_s: Per-run TTL for the ``/api/dashboard`` aggregate
            cache.
        live_window_s: A run is "live" if a manifest was modified within
            this many seconds.
    """
    app = web.Application()
    app[LOAD_TOKENIZER_KEY] = load_tokenizer
    app[RUN_CACHE_KEY] = OrderedDict()
    app[DASHBOARD_CACHE_KEY] = {}
    app[DASHBOARD_TTL_KEY] = dashboard_ttl_s
    app[LIVE_WINDOW_KEY] = live_window_s
    app[REGISTRY_DIR_KEY] = registry_dir

    if log_path is not None:
        # Single-run mode: endpoints at their original paths.
        app[SINGLE_CTX_KEY] = _make_run_ctx(str(log_path))
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
        if load_tokenizer:
            app.on_startup.append(_start_tokenizer_load)
    else:
        # Registry mode: dashboard + the same handlers mounted per run.
        if resolve_registry_dir(registry_dir) is None:
            raise ValueError(
                "The run registry is disabled (empty registry_dir / "
                "TINKER_TOKENDB_REGISTRY) and no log_path was given; nothing to serve."
            )
        app[SINGLE_CTX_KEY] = None
        app.router.add_get("/api/runs", _handle_runs)
        app.router.add_get("/api/dashboard", _handle_dashboard)
        prefix = "/api/runs/{run_id}"
        app.router.add_get(f"{prefix}/run", _handle_run)
        app.router.add_get(f"{prefix}/rollouts", _handle_rollouts)
        app.router.add_get(
            f"{prefix}/rollout/{{split}}/{{iteration}}/{{group_idx}}/{{traj_idx}}",
            _handle_rollout_detail,
        )
        app.router.add_post(f"{prefix}/search", _handle_search)
        app.router.add_post(f"{prefix}/sql", _handle_sql)
        app.router.add_get(f"{prefix}/labels", _handle_labels_get)
        app.router.add_post(f"{prefix}/labels", _handle_labels_post)
        app.router.add_post(f"{prefix}/tokens/decode", _handle_decode)
        app.router.add_get(f"{prefix}/ws", _handle_ws)
        app.router.add_get("/ws", _handle_ws)
        app.router.add_get("/ws/dashboard", _handle_ws_dashboard)

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

    return app


@chz.chz
class Config:
    """CLI config: ``python -m tinker_cookbook.tokendb.serve [log_path=...] [port=7423]``.

    With ``log_path`` the server views that one run; without it, the server
    runs in registry mode and shows every run in the local run registry.
    """

    log_path: str | None = None
    registry_dir: str | None = None
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"  # local viewer; not meant to be exposed


def run(config: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    app = build_app(config.log_path, registry_dir=config.registry_dir)
    if config.log_path is not None:
        logger.info(
            "Token DB viewer on http://%s:%d (log_path=%s)",
            config.host,
            config.port,
            config.log_path,
        )
    else:
        logger.info(
            "Token DB viewer on http://%s:%d (registry mode: %s)",
            config.host,
            config.port,
            resolve_registry_dir(config.registry_dir),
        )
    web.run_app(app, host=config.host, port=config.port)


if __name__ == "__main__":
    chz.nested_entrypoint(run)
