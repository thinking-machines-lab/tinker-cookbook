"""Chat agent for the token DB viewer: tool-use loop over tokendb tools.

The viewer's chat mode runs this server-side loop: the user's question goes to
an LLM (:mod:`tinker_cookbook.tokendb.llm`) together with tool definitions
bound to a run's :class:`~tinker_cookbook.tokendb.interface.TokenStoreBackend`
(``sql`` / ``search`` / ``get_rollout`` / ``publish_visual``; registry mode
adds ``list_runs`` / ``dashboard`` and per-run variants taking a ``run_id``).
Tool results are fed back until the model answers, up to
:data:`MAX_TOOL_ITERATIONS` rounds. :func:`run_chat_turn` streams the whole
exchange as JSON-ready event dicts (text deltas, tool calls/results, published
visuals, done/error) that the websocket handler forwards verbatim.

Persistence, all through the ``Storage`` protocol:

- Conversations: append-only JSONL, one message or event per line, under
  ``{log_path}/tokens/chats/{conversation_id}.jsonl`` (:class:`ChatStore`).
- Visuals: self-contained HTML files under ``{log_path}/tokens/visuals/``
  (:class:`VisualStore`), served by the viewer and rendered in sandboxed
  iframes. The registry-level (cross-run) chat stores both under the registry
  directory instead.

Results sent to the model are capped (row counts, long strings, long token
lists) so a broad ``SELECT *`` cannot blow the context window; truncation is
always reported so the model knows to aggregate instead.

Turns are decoupled from websocket lifetimes: :class:`TurnManager` runs each
turn as an asyncio task to completion and fans persisted transcript records
out to any number of subscribers, so navigating away from the viewer never
kills an in-flight turn. Every record carries a monotonically increasing
``seq`` per conversation, letting a client resume from the last record it saw
(replay from the JSONL, then tail live) with no gaps or duplicates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tinker_cookbook.stores.storage import Storage
from tinker_cookbook.tokendb.interface import TokenStoreBackend
from tinker_cookbook.tokendb.llm import (
    ErrorEvent,
    LLMClient,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolDef,
)
from tinker_cookbook.tokendb.reader import reconstruct_full_ob

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 20
# Streamed assistant text is persisted (and fanned out) as coalesced
# text_delta events: buffer until roughly this many characters. Small enough
# to still feel like streaming, large enough to keep transcripts compact.
TEXT_DELTA_COALESCE_CHARS = 80
MAX_SQL_ROWS_FOR_MODEL = 200
MAX_SEARCH_ROWS_FOR_MODEL = 50
MAX_STRING_CHARS_FOR_MODEL = 2000
MAX_LIST_ITEMS_FOR_MODEL = 200
MAX_VISUAL_BYTES = 512 * 1024
TOOL_RESULT_PREVIEW_CHARS = 400

CHATS_PREFIX = "tokens/chats"
VISUALS_PREFIX = "tokens/visuals"

_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VISUAL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.html$")


class ToolExecutionError(Exception):
    """Tool failure whose message goes back to the model as an error result."""


def new_conversation_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


def valid_conversation_id(conversation_id: str) -> bool:
    return bool(_CONVERSATION_ID_RE.match(conversation_id))


def valid_visual_name(name: str) -> bool:
    return bool(_VISUAL_NAME_RE.match(name))


# --- Truncation of tool results for the model ---


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS_FOR_MODEL:
        omitted = len(value) - MAX_STRING_CHARS_FOR_MODEL
        return value[:MAX_STRING_CHARS_FOR_MODEL] + f"... [truncated, {omitted} more chars]"
    if isinstance(value, (list, tuple)) and len(value) > MAX_LIST_ITEMS_FOR_MODEL:
        head = [_truncate_value(v) for v in value[:MAX_LIST_ITEMS_FOR_MODEL]]
        return [*head, f"... [truncated, {len(value) - MAX_LIST_ITEMS_FOR_MODEL} more items]"]
    if isinstance(value, dict):
        return {k: _truncate_value(v) for k, v in value.items()}
    return value


def _rows_for_model(rows: list[dict[str, Any]], max_rows: int) -> dict[str, Any]:
    """Cap row count and elide long values; report what was truncated."""
    payload: dict[str, Any] = {
        "n_rows": len(rows),
        "rows": [_truncate_value(row) for row in rows[:max_rows]],
    }
    if len(rows) > max_rows:
        payload["truncated"] = True
        payload["note"] = (
            f"Showing the first {max_rows} of {len(rows)} rows; refine the query "
            "or aggregate in SQL instead of paging."
        )
    return payload


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str)


# --- Visual publication ---


class VisualStore:
    """Published HTML visuals under a store prefix (append-only files)."""

    def __init__(self, storage: Storage, url_base: str, prefix: str = VISUALS_PREFIX) -> None:
        self._storage = storage
        self._prefix = prefix
        self._url_base = url_base.rstrip("/")

    def publish(self, title: str, description: str, html: str) -> dict[str, Any]:
        if not title or not html:
            raise ToolExecutionError("publish_visual needs a non-empty title and html")
        data = html.encode()
        if len(data) > MAX_VISUAL_BYTES:
            raise ToolExecutionError(
                f"visual is {len(data)} bytes; the limit is {MAX_VISUAL_BYTES} "
                "(keep visuals small and query the SQL endpoint for data)"
            )
        slug = re.sub(r"[^A-Za-z0-9]+", "-", title.lower()).strip("-")[:60] or "visual"
        name = f"{slug}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.html"
        self._storage.write(f"{self._prefix}/{name}", data)
        # No-op on LocalStorage; on staged cloud backends this makes sure the
        # visual (and anything else staged on this Storage) is durable now.
        self._storage.flush()
        return {
            "name": name,
            "url": f"{self._url_base}/{name}",
            "title": title,
            "description": description,
        }

    def list(self) -> list[dict[str, Any]]:
        try:
            names = [n for n in self._storage.list_dir(self._prefix) if valid_visual_name(n)]
        except FileNotFoundError:
            return []
        visuals = []
        for name in sorted(names):
            stat = self._storage.stat(f"{self._prefix}/{name}")
            visuals.append(
                {
                    "name": name,
                    "url": f"{self._url_base}/{name}",
                    "size": stat.size if stat else None,
                    "mtime": stat.mtime if stat else None,
                }
            )
        visuals.sort(key=lambda v: v["mtime"] or 0, reverse=True)
        return visuals

    def read(self, name: str) -> bytes:
        if not valid_visual_name(name):
            raise FileNotFoundError(name)
        return self._storage.read(f"{self._prefix}/{name}")


# --- Conversation persistence ---


class ChatStore:
    """Append-only JSONL conversation transcripts under a store prefix.

    One record per line: ``{"kind": "message", "role": ..., ...}`` for
    conversation messages (reloaded as model context on the next turn) and
    ``{"kind": "event", "type": ..., ...}`` for the streamed UI events of a
    turn (coalesced text deltas, tool calls/results, published visuals, and
    the terminal done/error/cancelled). Every record gets a monotonically
    increasing per-conversation ``seq`` so subscribers can resume from the
    last record they saw. Appends go through ``Storage.append``.
    """

    def __init__(self, storage: Storage, prefix: str = CHATS_PREFIX) -> None:
        self._storage = storage
        self._prefix = prefix
        self._next_seq: dict[str, int] = {}

    def _path(self, conversation_id: str) -> str:
        if not valid_conversation_id(conversation_id):
            raise ValueError(f"bad conversation id {conversation_id!r}")
        return f"{self._prefix}/{conversation_id}.jsonl"

    def append_message(self, conversation_id: str, message: Message) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": "message",
            "role": message.role,
            "content": message.content,
            "ts": datetime.now(UTC).isoformat(),
        }
        if message.tool_calls:
            record["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
            ]
        if message.tool_call_id is not None:
            record["tool_call_id"] = message.tool_call_id
        return self._append(conversation_id, record)

    def append_event(self, conversation_id: str, event: dict[str, Any]) -> dict[str, Any]:
        return self._append(
            conversation_id, {"kind": "event", "ts": datetime.now(UTC).isoformat(), **event}
        )

    def _append(self, conversation_id: str, record: dict[str, Any]) -> dict[str, Any]:
        record["seq"] = self._take_seq(conversation_id)
        self._storage.append(self._path(conversation_id), (_to_json(record) + "\n").encode())
        # FsspecStorage stages append()s locally and only uploads on flush();
        # nothing else flushes this storage, so without this chat transcripts
        # on gs:///s3:// stores would be silently lost when the viewer exits.
        # LocalStorage.flush() is a no-op, and FsspecStorage re-uploads only
        # files with staged appends (here: one small transcript).
        self._storage.flush()
        return record

    def _take_seq(self, conversation_id: str) -> int:
        """The next seq for this conversation (initialized from the JSONL)."""
        next_seq = self._next_seq.get(conversation_id)
        if next_seq is None:
            next_seq = self.last_seq(conversation_id) + 1
        self._next_seq[conversation_id] = next_seq + 1
        return next_seq

    def last_seq(self, conversation_id: str) -> int:
        """The highest persisted seq, or -1 for an empty/new conversation."""
        records = self.load_records(conversation_id)
        if not records:
            return -1
        return max(int(r.get("seq", -1)) for r in records)

    def load_records(self, conversation_id: str) -> list[dict[str, Any]]:
        try:
            raw = self._storage.read(self._path(conversation_id)).decode()
        except FileNotFoundError:
            return []
        records = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt chat line in %s", conversation_id)
        # Transcripts written before seq existed: backfill with the line
        # index so replay-from-seq works uniformly (indices stay below any
        # explicitly assigned seq, which starts at max+1).
        for index, record in enumerate(records):
            record.setdefault("seq", index)
        return records

    def load_messages(self, conversation_id: str) -> list[Message]:
        """Reconstruct the model-context messages from the transcript."""
        messages = []
        for record in self.load_records(conversation_id):
            if record.get("kind") != "message":
                continue
            messages.append(
                Message(
                    role=str(record.get("role", "user")),
                    content=str(record.get("content") or ""),
                    tool_calls=[
                        ToolCall(
                            id=str(c.get("id", "")),
                            name=str(c.get("name", "")),
                            arguments=dict(c.get("arguments") or {}),
                        )
                        for c in record.get("tool_calls", [])
                    ],
                    tool_call_id=record.get("tool_call_id"),
                )
            )
        return messages

    def list_conversations(self) -> list[dict[str, Any]]:
        """List conversations, newest activity first, with a title preview."""
        try:
            names = [n for n in self._storage.list_dir(self._prefix) if n.endswith(".jsonl")]
        except FileNotFoundError:
            return []
        conversations = []
        for name in names:
            conversation_id = name[: -len(".jsonl")]
            if not valid_conversation_id(conversation_id):
                continue
            stat = self._storage.stat(f"{self._prefix}/{name}")
            records = self.load_records(conversation_id)
            title = next(
                (
                    str(r.get("content", ""))[:120]
                    for r in records
                    if r.get("kind") == "message" and r.get("role") == "user"
                ),
                "",
            )
            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "title": title,
                    "n_records": len(records),
                    "mtime": stat.mtime if stat else None,
                }
            )
        conversations.sort(key=lambda c: c["mtime"] or 0, reverse=True)
        return conversations


# --- Tool boxes ---


_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "One read-only SELECT (or WITH ... SELECT) DuckDB statement.",
        }
    },
    "required": ["query"],
}
_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "regex": {"type": "string", "description": "Regex over the decoded text fields."},
        "fields": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["ac_text", "ob_text", "logs", "metrics", "attrs", "token_metrics"],
            },
            "description": "Fields the regex applies to (default ac_text and ob_text). "
            "For the map columns (metrics/attrs/token_metrics) the regex matches KEYS.",
        },
        "token_subsequence": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Contiguous token-ID subsequence to find inside ac_tokens.",
        },
        "limit": {"type": "integer", "description": "Max rows to return (default 50)."},
    },
}
_GET_ROLLOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "split": {"type": "string"},
        "iteration": {"type": "integer"},
        "group_idx": {"type": "integer"},
        "traj_idx": {"type": "integer"},
        "run_attempt": {
            "type": "integer",
            "description": "Specific attempt; omit for the latest.",
        },
    },
    "required": ["split", "iteration", "group_idx", "traj_idx"],
}
_PUBLISH_VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string", "description": "One-line summary shown in the UI."},
        "html": {
            "type": "string",
            "description": "Complete self-contained HTML document (inline JS/SVG, no CDNs).",
        },
    },
    "required": ["title", "description", "html"],
}


def _with_run_id(schema: dict[str, Any]) -> dict[str, Any]:
    """Registry-mode variant of a tool schema: adds a required run_id."""
    properties = {
        "run_id": {"type": "string", "description": "Run to query (see list_runs)."},
        **schema.get("properties", {}),
    }
    required = ["run_id", *schema.get("required", [])]
    return {"type": "object", "properties": properties, "required": required}


def _with_optional_run_id(schema: dict[str, Any]) -> dict[str, Any]:
    """Registry-mode variant of a tool schema: adds an optional run_id."""
    properties = {
        "run_id": {
            "type": "string",
            "description": "Optional: execute against one run's own store "
            "(omit for cross-run SQL over every registered run).",
        },
        **schema.get("properties", {}),
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.get("required", [])),
    }


@dataclass
class ToolOutcome:
    """One executed tool call: model-facing content plus optional UI frames."""

    content: str
    is_error: bool = False
    frames: list[dict[str, Any]] | None = None


def _execute_sql(reader: TokenStoreBackend, arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    if not query or not isinstance(query, str):
        raise ToolExecutionError("sql needs a 'query' string")
    # sql() is a backend-specific escape hatch, deliberately not part of the
    # TokenStoreBackend protocol.
    sql_fn = getattr(reader, "sql", None)
    if sql_fn is None:
        raise ToolExecutionError("this token store backend has no raw SQL escape hatch")
    rows = sql_fn(query)  # SELECT-only guard lives in the backend
    return _rows_for_model(rows, MAX_SQL_ROWS_FOR_MODEL)


def _execute_search(reader: TokenStoreBackend, arguments: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if arguments.get("regex"):
        kwargs["regex"] = str(arguments["regex"])
    if arguments.get("fields"):
        kwargs["fields"] = [str(f) for f in arguments["fields"]]
    if arguments.get("token_subsequence"):
        kwargs["token_subsequence"] = [int(t) for t in arguments["token_subsequence"]]
    if not kwargs:
        raise ToolExecutionError("search needs a regex and/or a token_subsequence")
    limit = int(arguments.get("limit") or MAX_SEARCH_ROWS_FOR_MODEL)
    rows = reader.search(limit=limit, **kwargs)
    payload = _rows_for_model(rows, MAX_SEARCH_ROWS_FOR_MODEL)
    payload["hit_counts_by_iteration"] = {
        str(k): v for k, v in sorted(reader.search_hit_counts(**kwargs).items())
    }
    return payload


def _execute_get_rollout(reader: TokenStoreBackend, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        split = str(arguments["split"])
        iteration = int(arguments["iteration"])
        group_idx = int(arguments["group_idx"])
        traj_idx = int(arguments["traj_idx"])
    except (KeyError, TypeError, ValueError) as e:
        raise ToolExecutionError(
            f"get_rollout needs split/iteration/group_idx/traj_idx: {e}"
        ) from e
    run_attempt = arguments.get("run_attempt")
    steps = reader.get_rollout(
        split, iteration, group_idx, traj_idx, int(run_attempt) if run_attempt is not None else None
    )
    if not steps:
        raise ToolExecutionError(f"rollout {split}/{iteration}/{group_idx}/{traj_idx} not found")
    for step, full_ob in zip(steps, reconstruct_full_ob(steps)):
        step["ob_full_tokens"] = full_ob
    return {"n_steps": len(steps), "steps": [_truncate_value(step) for step in steps]}


class RunToolbox:
    """Tools bound to one run's backend (single-run and per-run chats)."""

    def __init__(self, reader: TokenStoreBackend, visual_store: VisualStore) -> None:
        self._reader = reader
        self._visual_store = visual_store

    def tool_defs(self) -> list[ToolDef]:
        return [
            ToolDef(
                "sql",
                "Run a read-only DuckDB SELECT over this run's token DB "
                "(views: rollouts, rollouts_latest, trajectories, labels, runs, "
                "correct, parse_errors, context_overflows).",
                _SQL_SCHEMA,
            ),
            ToolDef(
                "search",
                "Search rollout rows by text regex and/or contiguous token-ID subsequence.",
                _SEARCH_SCHEMA,
            ),
            ToolDef(
                "get_rollout",
                "Fetch every turn of one trajectory, with full observations reconstructed.",
                _GET_ROLLOUT_SCHEMA,
            ),
            ToolDef(
                "publish_visual",
                "Publish a self-contained HTML visual; returns the URL it is served at.",
                _PUBLISH_VISUAL_SCHEMA,
            ),
        ]

    def execute(self, call: ToolCall) -> ToolOutcome:
        try:
            if call.name == "sql":
                return ToolOutcome(_to_json(_execute_sql(self._reader, call.arguments)))
            if call.name == "search":
                return ToolOutcome(_to_json(_execute_search(self._reader, call.arguments)))
            if call.name == "get_rollout":
                return ToolOutcome(_to_json(_execute_get_rollout(self._reader, call.arguments)))
            if call.name == "publish_visual":
                return _publish_visual_outcome(self._visual_store, call.arguments)
            raise ToolExecutionError(f"unknown tool {call.name!r}")
        except ToolExecutionError as e:
            return ToolOutcome(_to_json({"error": str(e)}), is_error=True)
        except Exception as e:  # DuckDB errors on model-written SQL, bad regexes, ...
            return ToolOutcome(_to_json({"error": str(e)}), is_error=True)


class RegistryToolbox:
    """Cross-run tools for the registry-level chat.

    ``sql`` runs against the cross-run *backend* (every registered run in
    one database, ``run_id`` as a column) when no ``run_id`` argument is
    given, and against that run's own store when one is. ``search`` and
    ``get_rollout`` stay per-run (they require a ``run_id``).
    """

    def __init__(
        self,
        list_runs_fn: Callable[[], list[dict[str, Any]]],
        dashboard_fn: Callable[[], list[dict[str, Any]]],
        resolve_reader: Callable[[str], TokenStoreBackend],
        visual_store: VisualStore,
        backend: TokenStoreBackend | None = None,
    ) -> None:
        self._list_runs_fn = list_runs_fn
        self._dashboard_fn = dashboard_fn
        self._resolve_reader = resolve_reader
        self._visual_store = visual_store
        self._backend = backend

    def tool_defs(self) -> list[ToolDef]:
        empty = {"type": "object", "properties": {}}
        return [
            ToolDef("list_runs", "List every registered run with liveness status.", empty),
            ToolDef(
                "dashboard",
                "Per-run aggregates: row counts, latest iteration, reward trend.",
                empty,
            ),
            ToolDef(
                "sql",
                "Run a read-only DuckDB SELECT over the token DB spanning EVERY "
                "registered run (views: rollouts, rollouts_latest, trajectories, "
                "labels, runs, correct, parse_errors, context_overflows; run_id is "
                "a normal column). Pass run_id to query one run's store instead.",
                _with_optional_run_id(_SQL_SCHEMA),
            ),
            ToolDef(
                "search",
                "Search one run's rows by text regex and/or token-ID subsequence.",
                _with_run_id(_SEARCH_SCHEMA),
            ),
            ToolDef(
                "get_rollout",
                "Fetch every turn of one trajectory of one run.",
                _with_run_id(_GET_ROLLOUT_SCHEMA),
            ),
            ToolDef(
                "publish_visual",
                "Publish a self-contained HTML visual; returns the URL it is served at.",
                _PUBLISH_VISUAL_SCHEMA,
            ),
        ]

    def execute(self, call: ToolCall) -> ToolOutcome:
        try:
            if call.name == "list_runs":
                return ToolOutcome(_to_json({"runs": self._list_runs_fn()}))
            if call.name == "dashboard":
                return ToolOutcome(_to_json({"runs": self._dashboard_fn()}))
            if call.name == "publish_visual":
                return _publish_visual_outcome(self._visual_store, call.arguments)
            if call.name == "sql":
                run_id = call.arguments.get("run_id")
                if run_id:
                    reader = self._resolve_reader(str(run_id))
                elif self._backend is not None:
                    reader = self._backend
                else:
                    raise ToolExecutionError("sql needs a run_id (see list_runs)")
                return ToolOutcome(_to_json(_execute_sql(reader, call.arguments)))
            if call.name in ("search", "get_rollout"):
                run_id = call.arguments.get("run_id")
                if not run_id:
                    raise ToolExecutionError(f"{call.name} needs a run_id (see list_runs)")
                reader = self._resolve_reader(str(run_id))
                executor = {
                    "search": _execute_search,
                    "get_rollout": _execute_get_rollout,
                }[call.name]
                return ToolOutcome(_to_json(executor(reader, call.arguments)))
            raise ToolExecutionError(f"unknown tool {call.name!r}")
        except ToolExecutionError as e:
            return ToolOutcome(_to_json({"error": str(e)}), is_error=True)
        except Exception as e:
            return ToolOutcome(_to_json({"error": str(e)}), is_error=True)


def _publish_visual_outcome(visual_store: VisualStore, arguments: dict[str, Any]) -> ToolOutcome:
    visual = visual_store.publish(
        str(arguments.get("title") or ""),
        str(arguments.get("description") or ""),
        str(arguments.get("html") or ""),
    )
    return ToolOutcome(
        _to_json({"published": True, "url": visual["url"], "name": visual["name"]}),
        frames=[{"type": "visual_published", **visual}],
    )


Toolbox = RunToolbox | RegistryToolbox


# --- The agent loop ---


async def run_chat_turn(
    client: LLMClient,
    toolbox: Toolbox,
    chat_store: ChatStore,
    conversation_id: str,
    user_text: str,
    system_prompt: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run one user turn of the chat agent, streaming persisted records.

    Every yielded frame is a transcript record (with its ``seq``) that was
    appended to the conversation's JSONL *before* being yielded, so a
    subscriber can attach at any point and replay records past its last seen
    seq with no gaps. The stream interleaves ``kind: "message"`` records (the
    user message, assistant segments, tool results as model context) with
    ``kind: "event"`` records: ``text_delta`` (assistant text, coalesced into
    ~:data:`TEXT_DELTA_COALESCE_CHARS`-char chunks), ``tool_call`` /
    ``tool_result`` (activity summaries), ``visual_published`` (with the
    served URL), and a terminal ``done`` or ``error``.
    """
    messages = chat_store.load_messages(conversation_id)
    user_message = Message(role="user", content=user_text)
    messages.append(user_message)
    yield chat_store.append_message(conversation_id, user_message)

    tool_defs = toolbox.tool_defs()
    for _ in range(MAX_TOOL_ITERATIONS):
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        pending_text: list[str] = []

        def _flush_text() -> dict[str, Any] | None:
            if not pending_text:
                return None
            record = chat_store.append_event(
                conversation_id, {"type": "text_delta", "text": "".join(pending_text)}
            )
            pending_text.clear()
            return record

        async for event in client.stream(system_prompt, messages, tool_defs):
            if isinstance(event, TextDelta):
                text_parts.append(event.text)
                pending_text.append(event.text)
                if sum(len(part) for part in pending_text) >= TEXT_DELTA_COALESCE_CHARS:
                    flushed = _flush_text()
                    if flushed is not None:
                        yield flushed
            elif isinstance(event, ToolCallEvent):
                flushed = _flush_text()
                if flushed is not None:
                    yield flushed
                calls.append(event.call)
                yield chat_store.append_event(
                    conversation_id,
                    {
                        "type": "tool_call",
                        "id": event.call.id,
                        "name": event.call.name,
                        "arguments": event.call.arguments,
                    },
                )
            elif isinstance(event, ErrorEvent):
                flushed = _flush_text()
                if flushed is not None:
                    yield flushed
                if text_parts:  # keep any partial assistant text in the transcript
                    partial = Message(role="assistant", content="".join(text_parts))
                    yield chat_store.append_message(conversation_id, partial)
                yield chat_store.append_event(
                    conversation_id, {"type": "error", "error": event.message}
                )
                return
            # Done carries only the stop reason; the presence/absence of tool
            # calls decides whether the loop continues.
        flushed = _flush_text()
        if flushed is not None:
            yield flushed

        assistant = Message(role="assistant", content="".join(text_parts), tool_calls=calls)
        messages.append(assistant)
        yield chat_store.append_message(conversation_id, assistant)
        if not calls:
            yield chat_store.append_event(
                conversation_id, {"type": "done", "conversation_id": conversation_id}
            )
            return

        for call in calls:
            outcome = await asyncio.to_thread(toolbox.execute, call)
            for frame in outcome.frames or []:
                yield chat_store.append_event(conversation_id, frame)
            preview = outcome.content[:TOOL_RESULT_PREVIEW_CHARS]
            yield chat_store.append_event(
                conversation_id,
                {
                    "type": "tool_result",
                    "id": call.id,
                    "name": call.name,
                    "is_error": outcome.is_error,
                    "preview": preview,
                },
            )
            tool_message = Message(role="tool", content=outcome.content, tool_call_id=call.id)
            messages.append(tool_message)
            yield chat_store.append_message(conversation_id, tool_message)

    yield chat_store.append_event(
        conversation_id,
        {
            "type": "error",
            "error": f"stopped after {MAX_TOOL_ITERATIONS} tool iterations without a final answer",
        },
    )


# --- Turn manager: server-owned background turns + subscriber fanout ---


class TurnBusyError(Exception):
    """A second turn was started on a conversation that already has one."""


class TurnManager:
    """Registry of running chat turns, keyed by ``(scope, conversation_id)``.

    :meth:`start_turn` drives a :func:`run_chat_turn` iterator inside an
    asyncio task that runs to completion regardless of any websocket: the
    records it yields are already persisted, and are additionally fanned out
    to every subscriber queue for that conversation. Disconnecting a
    subscriber never touches the turn; :meth:`cancel` (or :meth:`shutdown`)
    cancels the task, persisting a terminal ``cancelled`` event so the
    transcript keeps everything that streamed before the cancellation.
    """

    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._subscribers: dict[tuple[str, str], list[asyncio.Queue[dict[str, Any]]]] = {}

    def in_flight(self, scope: str, conversation_id: str) -> bool:
        return (scope, conversation_id) in self._tasks

    def in_flight_ids(self, scope: str) -> set[str]:
        """Conversation IDs with a running turn under this scope."""
        return {cid for (s, cid) in self._tasks if s == scope}

    def start_turn(
        self,
        scope: str,
        conversation_id: str,
        chat_store: ChatStore,
        frames: AsyncIterator[dict[str, Any]],
    ) -> asyncio.Task[None]:
        """Run *frames* (a :func:`run_chat_turn` iterator) to completion.

        Raises :class:`TurnBusyError` if this conversation already has a
        running turn.
        """
        key = (scope, conversation_id)
        if key in self._tasks:
            raise TurnBusyError(f"a turn is already running on conversation {conversation_id!r}")
        task = asyncio.create_task(self._run(key, chat_store, frames))
        self._tasks[key] = task
        return task

    async def _run(
        self,
        key: tuple[str, str],
        chat_store: ChatStore,
        frames: AsyncIterator[dict[str, Any]],
    ) -> None:
        _, conversation_id = key
        try:
            async for record in frames:
                self._publish(key, record)
        except asyncio.CancelledError:
            record = chat_store.append_event(
                conversation_id, {"type": "cancelled", "conversation_id": conversation_id}
            )
            self._publish(key, record)
            raise
        except Exception as e:  # keep the failure in the transcript + fanout
            logger.warning("Chat turn failed: %s", e, exc_info=True)
            record = chat_store.append_event(conversation_id, {"type": "error", "error": str(e)})
            self._publish(key, record)
        finally:
            self._tasks.pop(key, None)

    def _publish(self, key: tuple[str, str], record: dict[str, Any]) -> None:
        for queue in self._subscribers.get(key, []):
            queue.put_nowait(record)

    def subscribe(self, scope: str, conversation_id: str) -> asyncio.Queue[dict[str, Any]]:
        """A queue receiving every record of this conversation's live turns.

        Valid whether or not a turn is currently running (records start
        flowing when one starts). Pair with :meth:`unsubscribe`.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault((scope, conversation_id), []).append(queue)
        return queue

    def unsubscribe(
        self, scope: str, conversation_id: str, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        key = (scope, conversation_id)
        queues = self._subscribers.get(key)
        if queues is None:
            return
        if queue in queues:
            queues.remove(queue)
        if not queues:
            del self._subscribers[key]

    def cancel(self, scope: str, conversation_id: str) -> bool:
        """Cancel the running turn, if any; returns whether one was cancelled."""
        task = self._tasks.get((scope, conversation_id))
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        """Cancel every running turn and wait for the tasks to finish."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
