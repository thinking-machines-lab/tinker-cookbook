"""Tests for the tokendb chat agent: LLM client wire formats, tool executors,
and the tool-use loop. No network: a scripted SSE transport stands in for the
provider APIs."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")
pytest.importorskip("aiohttp")

from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.tokendb.agent import (
    MAX_LIST_ITEMS_FOR_MODEL,
    MAX_SQL_ROWS_FOR_MODEL,
    MAX_STRING_CHARS_FOR_MODEL,
    MAX_VISUAL_BYTES,
    ChatStore,
    RegistryToolbox,
    RunToolbox,
    ToolExecutionError,
    VisualStore,
    run_chat_turn,
)
from tinker_cookbook.tokendb.llm import (
    Done,
    ErrorEvent,
    LLMClient,
    LLMConfig,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolDef,
    build_anthropic_request,
    build_openai_responses_request,
    detect_default_provider,
)
from tinker_cookbook.tokendb.reader import ParquetSegmentReader
from tinker_cookbook.tokendb.schema import TokenRow
from tinker_cookbook.tokendb.writer import TokenDbWriter


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys must come from the test, never the developer's environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)


def make_row(**overrides) -> TokenRow:
    defaults: dict = {
        "split": "train",
        "iteration": 0,
        "group_idx": 0,
        "traj_idx": 0,
        "step_idx": 0,
        "ob_tokens": [1, 2, 3],
        "ac_tokens": [4, 5],
    }
    defaults.update(overrides)
    return TokenRow(**defaults)


# --- Scripted transport (shared with serve tests) ---


class ScriptedTransport:
    """SSETransport stub: replays scripted SSE event lists, records requests."""

    def __init__(self, scripts: list[list[tuple[str | None, dict]]]) -> None:
        self.scripts = list(scripts)
        self.requests: list[tuple[str, dict, dict]] = []

    async def stream_sse(self, url: str, headers: dict, payload: dict):
        self.requests.append((url, headers, payload))
        if not self.scripts:
            raise AssertionError("ScriptedTransport ran out of scripted responses")
        for event in self.scripts.pop(0):
            yield event


def anthropic_script(
    text: str | None = None, tool_calls: Sequence[tuple[str, str, dict]] = ()
) -> list[tuple[str | None, dict]]:
    """Anthropic-shaped SSE events for one model turn."""
    events: list[tuple[str | None, dict]] = [("message_start", {"type": "message_start"})]
    index = 0
    if text is not None:
        events += [
            (
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text"}},
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
        index += 1
    for call_id, name, arguments in tool_calls:
        raw = json.dumps(arguments)
        events += [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": call_id, "name": name},
                },
            ),
            # Input arrives as split partial-JSON deltas that must be joined.
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": raw[: len(raw) // 2]},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": raw[len(raw) // 2 :]},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
        index += 1
    stop = "tool_use" if tool_calls else "end_turn"
    events += [
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


def collect(aiter):
    async def main():
        return [event async for event in aiter]

    return asyncio.run(main())


# --- Provider wire formats ---


def _sample_conversation() -> tuple[list[Message], list[ToolDef]]:
    messages = [
        Message(role="user", content="how is reward trending?"),
        Message(
            role="assistant",
            content="Let me query.",
            tool_calls=[ToolCall(id="call_1", name="sql", arguments={"query": "SELECT 1"})],
        ),
        Message(role="tool", content='{"rows": []}', tool_call_id="call_1"),
    ]
    tools = [
        ToolDef(
            "sql",
            "Run SQL.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
    ]
    return messages, tools


def test_anthropic_request_shape():
    config = LLMConfig(provider="anthropic", max_tokens=1234)
    messages, tools = _sample_conversation()
    url, headers, payload = build_anthropic_request(config, "sk-test", "SYSTEM", messages, tools)
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"]
    assert payload["model"] == "claude-fable-5"
    assert payload["max_tokens"] == 1234
    assert payload["system"] == "SYSTEM"
    assert payload["stream"] is True
    assert payload["tools"] == [
        {"name": "sql", "description": "Run SQL.", "input_schema": tools[0].input_schema}
    ]
    assert payload["messages"][0] == {"role": "user", "content": "how is reward trending?"}
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me query."},
            {"type": "tool_use", "id": "call_1", "name": "sql", "input": {"query": "SELECT 1"}},
        ],
    }
    # Tool results are user-role tool_result blocks.
    assert payload["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"rows": []}'}],
    }


def test_openai_request_shape():
    config = LLMConfig(provider="openai", model="my-model")
    messages, tools = _sample_conversation()
    url, headers, payload = build_openai_responses_request(
        config, "sk-test", "SYSTEM", messages, tools
    )
    assert url == "https://api.openai.com/v1/responses"
    assert headers["authorization"] == "Bearer sk-test"
    assert payload["model"] == "my-model"
    assert payload["stream"] is True
    assert payload["max_output_tokens"] == config.max_tokens
    assert payload["store"] is False
    # System prompt travels as top-level instructions, not an input item.
    assert payload["instructions"] == "SYSTEM"
    # Reasoning stays at the API default unless explicitly configured.
    assert "reasoning" not in payload
    assert payload["input"][0] == {"role": "user", "content": "how is reward trending?"}
    # Assistant text and its tool calls are separate input items.
    assert payload["input"][1] == {"role": "assistant", "content": "Let me query."}
    assert payload["input"][2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "sql",
        "arguments": '{"query": "SELECT 1"}',
    }
    assert payload["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"rows": []}',
    }
    # Responses API function tools are flat (no nested "function" object).
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "sql",
            "description": "Run SQL.",
            "parameters": tools[0].input_schema,
        }
    ]


def test_openai_request_reasoning_effort_passthrough():
    config = LLMConfig(provider="openai", reasoning_effort="low")
    messages, tools = _sample_conversation()
    _, _, payload = build_openai_responses_request(config, "sk-test", "SYSTEM", messages, tools)
    assert payload["reasoning"] == {"effort": "low"}


# --- Stream normalization ---


def test_anthropic_stream_normalization():
    transport = ScriptedTransport(
        [anthropic_script(text="Hello", tool_calls=[("t1", "sql", {"query": "SELECT 1"})])]
    )
    client = LLMClient(LLMConfig(provider="anthropic", api_key="k"), transport=transport)
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert events[0] == TextDelta("Hello")
    assert isinstance(events[1], ToolCallEvent)
    assert events[1].call == ToolCall(id="t1", name="sql", arguments={"query": "SELECT 1"})
    assert events[-1] == Done("tool_use")


def test_openai_stream_normalization():
    """Responses SSE: text deltas, function-call args split across deltas."""
    chunks: list[tuple[str | None, dict]] = [
        ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant"},
            },
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hel"},
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "lo"},
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_1", "role": "assistant"},
            },
        ),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "t1",
                    "name": "sql",
                    "arguments": "",
                },
            },
        ),
        # Arguments arrive as string fragments that must be joined.
        (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"que',
            },
        ),
        (
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": 'ry": "SELECT 1"}',
            },
        ),
        (
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "t1",
                    "name": "sql",
                    "arguments": '{"query": "SELECT 1"}',
                },
            },
        ),
        (
            "response.completed",
            {"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}},
        ),
    ]
    client = LLMClient(
        LLMConfig(provider="openai", api_key="k"), transport=ScriptedTransport([chunks])
    )
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert events[0] == TextDelta("Hel")
    assert events[1] == TextDelta("lo")
    assert isinstance(events[2], ToolCallEvent)
    assert events[2].call == ToolCall(id="t1", name="sql", arguments={"query": "SELECT 1"})
    assert events[-1] == Done("tool_calls")


def test_openai_stream_error_frame():
    chunks: list[tuple[str | None, dict]] = [
        ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
        ("error", {"type": "error", "code": "server_error", "message": "boom", "param": None}),
    ]
    client = LLMClient(
        LLMConfig(provider="openai", api_key="k"), transport=ScriptedTransport([chunks])
    )
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert events == [ErrorEvent("boom")]


def test_openai_stream_failed_response():
    chunks: list[tuple[str | None, dict]] = [
        (
            "response.failed",
            {
                "type": "response.failed",
                "response": {"status": "failed", "error": {"message": "quota exceeded"}},
            },
        ),
    ]
    client = LLMClient(
        LLMConfig(provider="openai", api_key="k"), transport=ScriptedTransport([chunks])
    )
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert events == [ErrorEvent("quota exceeded")]


def test_missing_api_key_yields_error_event():
    client = LLMClient(LLMConfig(provider="anthropic"), transport=ScriptedTransport([]))
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "ANTHROPIC_API_KEY" in events[0].message


def test_detect_default_provider(monkeypatch: pytest.MonkeyPatch):
    # No keys at all (the autouse fixture cleared the env): anthropic fallback.
    assert detect_default_provider() == "anthropic"
    # Only openai has a key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert detect_default_provider() == "openai"
    # Only tinker has a key.
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("TINKER_API_KEY", "tml-key")
    assert detect_default_provider() == "tinker"
    # Preference order: anthropic > openai > tinker.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert detect_default_provider() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    assert detect_default_provider() == "anthropic"
    # A runtime key counts for every provider, so the first preference wins.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("TINKER_API_KEY")
    assert detect_default_provider(api_key="sk-runtime") == "anthropic"


# --- Tool executors ---


@pytest.fixture
def store_toolbox(tmp_path: Path):
    """A RunToolbox over a store with many rows and one long-fields rollout."""
    log_path = tmp_path / "run"
    long_text = "x" * (MAX_STRING_CHARS_FOR_MODEL + 500)
    long_tokens = list(range(MAX_LIST_ITEMS_FOR_MODEL + 50))
    with TokenDbWriter(log_path, context={"model_name": "test-model"}) as writer:
        writer.append_rows(
            [
                make_row(traj_idx=i, total_reward=float(i))
                for i in range(MAX_SQL_ROWS_FOR_MODEL + 50)
            ]
        )
        writer.append_rows(
            [
                make_row(
                    iteration=7,
                    ob_text=long_text,
                    ac_tokens=long_tokens,
                    ac_text="I give up",
                )
            ]
        )
    storage = storage_from_uri(str(log_path))
    reader = ParquetSegmentReader(storage)
    visual_store = VisualStore(storage, url_base="/visuals")
    return RunToolbox(reader, visual_store), storage, visual_store


def _run_tool(toolbox, name: str, arguments: dict):
    outcome = toolbox.execute(ToolCall(id="t", name=name, arguments=arguments))
    return outcome, json.loads(outcome.content)


def test_sql_tool_truncates_and_guards(store_toolbox):
    toolbox, _, _ = store_toolbox
    outcome, payload = _run_tool(toolbox, "sql", {"query": "SELECT * FROM rollouts"})
    assert not outcome.is_error
    assert payload["n_rows"] == MAX_SQL_ROWS_FOR_MODEL + 51
    assert len(payload["rows"]) == MAX_SQL_ROWS_FOR_MODEL
    assert payload["truncated"] is True
    assert "aggregate" in payload["note"]

    # SELECT-only guard is the reader's; surfaces as an error result.
    outcome, payload = _run_tool(toolbox, "sql", {"query": "DROP TABLE segment_rows"})
    assert outcome.is_error
    assert "SELECT" in payload["error"]

    outcome, payload = _run_tool(toolbox, "sql", {})
    assert outcome.is_error


def test_search_tool(store_toolbox):
    toolbox, _, _ = store_toolbox
    outcome, payload = _run_tool(toolbox, "search", {"regex": "give up"})
    assert not outcome.is_error
    assert payload["n_rows"] == 1
    assert payload["hit_counts_by_iteration"] == {"7": 1}

    outcome, payload = _run_tool(toolbox, "search", {})
    assert outcome.is_error


def test_get_rollout_truncates_long_fields(store_toolbox):
    toolbox, _, _ = store_toolbox
    outcome, payload = _run_tool(
        toolbox, "get_rollout", {"split": "train", "iteration": 7, "group_idx": 0, "traj_idx": 0}
    )
    assert not outcome.is_error
    step = payload["steps"][0]
    assert "truncated" in step["ob_text"]
    assert len(step["ob_text"]) < MAX_STRING_CHARS_FOR_MODEL + 100
    assert len(step["ac_tokens"]) == MAX_LIST_ITEMS_FOR_MODEL + 1  # head + truncation marker
    assert "truncated" in step["ac_tokens"][-1]

    outcome, payload = _run_tool(
        toolbox, "get_rollout", {"split": "train", "iteration": 99, "group_idx": 0, "traj_idx": 0}
    )
    assert outcome.is_error and "not found" in payload["error"]


def test_publish_visual_writes_file_and_size_guard(store_toolbox):
    toolbox, storage, visual_store = store_toolbox
    html = "<!doctype html><html><body>hi</body></html>"
    outcome, payload = _run_tool(
        toolbox, "publish_visual", {"title": "Reward Trend!", "description": "d", "html": html}
    )
    assert not outcome.is_error
    assert payload["published"] is True
    assert payload["url"].startswith("/visuals/reward-trend-")
    name = payload["name"]
    assert storage.read(f"tokens/visuals/{name}").decode() == html
    assert outcome.frames and outcome.frames[0]["type"] == "visual_published"
    assert visual_store.list()[0]["name"] == name

    with pytest.raises(ToolExecutionError, match="limit"):
        visual_store.publish("big", "d", "x" * (MAX_VISUAL_BYTES + 1))
    outcome, payload = _run_tool(
        toolbox,
        "publish_visual",
        {"title": "big", "description": "d", "html": "x" * (MAX_VISUAL_BYTES + 1)},
    )
    assert outcome.is_error


def test_visual_store_rejects_bad_names(store_toolbox):
    _, _, visual_store = store_toolbox
    with pytest.raises(FileNotFoundError):
        visual_store.read("../../etc/passwd")
    with pytest.raises(FileNotFoundError):
        visual_store.read("no-such.html")


# --- The agent loop ---


def test_chat_turn_multi_tool_and_persistence(store_toolbox, tmp_path: Path):
    toolbox, storage, _ = store_toolbox
    chat_store = ChatStore(storage)
    transport = ScriptedTransport(
        [
            anthropic_script(
                text="Let me look.",
                tool_calls=[
                    ("t1", "sql", {"query": "SELECT count(*) AS n FROM rollouts"}),
                    (
                        "t2",
                        "publish_visual",
                        {"title": "Counts", "description": "d", "html": "<html>ok</html>"},
                    ),
                ],
            ),
            anthropic_script(text="All done."),
        ]
    )
    client = LLMClient(LLMConfig(provider="anthropic", api_key="k"), transport=transport)
    frames = collect(
        run_chat_turn(client, toolbox, chat_store, "conv-1", "count rows please", "SYSTEM")
    )
    types = [f["type"] for f in frames]
    assert types == [
        "text_delta",
        "tool_call",
        "tool_call",
        "tool_result",
        "visual_published",
        "tool_result",
        "text_delta",
        "done",
    ]
    sql_result = next(f for f in frames if f["type"] == "tool_result" and f["name"] == "sql")
    assert not sql_result["is_error"]
    assert '"n"' in sql_result["preview"]

    # The second model call got the tool results threaded back.
    second_payload = transport.requests[1][2]
    roles = [m["role"] for m in second_payload["messages"]]
    assert roles == ["user", "assistant", "user", "user"]  # tool results as user blocks

    # Transcript: JSONL, one message/event per line, reloadable.
    records = chat_store.load_records("conv-1")
    kinds = [(r["kind"], r.get("role")) for r in records]
    assert kinds == [
        ("message", "user"),
        ("message", "assistant"),
        ("message", "tool"),
        ("event", None),  # visual_published
        ("message", "tool"),
        ("message", "assistant"),
    ]
    assert records[1]["tool_calls"][0]["name"] == "sql"
    messages = chat_store.load_messages("conv-1")
    assert messages[-1].content == "All done."
    assert chat_store.list_conversations()[0]["conversation_id"] == "conv-1"
    assert chat_store.list_conversations()[0]["title"] == "count rows please"


def test_chat_turn_llm_error_frame(store_toolbox):
    toolbox, storage, _ = store_toolbox
    chat_store = ChatStore(storage)
    client = LLMClient(
        LLMConfig(provider="anthropic", api_key="k"),
        transport=ScriptedTransport([[("error", {"error": {"message": "overloaded"}})]]),
    )
    frames = collect(run_chat_turn(client, toolbox, chat_store, "conv-2", "hi", "SYSTEM"))
    assert frames == [{"type": "error", "error": "overloaded"}]


def test_registry_toolbox_routes_by_run_id(tmp_path: Path):
    log_path = tmp_path / "run-a"
    with TokenDbWriter(log_path, context={"model_name": "m"}) as writer:
        writer.append_rows([make_row(total_reward=1.5)])
        run_id = writer.run_id
    readers = {run_id: ParquetSegmentReader(storage_from_uri(str(log_path)))}

    def resolve(rid: str) -> ParquetSegmentReader:
        from tinker_cookbook.tokendb.agent import ToolExecutionError

        if rid not in readers:
            raise ToolExecutionError(f"unknown run_id {rid!r}")
        return readers[rid]

    toolbox = RegistryToolbox(
        list_runs_fn=lambda: [{"run_id": run_id, "status": {"live": True}}],
        dashboard_fn=lambda: [{"run_id": run_id, "n_rows": 1}],
        resolve_reader=resolve,
        visual_store=VisualStore(
            storage_from_uri(str(tmp_path)), url_base="/visuals", prefix="visuals"
        ),
    )
    names = [t.name for t in toolbox.tool_defs()]
    assert names == ["list_runs", "dashboard", "sql", "search", "get_rollout", "publish_visual"]
    assert (
        "run_id" in next(t for t in toolbox.tool_defs() if t.name == "sql").input_schema["required"]
    )

    outcome, payload = _run_tool(toolbox, "list_runs", {})
    assert payload["runs"][0]["run_id"] == run_id
    outcome, payload = _run_tool(toolbox, "dashboard", {})
    assert payload["runs"][0]["n_rows"] == 1
    outcome, payload = _run_tool(
        toolbox, "sql", {"run_id": run_id, "query": "SELECT count(*) AS n FROM rollouts"}
    )
    assert payload["rows"][0]["n"] == 1
    outcome, payload = _run_tool(toolbox, "sql", {"query": "SELECT 1"})
    assert outcome.is_error and "run_id" in payload["error"]
    outcome, payload = _run_tool(toolbox, "sql", {"run_id": "nope", "query": "SELECT 1"})
    assert outcome.is_error and "unknown run_id" in payload["error"]
