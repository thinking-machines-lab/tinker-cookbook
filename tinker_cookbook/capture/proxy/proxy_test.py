"""Tests for the capture proxy: API shapes, streaming, and capture funneling.

Uses a fake sampling client (canned token sequences) and a fake renderer
implementing the same protocol; real renderers require HF tokenizer downloads
so they are not exercised here. The capture funnel is tested for real: the
fake client's ``sample_async`` is wrapped with the actual instrumentation
wrapper and records are asserted on the exporter's sink.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Any

import pytest
import pytest_asyncio
import tinker
from aiohttp.test_utils import TestClient, TestServer

from tinker_cookbook.capture import instrument as instrument_mod
from tinker_cookbook.capture.exporter import CaptureExporter, CaptureRecord
from tinker_cookbook.capture.instrument import _make_sample_async_wrapper
from tinker_cookbook.capture.proxy.address import parse_address
from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app
from tinker_cookbook.renderers import Message, ParseTermination

CANNED_TOKENS = [5, 6, 7]


class FakeSequence:
    def __init__(self) -> None:
        self.tokens = list(CANNED_TOKENS)
        self.logprobs = [-0.1, -0.2, -0.3]
        self.stop_reason = "stop"


class FakeResponse:
    def __init__(self) -> None:
        self.sequences = [FakeSequence()]


class FakeSamplingClient:
    """SamplingClient-shaped; records calls, returns canned tokens."""

    _sampling_session_id = "sess-proxy"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def sample_async(
        self, prompt: Any, num_samples: int, sampling_params: Any
    ) -> FakeResponse:
        self.calls.append(
            {"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params}
        )
        return FakeResponse()


class FakeRenderer:
    """Renderer-protocol fake: token count derives from message text lengths."""

    def create_conversation_prefix_with_tools(
        self, tools: list[Any], system_prompt: str = ""
    ) -> list[Message]:
        raise NotImplementedError  # like the Renderer base: no tool support

    def get_stop_sequences(self) -> list[str]:
        return ["<END>"]

    def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput:
        tokens = [len(str(m["content"])) for m in messages]
        return tinker.ModelInput.from_ints(tokens)

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        text = "decoded:" + ",".join(str(t) for t in response)
        return Message(role="assistant", content=text), ParseTermination.STOP_SEQUENCE


class ImmediateSink:
    def __init__(self) -> None:
        self.records: list[CaptureRecord] = []

    def export(self, records: Sequence[CaptureRecord], timeout: float | None = None) -> None:
        self.records.extend(records)


def _wait_records(sink: ImmediateSink, n: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(sink.records) >= n:
            return
        time.sleep(0.01)
    raise AssertionError(f"expected {n} records, got {len(sink.records)}")


@pytest.fixture
def sink() -> ImmediateSink:
    return ImmediateSink()


@pytest_asyncio.fixture
async def client(sink: ImmediateSink):  # type: ignore[no-untyped-def]
    """Proxy TestClient whose fake SDK client is instrumented for real."""
    exporter = CaptureExporter(sink, max_batch_size=1, flush_interval_sec=0.02)
    instrument_mod._exporter = exporter
    sampling_client = FakeSamplingClient()
    # Wrap the fake's sample_async with the real instrumentation wrapper, exactly
    # what instrument_tinker() does to tinker.SamplingClient.sample_async.
    sampling_client.sample_async = _make_sample_async_wrapper(  # type: ignore[method-assign]
        FakeSamplingClient.sample_async
    ).__get__(sampling_client)
    deps = ProxyDeps(
        renderer=FakeRenderer(), sampling_client=sampling_client, model_label="fake-model"
    )
    test_client = TestClient(TestServer(make_app(deps)))
    await test_client.start_server()
    test_client._fake_sampling_client = sampling_client  # type: ignore[attr-defined]
    yield test_client
    await test_client.close()
    instrument_mod._exporter = None
    exporter.shutdown()


# ── address parsing ───────────────────────────────────────────────────


def test_parse_address_maps_reserved_keys() -> None:
    assert parse_address("run/r1/attempt/2/iter/3/group/4/traj/5/split/train/purpose/eval") == {
        "run_id": "r1",
        "run_attempt": 2,
        "iteration": 3,
        "group_idx": 4,
        "traj_idx": 5,
        "split": "train",
        "purpose": "eval",
    }


def test_parse_address_preserves_unknown_keys_and_empty() -> None:
    assert parse_address("run/r1/experiment/abl-7") == {"run_id": "r1", "experiment": "abl-7"}
    assert parse_address("") == {}


def test_parse_address_rejects_odd_and_non_int() -> None:
    with pytest.raises(ValueError, match="odd number"):
        parse_address("run/r1/iter")
    with pytest.raises(ValueError, match="integer"):
        parse_address("iter/abc")


def test_parse_address_int32_range() -> None:
    # The store persists these coordinates as Int32; out-of-range values must
    # be rejected at parse time (not after a paid sample, at ingest).
    assert parse_address("iter/2147483647") == {"iteration": 2147483647}
    assert parse_address("iter/-2147483648") == {"iteration": -2147483648}
    with pytest.raises(ValueError, match="out of range"):
        parse_address("iter/2147483648")
    with pytest.raises(ValueError, match="out of range"):
        parse_address("attempt/-2147483649")


# ── Anthropic endpoint ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_non_stream_shape(client: TestClient) -> None:
    resp = await client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 64,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["id"].startswith("msg_")
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "fake-model"
    assert body["content"] == [{"type": "text", "text": "decoded:5,6,7"}]
    assert body["stop_reason"] == "end_turn"
    # FakeRenderer: one token per message (system + user), canned 3 out.
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 3}


@pytest.mark.asyncio
async def test_anthropic_stream_sequence(client: TestClient) -> None:
    resp = await client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    raw = (await resp.read()).decode()
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    deltas = dict(events)
    assert deltas["content_block_delta"]["delta"]["text"] == "decoded:5,6,7"
    assert deltas["message_delta"]["delta"]["stop_reason"] == "end_turn"
    assert deltas["message_delta"]["usage"]["output_tokens"] == 3


# ── OpenAI endpoint ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_non_stream_shape(client: TestClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-x",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "fake-model"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "decoded:5,6,7"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


@pytest.mark.asyncio
async def test_openai_stream_chunks(client: TestClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200
    lines = [line for line in (await resp.read()).decode().splitlines() if line]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "decoded:5,6,7"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"


# ── capture funnel (the whole point) ──────────────────────────────────


@pytest.mark.asyncio
async def test_address_lands_in_capture_records(client: TestClient, sink: ImmediateSink) -> None:
    resp = await client.post(
        "/r/run/r1/group/2/traj/0/experiment/abl/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    _wait_records(sink, 1)
    record = sink.records[0]
    assert record["kind"] == "sample"
    assert record["scope"]["run_id"] == "r1"
    assert record["scope"]["group_idx"] == 2
    assert record["scope"]["traj_idx"] == 0
    assert record["scope"]["experiment"] == "abl"
    assert record["scope"]["requested_model"] == "claude-x"
    assert record["prompt_tokens"] == [2]  # FakeRenderer: len("hi") per message
    assert record["samples"][0]["tokens"] == CANNED_TOKENS


@pytest.mark.asyncio
async def test_bare_path_captures_with_empty_address(
    client: TestClient, sink: ImmediateSink
) -> None:
    resp = await client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status == 200
    _wait_records(sink, 1)
    scope = sink.records[0]["scope"]
    assert "run_id" not in scope
    # No model field in the request either: scope is completely empty.
    assert scope == {}


@pytest.mark.asyncio
async def test_sampling_params_passthrough(client: TestClient) -> None:
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 17,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["\n\nHuman:"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    fake: FakeSamplingClient = client._fake_sampling_client  # type: ignore[attr-defined]
    params = fake.calls[-1]["sampling_params"]
    assert params.max_tokens == 17
    assert params.temperature == 0.3
    assert params.top_p == 0.9
    assert params.top_k == 40
    assert params.stop == ["<END>", "\n\nHuman:"]


# ── unsupported features and errors ───────────────────────────────────


@pytest.mark.asyncio
async def test_unsupported_features_400(client: TestClient) -> None:
    # Tools with a renderer that has no tool-calling support (FakeRenderer
    # lacks create_conversation_prefix_with_tools).
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "tools": [{"name": "bash"}],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["type"] == "error"
    assert "does not support tool calling" in body["error"]["message"]

    # Image content block.
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}],
        },
    )
    assert resp.status == 400
    assert "image" in (await resp.json())["error"]["message"]

    # Unknown role (OpenAI) and OpenAI error shape.
    resp = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "banana", "content": "out"}]},
    )
    assert resp.status == 400
    assert "role" in (await resp.json())["error"]["message"]

    # Malformed address.
    resp = await client.post(
        "/r/run/r1/iter/v1/messages",
        json={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 400

    # Out-of-Int32-range address coordinate.
    resp = await client.post(
        "/r/run/r1/iter/2147483648/v1/messages",
        json={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 400
    assert "out of range" in (await resp.json())["error"]["message"]

    # Empty messages.
    resp = await client.post("/v1/messages", json={"max_tokens": 8, "messages": []})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_healthz(client: TestClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status == 200
    assert await resp.json() == {"status": "ok", "model": "fake-model"}


class TokenStopRenderer(FakeRenderer):
    """Renderer whose stop conditions are token IDs (Qwen3/Llama3 style)."""

    def get_stop_sequences(self) -> list[int]:  # type: ignore[override]
        return [151645]


@pytest_asyncio.fixture
async def token_stop_client():  # type: ignore[no-untyped-def]
    deps = ProxyDeps(
        renderer=TokenStopRenderer(), sampling_client=FakeSamplingClient(), model_label="tok-model"
    )
    test_client = TestClient(TestServer(make_app(deps)))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_client_stops_rejected_with_token_id_renderer(token_stop_client: TestClient) -> None:
    """Client string stops cannot compose with token-id renderer stops
    (SamplingParams.stop is homogeneous): loud 400, never silent discard."""
    resp = await token_stop_client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "stop_sequences": ["END"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400
    assert "token-id stop" in (await resp.json())["error"]["message"]
    # Without client stops the same renderer works fine.
    resp = await token_stop_client.post(
        "/v1/messages",
        json={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_malformed_sampling_fields_400(client: TestClient) -> None:
    # Anthropic: bad max_tokens type.
    resp = await client.post(
        "/v1/messages",
        json={"max_tokens": "bad", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 400
    assert "max_tokens" in (await resp.json())["error"]["message"]

    # Anthropic: bad temperature.
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "temperature": "warm",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400

    # Anthropic: bad stop_sequences.
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "stop_sequences": [1, 2],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400

    # OpenAI: bad max_completion_tokens, OpenAI error shape.
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "max_completion_tokens": "bad",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 400
    assert "max_completion_tokens" in (await resp.json())["error"]["message"]


class ClientStopRenderer(FakeRenderer):
    """Decodes to text containing a client stop, with MALFORMED termination
    (the renderer's own stop signal did not fire)."""

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        return Message(role="assistant", content="hello XX tail"), ParseTermination.MALFORMED


@pytest_asyncio.fixture
async def client_stop_client():  # type: ignore[no-untyped-def]
    deps = ProxyDeps(
        renderer=ClientStopRenderer(),
        sampling_client=FakeSamplingClient(),
        model_label="stop-model",
    )
    test_client = TestClient(TestServer(make_app(deps)))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_client_stop_reported_and_stripped(client_stop_client: TestClient) -> None:
    """Generation ending on a client stop must report stop_reason
    "stop_sequence" with the matched value, and strip the stop text."""
    resp = await client_stop_client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "stop_sequences": ["XX"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    body = await resp.json()
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "XX"
    assert body["content"] == [{"type": "text", "text": "hello "}]

    # Streaming reports the same in message_delta.
    resp = await client_stop_client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "stream": True,
            "stop_sequences": ["XX"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    raw = (await resp.read()).decode()
    delta_line = next(
        line for line in raw.splitlines() if line.startswith("data: ") and "message_delta" in line
    )
    delta = json.loads(delta_line.removeprefix("data: "))
    assert delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "XX"}

    # OpenAI: finish_reason "stop", content stripped.
    resp = await client_stop_client.post(
        "/v1/chat/completions",
        json={"stop": "XX", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = await resp.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == "hello "

    # Renderer stop (no client stops) stays end_turn with null stop_sequence.
    resp = await client_stop_client.post(
        "/v1/messages",
        json={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    )
    body = await resp.json()
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None


@pytest.mark.asyncio
async def test_client_stop_identified_when_excluded_from_output(
    client: TestClient,
) -> None:
    """FakeRenderer terminates STOP_SEQUENCE (clean), so client stops that
    never fire keep end_turn; a MALFORMED parse with a single client stop is
    attributed to it even when the sampler excluded the stop text."""
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "stop_sequences": ["ZZ"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    body = await resp.json()
    # FakeRenderer parses cleanly, so this is a renderer stop: end_turn.
    assert body["stop_reason"] == "end_turn"
    assert body["stop_sequence"] is None


@pytest.mark.asyncio
async def test_openai_unsupported_semantic_fields_400(client: TestClient) -> None:
    base = {"messages": [{"role": "user", "content": "hi"}]}
    for extra in (
        {"n": 2},
        {"response_format": {"type": "json_object"}},
        {"logprobs": True},
        {"logit_bias": {"5": -100}},
        {"presence_penalty": 0.5},
    ):
        resp = await client.post("/v1/chat/completions", json={**base, **extra})
        assert resp.status == 400, extra
    # Harmless/falsy fields are accepted.
    resp = await client.post(
        "/v1/chat/completions",
        json={**base, "n": 1, "user": "u-1", "presence_penalty": 0, "logprobs": False},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_openai_seed_forwarded(client: TestClient) -> None:
    base = {"messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post("/v1/chat/completions", json={**base, "seed": 1234})
    assert resp.status == 200
    fake: FakeSamplingClient = client._fake_sampling_client  # type: ignore[attr-defined]
    assert fake.calls[-1]["sampling_params"].seed == 1234
    # Non-integer seed is a client mistake, not a silent ignore.
    resp = await client.post("/v1/chat/completions", json={**base, "seed": "bad"})
    assert resp.status == 400
    assert "seed" in (await resp.json())["error"]["message"]


@pytest.mark.asyncio
async def test_top_k_validation_400(client: TestClient) -> None:
    resp = await client.post(
        "/v1/messages",
        json={"max_tokens": 8, "top_k": "bad", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status == 400
    assert "top_k" in (await resp.json())["error"]["message"]


@pytest.mark.asyncio
async def test_openai_stream_include_usage(client: TestClient) -> None:
    """stream_options.include_usage must emit the final usage chunk (empty
    choices, totals) before [DONE], with usage: null on earlier chunks."""
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    lines = [line for line in (await resp.read()).decode().splitlines() if line]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}
    assert all(c["usage"] is None for c in chunks[:-1])

    # Without include_usage there is no usage chunk (unchanged behavior).
    resp = await client.post(
        "/v1/chat/completions",
        json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    lines = [line for line in (await resp.read()).decode().splitlines() if line]
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert all("usage" not in c for c in chunks)
    assert all(c["choices"] for c in chunks)

    # Malformed stream_options is a client error, rejected BEFORE sampling
    # (falsy non-objects included; `or {}` coercion must not mask them).
    fake: FakeSamplingClient = client._fake_sampling_client  # type: ignore[attr-defined]
    calls_before = len(fake.calls)
    for bad in (5, [], 0, False, ""):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "stream_options": bad,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status == 400, bad
    assert len(fake.calls) == calls_before  # no samples were submitted


def test_validate_bind() -> None:
    from tinker_cookbook.capture.proxy.serve import validate_bind

    validate_bind("127.0.0.1", None)  # loopback needs no token
    validate_bind("localhost", None)
    validate_bind("::1", None)
    validate_bind("0.0.0.0", "s3cret")  # non-loopback with token is fine
    for host in ("0.0.0.0", "10.0.0.5", "myhost.example"):
        with pytest.raises(SystemExit, match="refusing to bind"):
            validate_bind(host, None)
        with pytest.raises(SystemExit, match="refusing to bind"):
            validate_bind(host, "")


@pytest_asyncio.fixture
async def authed_client():  # type: ignore[no-untyped-def]
    deps = ProxyDeps(
        renderer=FakeRenderer(), sampling_client=FakeSamplingClient(), model_label="auth-model"
    )
    test_client = TestClient(TestServer(make_app(deps, auth_token="s3cret")))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_auth_token_enforced(authed_client: TestClient) -> None:
    payload = {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    resp = await authed_client.post("/v1/messages", json=payload)
    assert resp.status == 401
    assert (await resp.json())["error"]["type"] == "authentication_error"
    resp = await authed_client.post("/v1/messages", json=payload, headers={"x-api-key": "wrong"})
    assert resp.status == 401
    resp = await authed_client.post("/v1/messages", json=payload, headers={"x-api-key": "s3cret"})
    assert resp.status == 200
    resp = await authed_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert resp.status == 200
    # healthz stays open for liveness probes.
    resp = await authed_client.get("/healthz")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_anthropic_thinking_accepted_and_ignored(client: TestClient) -> None:
    """Claude Code sends thinking ENABLED by default: any object shape is
    accepted and served as plain text; only non-object shapes are 400s."""
    base = {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    resp = await client.post(
        "/v1/messages",
        json={**base, "thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["content"] == [{"type": "text", "text": "decoded:5,6,7"}]  # plain text
    resp = await client.post("/v1/messages", json={**base, "thinking": {"type": "disabled"}})
    assert resp.status == 200
    resp = await client.post("/v1/messages", json=base)
    assert resp.status == 200
    resp = await client.post("/v1/messages", json={**base, "thinking": "yes"})
    assert resp.status == 400  # shape validation stays


def test_validate_bind_normalizes_empty_token() -> None:
    """Empty/whitespace tokens are UNSET: refused for non-loopback binds and
    normalized to None (so no auth middleware) on loopback."""
    from tinker_cookbook.capture.proxy.serve import validate_bind

    for empty in ("", "   "):
        with pytest.raises(SystemExit, match="refusing to bind"):
            validate_bind("0.0.0.0", empty)
        assert validate_bind("127.0.0.1", empty) is None
    assert validate_bind("0.0.0.0", "  tok  ") == "tok"
    assert validate_bind("127.0.0.1", None) is None


@pytest.mark.asyncio
async def test_empty_auth_token_installs_no_auth() -> None:
    """make_app with an empty token must NOT install auth middleware that
    would 401 every request on a tokenless loopback deployment."""
    deps = ProxyDeps(renderer=FakeRenderer(), sampling_client=FakeSamplingClient(), model_label="m")
    test_client = TestClient(TestServer(make_app(deps, auth_token="")))
    await test_client.start_server()
    try:
        resp = await test_client.post(
            "/v1/messages",
            json={"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 200  # no token required
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_renderer_string_stop_excluded_not_misattributed() -> None:
    """A string-stop renderer's own stop firing with excluded stop text also
    parses non-clean; with the renderer's stops in play the terminating stop
    is ambiguous, so the response must report end_turn, never infer a client
    stop from parse cleanliness alone."""
    deps = ProxyDeps(
        # ClientStopRenderer inherits get_stop_sequences() == ["<END>"]
        # (renderer string stops present) and parses MALFORMED with text
        # that contains none of the client stops below.
        renderer=ClientStopRenderer(),
        sampling_client=FakeSamplingClient(),
        model_label="ambiguous-stop-model",
    )
    ambiguous_client = TestClient(TestServer(make_app(deps)))
    await ambiguous_client.start_server()
    try:
        for stops in (["AAA"], ["AAA", "BBB"]):
            resp = await ambiguous_client.post(
                "/v1/messages",
                json={
                    "max_tokens": 8,
                    "stop_sequences": stops,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            body = await resp.json()
            assert body["stop_reason"] == "end_turn", stops
            assert body["stop_sequence"] is None
            assert body["content"] == [{"type": "text", "text": "hello XX tail"}]
    finally:
        await ambiguous_client.close()


class NoStopRenderer(ClientStopRenderer):
    """MALFORMED parse, and the renderer contributes NO stop strings."""

    def get_stop_sequences(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_client_stop_attributed_when_only_client_stops_submitted() -> None:
    """With no renderer stops in play, a non-clean parse on stop can only be
    a client stop: named for a single candidate, null for several."""
    deps = ProxyDeps(
        renderer=NoStopRenderer(), sampling_client=FakeSamplingClient(), model_label="m"
    )
    solo_client = TestClient(TestServer(make_app(deps)))
    await solo_client.start_server()
    try:
        resp = await solo_client.post(
            "/v1/messages",
            json={
                "max_tokens": 8,
                "stop_sequences": ["ZZ"],  # excluded from the decoded text
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = await resp.json()
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] == "ZZ"

        resp = await solo_client.post(
            "/v1/messages",
            json={
                "max_tokens": 8,
                "stop_sequences": ["AAA", "BBB"],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        body = await resp.json()
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] is None  # unknowable among several
    finally:
        await solo_client.close()


class FakeToolRenderer(FakeRenderer):
    """Tool-capable fake: records the tools prefix and rendered messages,
    and parses the canned completion as one tool call plus text."""

    def __init__(self) -> None:
        self.prefix_calls: list[tuple[list[Any], str]] = []
        self.last_messages: list[Message] | None = None

    def create_conversation_prefix_with_tools(
        self, tools: list[Any], system_prompt: str = ""
    ) -> list[Message]:
        self.prefix_calls.append((tools, system_prompt))
        names = ",".join(t["name"] for t in tools)
        return [Message(role="system", content=f"TOOLS:{names};SYS:{system_prompt}")]

    def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput:
        self.last_messages = messages
        return super().build_generation_prompt(messages)

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        from tinker_cookbook.renderers import ToolCall

        message = Message(role="assistant", content="let me check")
        message["tool_calls"] = [
            ToolCall(
                id="toolu_01",
                function=ToolCall.FunctionBody(name="bash", arguments='{"cmd": "ls"}'),
            )
        ]
        return message, ParseTermination.STOP_SEQUENCE


@pytest_asyncio.fixture
async def tool_client():  # type: ignore[no-untyped-def]
    renderer = FakeToolRenderer()
    deps = ProxyDeps(
        renderer=renderer, sampling_client=FakeSamplingClient(), model_label="tool-model"
    )
    test_client = TestClient(TestServer(make_app(deps)))
    await test_client.start_server()
    test_client._tool_renderer = renderer  # type: ignore[attr-defined]
    yield test_client
    await test_client.close()


_ANTHROPIC_TOOL_REQUEST = {
    "max_tokens": 64,
    "system": "be safe",
    "tools": [
        {
            "name": "bash",
            "description": "run a command",
            "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ],
    "messages": [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running"},
                {"type": "tool_use", "id": "toolu_00", "name": "bash", "input": {"cmd": "pwd"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_00", "content": "/work"},
            ],
        },
    ],
}


@pytest.mark.asyncio
async def test_anthropic_tool_flow_end_to_end(tool_client: TestClient) -> None:
    """The Claude Code shape: tool catalog + tool_use/tool_result history in,
    a tool_use block out, everything rendered through the renderer's own
    tool machinery."""
    resp = await tool_client.post("/v1/messages", json=_ANTHROPIC_TOOL_REQUEST)
    assert resp.status == 200
    body = await resp.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0] == {"type": "text", "text": "let me check"}
    tool_block = body["content"][1]
    assert tool_block["type"] == "tool_use"
    assert tool_block["id"] == "toolu_01"
    assert tool_block["name"] == "bash"
    assert tool_block["input"] == {"cmd": "ls"}

    renderer: FakeToolRenderer = tool_client._tool_renderer  # type: ignore[attr-defined]
    tools, system_prompt = renderer.prefix_calls[0]
    assert [t["name"] for t in tools] == ["bash"]
    assert system_prompt == "be safe"
    rendered = renderer.last_messages
    assert rendered is not None
    assert rendered[0]["content"] == "TOOLS:bash;SYS:be safe"  # tools prefix first
    tool_message = next(m for m in rendered if m["role"] == "tool")
    assert tool_message["content"] == "/work"
    assert tool_message.get("tool_call_id") == "toolu_00"
    assert tool_message.get("name") == "bash"  # recovered from the tool_use block
    assistant = next(m for m in rendered if m["role"] == "assistant")
    assistant_calls = assistant.get("tool_calls")
    assert assistant_calls is not None
    assert assistant_calls[0].function.name == "bash"


@pytest.mark.asyncio
async def test_anthropic_tool_flow_streaming(tool_client: TestClient) -> None:
    resp = await tool_client.post("/v1/messages", json={**_ANTHROPIC_TOOL_REQUEST, "stream": True})
    assert resp.status == 200
    raw = (await resp.read()).decode()
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    names = [name for name, _ in events]
    # Two content blocks: text then tool_use.
    assert names.count("content_block_start") == 2
    starts = [data for name, data in events if name == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "text"
    assert starts[1]["content_block"] == {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "bash",
        "input": {},
    }
    deltas = [data for name, data in events if name == "content_block_delta"]
    assert deltas[1]["delta"]["type"] == "input_json_delta"
    assert json.loads(deltas[1]["delta"]["partial_json"]) == {"cmd": "ls"}
    message_delta = next(data for name, data in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_openai_tool_flow(tool_client: TestClient) -> None:
    resp = await tool_client.post(
        "/v1/chat/completions",
        json={
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "bash", "description": "run", "parameters": {}},
                }
            ],
            "messages": [
                {"role": "system", "content": "be safe"},
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_00",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"cmd": "pwd"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_00", "content": "/work"},
            ],
        },
    )
    assert resp.status == 200
    body = await resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "toolu_01",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
        }
    ]
    renderer: FakeToolRenderer = tool_client._tool_renderer  # type: ignore[attr-defined]
    assert renderer.prefix_calls[-1][1] == "be safe"  # system folded into prefix
    rendered = renderer.last_messages
    assert rendered is not None
    tool_message = next(m for m in rendered if m["role"] == "tool")
    assert tool_message.get("name") == "bash"  # recovered from assistant tool_calls


@pytest.mark.asyncio
async def test_openai_tool_flow_streaming(tool_client: TestClient) -> None:
    resp = await tool_client.post(
        "/v1/chat/completions",
        json={
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    lines = [line for line in (await resp.read()).decode().splitlines() if line]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    tool_chunk = chunks[-2]
    assert tool_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "bash"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_forced_tool_choice_400(tool_client: TestClient) -> None:
    base = {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    resp = await tool_client.post(
        "/v1/messages",
        json={**base, "tools": [{"name": "bash"}], "tool_choice": {"type": "tool", "name": "bash"}},
    )
    assert resp.status == 400
    assert "tool_choice" in (await resp.json())["error"]["message"]
    resp = await tool_client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            "tool_choice": "required",
        },
    )
    assert resp.status == 400
    # auto is fine.
    resp = await tool_client.post(
        "/v1/messages",
        json={**base, "tools": [{"name": "bash"}], "tool_choice": {"type": "auto"}},
    )
    assert resp.status == 200


@pytest.mark.asyncio
async def test_tool_choice_none_skips_tool_rendering(
    tool_client: TestClient, client: TestClient
) -> None:
    """Both API specs define tool_choice "none" as forbidding tool calls:
    the catalog must not be rendered into the prompt (asserted on the fake
    renderer), while the request itself is still accepted."""
    renderer: FakeToolRenderer = tool_client._tool_renderer  # type: ignore[attr-defined]
    prefix_calls_before = len(renderer.prefix_calls)

    # Anthropic: catalog + {"type": "none"}.
    resp = await tool_client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "system": "be safe",
            "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    assert len(renderer.prefix_calls) == prefix_calls_before  # catalog NOT rendered
    rendered = renderer.last_messages
    assert rendered is not None
    assert rendered[0] == {"role": "system", "content": "be safe"}  # plain system msg

    # OpenAI: "none" string form.
    resp = await tool_client.post(
        "/v1/chat/completions",
        json={
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            "tool_choice": "none",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    assert len(renderer.prefix_calls) == prefix_calls_before

    # Plain-text response guarantee: FakeRenderer (no tool support at all)
    # would 400 if the catalog were rendered; with "none" it serves text.
    resp = await client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "tools": [{"name": "bash"}],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["content"] == [{"type": "text", "text": "decoded:5,6,7"}]
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
            "tool_choice": "none",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    assert (await resp.json())["choices"][0]["message"]["content"] == "decoded:5,6,7"

    # And "auto" still renders the catalog.
    resp = await tool_client.post(
        "/v1/messages",
        json={
            "max_tokens": 8,
            "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status == 200
    assert len(renderer.prefix_calls) == prefix_calls_before + 1


@pytest.mark.asyncio
async def test_claude_code_beta_shape_system_in_messages(client: TestClient) -> None:
    """Claude Code (?beta=true) appends system-role messages inside
    messages[] carrying agent/skill text (shape from the captured
    cc-requests.jsonl); they fold into the system prompt in encounter
    order, and array-of-blocks content is accepted."""
    resp = await client.post(
        "/v1/messages?beta=true",
        json={
            "model": "claude-fable-5",
            "max_tokens": 8,
            "system": "top-level system",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                    ],
                },
                {"role": "system", "content": "Available agent types: ..."},
                {"role": "system", "content": [{"type": "text", "text": "skill text"}]},
            ],
        },
    )
    assert resp.status == 200
    fake: FakeSamplingClient = client._fake_sampling_client  # type: ignore[attr-defined]
    # FakeRenderer encodes one token per message: system prompt (folded from
    # 3 sources) + 1 user message = 2 rendered messages.
    prompt = fake.calls[-1]["prompt"]
    assert prompt.length == 2


@pytest.mark.asyncio
async def test_rejections_logged_at_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Every 400 leaves exactly one warn line: path + run_id + validation
    message (field names only, never request contents)."""
    import logging

    with caplog.at_level(logging.WARNING, logger="tinker_cookbook.capture.proxy.app"):
        resp = await client.post(
            "/r/run/r-log/v1/messages",
            json={"max_tokens": "bad", "messages": [{"role": "user", "content": "secret"}]},
        )
        assert resp.status == 400
    rejections = [r for r in caplog.records if "rejected request" in r.getMessage()]
    assert len(rejections) == 1
    line = rejections[0].getMessage()
    assert "/r/run/r-log/v1/messages" in line
    assert "run_id=r-log" in line
    assert "max_tokens" in line
    assert "secret" not in line  # no request contents


@pytest.mark.asyncio
async def test_tool_choice_none_suppresses_parsed_tool_calls(tool_client: TestClient) -> None:
    """tool_choice "none" must hold on the RESPONSE side too: FakeToolRenderer
    parses a ToolCall from every completion (a tool-capable model emitting
    its learned syntax without a catalog), but under "none" the response is
    plain text on both APIs, streaming included; under "auto" the tool call
    still comes through."""
    anthropic_none = {
        "max_tokens": 8,
        "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "none"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = await tool_client.post("/v1/messages", json=anthropic_none)
    assert resp.status == 200
    body = await resp.json()
    assert body["stop_reason"] == "end_turn"  # never tool_use
    assert body["content"] == [{"type": "text", "text": "let me check"}]  # no tool_use block

    # Streaming: no tool_use blocks in the event stream, end_turn delta.
    resp = await tool_client.post("/v1/messages", json={**anthropic_none, "stream": True})
    raw = (await resp.read()).decode()
    assert "tool_use" not in raw
    assert '"stop_reason": "end_turn"' in raw

    # OpenAI: no tool_calls, finish_reason stop (stream and non-stream).
    openai_none = {
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
        "tool_choice": "none",
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = await tool_client.post("/v1/chat/completions", json=openai_none)
    body = await resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "let me check"
    resp = await tool_client.post("/v1/chat/completions", json={**openai_none, "stream": True})
    raw = (await resp.read()).decode()
    assert "tool_calls" not in raw
    assert '"finish_reason": "stop"' in raw

    # Contrast: "auto" still emits the tool call.
    resp = await tool_client.post(
        "/v1/messages", json={**anthropic_none, "tool_choice": {"type": "auto"}}
    )
    body = await resp.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][1]["type"] == "tool_use"


# ── backend context overflow and client-abort resilience ─────────────────


def _bad_request_error(message: str) -> tinker.BadRequestError:
    import httpx

    return tinker.BadRequestError(
        message,
        response=httpx.Response(400, request=httpx.Request("POST", "http://backend/sample")),
        body=None,
    )


class FailingSamplingClient:
    """SamplingClient-shaped; raises a canned error from sample_async."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def sample_async(self, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        raise self.error


async def _failing_client(error: Exception) -> TestClient:
    deps = ProxyDeps(
        renderer=FakeRenderer(), sampling_client=FailingSamplingClient(error), model_label="fake"
    )
    test_client = TestClient(TestServer(make_app(deps)))
    await test_client.start_server()
    return test_client


@pytest.mark.asyncio
async def test_context_overflow_maps_to_anthropic_400() -> None:
    """A backend context-window 400 must reach the client as an Anthropic-shaped
    invalid_request_error whose message says the prompt is too long: streaming
    agent clients key their history-compaction behavior off that signal, and a
    generic 500 makes them retry the same over-long prompt instead."""
    error = _bad_request_error(
        "Prompt length plus max_tokens exceeds the model's context window: "
        "67984 prompt tokens + 4096 max_tokens > 65536"
    )
    test_client = await _failing_client(error)
    try:
        resp = await test_client.post(
            "/v1/messages",
            json={"max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "prompt is too long" in body["error"]["message"]
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_context_overflow_maps_to_openai_400() -> None:
    error = _bad_request_error("6000 prompt tokens + 1024 max_tokens > 4096")
    test_client = await _failing_client(error)
    try:
        resp = await test_client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "prompt is too long" in body["error"]["message"]
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_generic_context_window_mention_still_server_error() -> None:
    """A backend 400 that merely MENTIONS the context window without saying the
    prompt exceeds it (e.g. an invalid max_tokens parameter) must not be
    rewritten as "prompt is too long": no amount of history compaction can fix
    it, and the 400 would send agent clients into a useless compaction loop."""
    error = _bad_request_error("max_tokens must not exceed the context window")
    test_client = await _failing_client(error)
    try:
        resp = await test_client.post(
            "/v1/messages",
            json={"max_tokens": 999999, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 500
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_other_backend_400_still_server_error() -> None:
    """Non-overflow backend 400s keep the current behavior (server error), so
    genuine proxy/renderer bugs stay loud instead of masquerading as client
    mistakes."""
    error = _bad_request_error("Unknown model requested")
    test_client = await _failing_client(error)
    try:
        resp = await test_client.post(
            "/v1/messages",
            json={"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 500
    finally:
        await test_client.close()


@pytest.mark.asyncio
async def test_sse_client_abort_on_prepare_is_swallowed() -> None:
    """The client aborting an in-flight streaming request surfaces as
    a ConnectionResetError from response.prepare() (aiohttp 3.10+ raises its
    ClientConnectionResetError subclass; 3.9 raises the builtin); it must be
    swallowed (debug log), not escape the handler as an ERROR traceback."""
    from unittest import mock

    from aiohttp.test_utils import make_mocked_request

    from tinker_cookbook.capture.proxy.app import _serve_sse

    writer = mock.AsyncMock()
    writer.write_headers.side_effect = ConnectionResetError("Cannot write to closing transport")
    request = make_mocked_request("POST", "/v1/messages", writer=writer)
    response = await _serve_sse(request, [b"data: {}\n\n"])  # must not raise
    assert response is not None


@pytest.mark.asyncio
async def test_sse_client_abort_mid_stream_is_swallowed() -> None:
    from unittest import mock

    from aiohttp.test_utils import make_mocked_request

    from tinker_cookbook.capture.proxy.app import _serve_sse

    writer = mock.AsyncMock()
    # Builtin ConnectionResetError: available on all supported aiohttp
    # versions (aiohttp>=3.10 raises a subclass of it).
    writer.write.side_effect = ConnectionResetError("Cannot write to closing transport")
    request = make_mocked_request("POST", "/v1/messages", writer=writer)
    response = await _serve_sse(request, [b"data: {}\n\n", b"data: [DONE]\n\n"])  # must not raise
    assert response is not None
