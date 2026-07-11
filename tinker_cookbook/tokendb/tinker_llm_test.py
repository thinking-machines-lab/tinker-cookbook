"""Tests for the tinker chat-agent provider. No network: sampling is a
scripted fake behind the ``TinkerSession`` seam and renderers run on a tiny
offline tokenizer, so the real renderer prompt/parse machinery is exercised
end to end."""

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("tinker")

import tinker

from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.renderers.role_colon import RoleColonRenderer
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
)
from tinker_cookbook.tokendb.tinker_llm import (
    TinkerSession,
    base_model_name,
    parse_json_tool_blocks,
    pick_default_model,
    render_conversation,
)


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINKER_API_KEY", raising=False)


class FakeTokenizer:
    """Reversible offline tokenizer: special strings are single tokens,
    everything else is one token per character."""

    def __init__(self, specials: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")) -> None:
        self._specials = {s: i for i, s in enumerate(specials)}
        self._by_id = {i: s for s, i in self._specials.items()}
        self._offset = len(specials)
        self.bos_token = None
        self.eos_token_id = self._specials.get("<|im_end|>")
        self.name_or_path = "fake-tokenizer"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        i = 0
        while i < len(text):
            for special, token_id in self._specials.items():
                if text.startswith(special, i):
                    tokens.append(token_id)
                    i += len(special)
                    break
            else:
                tokens.append(ord(text[i]) + self._offset)
                i += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            self._by_id[t] if t in self._by_id else chr(t - self._offset) for t in tokens
        )


class ScriptedSampler:
    """TinkerSampler stub: records prompts, replays scripted completions."""

    def __init__(self, tokenizer: FakeTokenizer, completions: list[str]) -> None:
        self._tokenizer = tokenizer
        self._completions = list(completions)
        self.prompts: list[str] = []
        self.stops: list[object] = []

    async def sample(self, prompt: tinker.ModelInput, max_tokens: int, stop) -> list[int]:
        tokens = [t for chunk in prompt.chunks for t in chunk.tokens]
        self.prompts.append(self._tokenizer.decode(tokens))
        self.stops.append(stop)
        if not self._completions:
            raise AssertionError("ScriptedSampler ran out of completions")
        return self._tokenizer.encode(self._completions.pop(0))


SQL_TOOL = ToolDef(
    "sql",
    "Run SQL.",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)


def make_client(renderer, sampler, model: str = "test/model") -> LLMClient:
    session = TinkerSession(renderer=renderer, sampler=sampler)
    factories: list[tuple[str, str]] = []

    def factory(model_name: str, api_key: str) -> TinkerSession:
        factories.append((model_name, api_key))
        return session

    client = LLMClient(
        LLMConfig(provider="tinker", model=model, api_key="tml-test"),
        tinker_session_factory=factory,
    )
    client._factory_calls = factories  # type: ignore[attr-defined]
    return client


def collect(aiter):
    async def main():
        return [event async for event in aiter]

    return asyncio.run(main())


# --- Renderer-native tool calling (Qwen3) ---


def test_native_prompt_and_tool_call_parse():
    tokenizer = FakeTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    completion = (
        '<tool_call>\n{"name": "sql", "arguments": {"query": "SELECT 1"}}\n</tool_call><|im_end|>'
    )
    sampler = ScriptedSampler(tokenizer, [completion])
    client = make_client(renderer, sampler)
    events = collect(
        client.stream(
            "You analyze token DBs.", [Message(role="user", content="count rows")], [SQL_TOOL]
        )
    )
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].call.name == "sql"
    assert calls[0].call.arguments == {"query": "SELECT 1"}
    assert calls[0].call.id  # generated when the renderer format has none
    assert events[-1] == Done("tool_use")

    # The prompt went through the renderer: system prompt + tool schemas in
    # Qwen's <tools> block + the conversation + the assistant header.
    prompt = sampler.prompts[0]
    assert "You analyze token DBs." in prompt
    assert "<tools>" in prompt and '"name": "sql"' in prompt
    assert "<|im_start|>user\ncount rows<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert sampler.stops[0] == renderer.get_stop_sequences()
    assert client._factory_calls == [("test/model", "tml-test")]  # type: ignore[attr-defined]


def test_native_history_renders_tool_calls_and_results():
    tokenizer = FakeTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    sampler = ScriptedSampler(tokenizer, ["There is 1 row.<|im_end|>"])
    client = make_client(renderer, sampler)
    messages = [
        Message(role="user", content="count rows"),
        Message(
            role="assistant",
            content="Checking.",
            tool_calls=[ToolCall(id="c1", name="sql", arguments={"query": "SELECT 1"})],
        ),
        Message(role="tool", content='{"n_rows": 1}', tool_call_id="c1"),
    ]
    events = collect(client.stream("SYS", messages, [SQL_TOOL]))
    assert events == [TextDelta("There is 1 row."), Done("end_turn")]
    prompt = sampler.prompts[0]
    # Prior tool call rendered in Qwen's native format, result as tool_response.
    assert "<tool_call>" in prompt and '"SELECT 1"' in prompt
    assert '<tool_response>\n{"n_rows": 1}\n</tool_response>' in prompt


# --- JSON-in-text fallback (renderers without a tool convention) ---


def test_fallback_prompt_documents_protocol_and_parses_block():
    tokenizer = FakeTokenizer()
    renderer = RoleColonRenderer(tokenizer)
    completion = (
        'Let me check.\n\n```json\n{"tool": "sql", "arguments": {"query": "SELECT 1"}}\n```'
        "\n\nUser:"
    )
    sampler = ScriptedSampler(tokenizer, [completion])
    client = make_client(renderer, sampler)
    events = collect(client.stream("SYS", [Message(role="user", content="count rows")], [SQL_TOOL]))
    assert events[0] == TextDelta("Let me check.")
    assert isinstance(events[1], ToolCallEvent)
    assert events[1].call.name == "sql"
    assert events[1].call.arguments == {"query": "SELECT 1"}
    assert events[-1] == Done("tool_use")
    prompt = sampler.prompts[0]
    assert "SYS" in prompt
    assert '{"tool": "<tool name>", "arguments": {...}}' in prompt  # protocol docs
    assert '"name": "sql"' in prompt  # tool schema


def test_fallback_history_rendering():
    tokenizer = FakeTokenizer()
    renderer = RoleColonRenderer(tokenizer)
    messages = [
        Message(role="user", content="count rows"),
        Message(
            role="assistant",
            content="Checking.",
            tool_calls=[ToolCall(id="c1", name="sql", arguments={"query": "SELECT 1"})],
        ),
        Message(role="tool", content='{"n_rows": 1}', tool_call_id="c1"),
    ]
    rendered, native = render_conversation(renderer, "SYS", messages, [SQL_TOOL])
    assert native is False
    assert rendered[0]["role"] == "system"
    # Prior tool calls become fenced JSON blocks; results become user messages.
    assert '```json\n{"tool": "sql"' in rendered[2]["content"]
    assert rendered[3]["role"] == "user"
    assert rendered[3]["content"].startswith("Tool result for sql:")


def test_parse_json_tool_blocks():
    text, calls = parse_json_tool_blocks(
        'Before.\n```json\n{"tool": "sql", "arguments": {"query": "SELECT 1"}}\n```\nAfter.'
    )
    assert text == "Before.\n\nAfter."
    assert calls == [ToolCall(id="tinker_call_0", name="sql", arguments={"query": "SELECT 1"})]

    # Invalid JSON stays in the text so the failure is visible.
    text, calls = parse_json_tool_blocks('```json\n{"tool": "sql", oops}\n```')
    assert calls == []
    assert "oops" in text

    # Valid JSON that is not a tool call also stays.
    text, calls = parse_json_tool_blocks('```json\n{"data": [1, 2]}\n```')
    assert calls == []
    assert '"data"' in text

    # Missing arguments defaults to {}.
    _, calls = parse_json_tool_blocks('```json\n{"tool": "list_runs"}\n```')
    assert calls == [ToolCall(id="tinker_call_0", name="list_runs", arguments={})]

    # No block: plain text passthrough.
    text, calls = parse_json_tool_blocks("Just an answer.")
    assert text == "Just an answer." and calls == []


# --- Error paths and helpers ---


def test_missing_tinker_key_yields_error_event():
    client = LLMClient(LLMConfig(provider="tinker"))
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "TINKER_API_KEY" in events[0].message


def test_sampler_failure_surfaces_as_error_event():
    class FailingSampler:
        async def sample(self, prompt, max_tokens, stop):
            raise RuntimeError("service unavailable")

    client = make_client(RoleColonRenderer(FakeTokenizer()), FailingSampler())
    events = collect(client.stream("sys", [Message(role="user", content="hi")]))
    assert isinstance(events[-1], ErrorEvent)
    assert "service unavailable" in events[-1].message


def test_base_model_name_strips_capability_suffix():
    assert base_model_name("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
    assert base_model_name("moonshotai/Kimi-K2.5:peft:131072") == "moonshotai/Kimi-K2.5"


def test_pick_default_model_prefers_qwen3_instruct():
    models = [
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
    ]
    assert pick_default_model(models) == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert pick_default_model(["a/b", "c/d"]) == "a/b"
    assert pick_default_model([]) is None


def test_native_arguments_json_roundtrip():
    """Arguments survive the renderer's JSON-string encoding round trip."""
    tokenizer = FakeTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="sql", arguments={"query": 'SELECT "x"'})],
        )
    ]
    rendered, native = render_conversation(renderer, "SYS", messages, [SQL_TOOL])
    assert native is True
    rcall = rendered[-1]["tool_calls"][0]
    assert json.loads(rcall.function.arguments) == {"query": 'SELECT "x"'}
