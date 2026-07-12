"""Minimal provider-agnostic chat client for the tokendb chat agent.

Speaks the Anthropic Messages API and the OpenAI Responses API over plain
HTTP (aiohttp + server-sent events); no provider SDK dependencies. A third
provider, ``tinker``, samples from any Tinker-served model through the
tinker SDK and the cookbook's renderer machinery (see
:mod:`tinker_cookbook.tokendb_studio.tinker_llm`). All providers are normalized into
one internal representation:

- :class:`Message` / :class:`ToolCall` / :class:`ToolDef` describe the
  conversation and tools once; ``build_anthropic_request`` /
  ``build_openai_responses_request`` serialize them into each provider's wire
  format.
- :meth:`LLMClient.stream` yields typed :data:`LLMEvent` objects
  (:class:`TextDelta`, :class:`ToolCallEvent`, :class:`Done`,
  :class:`ErrorEvent`) regardless of provider.

API key resolution order: an explicitly configured key (held in memory, e.g.
set through the viewer's settings endpoint), then the provider's environment
variable (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``TINKER_API_KEY``).

The HTTP layer is behind the small :class:`SSETransport` protocol so tests can
inject a scripted transport and exercise the full parsing path offline.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_API_URL = "https://api.openai.com/v1/responses"

# Plain-constant defaults; override via LLMConfig(model=...) or the viewer's
# agent-config endpoint. The tinker default is a static fallback; the viewer
# usually resolves a better one from the fetched supported-model list.
DEFAULT_MODELS = {
    "anthropic": "claude-fable-5",
    "openai": "gpt-5.6-sol",
    "tinker": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}
# Curated per-provider suggestions surfaced by the viewer's settings UI.
# Not exhaustive; any model id the provider accepts still works via the
# "custom" option / LLMConfig(model=...). The tinker list is dynamic (server
# capabilities), so it is empty here and filled in by the config endpoint.
KNOWN_MODELS = {
    "anthropic": ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
    "tinker": [],
}
API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "tinker": "TINKER_API_KEY",
}
DEFAULT_MAX_TOKENS = 8192
# Order in which providers are considered when picking a default (first one
# with an available API key wins).
PROVIDER_PREFERENCE = ("anthropic", "openai", "tinker")


# --- Internal message / tool representation ---


@dataclass
class ToolDef:
    """One tool the model may call, with a JSON-schema parameter spec."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One conversation message in the provider-neutral representation.

    Roles: ``user``, ``assistant`` (may carry ``tool_calls``), and ``tool``
    (a tool result; ``tool_call_id`` links it to the assistant's call).
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


# --- Streaming events ---


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallEvent:
    call: ToolCall


@dataclass
class Done:
    stop_reason: str | None = None


@dataclass
class ErrorEvent:
    message: str


LLMEvent = TextDelta | ToolCallEvent | Done | ErrorEvent


# --- Config / key resolution ---


@dataclass
class LLMConfig:
    """Provider, model, and key configuration for :class:`LLMClient`."""

    provider: str = "anthropic"  # "anthropic" | "openai" | "tinker"
    model: str | None = None  # None resolves to DEFAULT_MODELS[provider]
    api_key: str | None = None  # explicit key beats the environment variable
    max_tokens: int = DEFAULT_MAX_TOKENS
    base_url: str | None = None  # override for proxies / compatible servers
    # OpenAI Responses API reasoning effort ("none" | "low" | "medium" | ...).
    # None omits the field entirely, leaving the model's default in place.
    reasoning_effort: str | None = None

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    def resolve_api_key(self) -> str | None:
        """Explicit runtime key first, then the provider's env var."""
        if self.api_key:
            return self.api_key
        env_var = API_KEY_ENV_VARS.get(self.provider)
        return os.environ.get(env_var) if env_var else None


def detect_default_provider(api_key: str | None = None) -> str:
    """The first provider (in PROVIDER_PREFERENCE order) with an available key.

    A key is available if ``api_key`` (a runtime-configured key) is set or the
    provider's environment variable is. Falls back to ``"anthropic"`` when no
    provider has a key, so the viewer still has something to prompt for.
    """
    for provider in PROVIDER_PREFERENCE:
        if LLMConfig(provider=provider, api_key=api_key).resolve_api_key() is not None:
            return provider
    return "anthropic"


# --- Provider wire formats ---


def build_anthropic_request(
    config: LLMConfig,
    api_key: str,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Serialize a conversation into an Anthropic Messages API request."""
    wire_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant":
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            wire_messages.append({"role": "assistant", "content": content})
        elif msg.role == "tool":
            # Tool results are user-role content blocks in the Messages API.
            wire_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                }
            )
        else:
            wire_messages.append({"role": "user", "content": msg.content})
    payload: dict[str, Any] = {
        "model": config.resolved_model(),
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": wire_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    return config.base_url or ANTHROPIC_API_URL, headers, payload


def build_openai_responses_request(
    config: LLMConfig,
    api_key: str,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Serialize a conversation into an OpenAI Responses API request.

    The Responses API (unlike Chat Completions) supports function tools on
    reasoning models without forcing ``reasoning_effort`` to ``"none"``. The
    system prompt travels as top-level ``instructions``; assistant tool calls
    and tool results become ``function_call`` / ``function_call_output`` input
    items linked by ``call_id``.
    """
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "assistant":
            if msg.content:
                input_items.append({"role": "assistant", "content": msg.content})
            for call in msg.tool_calls:
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
        elif msg.role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                }
            )
        else:
            input_items.append({"role": "user", "content": msg.content})
    payload: dict[str, Any] = {
        "model": config.resolved_model(),
        "instructions": system,
        "input": input_items,
        "max_output_tokens": config.max_tokens,
        "stream": True,
        # The full conversation is threaded client-side, so opt out of
        # server-side response storage.
        "store": False,
    }
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tools
        ]
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    return config.base_url or OPENAI_API_URL, headers, payload


# --- Transport (aiohttp SSE; injectable for tests) ---


class SSETransport(Protocol):
    """POST a JSON payload and yield ``(event_name, data)`` per SSE event.

    ``event_name`` is the SSE ``event:`` field (``None`` when absent) and
    ``data`` is the parsed JSON of the ``data:`` field. A ``data: [DONE]``
    sentinel is not yielded; the stream just ends.
    """

    def stream_sse(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]: ...


class AiohttpSSETransport:
    """Default transport: aiohttp POST with incremental SSE parsing."""

    async def stream_sse(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
        import aiohttp

        parser = _SSEParser()
        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=headers, json=payload) as resp,
        ):
            if resp.status != 200:
                body = (await resp.text())[:2000]
                raise RuntimeError(f"LLM API error {resp.status}: {body}")
            async for raw_line in resp.content:
                event = parser.feed(raw_line.decode("utf-8", errors="replace").rstrip("\r\n"))
                if event is not None:
                    yield event


class _SSEParser:
    """Incremental SSE parser: ``event:`` / ``data:`` lines into events.

    Comment lines and unknown fields are ignored; a blank line terminates one
    event. OpenAI's ``data: [DONE]`` sentinel is swallowed.
    """

    def __init__(self) -> None:
        self.event_name: str | None = None
        self.data_lines: list[str] = []

    def feed(self, line: str) -> tuple[str | None, dict[str, Any]] | None:
        """Feed one line (no trailing newline); return a completed event or None."""
        if line == "":
            data = "\n".join(self.data_lines)
            name = self.event_name
            self.event_name = None
            self.data_lines = []
            if not data or data == "[DONE]":
                return None
            return (name, json.loads(data))
        if line.startswith("event:"):
            self.event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            self.data_lines.append(line[len("data:") :].strip())
        return None


# --- Stream normalization ---


async def _stream_anthropic(
    sse: AsyncIterator[tuple[str | None, dict[str, Any]]],
) -> AsyncIterator[LLMEvent]:
    """Normalize Anthropic Messages streaming events."""
    stop_reason: str | None = None
    # In-flight tool_use blocks by content-block index (input arrives as
    # partial JSON deltas that must be accumulated).
    pending_tools: dict[int, dict[str, Any]] = {}
    async for name, data in sse:
        event_type = name or data.get("type")
        if event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                pending_tools[int(data["index"])] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "json": "",
                }
        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                yield TextDelta(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                pending = pending_tools.get(int(data["index"]))
                if pending is not None:
                    pending["json"] += delta.get("partial_json", "")
        elif event_type == "content_block_stop":
            pending = pending_tools.pop(int(data["index"]), None)
            if pending is not None:
                arguments = json.loads(pending["json"]) if pending["json"].strip() else {}
                yield ToolCallEvent(ToolCall(pending["id"], pending["name"], arguments))
        elif event_type == "message_delta":
            stop_reason = data.get("delta", {}).get("stop_reason") or stop_reason
        elif event_type == "error":
            error = data.get("error", data)
            yield ErrorEvent(str(error.get("message", error)))
            return
    yield Done(stop_reason)


async def _stream_openai_responses(
    sse: AsyncIterator[tuple[str | None, dict[str, Any]]],
) -> AsyncIterator[LLMEvent]:
    """Normalize OpenAI Responses API streaming events.

    Function-call arguments arrive as string deltas keyed by ``item_id`` and
    are accumulated until the item's ``response.output_item.done``, which also
    carries the authoritative complete ``arguments`` string.
    """
    stop_reason: str | None = None
    emitted_tool_call = False
    # In-flight function_call output items by item id (arguments accumulate
    # across response.function_call_arguments.delta events).
    pending_tools: dict[str, dict[str, Any]] = {}
    async for name, data in sse:
        event_type = name or data.get("type")
        if event_type == "response.output_text.delta":
            yield TextDelta(data.get("delta", ""))
        elif event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                pending_tools[str(item.get("id", ""))] = {
                    "call_id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "json": item.get("arguments") or "",
                }
        elif event_type == "response.function_call_arguments.delta":
            pending = pending_tools.get(str(data.get("item_id", "")))
            if pending is not None:
                pending["json"] += data.get("delta", "")
        elif event_type == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                pending = pending_tools.pop(str(item.get("id", "")), None) or {
                    "call_id": "",
                    "name": "",
                    "json": "",
                }
                raw = item.get("arguments") or pending["json"]
                arguments = json.loads(raw) if raw.strip() else {}
                call_id = item.get("call_id") or pending["call_id"]
                tool_name = item.get("name") or pending["name"]
                yield ToolCallEvent(ToolCall(call_id, tool_name, arguments))
                emitted_tool_call = True
        elif event_type == "response.completed":
            stop_reason = "tool_calls" if emitted_tool_call else "stop"
        elif event_type == "response.incomplete":
            details = (data.get("response") or {}).get("incomplete_details") or {}
            stop_reason = details.get("reason") or "incomplete"
        elif event_type == "response.failed":
            error = (data.get("response") or {}).get("error") or {}
            yield ErrorEvent(str(error.get("message", error)))
            return
        elif event_type == "error":
            yield ErrorEvent(str(data.get("message", data)))
            return
    yield Done(stop_reason)


class LLMClient:
    """Streaming chat client over an :class:`SSETransport`."""

    def __init__(
        self,
        config: LLMConfig,
        transport: SSETransport | None = None,
        tinker_session_factory: Any = None,
    ) -> None:
        if config.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider {config.provider!r} "
                "(expected 'anthropic', 'openai', or 'tinker')"
            )
        self.config = config
        self.transport: SSETransport = transport or AiohttpSSETransport()
        # Test seam for the tinker provider (tinker_llm.SessionFactory);
        # None uses the real SDK + renderer machinery.
        self.tinker_session_factory = tinker_session_factory

    async def stream(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolDef] = ()
    ) -> AsyncIterator[LLMEvent]:
        """Stream one model turn as normalized :data:`LLMEvent` objects.

        Always terminates with :class:`Done` or :class:`ErrorEvent`; transport
        and parse failures surface as :class:`ErrorEvent` rather than raising,
        so callers can render them.
        """
        api_key = self.config.resolve_api_key()
        if not api_key:
            env_var = API_KEY_ENV_VARS[self.config.provider]
            yield ErrorEvent(
                f"No API key configured for provider {self.config.provider!r}: "
                f"set {env_var} or configure a key in the viewer settings."
            )
            return
        if self.config.provider == "tinker":
            # Renderer + SDK path (see tinker_llm); imported lazily to keep
            # the HTTP providers importable without the tinker/renderer stack.
            from tinker_cookbook.tokendb_studio import tinker_llm

            try:
                async for event in tinker_llm.stream_tinker(
                    self.config,
                    api_key,
                    system,
                    messages,
                    tools,
                    session_factory=self.tinker_session_factory,
                ):
                    yield event
            except Exception as e:
                logger.warning("Tinker LLM turn failed: %s", e)
                yield ErrorEvent(str(e))
            return
        build = (
            build_anthropic_request
            if self.config.provider == "anthropic"
            else build_openai_responses_request
        )
        url, headers, payload = build(self.config, api_key, system, messages, tools)
        normalize = (
            _stream_anthropic if self.config.provider == "anthropic" else _stream_openai_responses
        )
        try:
            async for event in normalize(self.transport.stream_sse(url, headers, payload)):
                yield event
        except Exception as e:
            logger.warning("LLM stream failed: %s", e)
            yield ErrorEvent(str(e))
