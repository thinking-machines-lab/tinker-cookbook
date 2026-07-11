"""Minimal provider-agnostic chat-completions client for the tokendb chat agent.

Speaks the Anthropic Messages API and the OpenAI Chat Completions API over
plain HTTP (aiohttp + server-sent events); no provider SDK dependencies. Both
providers are normalized into one internal representation:

- :class:`Message` / :class:`ToolCall` / :class:`ToolDef` describe the
  conversation and tools once; ``build_anthropic_request`` /
  ``build_openai_request`` serialize them into each provider's wire format.
- :meth:`LLMClient.stream` yields typed :data:`LLMEvent` objects
  (:class:`TextDelta`, :class:`ToolCallEvent`, :class:`Done`,
  :class:`ErrorEvent`) regardless of provider.

API key resolution order: an explicitly configured key (held in memory, e.g.
set through the viewer's settings endpoint), then the provider's environment
variable (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``).

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
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Plain-constant defaults; override via LLMConfig(model=...) or the viewer's
# agent-config endpoint.
DEFAULT_MODELS = {
    "anthropic": "claude-fable-5",
    "openai": "gpt-5.6-sol",
}
# Curated per-provider suggestions surfaced by the viewer's settings UI.
# Not exhaustive; any model id the provider accepts still works via the
# "custom" option / LLMConfig(model=...).
KNOWN_MODELS = {
    "anthropic": ["claude-fable-5", "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "openai": ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
}
API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
DEFAULT_MAX_TOKENS = 8192


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

    provider: str = "anthropic"  # "anthropic" | "openai"
    model: str | None = None  # None resolves to DEFAULT_MODELS[provider]
    api_key: str | None = None  # explicit key beats the environment variable
    max_tokens: int = DEFAULT_MAX_TOKENS
    base_url: str | None = None  # override for proxies / compatible servers

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    def resolve_api_key(self) -> str | None:
        """Explicit runtime key first, then the provider's env var."""
        if self.api_key:
            return self.api_key
        env_var = API_KEY_ENV_VARS.get(self.provider)
        return os.environ.get(env_var) if env_var else None


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


def build_openai_request(
    config: LLMConfig,
    api_key: str,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Serialize a conversation into an OpenAI Chat Completions request."""
    wire_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        if msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in msg.tool_calls
                ]
            wire_messages.append(entry)
        elif msg.role == "tool":
            wire_messages.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
            )
        else:
            wire_messages.append({"role": "user", "content": msg.content})
    payload: dict[str, Any] = {
        "model": config.resolved_model(),
        "max_tokens": config.max_tokens,
        "messages": wire_messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
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

    ``event_name`` is the SSE ``event:`` field (``None`` when absent, as in
    OpenAI streams) and ``data`` is the parsed JSON of the ``data:`` field.
    OpenAI's ``data: [DONE]`` sentinel is not yielded; the stream just ends.
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


async def _stream_openai(
    sse: AsyncIterator[tuple[str | None, dict[str, Any]]],
) -> AsyncIterator[LLMEvent]:
    """Normalize OpenAI Chat Completions streaming chunks."""
    stop_reason: str | None = None
    # Tool-call fragments accumulate by index across chunks.
    pending_tools: dict[int, dict[str, Any]] = {}
    async for _, data in sse:
        if "error" in data:
            error = data["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            yield ErrorEvent(str(message))
            return
        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            yield TextDelta(delta["content"])
        for fragment in delta.get("tool_calls") or []:
            index = int(fragment.get("index", 0))
            pending = pending_tools.setdefault(index, {"id": "", "name": "", "json": ""})
            if fragment.get("id"):
                pending["id"] = fragment["id"]
            function = fragment.get("function") or {}
            if function.get("name"):
                pending["name"] += function["name"]
            if function.get("arguments"):
                pending["json"] += function["arguments"]
        if choice.get("finish_reason"):
            stop_reason = choice["finish_reason"]
    for _, pending in sorted(pending_tools.items()):
        arguments = json.loads(pending["json"]) if pending["json"].strip() else {}
        yield ToolCallEvent(ToolCall(pending["id"], pending["name"], arguments))
    yield Done(stop_reason)


class LLMClient:
    """Streaming chat-completions client over an :class:`SSETransport`."""

    def __init__(self, config: LLMConfig, transport: SSETransport | None = None) -> None:
        if config.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider {config.provider!r} (expected 'anthropic' or 'openai')"
            )
        self.config = config
        self.transport: SSETransport = transport or AiohttpSSETransport()

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
        build = (
            build_anthropic_request if self.config.provider == "anthropic" else build_openai_request
        )
        url, headers, payload = build(self.config, api_key, system, messages, tools)
        normalize = _stream_anthropic if self.config.provider == "anthropic" else _stream_openai
        try:
            async for event in normalize(self.transport.stream_sse(url, headers, payload)):
                yield event
        except Exception as e:
            logger.warning("LLM stream failed: %s", e)
            yield ErrorEvent(str(e))
