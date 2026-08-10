"""Anthropic/OpenAI-compatible chat proxy that captures via SDK instrumentation.

Black-box agent harnesses speak a chat API to this proxy (see
``proxy/README.md``); the proxy renders chat messages to tokens with a
cookbook renderer, samples through the (instrumented) Tinker SDK, and decodes
the sampled tokens back to a chat response. Because the actual sampling goes
through ``SamplingClient.sample_async``, everything is captured by the
in-process SDK instrumentation (:mod:`tinker_cookbook.capture.instrument`): this module contains NO export
logic of its own. Each request enters a ``capture(...)`` scope built from the
``/r/...`` address in the URL path, so wire rows are born addressed;
contextvars isolate concurrent requests since each aiohttp handler runs in
its own task.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import tinker
from aiohttp import web

from tinker_cookbook.capture.proxy.address import parse_address
from tinker_cookbook.capture.scope import capture
from tinker_cookbook.renderers import (
    Message,
    ParseTermination,
    ToolCall,
    ToolSpec,
    get_text_content,
)

logger = logging.getLogger(__name__)

_UNSUPPORTED_ANTHROPIC_KEYS = ("mcp_servers",)
_AUTH_EXEMPT_PATHS = frozenset({"/healthz"})
_UNSUPPORTED_OPENAI_KEYS = (
    "functions",
    "function_call",
    "response_format",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "modalities",
    "audio",
    "prediction",
)


class SampledSequenceLike(Protocol):
    """One sampled sequence: the slice of ``tinker.types.SampledSequence`` used here."""

    @property
    def tokens(self) -> Sequence[int]: ...

    @property
    def stop_reason(self) -> str | None: ...


class SampleResponseLike(Protocol):
    """The slice of ``tinker.types.SampleResponse`` the proxy reads."""

    @property
    def sequences(self) -> Sequence[SampledSequenceLike]: ...


class SamplingClientLike(Protocol):
    """The slice of ``tinker.SamplingClient`` the proxy uses."""

    def sample_async(
        self, prompt: tinker.ModelInput, num_samples: int, sampling_params: tinker.SamplingParams
    ) -> Awaitable[SampleResponseLike]: ...


class RendererLike(Protocol):
    """The slice of ``tinker_cookbook.renderers.Renderer`` the proxy uses."""

    def get_stop_sequences(self) -> list[str] | list[int]: ...

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]: ...

    def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput: ...

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]: ...


@dataclass
class ProxyDeps:
    """Dependencies wired at proxy startup (see ``proxy/serve.py``)."""

    renderer: RendererLike
    sampling_client: SamplingClientLike
    model_label: str
    """Reported in responses and /healthz; the model the proxy actually samples."""
    default_max_tokens: int = 1024
    """Used when the request does not specify max_tokens (OpenAI allows omitting it)."""


_DEPS_KEY: web.AppKey[ProxyDeps] = web.AppKey("proxy_deps", ProxyDeps)


class _BadRequest(Exception):
    """Client error carrying a message; shaped per-API by the handlers."""


# A backend 400 is classified as a context-window overflow only on
# prompt-specific evidence: the message must say the PROMPT (plus token
# budget) exceeds a limit, not merely mention the context window. Generic
# context-window mentions (e.g. "max_tokens must not exceed the context
# window") stay on the server-error path, since compacting history cannot
# fix them and a "prompt is too long" 400 would send agent clients into a
# useless compaction loop.
_CONTEXT_OVERFLOW_PATTERNS = (
    # Tinker's actual overflow message, observed from a live run:
    #   "Prompt length plus max_tokens exceeds the model's context window:
    #    67984 prompt tokens + 4096 max_tokens > 65536."
    re.compile(r"prompt length plus max_tokens exceeds", re.IGNORECASE),
    re.compile(r"\d+\s*prompt tokens\s*\+\s*\d+\s*max_tokens\s*>\s*\d+", re.IGNORECASE),
    # Anthropic's own overflow phrasing, in case a backend relays it verbatim.
    re.compile(r"prompt is too long", re.IGNORECASE),
)


def _is_context_overflow(message: str) -> bool:
    return any(pattern.search(message) for pattern in _CONTEXT_OVERFLOW_PATTERNS)


# ── request parsing ───────────────────────────────────────────────────


def _text_from_content(content: object) -> str:
    """Flatten a chat ``content`` field (string or text blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                kind = block.get("type", "unknown") if isinstance(block, dict) else type(block)
                raise _BadRequest(
                    f"unsupported content block type {kind!r}: only text content is supported "
                    "(no tool_use, tool_result, images, or documents)"
                )
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    raise _BadRequest(f"unsupported content type {type(content).__name__!r}")


def _validated_max_tokens(value: object, key: str, default: int) -> int:
    """Validate a client-supplied token budget; 400 (not 500) on bad types."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise _BadRequest(f"{key!r} must be a positive integer")
    if value <= 0:
        raise _BadRequest(f"{key!r} must be a positive integer")
    return value


def _validated_optional_positive_int(body: dict[str, Any], key: str) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _BadRequest(f"{key!r} must be a positive integer")
    return value


def _validated_optional_int(body: dict[str, Any], key: str) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _BadRequest(f"{key!r} must be an integer")
    return value


def _validated_number(body: dict[str, Any], key: str) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _BadRequest(f"{key!r} must be a number")
    return float(value)


def _validated_stop_strings(value: object, key: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise _BadRequest(f"{key!r} must be a string or a list of strings")


def _reject_unsupported(body: dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if body.get(key):
            raise _BadRequest(f"{key!r} is not supported by this proxy (see proxy/README.md)")


@dataclass
class _ParsedChat:
    """API-agnostic parse result: conversation, system text, tool catalog."""

    messages: list[Message]
    system_text: str | None
    tool_specs: list[ToolSpec]
    suppress_tool_calls: bool = False
    """tool_choice "none": enforce it on the RESPONSE side too (a tool-capable
    model can emit its learned tool syntax even without a catalog, especially
    when the history contains tool calls)."""


def _tool_choice_is_none(value: object) -> bool:
    """True for OpenAI ``"none"`` / Anthropic ``{"type": "none"}``."""
    return value == "none" or (isinstance(value, dict) and value.get("type") == "none")


def _check_tool_choice(value: object) -> None:
    """Accept absent/auto/none; forced tool choice has no renderer mechanism."""
    if value is None:
        return
    if isinstance(value, str) and value in ("auto", "none"):
        return
    if isinstance(value, dict) and value.get("type") in ("auto", "none"):
        return
    raise _BadRequest(
        "forced 'tool_choice' is not supported by this proxy; omit it or use auto/none"
    )


def _tool_spec(name: object, description: object, parameters: object) -> ToolSpec:
    if not isinstance(name, str) or not name:
        raise _BadRequest("each tool requires a non-empty string 'name'")
    if parameters is not None and not isinstance(parameters, dict):
        raise _BadRequest(f"tool {name!r} parameter schema must be an object")
    return ToolSpec(
        name=name,
        description=str(description or ""),
        parameters=parameters or {"type": "object", "properties": {}},
    )


def _tool_specs_from_anthropic(body: dict[str, Any]) -> list[ToolSpec]:
    raw = body.get("tools") or []
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise _BadRequest("'tools' must be a list of tool objects")
    return [
        _tool_spec(entry.get("name"), entry.get("description"), entry.get("input_schema"))
        for entry in raw
    ]


def _tool_specs_from_openai(body: dict[str, Any]) -> list[ToolSpec]:
    raw = body.get("tools") or []
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise _BadRequest("'tools' must be a list of tool objects")
    specs: list[ToolSpec] = []
    for entry in raw:
        if entry.get("type") != "function" or not isinstance(entry.get("function"), dict):
            raise _BadRequest("each OpenAI tool must be {'type': 'function', 'function': {...}}")
        function = entry["function"]
        specs.append(
            _tool_spec(
                function.get("name"), function.get("description"), function.get("parameters")
            )
        )
    return specs


def _tool_call(call_id: object, name: object, arguments: str) -> ToolCall:
    if not isinstance(name, str) or not name:
        raise _BadRequest("each tool call requires a non-empty string tool name")
    return ToolCall(
        id=str(call_id) if call_id else f"toolu_{uuid.uuid4().hex}",
        function=ToolCall.FunctionBody(name=name, arguments=arguments),
    )


def _parse_anthropic(body: dict[str, Any]) -> _ParsedChat:
    _reject_unsupported(body, _UNSUPPORTED_ANTHROPIC_KEYS)
    # `thinking` is accepted and ignored (shape-checked only): Claude Code
    # sends it ENABLED by default, so rejecting it blocks the flagship
    # harness at the door. The proxy serves plain text and never produces
    # thinking blocks; models with inline reasoning tags (Qwen-style
    # <think>) may surface them in the text. Documented in the README.
    thinking = body.get("thinking")
    if thinking is not None and not isinstance(thinking, dict):
        raise _BadRequest("'thinking' must be an object")
    _check_tool_choice(body.get("tool_choice"))
    tool_specs = _tool_specs_from_anthropic(body)
    suppress_tool_calls = _tool_choice_is_none(body.get("tool_choice"))
    if suppress_tool_calls:
        # Both API specs define "none" as forbidding tool calls. Dropping
        # the (already validated) catalog builds the prompt as if no tools
        # were sent, so the model cannot emit a call the caller forbade and
        # hand an agent harness an unexpected action; _complete also drops
        # any calls the model emits anyway (suppress_tool_calls).
        tool_specs = []
    system = body.get("system")
    system_texts: list[str] = []
    if system is not None:
        system_texts.append(_text_from_content(system))
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _BadRequest("'messages' must be a non-empty list")
    messages: list[Message] = []
    tool_names: dict[str, str] = {}  # tool_use id -> tool name (for results)
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise _BadRequest("each message must be a JSON object")
        role = raw.get("role")
        if role == "system":
            # Claude Code's beta shape (?beta=true) appends system-role
            # messages inside messages[] carrying agent/skill text. Renderers
            # cannot faithfully express a mid-conversation system turn, so
            # they are folded into the system prompt in encounter order
            # (documented in the README as the honest v0 behavior).
            system_texts.append(_text_from_content(raw.get("content")))
            continue
        if role not in ("user", "assistant"):
            raise _BadRequest(f"unsupported message role {role!r}: only user/assistant/system")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append(Message(role=role, content=content))
            continue
        if not isinstance(content, list):
            raise _BadRequest(f"unsupported content type {type(content).__name__!r}")
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                raise _BadRequest("each content block must be a JSON object")
            block_type = block.get("type")
            if block_type == "text":
                texts.append(str(block.get("text", "")))
            elif block_type == "tool_use" and role == "assistant":
                call = _tool_call(
                    block.get("id"), block.get("name"), json.dumps(block.get("input") or {})
                )
                assert call.id is not None  # _tool_call always assigns one
                tool_names[call.id] = call.function.name
                tool_calls.append(call)
            elif block_type == "tool_result" and role == "user":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id:
                    raise _BadRequest("'tool_result' blocks require a string tool_use_id")
                result = Message(
                    role="tool",
                    content=_text_from_content(block.get("content") or ""),
                    tool_call_id=call_id,
                )
                if call_id in tool_names:
                    # Some renderers (gpt-oss) render results by function
                    # name; recover it from the earlier tool_use block.
                    result["name"] = tool_names[call_id]
                messages.append(result)
            else:
                raise _BadRequest(
                    f"unsupported content block type {block_type!r} for role {role!r}: "
                    "only text, tool_use (assistant), and tool_result (user) are supported "
                    "(no images or documents)"
                )
        if texts or tool_calls:
            message = Message(role=role, content="".join(texts))
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
    system_text = "\n\n".join(text for text in system_texts if text) or None
    return _ParsedChat(
        messages=messages,
        system_text=system_text,
        tool_specs=tool_specs,
        suppress_tool_calls=suppress_tool_calls,
    )


def _parse_openai(body: dict[str, Any]) -> _ParsedChat:
    _reject_unsupported(body, _UNSUPPORTED_OPENAI_KEYS)
    # Semantic parameters the proxy would silently violate get a loud 400;
    # harmless/ignorable fields (user, metadata, store, stream_options, or
    # falsy values of the above) are accepted.
    if body.get("n") not in (None, 1):
        raise _BadRequest("'n' values other than 1 are not supported by this proxy")
    for key in ("presence_penalty", "frequency_penalty"):
        if body.get(key):
            raise _BadRequest(f"{key!r} is not supported by this proxy (see proxy/README.md)")
    _check_tool_choice(body.get("tool_choice"))
    tool_specs = _tool_specs_from_openai(body)
    suppress_tool_calls = _tool_choice_is_none(body.get("tool_choice"))
    if suppress_tool_calls:
        # See _parse_anthropic: "none" forbids tool calls on both the
        # request and response sides.
        tool_specs = []
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise _BadRequest("'messages' must be a non-empty list")
    messages: list[Message] = []
    system_texts: list[str] = []
    tool_names: dict[str, str] = {}
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise _BadRequest("each message must be a JSON object")
        role = raw.get("role")
        if role == "system":
            if tool_specs:
                if messages:
                    raise _BadRequest(
                        "with 'tools', system messages must precede the conversation "
                        "(they become the tool prefix's system prompt)"
                    )
                system_texts.append(_text_from_content(raw.get("content")))
            else:
                messages.append(
                    Message(role="system", content=_text_from_content(raw.get("content")))
                )
            continue
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise _BadRequest("tool messages require a string 'tool_call_id'")
            result = Message(
                role="tool",
                content=_text_from_content(raw.get("content") or ""),
                tool_call_id=call_id,
            )
            name = raw.get("name") or tool_names.get(call_id)
            if name:
                result["name"] = str(name)
            messages.append(result)
            continue
        if role not in ("user", "assistant"):
            raise _BadRequest(f"unsupported message role {role!r}: only system/user/assistant/tool")
        message = Message(role=role, content=_text_from_content(raw.get("content") or ""))
        raw_calls = raw.get("tool_calls")
        if raw_calls:
            if role != "assistant" or not isinstance(raw_calls, list):
                raise _BadRequest("'tool_calls' is only valid on assistant messages")
            calls: list[ToolCall] = []
            for entry in raw_calls:
                if not isinstance(entry, dict) or not isinstance(entry.get("function"), dict):
                    raise _BadRequest("each tool_call must have a 'function' object")
                function = entry["function"]
                call = _tool_call(
                    entry.get("id"), function.get("name"), str(function.get("arguments") or "{}")
                )
                assert call.id is not None
                tool_names[call.id] = call.function.name
                calls.append(call)
            message["tool_calls"] = calls
        messages.append(message)
    system_text = "\n\n".join(system_texts) if system_texts else None
    return _ParsedChat(
        messages=messages,
        system_text=system_text,
        tool_specs=tool_specs,
        suppress_tool_calls=suppress_tool_calls,
    )


# ── sampling core ─────────────────────────────────────────────────────


@dataclass
class _Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str  # tinker StopReason: "stop" | "length"
    tool_calls: list[ToolCall] = field(default_factory=list)
    """Structured tool calls parsed from the completion by the renderer."""
    client_stop_hit: bool = False
    """Generation ended on a client-supplied stop sequence (not the renderer's)."""
    matched_stop: str | None = None
    """The client stop sequence that ended generation, when identifiable."""


async def _complete(
    deps: ProxyDeps,
    parsed: _ParsedChat,
    *,
    address: dict[str, str | int],
    requested_model: str | None,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    extra_stop: list[str] | None,
    seed: int | None = None,
) -> _Completion:
    """Render, sample through the instrumented SDK inside a capture scope, decode."""
    if parsed.tool_specs:
        # The renderer owns the tool wire format: its tools prefix replaces
        # the plain system message (the system text rides inside it).
        try:
            prefix = deps.renderer.create_conversation_prefix_with_tools(
                parsed.tool_specs, system_prompt=parsed.system_text or ""
            )
        except (NotImplementedError, AttributeError):
            # The Renderer base raises NotImplementedError; duck-typed
            # renderers may simply lack the method.
            raise _BadRequest(
                "the configured renderer does not support tool calling; omit 'tools'"
            ) from None
        messages = [*prefix, *parsed.messages]
    elif parsed.system_text is not None:
        messages = [Message(role="system", content=parsed.system_text), *parsed.messages]
    else:
        messages = parsed.messages
    model_input = deps.renderer.build_generation_prompt(messages)
    stop: list[str] | list[int] = deps.renderer.get_stop_sequences()
    renderer_has_string_stops = bool(stop) and isinstance(stop[0], str)
    if extra_stop:
        if stop and not isinstance(stop[0], str):
            # SamplingParams.stop is homogeneous (Sequence[str] OR
            # Sequence[int]); client strings cannot be combined with this
            # renderer's token-id stops, so reject loudly instead of
            # silently ignoring the client's boundaries.
            raise _BadRequest(
                "client stop sequences are not supported with this model's renderer, "
                "which uses token-id stop conditions; omit stop/stop_sequences"
            )
        stop = [*[s for s in stop if isinstance(s, str)], *extra_stop]
    params_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "stop": stop}
    if temperature is not None:
        params_kwargs["temperature"] = temperature
    if top_p is not None:
        params_kwargs["top_p"] = top_p
    if top_k is not None:
        params_kwargs["top_k"] = top_k
    if seed is not None:
        params_kwargs["seed"] = seed
    try:
        sampling_params = tinker.SamplingParams(**params_kwargs)
    except (ValueError, TypeError) as e:
        # pydantic validation errors are client mistakes, not server faults.
        raise _BadRequest(f"invalid sampling parameters: {e}") from None

    scope_pairs: dict[str, str | int] = dict(address)
    if requested_model:
        # The client-side "model" field is recorded but never overrides the
        # configured model (see README); it lands in the scope so records
        # keep it.
        scope_pairs.setdefault("requested_model", str(requested_model))
    with capture(**scope_pairs):
        try:
            response = await deps.sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
        except tinker.BadRequestError as e:
            # A context-window overflow must surface as a client-shaped 400
            # whose message says the prompt is too long: agent harnesses key
            # their history-compaction behavior off exactly that signal, and
            # a generic 500 instead makes them retry the same over-long
            # prompt. Other backend 400s keep the current behavior.
            if _is_context_overflow(str(e)):
                raise _BadRequest(
                    f"prompt is too long: {model_input.length} prompt tokens plus "
                    f"{max_tokens} max_tokens exceed the model's context window"
                ) from e
            raise
    sequence = response.sequences[0]
    tokens = list(sequence.tokens)
    message, termination = deps.renderer.parse_response(tokens)
    text = get_text_content(message)
    tool_calls = list(message.get("tool_calls") or [])
    if parsed.suppress_tool_calls and tool_calls:
        # tool_choice "none" holds on the response side too: the model can
        # still emit its learned tool syntax (especially with tool calls in
        # the history), but the caller forbade tool use, so the parsed calls
        # are dropped and the completion is served as plain text. The text
        # is the renderer-parsed content; tool markup the renderer extracted
        # into the (dropped) calls is not reconstructed. Documented in the
        # README.
        tool_calls = []
    stop_reason = str(sequence.stop_reason or "stop")
    client_stop_hit = False
    matched_stop: str | None = None
    if tool_calls:
        # A parsed tool call ends the turn on the renderer's own format;
        # client-stop attribution does not apply.
        extra_stop = None
    if extra_stop and stop_reason == "stop" and not termination.is_clean:
        # The sampler stopped and the parse did not terminate cleanly. That
        # alone does NOT prove a client stop fired: a STRING-stop renderer's
        # own stop, excluded from the output by the sampler, also parses as
        # non-clean. Attribute using the actual stop candidates instead of
        # parse cleanliness alone:
        hits = [(text.find(s), s) for s in extra_stop if s in text]
        if hits:
            # A client stop is visible in the output: exact attribution.
            idx, matched_stop = min(hits)
            text = text[:idx]
            client_stop_hit = True
        elif not renderer_has_string_stops:
            # Only client stops were submitted to the sampler, so one of
            # them fired even though its text was excluded; with a single
            # candidate it is known, otherwise the matched stop is
            # unknowable (no matched-stop metadata in the response) and
            # stop_sequence stays null (nullable per the API).
            client_stop_hit = True
            if len(extra_stop) == 1:
                matched_stop = extra_stop[0]
        # else: the renderer contributed its own stop strings, so the
        # terminating stop is ambiguous between renderer and client; do not
        # infer a client stop (the response reports end_turn). Documented
        # in the README.
    return _Completion(
        text=text,
        prompt_tokens=model_input.length,
        completion_tokens=len(tokens),
        stop_reason=stop_reason,
        tool_calls=tool_calls,
        client_stop_hit=client_stop_hit,
        matched_stop=matched_stop,
    )


# ── shared handler plumbing ───────────────────────────────────────────


def _address_from_request(request: web.Request) -> dict[str, str | int]:
    return parse_address(request.match_info.get("address", ""))


async def _json_body(request: web.Request) -> dict[str, Any]:
    """Parse the request body as a JSON object.

    The values are arbitrary client-supplied JSON, validated field by field
    by the parsing helpers, so ``Any`` is the honest value type here.
    """
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise _BadRequest("request body must be valid JSON") from None
    if not isinstance(body, dict):
        raise _BadRequest("request body must be a JSON object")
    return body


def _log_rejection(request: web.Request, message: str) -> None:
    """One warn line per client rejection.

    Includes only the path, the parsed run_id (when the address parses), and
    the validation message, which names the offending FIELD but never its
    contents; request bodies are deliberately not logged.
    """
    run_id: object = None
    with contextlib.suppress(ValueError):
        run_id = parse_address(request.match_info.get("address", "")).get("run_id")
    logger.warning("rejected request: path=%s run_id=%s: %s", request.path, run_id, message)


def _anthropic_error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        status=status,
    )


def _openai_error(status: int, message: str) -> web.Response:
    return web.json_response(
        {"error": {"type": "invalid_request_error", "message": message, "code": None}},
        status=status,
    )


def _sse_event(event: str, data: dict[str, Any]) -> bytes:
    """One named SSE event (Anthropic-style)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _sse_data(data: object) -> bytes:
    """One data-only SSE line (OpenAI-style); strings pass through verbatim."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"data: {payload}\n\n".encode()


async def _serve_sse(request: web.Request, chunks: Sequence[bytes]) -> web.StreamResponse:
    """Write an SSE response, tolerating a client that has gone away.

    Streaming clients abort in-flight requests as part of normal operation
    (cancellation, retries), which surfaces here as a ``ConnectionResetError``
    from ``prepare()`` or a write (aiohttp's ``ClientConnectionResetError``
    subclasses it). That is peer behavior, not a server fault, so it is
    logged at DEBUG and swallowed instead of escaping as an ERROR traceback.
    """
    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    try:
        await response.prepare(request)
        for chunk in chunks:
            await response.write(chunk)
        await response.write_eof()
    except ConnectionResetError:
        logger.debug("client disconnected during SSE response: path=%s", request.path)
    return response


def _tool_input(call: ToolCall) -> dict[str, Any]:
    """Best-effort JSON object for a tool_use block's ``input``."""
    try:
        parsed = json.loads(call.function.arguments)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # Never fabricate structure the model did not produce: expose the raw
    # argument text so the harness sees what was actually sampled.
    return {"_raw_arguments": call.function.arguments}


def _anthropic_content_blocks(completion: _Completion) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if completion.text or not completion.tool_calls:
        blocks.append({"type": "text", "text": completion.text})
    for call in completion.tool_calls:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id or f"toolu_{uuid.uuid4().hex}",
                "name": call.function.name,
                "input": _tool_input(call),
            }
        )
    return blocks


def _openai_tool_calls(completion: _Completion) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id or f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }
        for call in completion.tool_calls
    ]


# ── Anthropic Messages API ────────────────────────────────────────────


async def _handle_anthropic(request: web.Request) -> web.StreamResponse:
    deps = request.app[_DEPS_KEY]
    try:
        address = _address_from_request(request)
    except ValueError as e:
        _log_rejection(request, str(e))
        return _anthropic_error(400, str(e))
    try:
        body = await _json_body(request)
        parsed = _parse_anthropic(body)
        completion = await _complete(
            deps,
            parsed,
            address=address,
            requested_model=body.get("model"),
            max_tokens=_validated_max_tokens(
                body.get("max_tokens"), "max_tokens", deps.default_max_tokens
            ),
            temperature=_validated_number(body, "temperature"),
            top_p=_validated_number(body, "top_p"),
            top_k=_validated_optional_positive_int(body, "top_k"),
            extra_stop=_validated_stop_strings(body.get("stop_sequences"), "stop_sequences"),
        )
    except _BadRequest as e:
        _log_rejection(request, str(e))
        return _anthropic_error(400, str(e))

    message_id = f"msg_{uuid.uuid4().hex}"
    if completion.tool_calls:
        stop_reason = "tool_use"
    elif completion.stop_reason == "length":
        stop_reason = "max_tokens"
    elif completion.client_stop_hit:
        stop_reason = "stop_sequence"
    else:
        stop_reason = "end_turn"
    stop_sequence = completion.matched_stop
    content_blocks = _anthropic_content_blocks(completion)
    usage = {
        "input_tokens": completion.prompt_tokens,
        "output_tokens": completion.completion_tokens,
    }
    if not body.get("stream"):
        return web.json_response(
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": deps.model_label,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
                "usage": usage,
            }
        )

    # SSE: each content block is delivered in ONE delta (text_delta for text
    # blocks, input_json_delta for tool_use blocks); the event sequence is
    # valid for streaming clients like Claude Code, and token-by-token
    # streaming is future work (see README).
    chunks: list[bytes] = [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": deps.model_label,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": completion.prompt_tokens, "output_tokens": 0},
                },
            },
        )
    ]
    for index, block in enumerate(content_blocks):
        if block["type"] == "text":
            start_block: dict[str, Any] = {"type": "text", "text": ""}
            delta: dict[str, Any] = {"type": "text_delta", "text": block["text"]}
        else:
            start_block = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        chunks.append(
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": start_block},
            )
        )
        chunks.append(
            _sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
        )
        chunks.append(
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        )
    chunks.append(
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {"output_tokens": completion.completion_tokens},
            },
        )
    )
    chunks.append(_sse_event("message_stop", {"type": "message_stop"}))
    return await _serve_sse(request, chunks)


# ── OpenAI Chat Completions API ───────────────────────────────────────


async def _handle_openai(request: web.Request) -> web.StreamResponse:
    deps = request.app[_DEPS_KEY]
    try:
        address = _address_from_request(request)
    except ValueError as e:
        _log_rejection(request, str(e))
        return _openai_error(400, str(e))
    try:
        body = await _json_body(request)
        parsed = _parse_openai(body)
        # Validate BEFORE sampling: a malformed value must not cost a Tinker
        # sample (and a captured record) before its 400. The `is not None`
        # check (no truthiness coercion) also catches falsy non-objects
        # like [] or false.
        stream_options = body.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, dict):
            raise _BadRequest("'stream_options' must be an object")
        include_usage = bool(stream_options.get("include_usage")) if stream_options else False
        raw_max = body.get("max_completion_tokens")
        max_key = "max_completion_tokens"
        if raw_max is None:
            raw_max, max_key = body.get("max_tokens"), "max_tokens"
        completion = await _complete(
            deps,
            parsed,
            address=address,
            requested_model=body.get("model"),
            max_tokens=_validated_max_tokens(raw_max, max_key, deps.default_max_tokens),
            temperature=_validated_number(body, "temperature"),
            top_p=_validated_number(body, "top_p"),
            top_k=_validated_optional_positive_int(body, "top_k"),
            extra_stop=_validated_stop_strings(body.get("stop"), "stop"),
            seed=_validated_optional_int(body, "seed"),
        )
    except _BadRequest as e:
        _log_rejection(request, str(e))
        return _openai_error(400, str(e))

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if completion.tool_calls:
        finish_reason = "tool_calls"
    elif completion.stop_reason == "length":
        finish_reason = "length"
    else:
        finish_reason = "stop"
    tool_call_payload = _openai_tool_calls(completion)
    usage = {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.prompt_tokens + completion.completion_tokens,
    }
    if not body.get("stream"):
        message_payload: dict[str, Any] = {
            "role": "assistant",
            "content": completion.text if completion.text or not tool_call_payload else None,
        }
        if tool_call_payload:
            message_payload["tool_calls"] = tool_call_payload
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": deps.model_label,
                "choices": [
                    {
                        "index": 0,
                        "message": message_payload,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        )

    # One-chunk SSE stream (see README; token-by-token streaming is future work).
    chunk_base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": deps.model_label,
    }
    stream_choices: list[dict[str, Any]] = [
        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None},
        {"index": 0, "delta": {"content": completion.text}, "finish_reason": None},
    ]
    if tool_call_payload:
        stream_choices.append(
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": call_index, **call}
                        for call_index, call in enumerate(tool_call_payload)
                    ]
                },
                "finish_reason": None,
            }
        )
    stream_choices.append({"index": 0, "delta": {}, "finish_reason": finish_reason})
    chunks: list[bytes] = []
    for choice in stream_choices:
        payload = {**chunk_base, "choices": [choice]}
        if include_usage:
            # Per the OpenAI spec, with include_usage every chunk carries
            # "usage": null and one final chunk (empty choices) carries the
            # totals.
            payload["usage"] = None
        chunks.append(_sse_data(payload))
    if include_usage:
        chunks.append(_sse_data({**chunk_base, "choices": [], "usage": usage}))
    chunks.append(_sse_data("[DONE]"))
    return await _serve_sse(request, chunks)


# ── app ───────────────────────────────────────────────────────────────


async def _healthz(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "model": request.app[_DEPS_KEY].model_label})


def _make_auth_middleware(auth_token: str) -> Any:
    """Require the token on every request except ``/healthz``.

    Accepted as ``x-api-key: <token>`` (Anthropic-style) or
    ``Authorization: Bearer <token>`` (OpenAI-style). Comparison is
    constant-time.
    """

    @web.middleware
    async def middleware(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        if request.path in _AUTH_EXEMPT_PATHS:
            return await handler(request)
        presented = request.headers.get("x-api-key")
        if presented is None:
            authorization = request.headers.get("Authorization", "")
            if authorization.startswith("Bearer "):
                presented = authorization.removeprefix("Bearer ")
        if presented is None or not hmac.compare_digest(presented, auth_token):
            return web.json_response(
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid proxy token"},
                },
                status=401,
            )
        return await handler(request)

    return middleware


def make_app(deps: ProxyDeps, *, auth_token: str | None = None) -> web.Application:
    """Build the proxy aiohttp application.

    Both API paths are registered bare (empty capture address) and under
    ``/r/{address}`` where ``address`` is ``key/value/...`` pairs (see
    :mod:`tinker_cookbook.capture.proxy.address`).

    Args:
        deps: Wired renderer/sampling dependencies.
        auth_token: When set, every request except ``/healthz`` must present
            it (``x-api-key`` or ``Authorization: Bearer``). Required by
            ``serve.py`` for non-loopback binds, which would otherwise expose
            an unauthenticated endpoint that spends Tinker credits.
    """
    # Truthiness, not `is not None`: an empty token means "no auth" (it
    # would otherwise install middleware that rejects every request).
    middlewares = [_make_auth_middleware(auth_token)] if auth_token else []
    app = web.Application(middlewares=middlewares)
    app[_DEPS_KEY] = deps
    app.router.add_get("/healthz", _healthz)
    app.router.add_post("/v1/messages", _handle_anthropic)
    app.router.add_post("/r/{address:.+}/v1/messages", _handle_anthropic)
    app.router.add_post("/v1/chat/completions", _handle_openai)
    app.router.add_post("/r/{address:.+}/v1/chat/completions", _handle_openai)
    return app
