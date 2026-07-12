"""Tinker provider for the tokendb chat agent.

Lets the chat agent run against any model served by Tinker: the model list
comes from ``service_client.get_server_capabilities().supported_models`` and
sampling goes through a ``tinker.SamplingClient``. Prompts are built with the
cookbook's renderer machinery (``model_info.get_recommended_renderer_name`` +
``renderers.get_renderer``), the same pattern as the tool-use recipes.

Tool calling has two paths:

- **Renderer-native**: if the model's renderer implements
  ``create_conversation_prefix_with_tools`` (Qwen3, DeepSeek, Kimi, GPT-OSS,
  Nemotron, ...), tool schemas are injected in the renderer's own format and
  tool calls are parsed out of the sampled tokens by
  ``renderer.parse_response`` (``message["tool_calls"]``).
- **JSON-in-text fallback**: for renderers with no tool convention (e.g.
  ``role_colon`` for base models), the system prompt documents a fenced
  `````json {"tool": ..., "arguments": ...}````` protocol and
  :func:`parse_json_tool_blocks` extracts those blocks from the sampled text.

There is no token-level streaming: one sample per model turn, emitted as a
single ``TextDelta`` followed by any ``ToolCallEvent``s and ``Done``.

The SDK + renderer boundary is behind :class:`TinkerSession` /
``SessionFactory`` so tests can inject a fake sampler and an offline renderer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from tinker_cookbook.tokendb.llm import (
    Done,
    LLMConfig,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolDef,
)

if TYPE_CHECKING:
    import tinker

    from tinker_cookbook.renderers.base import Message as RendererMessage
    from tinker_cookbook.renderers.base import Renderer, ToolSpec

logger = logging.getLogger(__name__)

# The agent writes SQL and HTML; sample well below temperature 1.0.
TINKER_TEMPERATURE = 0.7

_JSON_TOOL_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)

FALLBACK_TOOL_INSTRUCTIONS = """\
# Tools

You can call tools. The available tools are described by the JSON schemas \
below:

{tool_schemas}

To call a tool, emit a fenced JSON block of the form:

```json
{{"tool": "<tool name>", "arguments": {{...}}}}
```

Emit at most one such block per response and nothing after it; the tool \
result will arrive in the next user message. When you have the final answer, \
reply without any JSON tool block."""


# --- SDK / renderer boundary (injectable in tests) ---


class TinkerSampler(Protocol):
    """One sampled completion from the chosen model (test seam)."""

    async def sample(
        self, prompt: tinker.ModelInput, max_tokens: int, stop: list[str] | list[int]
    ) -> list[int]: ...


@dataclass
class TinkerSession:
    """Renderer + sampler for one model, ready to run chat turns."""

    renderer: Renderer
    sampler: TinkerSampler


# (model_name, api_key) -> session. May block (tokenizer load, SDK client
# construction); called via asyncio.to_thread.
SessionFactory = Callable[[str, str], TinkerSession]


class _SdkSampler:
    """Default sampler over a ``tinker.SamplingClient``."""

    def __init__(self, sampling_client: Any) -> None:
        self._sampling_client = sampling_client

    async def sample(
        self, prompt: tinker.ModelInput, max_tokens: int, stop: list[str] | list[int]
    ) -> list[int]:
        import tinker

        result = await self._sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=TINKER_TEMPERATURE, stop=stop
            ),
        )
        return list(result.sequences[0].tokens)


def base_model_name(model_name: str) -> str:
    """Strip capability suffixes (``:peft:...`` context variants) for
    renderer/tokenizer lookup."""
    return model_name.split(":", 1)[0]


def build_tinker_session(model_name: str, api_key: str) -> TinkerSession:
    """Real session factory: cookbook renderer + SDK sampling client."""
    import tinker

    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    base = base_model_name(model_name)
    try:
        renderer_name = model_info.get_recommended_renderer_name(base)
    except Exception:
        # Model not in the cookbook registry: fall back to the plain
        # role-colon format (tool calls then use the JSON-in-text protocol).
        renderer_name = "role_colon"
    tokenizer = get_tokenizer(base)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=base)
    service_client = tinker.ServiceClient(api_key=api_key)
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    return TinkerSession(renderer=renderer, sampler=_SdkSampler(sampling_client))


# --- Model listing ---


def fetch_tinker_models(api_key: str) -> list[str]:
    """Fetch the supported model names from the Tinker service (blocking)."""
    import tinker

    service_client = tinker.ServiceClient(api_key=api_key)
    capabilities = service_client.get_server_capabilities()
    return [m.model_name for m in capabilities.supported_models if m.model_name]


def pick_default_model(models: Sequence[str]) -> str | None:
    """A sensible chat default: the first Qwen3+ instruct-ish model, else the
    first entry."""
    for model in models:
        lowered = model.lower()
        if "qwen3" in lowered and "instruct" in lowered and "base" not in lowered:
            return model
    return models[0] if models else None


# --- Conversation rendering ---


def _tool_specs(tools: Sequence[ToolDef]) -> list[ToolSpec]:
    """Our ToolDefs as renderer ToolSpecs (OpenAI function format)."""
    from tinker_cookbook.renderers.base import ToolSpec

    return [
        ToolSpec(name=t.name, description=t.description, parameters=t.input_schema) for t in tools
    ]


def _renderer_tool_call(call: ToolCall) -> Any:
    from tinker_cookbook.renderers.base import ToolCall as RToolCall

    return RToolCall(
        id=call.id or None,
        function=RToolCall.FunctionBody(name=call.name, arguments=json.dumps(call.arguments)),
    )


def _fallback_system_prompt(system: str, tools: Sequence[ToolDef]) -> str:
    if not tools:
        return system
    schemas = "\n".join(
        json.dumps({"name": t.name, "description": t.description, "parameters": t.input_schema})
        for t in tools
    )
    instructions = FALLBACK_TOOL_INSTRUCTIONS.format(tool_schemas=schemas)
    return f"{system}\n\n{instructions}" if system else instructions


def _tool_call_as_json_block(call: ToolCall) -> str:
    payload = json.dumps({"tool": call.name, "arguments": call.arguments})
    return f"```json\n{payload}\n```"


def render_conversation(
    renderer: Renderer,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
) -> tuple[list[RendererMessage], bool]:
    """Convert the provider-neutral conversation into renderer messages.

    Returns ``(renderer_messages, native_tools)``. ``native_tools`` is True
    when the renderer's own tool-call format is in play; False means the
    JSON-in-text fallback protocol (tool calls rendered as fenced JSON blocks,
    tool results as plain user messages).
    """
    from tinker_cookbook.renderers.base import Message as RMessage

    try:
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=_tool_specs(tools), system_prompt=system
        )
        native = True
    except NotImplementedError:
        prefix = [RMessage(role="system", content=_fallback_system_prompt(system, tools))]
        native = False

    rendered: list[RendererMessage] = list(prefix)
    call_names: dict[str, str] = {}  # tool_call_id -> tool name (for results)
    for msg in messages:
        if msg.role == "assistant":
            if native:
                entry = RMessage(role="assistant", content=msg.content)
                if msg.tool_calls:
                    entry["tool_calls"] = [_renderer_tool_call(c) for c in msg.tool_calls]
            else:
                parts = [msg.content] if msg.content else []
                parts.extend(_tool_call_as_json_block(c) for c in msg.tool_calls)
                entry = RMessage(role="assistant", content="\n\n".join(parts))
            for call in msg.tool_calls:
                call_names[call.id] = call.name
            rendered.append(entry)
        elif msg.role == "tool":
            name = call_names.get(msg.tool_call_id or "", "")
            if native:
                entry = RMessage(role="tool", content=msg.content)
                if msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                if name:
                    entry["name"] = name
                rendered.append(entry)
            else:
                label = f"Tool result for {name or 'tool call'}:\n"
                rendered.append(RMessage(role="user", content=label + msg.content))
        else:
            rendered.append(RMessage(role="user", content=msg.content))
    return rendered, native


# --- Response parsing ---


def parse_json_tool_blocks(text: str) -> tuple[str, list[ToolCall]]:
    """Extract fenced ``json`` tool blocks from sampled text (fallback path).

    A block parses into a tool call when it is valid JSON of the form
    ``{"tool": <str>, "arguments": <dict>}``; parsed blocks are removed from
    the text. Anything else (invalid JSON, missing/odd fields) is left in the
    text verbatim so the failure is visible in the chat.
    """
    calls: list[ToolCall] = []

    def _replace(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(payload, dict) or not isinstance(payload.get("tool"), str):
            return match.group(0)
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return match.group(0)
        calls.append(
            ToolCall(id=f"tinker_call_{len(calls)}", name=payload["tool"], arguments=arguments)
        )
        return ""

    remaining = _JSON_TOOL_BLOCK_RE.sub(_replace, text)
    return remaining.strip(), calls


def _normalize_native_tool_calls(parsed: RendererMessage) -> list[ToolCall]:
    """Renderer ToolCalls (JSON-string arguments) into normalized ToolCalls."""
    calls: list[ToolCall] = []
    for index, rcall in enumerate(parsed.get("tool_calls") or []):
        try:
            arguments = json.loads(rcall.function.arguments) if rcall.function.arguments else {}
        except json.JSONDecodeError:
            logger.warning("Dropping tool call with invalid arguments JSON: %s", rcall)
            continue
        if not isinstance(arguments, dict):
            logger.warning("Dropping tool call with non-object arguments: %s", rcall)
            continue
        calls.append(
            ToolCall(
                id=rcall.id or f"tinker_call_{index}",
                name=rcall.function.name,
                arguments=arguments,
            )
        )
    return calls


# --- The provider ---


async def stream_tinker(
    config: LLMConfig,
    api_key: str,
    system: str,
    messages: Sequence[Message],
    tools: Sequence[ToolDef],
    session_factory: SessionFactory | None = None,
) -> AsyncIterator[LLMEvent]:
    """One model turn against a Tinker-served model, as normalized events.

    Exceptions propagate to the caller (:meth:`LLMClient.stream` converts
    them into ``ErrorEvent``s).
    """
    factory = session_factory or build_tinker_session
    session = await asyncio.to_thread(factory, config.resolved_model(), api_key)
    renderer = session.renderer
    rendered, native = render_conversation(renderer, system, messages, tools)
    prompt = renderer.build_generation_prompt(rendered)
    tokens = await session.sampler.sample(
        prompt, max_tokens=config.max_tokens, stop=renderer.get_stop_sequences()
    )
    parsed, _termination = renderer.parse_response(tokens)

    from tinker_cookbook.renderers import get_text_content

    text = get_text_content(parsed)
    if native:
        calls = _normalize_native_tool_calls(parsed)
    else:
        text, calls = parse_json_tool_blocks(text)
    if text:
        yield TextDelta(text)
    for call in calls:
        yield ToolCallEvent(call)
    yield Done("tool_use" if calls else "end_turn")
