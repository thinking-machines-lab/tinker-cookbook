"""A verifiers v1 training client backed by Tinker sampling."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from functools import cache

import tinker
import verifiers.v1 as vf
from renderers import RendererPool, create_renderer_pool
from verifiers.v1.dialects import ChatDialect, Dialect, parse_tools
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.graph import PendingTurn


def _tool_to_wire(tool: vf.Tool) -> dict:
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def _serialize_response(response: vf.Response) -> dict:
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in response.message.tool_calls
        ]
    usage = response.usage
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": (
            {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage is not None
            else None
        ),
    }


def _is_incremental_tail(messages: list[dict]) -> bool:
    roles = [message.get("role") for message in messages]
    if not roles:
        return False
    if roles[-1] == "user":
        roles = roles[:-1]
    return all(role == "tool" for role in roles)


@cache
def _renderer_pool(model: str, size: int) -> RendererPool:
    return create_renderer_pool(model, size=size)


class TinkerClient(vf.Client):
    """Render chat requests like ``vf.TrainClient`` and sample them with Tinker."""

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer_model_name: str,
        renderer_pool_size: int = 1,
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = _renderer_pool(renderer_model_name, renderer_pool_size)

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: vf.SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> vf.Response:
        del session_id, headers
        if not isinstance(dialect, ChatDialect):
            raise NotImplementedError(
                "TinkerClient supports the chat-completions dialect, matching vf.TrainClient; "
                f"got {type(dialect).__name__}"
            )

        if turn is not None:
            prompt = turn.prompt
            tools = parse_tools(body.get("tools"))
            tail = [message_to_wire(message) for message in turn.tail]
        else:
            prompt, tools = dialect.parse_request(body)
            tail = []
        wire_tools = [_tool_to_wire(tool) for tool in tools] if tools else None

        rendered = None
        bridged_turn = None
        if turn is not None and _is_incremental_tail(tail):
            previous = turn.previous_token_ids()
            if previous is not None:
                rendered = await asyncio.to_thread(
                    self.renderer.bridge_to_next_turn,
                    previous[0],
                    previous[1],
                    tail,
                    tools=wire_tools,
                )
                if rendered is not None:
                    bridged_turn = turn
        if rendered is None:
            wire_messages = [message_to_wire(message) for message in prompt]
            rendered = await asyncio.to_thread(
                self.renderer.render,
                wire_messages,
                tools=wire_tools,
                add_generation_prompt=True,
            )
        if rendered.multi_modal_data is not None and not rendered.multi_modal_data.is_empty():
            raise NotImplementedError("TinkerClient currently supports text-only rendered prompts")

        raw_sampling = sampling_args.model_dump(exclude_none=True)
        extra = raw_sampling.pop("extra_body", None) or {}
        if not isinstance(extra, dict):
            raise TypeError("sampling extra_body must be an object")
        params = {**extra, **raw_sampling}
        unsupported = set(params) - {"max_tokens", "temperature", "top_p", "top_k", "seed"}
        if unsupported:
            raise ValueError(f"Unsupported Tinker sampling arguments: {sorted(unsupported)}")

        sampled = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints(rendered.token_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=int(params.get("max_tokens", 512)),
                temperature=float(params.get("temperature", 1.0)),
                top_p=float(params.get("top_p", 1.0)),
                top_k=int(params.get("top_k", -1)),
                seed=int(params["seed"]) if params.get("seed") is not None else None,
                stop=self.renderer.get_stop_token_ids(),
            ),
        )
        sequence = sampled.sequences[0]
        completion_ids = sequence.tokens
        logprobs = sequence.logprobs or [0.0] * len(completion_ids)
        parsed = await asyncio.to_thread(
            self.renderer.parse_response, completion_ids, tools=wire_tools
        )
        tool_calls = [
            vf.ToolCall(
                id=call.id or f"call_{i}",
                name=call.name,
                arguments=(
                    call.arguments
                    if isinstance(call.arguments, str)
                    else json.dumps(call.arguments or {})
                ),
            )
            for i, call in enumerate(parsed.tool_calls)
            if call.name is not None
        ] or None
        spans = (
            bridged_turn.prompt_message_spans(rendered)
            if bridged_turn is not None
            else rendered.message_token_spans()
        )
        response = vf.Response(
            id="tinker-chatcmpl",
            created=int(time.time()),
            model=model,
            message=vf.AssistantMessage(
                content=parsed.content or None,
                reasoning_content=parsed.reasoning_content,
                tool_calls=tool_calls,
            ),
            finish_reason="stop" if sequence.stop_reason == "stop" else "length",
            usage=vf.Usage(
                prompt_tokens=len(rendered.token_ids),
                completion_tokens=len(completion_ids),
            ),
            tokens=vf.TurnTokens(
                prompt_ids=rendered.token_ids,
                completion_ids=completion_ids,
                completion_logprobs=logprobs,
                message_spans=spans,
                is_content=rendered.is_content,
            ),
        )
        response.raw = _serialize_response(response)
        return response
