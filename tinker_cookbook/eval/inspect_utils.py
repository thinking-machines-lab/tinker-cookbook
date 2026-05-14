"""
Shared utilities for inspect evaluation.

This module contains the common classes and functions used by both
run_inspect_evals.py and inspect_evaluator.py to avoid code duplication.
"""

import json
import logging
import time
from collections.abc import Sequence

import tinker
from inspect_ai.model import ChatCompletionChoice as InspectAIModelOutputChoice
from inspect_ai.model import ChatMessage as InspectAIChatMessage
from inspect_ai.model import ChatMessageAssistant as InspectAIChatMessageAssistant
from inspect_ai.model import ChatMessageSystem, Content
from inspect_ai.model import ChatMessageTool as InspectAIChatMessageTool
from inspect_ai.model import ContentReasoning as InspectAIContentReasoning
from inspect_ai.model import ContentText as InspectAIContentText
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import ModelAPI as InspectAIModelAPI
from inspect_ai.model import ModelOutput as InspectAIModelOutput
from inspect_ai.model import ModelUsage as InspectAIModelUsage
from inspect_ai.model._registry import modelapi_register
from inspect_ai.tool import ToolCall as InspectAIToolCall
from inspect_ai.tool import ToolChoice as InspectAIToolChoice
from inspect_ai.tool import ToolFunction as InspectAIToolFunction
from inspect_ai.tool import ToolInfo as InspectAIToolInfo
from termcolor import colored

from tinker_cookbook import renderers
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.renderers.base import ensure_list
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


def _inspect_tool_call_to_renderer_tool_call(tool_call: InspectAIToolCall) -> renderers.ToolCall:
    return renderers.ToolCall(
        id=tool_call.id,
        function=renderers.ToolCall.FunctionBody(
            name=tool_call.function,
            arguments=json.dumps(tool_call.arguments, separators=(",", ":")),
        ),
    )


def _renderer_tool_call_to_inspect_tool_call(
    tool_call: renderers.ToolCall, *, choice_index: int, tool_index: int
) -> InspectAIToolCall:
    parse_error = None
    arguments: dict[str, object] = {}
    try:
        decoded_arguments = json.loads(tool_call.function.arguments)
        if isinstance(decoded_arguments, dict):
            arguments = decoded_arguments
        else:
            parse_error = (
                f"Expected tool call arguments to decode to a JSON object, got "
                f"{type(decoded_arguments).__name__}"
            )
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    return InspectAIToolCall(
        id=tool_call.id or f"call_{choice_index}_{tool_index}",
        function=tool_call.function.name,
        arguments=arguments,
        parse_error=parse_error,
    )


def _message_to_inspect_tool_calls(
    message: renderers.Message, *, choice_index: int
) -> list[InspectAIToolCall] | None:
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None
    return [
        _renderer_tool_call_to_inspect_tool_call(
            tool_call, choice_index=choice_index, tool_index=tool_index
        )
        for tool_index, tool_call in enumerate(tool_calls)
    ]


def _inspect_tool_info_to_renderer_tool_spec(tool_info: InspectAIToolInfo) -> renderers.ToolSpec:
    return renderers.ToolSpec(
        name=tool_info.name,
        description=tool_info.description,
        parameters=tool_info.parameters.model_dump(exclude_none=True),
    )


def _select_tools_for_choice(
    tools: list[InspectAIToolInfo], tool_choice: InspectAIToolChoice
) -> list[renderers.ToolSpec]:
    if tool_choice == "none":
        return []

    tool_specs = [_inspect_tool_info_to_renderer_tool_spec(tool) for tool in tools]
    if isinstance(tool_choice, InspectAIToolFunction):
        matching_tool_specs = [tool for tool in tool_specs if tool["name"] == tool_choice.name]
        if not matching_tool_specs:
            logger.warning(f"Inspect requested unknown tool_choice function: {tool_choice.name}")
        return matching_tool_specs
    return tool_specs


def _conversation_with_tool_declarations(
    renderer: renderers.Renderer,
    convo: list[renderers.Message],
    tools: list[InspectAIToolInfo],
    tool_choice: InspectAIToolChoice,
) -> list[renderers.Message]:
    tool_specs = _select_tools_for_choice(tools, tool_choice)
    if not tool_specs:
        return convo

    system_prompt = ""
    messages_after_system = convo
    if convo and convo[0]["role"] == "system":
        system_prompt = renderers.format_content_as_string(convo[0]["content"])
        messages_after_system = convo[1:]

    try:
        prefix = renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
    except NotImplementedError as exc:
        raise ConfigurationError(
            f"Renderer {type(renderer).__name__} does not support Inspect AI tool-use evals"
        ) from exc
    return prefix + messages_after_system


def get_model_usage(
    tokenized_prompt: Sequence[int], responses: Sequence[tinker.SampledSequence]
) -> InspectAIModelUsage:
    """
    Given a tokenized prompt and a list of responses, return the number of tokens used/generated by the model.
    """
    num_input_tokens = len(tokenized_prompt)
    num_output_tokens = sum(len(r.tokens) for r in responses)
    total_tokens = num_input_tokens + num_output_tokens
    usage = InspectAIModelUsage(
        input_tokens=num_input_tokens, output_tokens=num_output_tokens, total_tokens=total_tokens
    )
    return usage


def convert_inspect_messages(messages: list[InspectAIChatMessage]) -> list[renderers.Message]:
    result: list[renderers.Message] = []
    for m in messages:
        content = m.content
        if isinstance(content, str):
            message = renderers.Message(role=m.role, content=content.strip())
        else:
            # Structured content list from inspect_ai
            parts: list[renderers.ContentPart] = []
            for item in content:
                if isinstance(item, InspectAIContentText):
                    parts.append(renderers.TextPart(type="text", text=item.text))
                elif isinstance(item, InspectAIContentReasoning):
                    parts.append(renderers.ThinkingPart(type="thinking", thinking=item.reasoning))
                else:
                    logger.warning(
                        f"Skipping unsupported inspect content type: {type(item).__name__}"
                    )
            # For non-assistant roles, flatten to string (reasoning in user/system is meaningless)
            if m.role != "assistant" or not parts:
                text = " ".join(
                    p["text"] if p["type"] == "text" else p["thinking"]  # type: ignore[typeddict-item]
                    for p in parts
                ).strip()
                message = renderers.Message(role=m.role, content=text)
            else:
                message = renderers.Message(role=m.role, content=parts)
        if isinstance(m, InspectAIChatMessageAssistant) and m.tool_calls:
            message["tool_calls"] = [
                _inspect_tool_call_to_renderer_tool_call(tool_call) for tool_call in m.tool_calls
            ]
        if isinstance(m, InspectAIChatMessageTool):
            if m.tool_call_id is not None:
                message["tool_call_id"] = m.tool_call_id
            if m.function is not None:
                message["name"] = m.function
        result.append(message)
    return result


def _message_to_inspect_content(
    message: renderers.Message,
) -> list[Content]:
    """Convert a renderer Message's content parts to inspect_ai content types."""
    parts = ensure_list(message["content"])
    result: list[Content] = []
    for part in parts:
        if part["type"] == "thinking":
            result.append(InspectAIContentReasoning(reasoning=part["thinking"]))
        elif part["type"] == "text":
            result.append(InspectAIContentText(text=part["text"]))
        else:
            logger.warning(f"Skipping unsupported content part type in response: {part['type']}")
    return result


class InspectAPIFromTinkerSampling(InspectAIModelAPI):
    """
    A model API wrapper that adapts tinker sampling clients to the inspect API interface.

    This class can be initialized either with a model_path (for standalone use)
    or with a sampling_client (for use in evaluators).
    """

    def __init__(
        self,
        renderer_name: str,
        model_name: str,
        model_path: str | None = None,
        sampling_client: tinker.SamplingClient | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: InspectAIGenerateConfig = InspectAIGenerateConfig(),
        verbose: bool = False,
        include_reasoning: bool = False,
    ):
        if api_key_vars is None:
            api_key_vars = []
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=api_key_vars,
            config=config,
        )

        # Initialize sampling client
        if sampling_client is not None:
            self.sampling_client = sampling_client
        elif model_path is not None:
            service_client = tinker.ServiceClient(api_key=api_key)
            self.sampling_client = service_client.create_sampling_client(model_path=model_path)
        else:
            raise ConfigurationError("Either model_path or sampling_client must be provided")

        # Initialize renderer and tokenizer
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.verbose = verbose
        self.include_reasoning = include_reasoning

    async def generate(
        self,
        input: list[InspectAIChatMessage],
        tools: list[InspectAIToolInfo],
        tool_choice: InspectAIToolChoice,
        config: InspectAIGenerateConfig,
    ) -> InspectAIModelOutput:
        """
        The main interface that needs to be implemented to test a new model.
        """
        if config.system_message:
            input = [ChatMessageSystem(content=config.system_message)] + input
        convo = convert_inspect_messages(input)
        convo = _conversation_with_tool_declarations(self.renderer, convo, tools, tool_choice)
        prompt = self.renderer.build_generation_prompt(convo)
        num_responses = 1 if config.num_choices is None else config.num_choices
        sampling_params = tinker.SamplingParams(
            temperature=config.temperature if config.temperature is not None else 1.0,
            max_tokens=config.max_tokens or 128,
            stop=self.renderer.get_stop_sequences(),
            top_p=config.top_p if config.top_p is not None else 1.0,
            top_k=config.top_k if config.top_k is not None else -1,
            seed=config.seed,
        )

        start_time = time.time()
        sample_result = await self.sampling_client.sample_async(
            prompt=prompt, sampling_params=sampling_params, num_samples=num_responses
        )
        sampled_token_sequences = sample_result.sequences

        # Optional verbose output (only for standalone use)
        if self.verbose:
            prompt_text = colored(self.renderer.tokenizer.decode(prompt.to_ints()), "green")
            logger.info(f"[Prompt]\n{prompt_text}")
            for i, seq in enumerate(sampled_token_sequences):
                response_text = colored(self.renderer.tokenizer.decode(seq.tokens), "red")
                logger.info(f"[Response {i + 1}/{num_responses}]\n{response_text}")

        end_time = time.time()

        parsed_responses = [
            self.renderer.parse_response(r.tokens)[0] for r in sampled_token_sequences
        ]
        if self.include_reasoning:
            all_choices = [
                InspectAIModelOutputChoice(
                    message=InspectAIChatMessageAssistant(
                        content=_message_to_inspect_content(r),
                        model=self.model_name,
                        tool_calls=_message_to_inspect_tool_calls(r, choice_index=choice_index),
                    ),
                    stop_reason="stop",
                )
                for choice_index, r in enumerate(parsed_responses)
            ]
        else:
            all_choices = [
                InspectAIModelOutputChoice(
                    message=InspectAIChatMessageAssistant(
                        content=renderers.get_text_content(r),
                        model=self.model_name,
                        tool_calls=_message_to_inspect_tool_calls(r, choice_index=choice_index),
                    ),
                    stop_reason="stop",
                )
                for choice_index, r in enumerate(parsed_responses)
            ]
        usage = get_model_usage(prompt.to_ints(), sampled_token_sequences)

        return InspectAIModelOutput(
            model=self.model_name, choices=all_choices, time=end_time - start_time, usage=usage
        )


# Register with inspect_ai's model registry.
# Using modelapi_register instead of @modelapi decorator preserves the __init__ signature for pyright.
modelapi_register(InspectAPIFromTinkerSampling, "tinker-sampling")
