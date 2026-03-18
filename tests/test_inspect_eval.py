"""Smoke tests for inspect evaluation integration.

Tests the include_reasoning parameter across thinking and non-thinking models
by calling api.generate() directly to verify the adapter returns the correct
content types to inspect_ai.

Test matrix:
  - Thinking model (Qwen3) + include_reasoning=True  → [ContentReasoning, ContentText]
  - Thinking model (Qwen3) + include_reasoning=False → plain string, no <think> tags
  - Non-thinking model (Llama 3.1) + include_reasoning=True  → [ContentText] only
  - Non-thinking model (Llama 3.1) + include_reasoning=False → plain string
"""

import asyncio

import pytest
import tinker
from inspect_ai.model import ChatMessage as InspectAIChatMessage
from inspect_ai.model import ChatMessageUser as InspectAIChatMessageUser
from inspect_ai.model import ContentReasoning as InspectAIContentReasoning
from inspect_ai.model import ContentText as InspectAIContentText
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import ModelOutput as InspectAIModelOutput

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

THINKING_MODEL = "Qwen/Qwen3-8B"
THINKING_RENDERER = "qwen3"

NON_THINKING_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NON_THINKING_RENDERER = "llama3"

PROMPT: list[InspectAIChatMessage] = [
    InspectAIChatMessageUser(content="What is 1 + 1? Reply with just the number.")
]
GENERATE_CONFIG = InspectAIGenerateConfig(temperature=0.6, max_tokens=1024)


def _create_api(
    model_name: str, renderer_name: str, include_reasoning: bool
) -> InspectAPIFromTinkerSampling:
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    return InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=model_name,
        sampling_client=sampling_client,
        include_reasoning=include_reasoning,
    )


async def _generate(api: InspectAPIFromTinkerSampling) -> InspectAIModelOutput:
    return await api.generate(input=PROMPT, tools=[], tool_choice="auto", config=GENERATE_CONFIG)


def _log_response(result: InspectAIModelOutput) -> None:
    """Print response content for CI debuggability."""
    content = result.choices[0].message.content
    print(f"\n  Content type: {type(content).__name__}")
    if isinstance(content, str):
        print(f"  Text: {content[:300]!r}")
    else:
        for i, part in enumerate(content):
            if isinstance(part, InspectAIContentReasoning):
                print(f"  Part {i} [ContentReasoning]: {part.reasoning[:200]!r}")
            elif isinstance(part, InspectAIContentText):
                print(f"  Part {i} [ContentText]: {part.text[:200]!r}")
            else:
                print(f"  Part {i} [{type(part).__name__}]: {repr(part)[:200]}")
    usage = result.usage
    if usage:
        print(f"  Tokens: {usage.input_tokens} in, {usage.output_tokens} out")


@pytest.mark.integration
def test_thinking_model_include_reasoning():
    """Thinking model + include_reasoning=True: response has ContentReasoning + ContentText."""
    api = _create_api(THINKING_MODEL, THINKING_RENDERER, include_reasoning=True)
    result = asyncio.run(_generate(api))
    _log_response(result)

    content = result.choices[0].message.content
    assert isinstance(content, list), f"Expected list content, got {type(content)}"

    reasoning_parts = [c for c in content if isinstance(c, InspectAIContentReasoning)]
    text_parts = [c for c in content if isinstance(c, InspectAIContentText)]
    assert len(reasoning_parts) > 0, "Expected ContentReasoning from thinking model"
    assert len(text_parts) > 0, "Expected ContentText from thinking model"
    assert len(reasoning_parts[0].reasoning) > 0, "Reasoning content should not be empty"


@pytest.mark.integration
def test_thinking_model_exclude_reasoning():
    """Thinking model + include_reasoning=False: response is plain string without <think> tags."""
    api = _create_api(THINKING_MODEL, THINKING_RENDERER, include_reasoning=False)
    result = asyncio.run(_generate(api))
    _log_response(result)

    content = result.choices[0].message.content
    assert isinstance(content, str), f"Expected string content, got {type(content)}"
    assert "<think>" not in content, "Reasoning should be stripped from string content"


@pytest.mark.integration
def test_non_thinking_model_include_reasoning():
    """Non-thinking model + include_reasoning=True: response has ContentText only, no crash."""
    api = _create_api(NON_THINKING_MODEL, NON_THINKING_RENDERER, include_reasoning=True)
    result = asyncio.run(_generate(api))
    _log_response(result)

    content = result.choices[0].message.content
    assert isinstance(content, list), f"Expected list content, got {type(content)}"

    reasoning_parts = [c for c in content if isinstance(c, InspectAIContentReasoning)]
    text_parts = [c for c in content if isinstance(c, InspectAIContentText)]
    assert len(reasoning_parts) == 0, "Non-thinking model should not produce ContentReasoning"
    assert len(text_parts) > 0, "Expected ContentText from non-thinking model"


@pytest.mark.integration
def test_non_thinking_model_exclude_reasoning():
    """Non-thinking model + include_reasoning=False: response is plain string (baseline)."""
    api = _create_api(NON_THINKING_MODEL, NON_THINKING_RENDERER, include_reasoning=False)
    result = asyncio.run(_generate(api))
    _log_response(result)

    content = result.choices[0].message.content
    assert isinstance(content, str), f"Expected string content, got {type(content)}"
