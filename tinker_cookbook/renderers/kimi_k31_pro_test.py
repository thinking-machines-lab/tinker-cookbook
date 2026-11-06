from __future__ import annotations

import json
from importlib import import_module
from typing import Any, cast

import pytest

from tinker_cookbook.renderers import Message, TextPart, ThinkingPart, ToolCall, get_renderer
from tinker_cookbook.renderers.base import ParseTermination
from tinker_cookbook.renderers.kimi_k31_pro import KimiK31ProRenderer
from tinker_cookbook.third_party.tml_renderers_compat import tinker_messages_to_message_list
from tinker_cookbook.tokenizer_utils import SupportsTmlTokenizer, get_tokenizer

MODEL = "moonshotai/Kimi-K3.1-Pro"


def _renderer(name: str = "kimi_k31_pro_max_reasoning") -> KimiK31ProRenderer:
    return cast(KimiK31ProRenderer, get_renderer(name, get_tokenizer(MODEL)))


def _messages() -> list[Message]:
    return [
        Message(role="system", content="You are concise."),
        Message(role="user", content="Say hello."),
        Message(
            role="assistant",
            content=[
                ThinkingPart(type="thinking", thinking="Greet back."),
                TextPart(type="text", text="Hello."),
            ],
        ),
    ]


@pytest.mark.parametrize("model_name", [MODEL, "moonshotai/Kimi-K3.1-Pro:peft:262144"])
def test_kimi_k31_pro_models_use_the_wheel_tokenizer(model_name: str) -> None:
    tokenizer = get_tokenizer(model_name)

    assert isinstance(tokenizer, SupportsTmlTokenizer)
    k3_tokenizer = import_module("tml_renderers.tokenizers").KimiK3Tokenizer
    assert isinstance(tokenizer.tml_tokenizer, k3_tokenizer)


@pytest.mark.parametrize(
    ("name", "wheel_preset", "effort_line"),
    [
        ("kimi_k31_pro_max_reasoning", "__call__", "thinking_effort=max`"),
        ("kimi_k31_pro_high_reasoning", "high", "thinking_effort=high`"),
        ("kimi_k31_pro_low_reasoning", "low", "thinking_effort=low`"),
        ("kimi_k31_pro_disable_thinking", "disable_thinking", None),
        ("kimi_k31_pro_no_reasoning_effort", "no_reasoning_effort", None),
    ],
)
def test_generation_prompt_matches_the_wheel(
    name: str, wheel_preset: str, effort_line: str | None
) -> None:
    wheel_class = import_module("tml_renderers.renderers").KimiK31Pro
    wheel = wheel_class() if wheel_preset == "__call__" else getattr(wheel_class, wheel_preset)()
    _assert_prompt_matches_the_wheel(name, wheel, effort_line)


def _assert_prompt_matches_the_wheel(name: str, wheel: Any, effort_line: str | None) -> None:
    renderer = _renderer(name)
    prompt = _messages()[:2]
    spans, _ = wheel.render_for_completion(tinker_messages_to_message_list(prompt))
    expected = import_module("tml_renderers.tinker").token_spans_to_tinker_model_input(spans)

    actual = renderer.build_generation_prompt(prompt)
    assert actual.to_ints() == expected.to_ints()
    text = renderer.tokenizer.decode(actual.to_ints())
    if effort_line is None:
        assert "thinking_effort=" not in text
    else:
        assert effort_line in text


def test_disable_thinking_drops_reasoning_and_round_trips_the_response() -> None:
    renderer = _renderer("kimi_k31_pro_disable_thinking")

    completion = _completion(renderer, _messages())
    assert renderer.tokenizer.decode(completion) == (
        "Hello.<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
    )

    message, termination = renderer.parse_response(completion)
    assert termination == ParseTermination.STOP_SEQUENCE
    assert message == {"role": "assistant", "content": "Hello."}


def test_generation_prompt_is_prefix_of_supervised_example() -> None:
    renderer = _renderer()

    prompt_ints = renderer.build_generation_prompt(_messages()[:2]).to_ints()
    supervised_input, weights = renderer.build_supervised_example(_messages())

    assert supervised_input.to_ints()[: len(prompt_ints)] == prompt_ints
    assert float(weights[: len(prompt_ints)].sum()) == 0
    assert float(weights.sum()) > 0


def _completion(renderer: KimiK31ProRenderer, messages: list[Message]) -> list[int]:
    """Return the tokens the model samples after the generation prompt for ``messages``."""
    prompt_len = len(renderer.build_generation_prompt(messages[:-1]).to_ints())
    supervised_input, _ = renderer.build_supervised_example(messages)
    return supervised_input.to_ints()[prompt_len:]


def test_parse_response_round_trips_a_sampled_turn() -> None:
    renderer = _renderer()

    message, termination = renderer.parse_response(_completion(renderer, _messages()))

    assert termination == ParseTermination.STOP_SEQUENCE
    tml_tokenizer = cast(SupportsTmlTokenizer, renderer.tokenizer).tml_tokenizer
    assert renderer.get_stop_sequences() == [tml_tokenizer.encode_special("end_of_msg")]
    content = message["content"]
    assert isinstance(content, list)
    assert [(part["type"], part.get("thinking") or part.get("text")) for part in content] == [
        ("thinking", "Greet back."),
        ("text", "Hello."),
    ]


def test_parse_response_returns_malformed_for_truncated_output() -> None:
    renderer = _renderer()
    completion = _completion(renderer, _messages())

    for cut in range(1, len(completion)):
        _message, termination = renderer.parse_response(completion[:cut])
        assert termination == ParseTermination.MALFORMED


def test_tool_call_round_trips_through_tool_declarations() -> None:
    renderer = _renderer()
    prefix = renderer.create_conversation_prefix_with_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]
    )
    call = ToolCall(
        function=ToolCall.FunctionBody(name="get_weather", arguments='{"city":"SF"}'),
        id="call_1",
    )
    messages = prefix + [
        Message(role="user", content="Weather in SF?"),
        Message(role="assistant", content="", tool_calls=[call]),
    ]

    prompt = renderer.tokenizer.decode(renderer.build_generation_prompt(messages[:-1]).to_ints())
    assert '"name":"get_weather"' in prompt

    message, termination = renderer.parse_response(_completion(renderer, messages))
    assert termination == ParseTermination.STOP_SEQUENCE
    tool_calls = cast(Any, message.get("tool_calls"))
    assert [(c.function.name, json.loads(c.function.arguments)) for c in tool_calls] == [
        ("get_weather", {"city": "SF"})
    ]


def test_kimi_k31_pro_rejects_another_tokenizer() -> None:
    with pytest.raises(TypeError, match=r"Kimi K3\.1-Pro tokenizer"):
        KimiK31ProRenderer(get_tokenizer("thinkingmachines/Inkling"))
