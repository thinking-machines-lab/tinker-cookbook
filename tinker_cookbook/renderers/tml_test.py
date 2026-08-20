from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from tml_renderers import chat as tml_chat
from tml_renderers.renderer import Renderer as PublicRenderer

from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers import tml
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
    TextPart,
    ThinkingPart,
)
from tinker_cookbook.renderers.tml_conversions import TmlRenderInput


class _Tokenizer:
    def encode_ordinary(self, text: str) -> list[int]:
        assert text == "prefix"
        return [9]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "decoded:" + ",".join(str(token) for token in token_ids)


class _Parser:
    def __init__(self) -> None:
        self.parsed_tokens: list[list[int]] = []

    @staticmethod
    def _message(content: tml_chat.Content) -> tml_chat.Message:
        return tml_chat.Message(content, tml_chat.Author(tml_chat.AuthorKind.Model))

    def parse_tokens(self, tokens: Sequence[int]) -> list[tml_chat.Message]:
        self.parsed_tokens.append(list(tokens))
        return [self._message(tml_chat.Text("answer")), self._message(tml_chat.ModelEndSampling())]

    def parse_token(self, token: int) -> list[tml_chat.ParseUpdate]:
        author = tml_chat.Author(tml_chat.AuthorKind.Model)
        if token == 7:
            return [
                tml_chat.ParseUpdate(tml_chat.StreamingMessageHeader(author)),
                tml_chat.ParseUpdate(tml_chat.StreamingContent(0, tml_chat.Thinking("work"))),
                tml_chat.ParseUpdate(self._message(tml_chat.Thinking("work"))),
            ]
        return [
            tml_chat.ParseUpdate(tml_chat.StreamingMessageHeader(author)),
            tml_chat.ParseUpdate(tml_chat.StreamingContent(1, tml_chat.Text("answer"))),
            tml_chat.ParseUpdate(self._message(tml_chat.Text("answer"))),
            tml_chat.ParseUpdate(self._message(tml_chat.ModelEndSampling())),
        ]

    def parse_updates(self, tokens: Sequence[int]) -> list[tml_chat.ParseUpdate]:
        return [update for token in tokens for update in self.parse_token(token)]

    def flush_updates(self) -> list[tml_chat.ParseUpdate]:
        return []


class _Renderer:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.rendered: TmlRenderInput | None = None
        self.sft_input: TmlRenderInput | None = None
        self.parsers: list[_Parser] = []

    def render_for_completion(
        self, messages: TmlRenderInput
    ) -> tuple[list[tml_chat.TokenSpan], _Parser]:
        self.rendered = messages
        parser = _Parser()
        self.parsers.append(parser)
        return [tml_chat.TokenSpan(tml_chat.EncodedTextTokenSpan([1, 2, 3]))], parser

    def render_for_sft(
        self,
        messages: TmlRenderInput,
        *,
        split_non_extension_history: bool = True,
    ) -> list[tml_chat.TrainingExample]:
        del split_non_extension_history
        self.sft_input = messages
        return [
            tml_chat.TrainingExample(
                [tml_chat.TokenSpan(tml_chat.EncodedTextTokenSpan([4, 5]))],
                [0.0, 1.0],
            )
        ]

    def stop(self) -> list[int] | list[str] | str | None:
        return [42]


class _StopRenderer(_Renderer):
    def __init__(self, stop: list[int] | list[str] | str | None):
        super().__init__()
        self._stop = stop

    def stop(self) -> list[int] | list[str] | str | None:
        return self._stop


def test_adapter_takes_only_the_public_renderer() -> None:
    public_renderer = _Renderer()

    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, public_renderer))
    prompt = adapter.build_generation_prompt([Message(role="user", content="hello")])
    parsed, termination = adapter.parse_response([7, 42])

    assert adapter.tokenizer is public_renderer.tokenizer
    assert prompt.to_ints() == [1, 2, 3]
    assert tml_chat.OpenAIMessage.to_oss_messages(
        tml_chat.OpenAIMessage.from_messages(
            cast(Sequence[tml_chat.Message], public_renderer.rendered)
        )
    ) == [{"role": "user", "content": "hello"}]
    assert parsed == Message(role="assistant", content="answer")
    assert termination == ParseTermination.STOP_SEQUENCE
    assert len(public_renderer.parsers) == 1
    assert public_renderer.parsers[0].parsed_tokens == [[7, 42]]
    assert adapter.get_stop_sequences() == [42]


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        (None, []),
        ("stop", ["stop"]),
        (["first", "second"], ["first", "second"]),
        ([1, 2], [1, 2]),
    ],
)
def test_adapter_normalizes_public_stop_condition(
    stop: list[int] | list[str] | str | None,
    expected: list[int] | list[str],
) -> None:
    renderer = _StopRenderer(stop)

    actual = tml.TmlRendererAdapter(cast(PublicRenderer, renderer)).get_stop_sequences()

    assert actual == expected
    if isinstance(stop, list):
        assert actual is stop


def test_adapter_rejects_caller_prefill() -> None:
    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, _Renderer()))

    with pytest.raises(NotImplementedError, match="caller-provided prefill"):
        adapter.build_generation_prompt([Message(role="user", content="hello")], prefill="prefix")


def test_adapter_uses_public_sft_examples() -> None:
    public_renderer = _Renderer()
    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, public_renderer))

    model_input, weights = adapter.build_supervised_example(
        [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
    )

    assert model_input.to_ints() == [4, 5]
    assert weights.tolist() == [0.0, 1.0]
    sft_input = cast(Sequence[tml_chat.Message], public_renderer.sft_input)
    assert [message.author.kind for message in sft_input] == [
        tml_chat.AuthorKind.User,
        tml_chat.AuthorKind.Model,
        tml_chat.AuthorKind.Model,
    ]
    assert isinstance(sft_input[-1].content, tml_chat.ModelEndSampling)


def test_adapter_rejects_context_free_message_rendering() -> None:
    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, _Renderer()))

    with pytest.raises(NotImplementedError, match="complete conversations"):
        adapter.render_message(
            Message(role="tool", content="result"),
            RenderContext(idx=0, is_last=False, prev_message=None),
        )


def test_adapter_translates_public_render_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(messages: object, **kwargs: object) -> None:
        del messages, kwargs
        raise ValueError("bad messages")

    completion_renderer = _Renderer()
    monkeypatch.setattr(completion_renderer, "render_for_completion", fail)
    with pytest.raises(RendererError, match="bad messages"):
        tml.TmlRendererAdapter(cast(PublicRenderer, completion_renderer)).build_generation_prompt(
            []
        )

    sft_renderer = _Renderer()
    monkeypatch.setattr(sft_renderer, "render_for_sft", fail)
    with pytest.raises(RendererError, match="bad messages"):
        tml.TmlRendererAdapter(cast(PublicRenderer, sft_renderer)).build_supervised_examples([])


def test_adapter_translates_public_streaming_updates() -> None:
    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, _Renderer()))
    adapter.build_generation_prompt([Message(role="user", content="hello")])

    events = list(adapter.parse_response_streaming([7, 42]))

    assert isinstance(events[0], StreamingMessageHeader)
    assert events[1] == StreamingThinkingDelta(thinking="work", content_index=0)
    assert events[2] == StreamingTextDelta(text="answer", content_index=1)
    assert events[3] == Message(
        role="assistant",
        content=[
            ThinkingPart(type="thinking", thinking="work"),
            TextPart(type="text", text="answer"),
        ],
    )


def test_adapter_enforces_single_pending_completion() -> None:
    adapter = tml.TmlRendererAdapter(cast(PublicRenderer, _Renderer()))
    messages = [Message(role="user", content="hello")]

    with pytest.raises(RuntimeError, match="render a completion prompt"):
        adapter.parse_response([42])

    adapter.build_generation_prompt(messages)
    with pytest.raises(RuntimeError, match="parse the pending completion"):
        adapter.build_generation_prompt(messages)
    with pytest.raises(RuntimeError, match="unparsed completion"):
        adapter.__reduce__()

    adapter.parse_response([42])
    adapter.build_generation_prompt(messages)
