from __future__ import annotations

import pickle
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest
import tinker

from tinker_cookbook.renderers import tml, tml_v0
from tinker_cookbook.renderers.base import (
    AudioPart,
    Message,
    ParseTermination,
    RenderContext,
    StreamingMessageHeader,
    StreamingTextDelta,
    StreamingThinkingDelta,
)


class _Tokenizer:
    def encode_ordinary(self, text: str) -> list[int]:
        assert text == "prefix"
        return [9]

    def decode(self, token_ids: list[int]) -> str:
        return "decoded:" + ",".join(str(token) for token in token_ids)


class _OpenAIMessage:
    def __init__(self, source: dict[str, object]):
        self.source = source

    @classmethod
    def from_oss_messages(cls, messages: object) -> list[_OpenAIMessage]:
        return [cls(message) for message in cast(Sequence[dict[str, object]], messages)]

    @classmethod
    def from_messages(cls, messages: object) -> list[object]:
        return list(cast(Sequence[object], messages))

    @classmethod
    def to_oss_messages(cls, messages: object) -> list[dict[str, str]]:
        del messages
        return [{"role": "assistant", "content": "answer"}]

    def to_messages(self) -> list[_ParsedMessage]:
        return [_ParsedMessage(_Text(str(self.source["content"])), role=str(self.source["role"]))]


class _Text:
    def __init__(self, text: str):
        self.text = text


class _Thinking(_Text):
    pass


class _StreamingMessageHeader:
    pass


class _StreamingContent:
    def __init__(self, content: object):
        self.content = content


class _ParseUpdate:
    def __init__(self, update: object):
        self.update = update


class _ModelEndSampling:
    pass


class _ParsedMessage:
    def __init__(self, content: object, role: str = "assistant"):
        self.content = content
        self.author = SimpleNamespace(kind="model" if role == "assistant" else role)


class _MessageList:
    pass


class _TrainingMetadata:
    def __init__(self, weight: float, synthetic: bool):
        self.weight = weight
        self.synthetic = synthetic


class _MessageMetadata:
    def __init__(self, training_metadata: _TrainingMetadata):
        self.training_metadata = training_metadata


class _TrainingExample:
    pass


class _Parser:
    def __init__(self) -> None:
        self.parsed_tokens: list[list[int]] = []

    def parse_tokens(self, tokens: list[int]) -> list[_ParsedMessage]:
        self.parsed_tokens.append(tokens)
        return [_ParsedMessage(_Text("answer")), _ParsedMessage(_ModelEndSampling())]

    def parse_token(self, token: int) -> list[_ParseUpdate]:
        if token == 7:
            return [
                _ParseUpdate(_StreamingMessageHeader()),
                _ParseUpdate(_StreamingContent(_Thinking("work"))),
                _ParseUpdate(_ParsedMessage(_Thinking("work"))),
            ]
        return [
            _ParseUpdate(_StreamingMessageHeader()),
            _ParseUpdate(_StreamingContent(_Text("answer"))),
            _ParseUpdate(_ParsedMessage(_Text("answer"))),
            _ParseUpdate(_ParsedMessage(_ModelEndSampling())),
        ]

    def flush_updates(self) -> list[_ParseUpdate]:
        return []


class _Renderer:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.rendered: object = None
        self.sft_input: object = None
        self.parsers: list[_Parser] = []

    def render_for_completion(self, messages: object) -> tuple[list[object], _Parser]:
        self.rendered = messages
        parser = _Parser()
        self.parsers.append(parser)
        return [object()], parser

    def render_for_sft(self, messages: object) -> list[object]:
        self.sft_input = messages
        return [_TrainingExample()]

    def stop(self) -> list[int]:
        return [42]


class _TinkerBridge:
    @staticmethod
    def token_spans_to_tinker_model_input(spans: object) -> tinker.ModelInput:
        if isinstance(spans, list) and all(isinstance(token, int) for token in spans):
            return tinker.ModelInput.from_ints(cast(list[int], spans))
        return tinker.ModelInput.from_ints([1, 2, 3])

    @staticmethod
    def training_example_to_tinker_model_input_and_weights(
        example: object,
    ) -> tuple[tinker.ModelInput, list[float]]:
        assert isinstance(example, _TrainingExample)
        return tinker.ModelInput.from_ints([4, 5]), [0.0, 1.0]


def _chat_module() -> SimpleNamespace:
    return SimpleNamespace(
        OpenAIMessage=_OpenAIMessage,
        Message=_ParsedMessage,
        MessageList=_MessageList,
        ModelEndSampling=_ModelEndSampling,
        ParseUpdate=_ParseUpdate,
        StreamingContent=_StreamingContent,
        StreamingMessageHeader=_StreamingMessageHeader,
        Text=_Text,
        Thinking=_Thinking,
        MessageMetadata=_MessageMetadata,
        TrainingMetadata=_TrainingMetadata,
        AuthorKind=SimpleNamespace(Model="model"),
    )


def _patch_public_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(name: str) -> object:
        return _chat_module() if name == "tml_renderers.chat" else _TinkerBridge

    monkeypatch.setattr(tml, "import_module", load)
    monkeypatch.setattr(tml_v0, "import_module", load)


def test_adapter_takes_only_the_public_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_modules(monkeypatch)
    public_renderer = _Renderer()

    adapter = tml.TmlRendererAdapter(public_renderer)
    prompt = adapter.build_generation_prompt(
        [Message(role="user", content="hello")], prefill="prefix"
    )
    parsed, termination = adapter.parse_response([7, 42])

    assert adapter.tokenizer is public_renderer.tokenizer
    assert prompt.to_ints() == [1, 2, 3, 9]
    assert [
        message.source for message in cast(Sequence[_OpenAIMessage], public_renderer.rendered)
    ] == [{"role": "user", "content": "hello"}]
    assert parsed == Message(role="assistant", content="answer")
    assert termination == ParseTermination.STOP_SEQUENCE
    assert len(public_renderer.parsers) == 1
    assert public_renderer.parsers[0].parsed_tokens == [[7, 42]]
    assert adapter.get_stop_sequences() == [42]


def test_adapter_uses_public_sft_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_modules(monkeypatch)
    public_renderer = _Renderer()
    adapter = tml.TmlRendererAdapter(public_renderer)

    model_input, weights = adapter.build_supervised_example(
        [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
    )

    assert model_input.to_ints() == [4, 5]
    assert weights.tolist() == [0.0, 1.0]
    assert len(cast(Sequence[object], public_renderer.sft_input)) == 2


def test_adapter_rejects_context_free_message_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_public_modules(monkeypatch)
    adapter = tml.TmlRendererAdapter(_Renderer())

    with pytest.raises(NotImplementedError, match="complete conversations"):
        adapter.render_message(
            Message(role="tool", content="result"),
            RenderContext(idx=0, is_last=False, prev_message=None),
        )


def test_adapter_normalizes_cookbook_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_modules(monkeypatch)
    public_renderer = _Renderer()
    adapter = tml.TmlRendererAdapter(public_renderer)

    adapter.build_generation_prompt(
        [
            Message(
                role="user",
                content=[AudioPart(type="audio", audio=b"RIFF-test", format="wav")],
            )
        ]
    )

    [converted] = cast(Sequence[_OpenAIMessage], public_renderer.rendered)
    [audio] = cast(list[dict[str, object]], converted.source["content"])
    assert audio["type"] == "input_audio"
    assert cast(dict[str, object], audio["input_audio"])["format"] == "wav"


def test_adapter_translates_public_streaming_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_modules(monkeypatch)
    adapter = tml.TmlRendererAdapter(_Renderer())
    adapter.build_generation_prompt([Message(role="user", content="hello")])

    events = list(adapter.parse_response_streaming([7, 42]))

    assert isinstance(events[0], StreamingMessageHeader)
    assert events[1] == StreamingThinkingDelta(thinking="work", content_index=0)
    assert events[2] == StreamingTextDelta(text="answer", content_index=1)
    assert events[3] == Message(role="assistant", content="answer")


def test_adapter_enforces_single_pending_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_modules(monkeypatch)
    adapter = tml.TmlRendererAdapter(_Renderer())
    messages = [Message(role="user", content="hello")]

    adapter.build_generation_prompt(messages)
    with pytest.raises(RuntimeError, match="parse the pending completion"):
        adapter.build_generation_prompt(messages)
    with pytest.raises(RuntimeError, match="unparsed completion"):
        pickle.dumps(adapter)

    adapter.parse_response([42])
    adapter.build_generation_prompt(messages)


def test_adapter_rejects_parsing_without_a_render(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_public_modules(monkeypatch)
    adapter = tml.TmlRendererAdapter(_Renderer())

    with pytest.raises(RuntimeError, match="render a completion prompt"):
        adapter.parse_response([42])


def test_adapter_pickles_the_renderer_object() -> None:
    restored = pickle.loads(pickle.dumps(tml.TmlRendererAdapter(_Renderer())))

    assert isinstance(restored, tml.TmlRendererAdapter)
    assert isinstance(restored._tml_renderer, _Renderer)
