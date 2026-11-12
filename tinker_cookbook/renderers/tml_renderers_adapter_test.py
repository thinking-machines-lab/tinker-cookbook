from __future__ import annotations

from typing import Any, cast

import pytest

from tinker_cookbook.renderers import TrainOnWhat, tml_renderers_adapter


class _FakeTmlRenderersChat:
    class AuthorKind:
        User = "user"

    class MessageList:
        pass

    class Message:
        def __init__(self, kind):
            self.kind = kind

    class OpenAIMessage:
        def __init__(self, role):
            self.role = role


@pytest.fixture
def mock_tml_renderers_chat(monkeypatch: pytest.MonkeyPatch) -> type[_FakeTmlRenderersChat]:
    monkeypatch.setattr(
        tml_renderers_adapter,
        "import_module",
        lambda module="tml_renderers": _FakeTmlRenderersChat,
    )
    return _FakeTmlRenderersChat


@pytest.mark.parametrize("version", ["2.10.0", "2.12.0+cu130", "3.0.0"])
def test_validate_torch_version_accepts_supported_versions(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(tml_renderers_adapter.torch, "__version__", version)
    tml_renderers_adapter._validate_torch_version()


def test_validate_torch_version_rejects_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tml_renderers_adapter.torch, "__version__", "2.9.1")
    with pytest.raises(RuntimeError, match=r"requires PyTorch 2\.10 or newer; found 2\.9\.1"):
        tml_renderers_adapter._validate_torch_version()


def test_native_tml_renderers_inputs_pass_through(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    messages = [
        mock_tml_renderers_chat.Message(mock_tml_renderers_chat.AuthorKind.User),
        mock_tml_renderers_chat.OpenAIMessage("user"),
    ]
    message_list = mock_tml_renderers_chat.MessageList()

    assert tml_renderers_adapter._messages_to_render_input(cast(Any, messages)) is messages
    assert tml_renderers_adapter._messages_to_render_input(cast(Any, message_list)) is message_list


def test_selective_sft_rejects_native_tml_renderers_inputs(
    mock_tml_renderers_chat: type[_FakeTmlRenderersChat],
) -> None:
    with pytest.raises(NotImplementedError, match="selective train_on_what"):
        tml_renderers_adapter._cookbook_messages_to_sft_input(
            cast(Any, [mock_tml_renderers_chat.OpenAIMessage("assistant")]),
            TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            "tml_v0",
        )
