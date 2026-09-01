"""Cookbook renderer backed by the public ``tml_renderers.v0`` renderer.

This is intentionally a thin integration layer. ``tml_renderers`` owns TMLv0
framing and unshifted SFT masks; cookbook owns final Datum construction through
``datum_from_model_input_weights``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from typing import TYPE_CHECKING, cast

import tinker
import torch

from tinker_cookbook.renderers.base import (
    Message,
    Role,
    TrainOnWhat,
)
from tinker_cookbook.renderers.tml import TmlRendererAdapter
from tinker_cookbook.renderers.tml_conversions import TmlRenderInput
from tinker_cookbook.tokenizer_utils import (
    SupportsTmlTokenizer,
    TmlTokenizer,
    Tokenizer,
    ensure_tml_renderers_importable,
)

if TYPE_CHECKING:
    # tml_renderers is imported lazily at runtime; import it here for
    # annotations only. It ships py.typed stubs, so pyright checks these types.
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]
    from tml_renderers import v0 as tml_v0  # pyright: ignore[reportMissingImports]

_MINIMUM_TORCH_VERSION = (2, 10)

# Effort values must be in [0.0, 1.0).
DEFAULT_EFFORT: float = 0.9


def _validate_torch_version() -> None:
    try:
        major, minor = (int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"TmlV0Renderer could not determine the installed PyTorch version: "
            f"{torch.__version__!r}"
        ) from exc

    if (major, minor) < _MINIMUM_TORCH_VERSION:
        raise RuntimeError(
            f"TmlV0Renderer requires PyTorch 2.10 or newer; found {torch.__version__}. "
            'Reinstall tinker-cookbook or run `pip install "torch>=2.10"`.'
        )


def _validate_effort(effort: float) -> None:
    if not math.isfinite(effort) or not 0.0 <= effort < 1.0:
        raise ValueError(f"thinking effort must be a finite number in [0, 1), got {effort}")


def _ensure_model_end_sampling(
    messages: list[tml_chat.Message],
) -> list[tml_chat.Message]:
    """Terminate every model turn with ``ModelEndSampling``.

    ``ModelEndSampling`` is the stop-token supervision: without it the model
    never learns where to end its turn, and rendering silently drops that
    weighted token. ``from_oss_messages`` adds the boundary for dict inputs;
    this makes native-message SFT input behave the same. Idempotent, so
    callers that already include the boundary are unchanged.
    """
    chat = import_module("tml_renderers.chat")
    result: list[tml_chat.Message] = []
    for i, message in enumerate(messages):
        result.append(message)
        if message.author.kind != chat.AuthorKind.Model or isinstance(
            message.content, chat.ModelEndSampling
        ):
            continue
        # A model turn can span several messages (thinking, text, tool calls);
        # only close the turn when the run of model messages ends.
        next_message = messages[i + 1] if i + 1 < len(messages) else None
        if next_message is None or next_message.author.kind != chat.AuthorKind.Model:
            result.append(
                chat.Message(
                    content=chat.ModelEndSampling(),
                    author=chat.Author(chat.AuthorKind.Model),
                )
            )
    return result


def _prepare_sft_input(messages: TmlRenderInput, effort: float) -> list[tml_chat.Message]:
    """Normalize SFT input to native messages ready for ``render_for_sft``.

    Expands ``OpenAIMessage`` inputs, terminates model turns with
    ``ModelEndSampling``, and inserts the same effort message used by
    completion rendering.
    """
    _validate_effort(effort)
    chat = import_module("tml_renderers.chat")

    # isinstance against the lazily imported classes can't narrow for pyright,
    # so the checked branches restate the types with casts.
    source: Sequence[tml_chat.Message | tml_chat.OpenAIMessage]
    if isinstance(messages, chat.MessageList):
        source = cast("tml_chat.MessageList", messages).messages
    else:
        source = cast("Sequence[tml_chat.Message | tml_chat.OpenAIMessage]", messages)
    native_messages: list[tml_chat.Message] = []
    for message in source:
        if isinstance(message, chat.OpenAIMessage):
            native_messages.extend(cast("tml_chat.OpenAIMessage", message).to_messages())
        else:
            native_messages.append(cast("tml_chat.Message", message))
    native_messages = _ensure_model_end_sampling(native_messages)

    # ThinkingEffort stores thousandths; tml-renderers owns its display rounding.
    effort_message = chat.Message(
        content=chat.ThinkingEffort(round(effort * 1000)),
        author=chat.Author(chat.AuthorKind.System),
    )
    insertion_index = 0
    while (
        insertion_index < len(native_messages)
        and native_messages[insertion_index].author.kind == chat.AuthorKind.System
    ):
        insertion_index += 1
    native_messages.insert(insertion_index, effort_message)
    return native_messages


def _unwrap_tml_tokenizer(tokenizer: Tokenizer) -> TmlTokenizer:
    if isinstance(tokenizer, SupportsTmlTokenizer):
        return tokenizer.tml_tokenizer
    raise TypeError(
        "TmlV0Renderer requires the TML tokenizer adapter. "
        "Use get_tokenizer('thinkingmachines/Inkling') or another "
        "tml-renderers-backed model name."
    )


class TmlV0Renderer(TmlRendererAdapter):
    """Renderer adapter for Inkling models."""

    supports_streaming = False

    def __init__(self, tokenizer: Tokenizer):
        _validate_torch_version()
        ensure_tml_renderers_importable()
        renderer = cast(
            "tml_v0.Renderer",
            import_module("tml_renderers.v0").Renderer(_unwrap_tml_tokenizer(tokenizer)),
        )
        super().__init__(renderer)
        self.tokenizer = tokenizer

    @property
    def has_extension_property(self) -> bool:
        """TMLv0 frames each message independently, so nothing is stripped or re-headered
        by position. Shorter prompts stay token-prefixes of longer ones.
        """
        return True

    @staticmethod
    def _validate_generation_options(role: Role, prefill: str | None) -> None:
        if role != "assistant":
            raise NotImplementedError("tml_v0 only supports assistant generation")
        if prefill is not None:
            raise NotImplementedError(
                "TMLv0 sampling does not accept partial assistant messages. "
                "Pass complete messages and let the model start a new assistant response."
            )

    def build_generation_prompt(
        self,
        messages: list[Message] | TmlRenderInput,
        role: Role = "assistant",
        prefill: str | None = None,
        effort: float = DEFAULT_EFFORT,
    ) -> tinker.ModelInput:
        """Build a generation prompt with reasoning-effort conditioning.

        ``effort`` must be a finite value in ``[0.0, 1.0)`` and defaults to
        high. Insertion of the system-level effort directive is delegated to
        ``tml-renderers``.
        """
        _validate_effort(effort)
        renderer = cast("tml_v0.Renderer", self._tml_renderer)
        return self._build_generation_prompt(
            messages,
            role,
            prefill,
            lambda render_input: renderer.render_for_completion_with_effort(render_input, effort),
        )

    def build_supervised_examples(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        """Build SFT examples with the same effort conditioning used for generation.

        The inserted effort message is token-identical to the one
        ``build_generation_prompt`` renders, so supervised data matches sampling.

        Generic supervised dataset builders currently use the default effort
        (``0.9``). We plan to expose per-example effort through those builders;
        until then, call this method directly to render a conversation at a
        specific effort level.
        """
        render_input = self._sft_render_input(messages, train_on_what)
        return self._build_supervised_examples(_prepare_sft_input(render_input, effort))

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return self._single_supervised_example(
            self.build_supervised_examples(messages, train_on_what, effort=effort)
        )
