"""Cookbook renderer backed by the public ``tml_renderers.v0`` renderer."""

from __future__ import annotations

import math
from importlib import import_module
from typing import TYPE_CHECKING

import tinker
import torch

from tinker_cookbook.renderers.base import Message, Role, TrainOnWhat
from tinker_cookbook.renderers.tml_renderers_adapter import (
    TmlRenderersRenderer,
    TmlRenderInput,
    _cookbook_messages_to_sft_input,
    _messages_to_render_input,
    _native_sft_messages,
)
from tinker_cookbook.tokenizer_utils import TmlTokenizer

if TYPE_CHECKING:
    from tml_renderers import chat as tml_chat  # pyright: ignore[reportMissingImports]
    from tml_renderers import v0 as tml_v0  # pyright: ignore[reportMissingImports]

# Effort values must be in [0.0, 1.0).
DEFAULT_EFFORT: float = 0.9


def _validate_effort(effort: float) -> None:
    if not math.isfinite(effort) or not 0.0 <= effort < 1.0:
        raise ValueError(f"thinking effort must be a finite number in [0, 1), got {effort}")


def _prepare_sft_input(messages: TmlRenderInput, effort: float) -> list[tml_chat.Message]:
    """Normalize SFT input to native messages ready for ``render_for_sft``.

    Expands ``OpenAIMessage`` inputs, terminates model turns with
    ``ModelEndSampling``, and inserts the same effort message used by
    completion rendering.
    """
    _validate_effort(effort)
    chat = import_module("tml_renderers.chat")
    native_messages = _native_sft_messages(messages)

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


class TmlV0Renderer(TmlRenderersRenderer):
    """Renderer adapter for Inkling models."""

    renderer_name = "tml_v0"
    _tml_renderer: tml_v0.Renderer

    def _build_tml_renderer(self, tml_tokenizer: TmlTokenizer) -> tml_v0.Renderer:
        return import_module("tml_renderers.v0").Renderer(tml_tokenizer)

    def _parse_error_types(self) -> tuple[type[Exception], ...]:
        return (import_module("tml_renderers.v0").ParseError,)

    @property
    def has_extension_property(self) -> bool:
        """TMLv0 frames each message independently, so nothing is stripped or re-headered
        by position. Shorter prompts stay token-prefixes of longer ones.
        """
        return True

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
        self._validate_generation_options(role, prefill)
        _validate_effort(effort)
        render_input = _messages_to_render_input(messages)
        spans, _parser = self._tml_renderer.render_for_completion_with_effort(render_input, effort)
        return self._token_spans_to_model_input(spans)

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
        render_input = _cookbook_messages_to_sft_input(messages, train_on_what, self.renderer_name)
        return self._render_sft_examples(_prepare_sft_input(render_input, effort))

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        examples = self.build_supervised_examples(messages, train_on_what, effort=effort)
        return self._single_example(examples)
