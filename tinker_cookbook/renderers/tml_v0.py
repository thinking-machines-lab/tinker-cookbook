"""Cookbook renderer backed by the public ``tml_renderers`` Inkling renderers."""

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
from tinker_cookbook.tokenizer_utils import TmlTokenizer, Tokenizer

if TYPE_CHECKING:
    from tml_renderers import renderers as tml_renderers  # pyright: ignore[reportMissingImports]

# Effort values must be in [0.0, 1.0).
DEFAULT_EFFORT: float = 0.9


def _validate_effort(effort: float) -> None:
    if not math.isfinite(effort) or not 0.0 <= effort < 1.0:
        raise ValueError(f"thinking effort must be a finite number in [0, 1), got {effort}")


class TmlV0Renderer(TmlRenderersRenderer):
    """Renderer adapter for Inkling models."""

    renderer_name = "tml_v0"
    _tml_renderer: tml_renderers.Inkling | tml_renderers.Inkling2

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        self._renderers_by_effort: dict[float, tml_renderers.Inkling | tml_renderers.Inkling2] = {
            DEFAULT_EFFORT: self._tml_renderer
        }

    def _build_tml_renderer(
        self, tml_tokenizer: TmlTokenizer
    ) -> tml_renderers.Inkling | tml_renderers.Inkling2:
        return self._build_inkling(tml_tokenizer, DEFAULT_EFFORT)

    @staticmethod
    def _build_inkling(
        tml_tokenizer: TmlTokenizer, effort: float
    ) -> tml_renderers.Inkling | tml_renderers.Inkling2:
        # Each Inkling generation counts image tokens differently, so build the model's own
        # wheel renderer. The tokenizer was chosen per model (`_tml_tokenizer_family`).
        renderers = import_module("tml_renderers.renderers")
        tokenizers = import_module("tml_renderers.tokenizers")
        if isinstance(tml_tokenizer, tokenizers.O200kBaseChatTokenizer):
            return renderers.Inkling(effort)
        if isinstance(tml_tokenizer, tokenizers.H200kBaseChatTokenizer):
            return renderers.Inkling2(effort)
        raise TypeError(
            f"tml_v0 needs an Inkling tokenizer, got {type(tml_tokenizer).__name__}. "
            "Use get_tokenizer('thinkingmachines/Inkling') or another Inkling model."
        )

    def _renderer_for_effort(self, effort: float) -> tml_renderers.Inkling | tml_renderers.Inkling2:
        _validate_effort(effort)
        renderer = self._renderers_by_effort.get(effort)
        if renderer is None:
            renderer = self._build_inkling(self._tml_tokenizer, effort)
            self._renderers_by_effort[effort] = renderer
        return renderer

    def _parse_error_types(self) -> tuple[type[Exception], ...]:
        return (ValueError,)

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
        renderer = self._renderer_for_effort(effort)
        spans, _parser = renderer.render_for_completion(_messages_to_render_input(messages))
        return self._token_spans_to_model_input(spans)

    def build_supervised_examples(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> list[tuple[tinker.ModelInput, torch.Tensor]]:
        """Build SFT examples with the same effort conditioning used for generation.

        Both methods render with the same wheel renderer for ``effort``, so the
        effort message in supervised data matches the one used for sampling.

        Generic supervised dataset builders currently use the default effort
        (``0.9``). We plan to expose per-example effort through those builders;
        until then, call this method directly to render a conversation at a
        specific effort level.
        """
        render_input = _cookbook_messages_to_sft_input(messages, train_on_what, self.renderer_name)
        renderer = self._renderer_for_effort(effort)
        return self._render_sft_examples(_native_sft_messages(render_input), renderer)

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        examples = self.build_supervised_examples(messages, train_on_what, effort=effort)
        return self._single_example(examples)
