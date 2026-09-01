"""Cookbook renderer backed by the public ``tml_renderers.v0`` renderer.

This is intentionally a thin integration layer. ``tml_renderers`` owns TMLv0
framing and unshifted SFT masks; cookbook owns final Datum construction through
``datum_from_model_input_weights``.
"""

from __future__ import annotations

import math

import tinker
import torch
from tml_renderers import chat as tml_chat
from tml_renderers import v0 as tml_v0

from tinker_cookbook.renderers.base import (
    Message,
    Role,
    TrainOnWhat,
)
from tinker_cookbook.renderers.tml import TmlRendererAdapter, TmlRenderInput
from tinker_cookbook.tokenizer_utils import (
    SupportsTmlTokenizer,
    Tokenizer,
)

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


class TmlV0Renderer(TmlRendererAdapter):
    """Renderer adapter for Inkling models."""

    supports_streaming = False
    _tml_renderer: tml_v0.Renderer

    def __init__(self, tokenizer: Tokenizer):
        _validate_torch_version()
        if not isinstance(tokenizer, SupportsTmlTokenizer):
            raise TypeError(
                "TmlV0Renderer requires the TML tokenizer adapter. "
                "Use get_tokenizer('thinkingmachines/Inkling') or another "
                "tml-renderers-backed model name."
            )
        renderer = tml_v0.Renderer(tokenizer.tml_tokenizer)
        super().__init__(renderer)
        self.tokenizer = tokenizer

    @staticmethod
    def _prepare_sft_input(messages: TmlRenderInput, effort: float) -> list[tml_chat.Message]:
        """Terminate native model turns and add v0 reasoning-effort conditioning."""
        _validate_effort(effort)
        messages = tml_chat.MessageList.from_messages(messages).messages
        render_input: list[tml_chat.Message] = []
        for index, message in enumerate(messages):
            render_input.append(message)
            if message.author.kind != tml_chat.AuthorKind.Model or isinstance(
                message.content, tml_chat.ModelEndSampling
            ):
                continue
            next_message = messages[index + 1] if index + 1 < len(messages) else None
            if next_message is None or next_message.author.kind != tml_chat.AuthorKind.Model:
                render_input.append(
                    tml_chat.Message(
                        content=tml_chat.ModelEndSampling(),
                        author=tml_chat.Author(tml_chat.AuthorKind.Model),
                    )
                )

        effort_message = tml_chat.Message(
            content=tml_chat.ThinkingEffort(round(effort * 1000)),
            author=tml_chat.Author(tml_chat.AuthorKind.System),
        )
        insertion_index = 0
        while (
            insertion_index < len(render_input)
            and render_input[insertion_index].author.kind == tml_chat.AuthorKind.System
        ):
            insertion_index += 1
        render_input.insert(insertion_index, effort_message)
        return render_input

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
        return self._build_generation_prompt(
            messages,
            role,
            prefill,
            lambda render_input: self._tml_renderer.render_for_completion_with_effort(
                render_input, effort
            ),
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
        return self._build_supervised_examples(self._prepare_sft_input(render_input, effort))

    def build_supervised_example(
        self,
        messages: list[Message] | TmlRenderInput,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float = DEFAULT_EFFORT,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        return self._single_supervised_example(
            self.build_supervised_examples(messages, train_on_what, effort=effort)
        )
