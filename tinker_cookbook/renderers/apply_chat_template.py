"""A model-agnostic Renderer that delegates to ``tokenizer.apply_chat_template``.

The cookbook's RL loop is coupled to the ``Renderer`` ABC, so we need *some*
class to plug in. This is the smallest such class: one method does real work,
the rest are stubs the cookbook's RL path never touches.
"""

from __future__ import annotations

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    Renderer,
    RenderContext,
    RenderedMessage,
    parse_response_for_stop_token,
)


class TitoRenderer(Renderer):
    """``tokenizer.apply_chat_template`` is the renderer. Zero per-family code."""

    has_extension_property = True  # caller picks a §6-passing template

    def build_generation_prompt(
        self, messages: list[Message], role="assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        ids = list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True,
        ))
        if prefill:
            ids.extend(self.tokenizer.encode(prefill, add_special_tokens=False))
        return tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=ids)])

    def get_stop_sequences(self) -> list[int]:
        eos = self.tokenizer.eos_token_id
        return [eos] if eos is not None else []

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        eos = self.tokenizer.eos_token_id
        return parse_response_for_stop_token(response, self.tokenizer, eos)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # ABC requirement. The cookbook's RL loop never calls this for us —
        # ``build_generation_prompt`` is the only rendering path we expose.
        raise NotImplementedError("TitoRenderer renders via apply_chat_template only")
