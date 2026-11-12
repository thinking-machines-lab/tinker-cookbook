"""Cookbook renderer backed by the ``tml_renderers`` Kimi K3.1-Pro renderer."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal

from tinker_cookbook.renderers.tml_renderers_adapter import TmlRenderersRenderer
from tinker_cookbook.tokenizer_utils import TmlTokenizer, Tokenizer

if TYPE_CHECKING:
    from tml_renderers import renderer as tml_renderer  # pyright: ignore[reportMissingImports]

KimiK31ProReasoningEffort = Literal["low", "high", "max"]


class KimiK31ProRenderer(TmlRenderersRenderer):
    """Renderer adapter for Kimi K3.1-Pro."""

    renderer_name = "kimi_k31_pro"

    def __init__(
        self,
        tokenizer: Tokenizer,
        reasoning_effort: KimiK31ProReasoningEffort | None = "max",
        support_thinking: bool = True,
    ):
        self.reasoning_effort: KimiK31ProReasoningEffort | None = reasoning_effort
        self.support_thinking = support_thinking
        super().__init__(tokenizer)

    def _build_tml_renderer(self, tml_tokenizer: TmlTokenizer) -> tml_renderer.Renderer:
        renderers = import_module("tml_renderers.renderers")
        # The wheel renderer builds its own tokenizer, so only check that the two match.
        k3_tokenizer = import_module("tml_renderers.tokenizers").KimiK3Tokenizer
        if not isinstance(tml_tokenizer, k3_tokenizer):
            raise TypeError(
                f"kimi_k31_pro needs the Kimi K3.1-Pro tokenizer, got {type(tml_tokenizer).__name__}. "
                "Use get_tokenizer('moonshotai/Kimi-K3.1-Pro')."
            )
        return renderers.KimiK31Pro(
            reasoning_effort=self.reasoning_effort, support_thinking=self.support_thinking
        )

    def _parse_error_types(self) -> tuple[type[Exception], ...]:
        return (ValueError,)

    def _response_stopped(self, response: list[int], saw_end: bool) -> bool:
        # The K3.1-Pro parser also ends a cut-off response with ModelEndSampling, and Python
        # cannot see that it is synthetic. Only the sampled stop token ends the turn.
        return saw_end and any(token in response for token in self.get_stop_sequences())
