"""Renderer for Moonshot AI's Kimi K2.6 models."""

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers.kimi_k25 import KimiK25Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class KimiK26PreserveThinkingRenderer(KimiK25Renderer):
    """Kimi K2.6 with thinking enabled and preserved across history.

    Matches the K2.6 HF chat template with ``preserve_thinking=true``:
    retains full ``<think>...</think>`` reasoning blocks on historical
    assistant turns instead of collapsing them to ``<think></think>``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor | None = None,
    ):
        super().__init__(
            tokenizer,
            image_processor=image_processor,
            strip_thinking_from_history=False,
        )
