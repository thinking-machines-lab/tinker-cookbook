"""Renderer for Moonshot AI's Kimi K2.6 models.

K2.6 is a near-drop-in replacement for K2.5 at the renderer level — the
only template difference is the new opt-in ``preserve_thinking`` flag.
The ``kimi_k26`` and ``kimi_k26_disable_thinking`` renderer names
therefore dispatch directly to ``KimiK25Renderer`` /
``KimiK25DisableThinkingRenderer`` in the factory; only the
preserve-thinking variant lives here as a thin subclass.
"""

from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers.kimi_k25 import KimiK25Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class KimiK26PreserveThinkingRenderer(KimiK25Renderer):
    """Kimi K2.6 with thinking enabled and preserved across history.

    Matches the K2.6 HF chat template rendered with
    ``preserve_thinking=true``: every historical assistant turn keeps its
    full ``<think>...</think>`` reasoning block, instead of the default
    K2.5/K2.6 behavior that collapses prior reasoning to empty
    ``<think></think>`` and preserves only the current turn's thinking.

    Use this variant when reasoning traces on prior turns are load-bearing
    at inference or training time — most commonly:

    - Long-horizon coding agents, which is the scenario Moonshot cites
      when recommending the flag. See
      https://huggingface.co/moonshotai/Kimi-K2.6
    - Multi-turn RL training that requires the renderer to satisfy the
      extension property (a shorter prefix of a conversation must tokenize
      to a prefix of the full conversation).

    Under the hood this is a thin wrapper that forwards
    ``strip_thinking_from_history=False`` to :class:`KimiK25Renderer`. The
    two flags are negations of each other: HF's ``preserve_thinking=true``
    is byte-equivalent to tinker-cookbook's
    ``strip_thinking_from_history=False``. We expose it as a named
    renderer variant (rather than a constructor kwarg on the default
    K2.6 renderer) to match Moonshot's user-facing flag name and to
    keep the factory dispatch a pure ``name -> class`` table.
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
