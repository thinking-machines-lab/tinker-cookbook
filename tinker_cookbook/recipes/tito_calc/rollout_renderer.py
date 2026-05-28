"""Renderer-driven training-sample construction.

Standard tinker-cookbook code path: a hand-coded ``Renderer`` subclass via
``get_renderer(name, tok)`` plus ``build_supervised_example``. Returned
``ModelInput`` is flattened to ``list[int]`` so the equivalence harness can
compare it apples-to-apples with the TITO-driven path.
"""

from __future__ import annotations

import tinker

from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer, TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _flatten_chunks(mi: tinker.ModelInput) -> list[int]:
    out: list[int] = []
    for chunk in mi.chunks:
        out.extend(chunk.tokens)
    return out


def make_renderer(
    renderer_name: str,
    model_name: str,
    **renderer_kwargs,
) -> Renderer:
    """Construct a cookbook renderer with optional per-class kwargs.

    Most renderers (e.g. Qwen3Renderer) take ``strip_thinking_from_history``;
    for multi-turn RL we typically want ``strip_thinking_from_history=False``
    so the cookbook's own ``has_extension_property`` returns True.
    """
    tokenizer = get_tokenizer(model_name)
    return get_renderer(renderer_name, tokenizer, **renderer_kwargs)


def build_via_renderer(
    messages: list[Message],
    renderer: Renderer,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> tuple[list[int], list[int]]:
    """Build ``(token_ids, weights)`` via the cookbook's hand-coded Renderer."""
    model_input, weights = renderer.build_supervised_example(
        messages, train_on_what=train_on_what
    )
    token_ids = _flatten_chunks(model_input)
    weight_list = [int(w) for w in weights.tolist()]
    assert len(token_ids) == len(weight_list), (
        f"tokens/weights length mismatch via renderer: "
        f"{len(token_ids)} vs {len(weight_list)}"
    )
    return token_ids, weight_list
