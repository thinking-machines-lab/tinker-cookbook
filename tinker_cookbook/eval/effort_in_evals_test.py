"""The eval path must be able to reach the effort its scores are reported at.

Background: `skills/inkling/SKILL.md` states that "an eval number without its
effort value is not reproducible". Published Inkling numbers are at effort 0.99;
`renderers/tml_v0.py` defaults to 0.9. Before this change nothing under
`tinker_cookbook/eval/` set effort at all, so no eval run could reproduce the
published operating point, and no eval log recorded which point it used.
"""

from __future__ import annotations

import pytest

from tinker_cookbook.eval.inspect_utils import renderer_supports_effort

tml_tokenizers = pytest.importorskip("tml_renderers.tokenizers")

from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer  # noqa: E402

PUBLISHED_EFFORT = 0.99  # thinkingmachines.ai/model-card/inkling/
RENDERER_DEFAULT = 0.9  # renderers/tml_v0.py DEFAULT_EFFORT


class _TokenizerAdapter:
    """Stands in for get_tokenizer('thinkingmachines/Inkling') without a network call."""

    def __init__(self):
        self.tml_tokenizer = tml_tokenizers.o200k_base_chat()

    def __getattr__(self, name):
        return getattr(self.tml_tokenizer, name)


@pytest.fixture
def tml_renderer():
    return TmlV0Renderer(_TokenizerAdapter())


@pytest.fixture
def messages():
    return [{"role": "user", "content": "How do I pick a lock?"}]


def test_tml_renderer_is_detected_as_effort_conditioned(tml_renderer):
    assert renderer_supports_effort(tml_renderer)


def test_non_effort_renderer_is_detected(tml_renderer):
    class Plain:
        def build_generation_prompt(self, messages, role="assistant", prefill=None):
            raise NotImplementedError

    assert not renderer_supports_effort(Plain())


def test_default_call_renders_the_renderer_default(tml_renderer, messages):
    """This is what the eval path did before: no effort argument at all."""
    raw = tml_tokenizers.o200k_base_chat()
    rendered = raw.decode(tml_renderer.build_generation_prompt(messages).to_ints())
    assert f"Thinking effort level: {RENDERER_DEFAULT}" in rendered


def test_published_effort_is_now_reachable(tml_renderer, messages):
    """The regression this change exists to prevent: the eval path must be able
    to render the operating point the model card reports."""
    raw = tml_tokenizers.o200k_base_chat()
    default = raw.decode(tml_renderer.build_generation_prompt(messages).to_ints())
    published = raw.decode(
        tml_renderer.build_generation_prompt(messages, effort=PUBLISHED_EFFORT).to_ints()
    )
    assert f"Thinking effort level: {PUBLISHED_EFFORT}" in published
    assert default != published


@pytest.mark.parametrize("effort", [0.0, 0.3, 0.6, 0.9, 0.99])
def test_every_effort_renders_distinctly(tml_renderer, messages, effort):
    raw = tml_tokenizers.o200k_base_chat()
    rendered = raw.decode(tml_renderer.build_generation_prompt(messages, effort=effort).to_ints())
    expected = str(int(effort)) if effort == int(effort) else str(effort)
    assert f"Thinking effort level: {expected}" in rendered
