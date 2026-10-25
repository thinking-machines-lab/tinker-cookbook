"""Unit tests for the shared audio grading helpers.

Pure-text tests run everywhere; WER tests skip without ``jiwer`` (installed
by the ``audio`` extra). No network, no audio files, no ``tml_renderers``.
"""

from __future__ import annotations

from typing import cast

import pytest

from tinker_cookbook.recipes.audio.grading import (
    clip_wer,
    corpus_wer,
    normalize_text,
    parse_response_text,
)
from tinker_cookbook.renderers import Message, ParseTermination, Renderer


def test_normalize_text_strips_punctuation_and_case() -> None:
    assert normalize_text("Hello,   WORLD!") == "hello world"
    assert normalize_text("The (quick) brown-fox.") == "the quick brown fox"


def test_normalize_text_keeps_apostrophes() -> None:
    assert normalize_text("Don't STOP me now") == "don't stop me now"


def test_normalize_text_empty() -> None:
    assert normalize_text("...!?") == ""


class _StubRenderer:
    """Just enough of the Renderer surface for parse_response_text."""

    def __init__(
        self,
        content: str | None = None,
        termination: ParseTermination = ParseTermination.STOP_SEQUENCE,
    ):
        self._content = content  # None = raise like a mid-UTF-8 truncation
        self._termination = termination

    def parse_response(self, action: list[int]) -> tuple[Message, ParseTermination]:
        if self._content is None:
            raise ValueError("invalid utf-8 sequence")
        message: Message = {"role": "assistant", "content": self._content}
        return message, self._termination


def test_parse_response_text_strips_special_tokens() -> None:
    renderer = cast(Renderer, _StubRenderer("  hello world<|return|>"))
    text, termination = parse_response_text(renderer, [1, 2, 3])
    assert text == "hello world"
    assert termination == ParseTermination.STOP_SEQUENCE


def test_parse_response_text_grades_parser_rejection_as_malformed() -> None:
    renderer = cast(Renderer, _StubRenderer(content=None))
    text, termination = parse_response_text(renderer, [1, 2, 3])
    assert text == ""
    assert termination == ParseTermination.MALFORMED


def test_parse_response_text_passes_text_through_on_any_termination() -> None:
    """Grading is lenient by design: the text is returned whatever way the
    rollout ended, and the termination rides along for format metrics."""
    eos = cast(Renderer, _StubRenderer("hello", termination=ParseTermination.EOS))
    assert parse_response_text(eos, [1]) == ("hello", ParseTermination.EOS)


def test_clip_wer() -> None:
    pytest.importorskip("jiwer")
    assert clip_wer("hello world", "hello world") == 0.0
    assert clip_wer("Hello, World!", "hello world") == 0.0  # normalized before scoring
    assert clip_wer("hello world", "hello there world") == pytest.approx(0.5)
    assert clip_wer("hello world", "") == pytest.approx(1.0)
    assert clip_wer("", "anything at all") == 0.0  # empty-reference convention


def test_corpus_wer_is_corpus_level_not_mean() -> None:
    pytest.importorskip("jiwer")
    refs = ["a b c d", "x"]
    hyps = ["a b c d", "y"]
    # 1 error over 5 reference words, not mean(0.0, 1.0) = 0.5.
    assert corpus_wer(refs, hyps) == pytest.approx(0.2)


def test_corpus_wer_skips_empty_references() -> None:
    pytest.importorskip("jiwer")
    assert corpus_wer(["", "a b"], ["ignored", "a b"]) == 0.0
    assert corpus_wer([], []) == 0.0


def test_corpus_wer_is_unbounded_under_insertions() -> None:
    """Documents the known sharp edge: one rambling hypothesis contributes
    WER >> 1 on its clip (insertions count as errors), so early evals of an
    untrained model at temperature 1 can spike well past 1.0."""
    pytest.importorskip("jiwer")
    assert corpus_wer(["a b c d"], ["x " * 100]) > 1.0
