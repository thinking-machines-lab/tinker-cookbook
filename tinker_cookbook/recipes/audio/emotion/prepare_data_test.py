"""Unit tests for the pure split logic in prepare_data (no download/ffmpeg)."""

from __future__ import annotations

from collections import Counter

from tinker_cookbook.recipes.audio.emotion.prepare_data import _balanced_unique_indices


def test_one_index_per_unique_text() -> None:
    texts = ["a", "a", "b", "b", "c"]
    styles = ["happy", "sad", "happy", "sad", "happy"]
    picked = _balanced_unique_indices(texts, styles)
    assert len(picked) == 3
    assert sorted({texts[i] for i in picked}) == ["a", "b", "c"]


def test_style_balanced_greedy_pick() -> None:
    # Every sentence has a happy rendition first; a uniform first-pick would
    # produce 4x happy. The greedy pick alternates styles instead.
    texts = ["a", "a", "b", "b", "c", "c", "d", "d"]
    styles = ["happy", "sad"] * 4
    picked = _balanced_unique_indices(texts, styles)
    counts = Counter(styles[i] for i in picked)
    assert counts["happy"] == counts["sad"] == 2


def test_ties_prefer_first_seen_rendition() -> None:
    picked = _balanced_unique_indices(["a", "a"], ["happy", "sad"])
    assert picked == [0]
