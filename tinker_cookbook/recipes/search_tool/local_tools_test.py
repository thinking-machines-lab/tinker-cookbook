"""Unit tests for the local BM25 search tool.

These tests exercise the BM25 index on a small in-memory corpus — no network
calls, no HuggingFace downloads — so they stay fast and hermetic.
"""

from __future__ import annotations

import pytest

from tinker_cookbook.recipes.search_tool.local_tools import (
    BM25Document,
    BM25Index,
    tokenize,
)


@pytest.fixture
def toy_corpus() -> list[BM25Document]:
    return [
        BM25Document(
            title="Paris",
            text="Paris is the capital and most populous city of France.",
        ),
        BM25Document(
            title="London",
            text="London is the capital and largest city of England and the United Kingdom.",
        ),
        BM25Document(
            title="Tokyo",
            text="Tokyo is the capital of Japan and one of the most populous cities in the world.",
        ),
        BM25Document(
            title="Python",
            text="Python is a high-level general-purpose programming language.",
        ),
    ]


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_bm25_returns_most_relevant_document_first(toy_corpus: list[BM25Document]) -> None:
    index = BM25Index(toy_corpus)
    results = index.search("capital of France", n_results=2)

    assert len(results) > 0
    assert results[0].title == "Paris"


def test_bm25_ranks_programming_query_correctly(toy_corpus: list[BM25Document]) -> None:
    index = BM25Index(toy_corpus)
    results = index.search("programming language", n_results=1)

    assert len(results) == 1
    assert results[0].title == "Python"


def test_bm25_returns_empty_list_for_unknown_query(toy_corpus: list[BM25Document]) -> None:
    index = BM25Index(toy_corpus)
    results = index.search("xyzzy nonsense tokens", n_results=3)

    assert results == []


def test_bm25_returns_empty_list_for_empty_query(toy_corpus: list[BM25Document]) -> None:
    index = BM25Index(toy_corpus)
    assert index.search("", n_results=3) == []


def test_bm25_respects_n_results(toy_corpus: list[BM25Document]) -> None:
    index = BM25Index(toy_corpus)
    results = index.search("capital city", n_results=2)

    assert len(results) <= 2


def test_bm25_requires_non_empty_corpus() -> None:
    with pytest.raises(ValueError):
        BM25Index([])
