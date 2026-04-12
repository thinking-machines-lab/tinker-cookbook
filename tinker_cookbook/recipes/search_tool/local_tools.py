"""Lightweight in-memory search tool for the Search-R1 recipe.

This is a drop-in alternative to ``ChromaTool`` that runs entirely on a laptop:
no 160 GB vector database, no external embedding API, no Chroma service.

The backend is a pure-numpy BM25 index over a small Wikipedia passage corpus
(a few thousand documents). This trades retrieval quality for accessibility —
you will not reach the same benchmark numbers as the full Search-R1 setup,
but the recipe becomes runnable locally, which makes it useful for learning
and iterating on the training loop itself.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, cast

import numpy as np
from datasets import Dataset, load_dataset

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

logger = logging.getLogger(__name__)

# BM25 hyperparameters — standard defaults from the Robertson/Zaragoza tutorial.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Intentionally simple — BM25 is robust to it."""
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Document:
    """A single indexed passage."""

    title: str
    text: str


class BM25Index:
    """Pure-numpy BM25 index. Suitable for corpora up to ~100k passages.

    Memory usage is dominated by the term-frequency dictionaries. For a 10k-passage
    corpus this is well under 1 GB, vs 160 GB for the full Wikipedia Chroma index.
    """

    def __init__(
        self,
        documents: Sequence[BM25Document],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if len(documents) == 0:
            raise ValueError("BM25Index requires at least one document")
        self._documents = list(documents)
        self._k1 = k1
        self._b = b

        tokenized = [tokenize(doc.text) for doc in self._documents]
        self._doc_lens = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
        self._avg_doc_len = float(self._doc_lens.mean())

        # Document frequency per term
        doc_freq: Counter[str] = Counter()
        self._term_freqs: list[Counter[str]] = []
        for tokens in tokenized:
            counts = Counter(tokens)
            self._term_freqs.append(counts)
            for term in counts:
                doc_freq[term] += 1

        n_docs = len(self._documents)
        # Standard BM25 IDF with +1 smoothing to avoid negative values
        self._idf: dict[str, float] = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._documents)

    def search(self, query: str, n_results: int) -> list[BM25Document]:
        """Return the top-n documents for a query, ordered by BM25 score (highest first)."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(len(self._documents), dtype=np.float32)
        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in enumerate(self._term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                norm = 1.0 - self._b + self._b * (self._doc_lens[doc_idx] / self._avg_doc_len)
                scores[doc_idx] += idf * (freq * (self._k1 + 1.0)) / (freq + self._k1 * norm)

        if not np.any(scores > 0):
            return []

        top_k = np.argsort(-scores)[:n_results]
        return [self._documents[int(i)] for i in top_k if scores[int(i)] > 0]


# Default corpus: a small Wikipedia subset curated for RAG benchmarks.
# ~3000 passages, <10 MB on disk. Downloaded once via HuggingFace datasets.
DEFAULT_CORPUS_REPO = "rag-datasets/rag-mini-wikipedia"
DEFAULT_CORPUS_CONFIG = "text-corpus"
DEFAULT_CORPUS_SPLIT = "passages"


def load_default_corpus() -> list[BM25Document]:
    """Download and materialize the default mini-wikipedia corpus."""
    logger.info(
        "Loading default corpus %s (%s/%s)",
        DEFAULT_CORPUS_REPO,
        DEFAULT_CORPUS_CONFIG,
        DEFAULT_CORPUS_SPLIT,
    )
    dataset = cast(
        Dataset,
        load_dataset(DEFAULT_CORPUS_REPO, DEFAULT_CORPUS_CONFIG, split=DEFAULT_CORPUS_SPLIT),
    )
    documents: list[BM25Document] = []
    for row in dataset:
        row_dict = cast(dict, row)
        passage = row_dict.get("passage") or row_dict.get("text") or ""
        title = row_dict.get("title") or f"doc-{row_dict.get('id', len(documents))}"
        if passage:
            documents.append(BM25Document(title=str(title), text=str(passage)))
    if not documents:
        raise RuntimeError(f"Loaded empty corpus from {DEFAULT_CORPUS_REPO}")
    logger.info("Loaded %d documents", len(documents))
    return documents


class LocalSearchTool:
    """In-memory BM25 search tool, exposing the same ``search`` tool method as ``ChromaTool``.

    Use ``LocalSearchTool.build()`` to construct — it handles corpus loading and index build.
    The tool is stateless after construction, so it pickles cleanly for multi-worker RL.
    """

    def __init__(self, index: BM25Index, n_results: int) -> None:
        self._index = index
        self._n_results = n_results

    @staticmethod
    def build(
        corpus: Sequence[BM25Document] | None = None,
        n_results: int = 3,
    ) -> LocalSearchTool:
        """Build a LocalSearchTool from an optional corpus (defaults to mini-wikipedia)."""
        documents = list(corpus) if corpus is not None else load_default_corpus()
        index = BM25Index(documents)
        return LocalSearchTool(index=index, n_results=n_results)

    @tool
    async def search(
        self,
        query_list: Annotated[
            list[str],
            "A list of fully-formed semantic queries. The tool will return search results for each query.",
        ],
    ) -> ToolResult:
        """Search Wikipedia for relevant information based on the given query."""
        message_content = ""
        for query in query_list:
            results = self._index.search(query, self._n_results)
            message_content += f"Query: {query}\n"
            if not results:
                message_content += "No relevant documents found.\n"
                continue
            for doc_i, doc in enumerate(results):
                message_content += f"Document {doc_i + 1} ({doc.title}):\n"
                message_content += f"{doc.text}\n"
        return simple_tool_result(message_content)
