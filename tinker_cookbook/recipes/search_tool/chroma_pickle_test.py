"""Tests for picklability of ChromaTool."""

import pytest

try:
    import chromadb as _chromadb  # noqa: F401

    _has_chromadb = True
except ImportError:
    _has_chromadb = False


@pytest.mark.skipif(not _has_chromadb, reason="chromadb not installed")
class TestChromatoolPickle:
    def test_pickle_excludes_clients(self) -> None:
        """ChromaTool excludes async clients from pickle state and preserves connection params."""
        from unittest.mock import MagicMock

        from tinker_cookbook.recipes.search_tool.tools import ChromaTool, RetrievalConfig

        tool = ChromaTool(
            chroma_client=MagicMock(),
            gemini_client=MagicMock(),
            collection_name="wiki_chunks",
            retrieval_config=RetrievalConfig(),
            max_retries=5,
            initial_retry_delay=2,
            chroma_host="localhost",
            chroma_port=8000,
        )
        state = tool.__getstate__()
        assert state["_chroma_client"] is None
        assert state["_gemini_client"] is None
        assert state["_chroma_host"] == "localhost"
        assert state["_chroma_port"] == 8000
        assert state["_collection_name"] == "wiki_chunks"
        assert state["_max_retries"] == 5
