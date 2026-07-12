"""Tests for search_tool token DB capture metadata.

Per-call tool usage (name, args, error type) flows through
``AgentToolMessageEnv``'s structured ``tool_calls`` records, covered by
``tool_use/agent_tool_message_env_test.py`` and
``tokendb/capture_test.py::TestAgentToolEnvEndToEnd``.
"""

from typing import Any, cast

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("google.genai")

from tinker_cookbook.recipes.search_tool.search_env import (
    SearchEnvGroupBuilder,
    SearchR1Datum,
)
from tinker_cookbook.recipes.search_tool.tools import ChromaTool, RetrievalConfig


def test_metadata_data_source_and_corpus():
    tool = ChromaTool(cast(Any, None), cast(Any, None), "wiki_2018", RetrievalConfig(), 1, 1)
    datum: SearchR1Datum = {"question": "q", "answer": ["a"], "data_source": "nq"}
    builder = SearchEnvGroupBuilder(
        datum=datum,
        model_name="m",
        renderer_name=None,
        max_turns=1,
        group_size=1,
        chroma_tool=tool,
    )
    assert builder.metadata() == {"data_source": "nq", "corpus": "wiki_2018"}
