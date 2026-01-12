"""Search tool using ChromaDB + Gemini embeddings.

ChromaTool replaces ChromaToolClient with @tool interface.
Internals copy-pasted from tinker_cookbook/recipes/search_tool/tools.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated, cast

import chromadb
import chz
import google.genai as genai
from chromadb.api import AsyncClientAPI
from chromadb.api.types import QueryResult
from chromadb.config import Settings
from tinker_cookbook.recipes.search_tool.embedding import (
    get_gemini_client,
    get_gemini_embedding,
)
from tinker_cookbook.recipes.search_tool.search_env import normalize_answer
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.tool_use import ToolInterface, tool

logger = logging.getLogger(__name__)


@chz.chz
class ChromaToolConfig:
    """Config for ChromaTool - mirrors ChromaToolClientConfig structure."""

    chroma_host: str
    chroma_port: int
    chroma_collection_name: str
    n_results: int = 3
    max_retries: int = 10
    initial_retry_delay: int = 1
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768


class ChromaTool:
    """Search tool using ChromaDB + Gemini embeddings.

    Replaces ChromaToolClient with @tool interface.
    Internals copy-pasted from tinker_cookbook/recipes/search_tool/tools.py.
    """

    def __init__(
        self,
        chroma_client: AsyncClientAPI,
        gemini_client: genai.Client,
        config: ChromaToolConfig,
    ):
        self._chroma_client = chroma_client
        self._gemini_client = gemini_client
        self._config = config

    # TODO(tgriggs): Change this factory function method. Do it in ctor?
    @staticmethod
    async def create(config: ChromaToolConfig) -> ChromaTool:
        """Async factory - same pattern as ChromaToolClient.create()."""
        chroma_client = await chromadb.AsyncHttpClient(
            host=config.chroma_host,
            port=config.chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )
        gemini_client = get_gemini_client()
        return ChromaTool(chroma_client, gemini_client, config)

    # === Copy-pasted internals from ChromaToolClient ===

    async def _get_embeddings_with_retry(self, query_list: list[str]) -> list[list[float]]:
        return await get_gemini_embedding(
            self._gemini_client,
            query_list,
            self._config.embedding_model,
            self._config.embedding_dim,
            "RETRIEVAL_QUERY",
        )

    async def _query_chroma_with_retry(self, query_embeddings: list[list[float]]) -> QueryResult:
        for attempt in range(self._config.max_retries):
            collection = await self._chroma_client.get_collection(
                self._config.chroma_collection_name
            )
            try:
                results = await collection.query(
                    query_embeddings=query_embeddings,  # pyright: ignore[reportArgumentType]
                    n_results=self._config.n_results,
                )
                return results
            except Exception as e:
                if attempt < self._config.max_retries - 1:
                    wait_time = self._config.initial_retry_delay * (1.5**attempt)
                    logger.error(
                        f"ChromaDB query attempt {attempt + 1}/{self._config.max_retries} "
                        f"failed: {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise e

        raise RuntimeError("All ChromaDB query attempts failed")

    # === New @tool interface ===

    @tool
    async def search(
        self,
        query_list: Annotated[
            list[str],
            "A list of fully-formed semantic queries. The tool will return search results for each query.",
        ],
    ) -> str:
        """Search Wikipedia for relevant information based on the given query."""
        embeddings = await self._get_embeddings_with_retry(query_list)
        results = await self._query_chroma_with_retry(embeddings)

        # Format same as original ChromaToolClient.invoke()
        message_content = ""
        documents_list = results["documents"] or []
        for query, documents in zip(query_list, documents_list):
            message_content += f"Query: {query}\n"
            for doc_i, doc in enumerate(documents):
                message_content += f"Document {doc_i + 1}:\n"
                message_content += f"{doc}\n"

        return message_content


def build_tools(chroma_tool: ChromaTool) -> list[ToolInterface]:
    """Build list of tools from ChromaTool instance."""
    return cast(list[ToolInterface], [chroma_tool.search])


@dataclass
class TextAnswerReward:
    """Reward function matching original SearchEnv logic exactly.

    formula: format_coef * (correct_format - 1) + correct_answer
    """

    gold_answers: list[str]
    format_coef: float = 0.1

    def __call__(
        self, results: list[Message], message: Message
    ) -> tuple[float, bool, dict[str, float]]:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        # Matches SearchEnv.check_format and check_answer
        correct_format = float(self._extract_answer(content) is not None)
        correct_answer = float(self._check_answer(content))

        reward = self.format_coef * (correct_format - 1) + correct_answer
        return reward, True, {"format": correct_format, "correct": correct_answer}

    def _extract_answer(self, text: str) -> str | None:
        """Matches SearchEnv._extract_answer exactly."""
        if "Answer:" not in text:
            return None
        parts = text.split("Answer:")
        if len(parts) != 2:
            return None
        return parts[1].strip()

    def _check_answer(self, text: str) -> bool:
        """Matches SearchEnv.check_answer exactly."""
        model_answer = self._extract_answer(text)
        if model_answer is None or len(self.gold_answers) == 0:
            return False
        for gold in self.gold_answers:
            if normalize_answer(model_answer) == normalize_answer(gold):
                return True
        return False
