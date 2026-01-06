import asyncio
import json
import logging
from abc import ABC, abstractmethod
import chromadb
import chz
from chromadb.api import AsyncClientAPI
from chromadb.api.types import QueryResult
from chromadb.config import Settings
import google.genai as genai
from tinker_cookbook.recipes.search_tool.embedding import (
    get_gemini_client,
    get_gemini_embedding,
)
from tinker_cookbook.renderers import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[ToolSpec]: ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]: ...


@chz.chz
class EmbeddingConfig:
    model_name: str = "gemini-embedding-001"
    embedding_dim: int = 768
    task_type: str = "RETRIEVAL_QUERY"


@chz.chz
class RetrievalConfig:
    n_results: int = 3
    embedding_config: EmbeddingConfig = EmbeddingConfig()


@chz.chz
class ChromaToolClientConfig:
    chroma_host: str
    chroma_port: int
    chroma_collection_name: str
    retrieval_config: RetrievalConfig
    max_retries: int = 10
    initial_retry_delay: int = 1


class ChromaToolClient(ToolClientInterface):
    def __init__(
        self,
        chroma_client: AsyncClientAPI,
        gemini_client: genai.Client,
        chroma_collection_name: str,
        retrieval_config: RetrievalConfig,
        max_retries: int,
        initial_retry_delay: int,
        embedding_config: EmbeddingConfig,
    ):
        self.chroma_client: AsyncClientAPI = chroma_client
        self.gemini_client: genai.Client = gemini_client
        self.chroma_collection_name: str = chroma_collection_name
        self.n_results: int = retrieval_config.n_results
        self.max_retries: int = max_retries
        self.initial_retry_delay: int = initial_retry_delay
        self.embedding_model: str = embedding_config.model_name
        self.embedding_dim: int = embedding_config.embedding_dim

    @staticmethod
    async def create(chroma_config: ChromaToolClientConfig) -> "ChromaToolClient":
        chroma_client = await chromadb.AsyncHttpClient(
            host=chroma_config.chroma_host,
            port=chroma_config.chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )
        gemini_client = get_gemini_client()
        # list available collections
        return ChromaToolClient(
            chroma_client,
            gemini_client,
            chroma_config.chroma_collection_name,
            chroma_config.retrieval_config,
            chroma_config.max_retries,
            chroma_config.initial_retry_delay,
            chroma_config.retrieval_config.embedding_config,
        )

    def get_tool_schemas(self) -> list[ToolSpec]:
        return [
            {
                "name": "search",
                "description": "Searches Wikipedia for relevant information based on the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of fully-formed semantic queries. The tool will return search results for each query.",
                        }
                    },
                    "required": ["query_list"],
                },
            }
        ]

    async def _get_embeddings_with_retry(self, query_list: list[str]) -> list[list[float]]:
        return await get_gemini_embedding(
            self.gemini_client,
            query_list,
            self.embedding_model,
            self.embedding_dim,
            "RETRIEVAL_QUERY",
        )

    async def _query_chroma_with_retry(self, query_embeddings: list[list[float]]) -> QueryResult:
        for attempt in range(self.max_retries):
            collection = await self.chroma_client.get_collection(self.chroma_collection_name)
            try:
                results = await collection.query(
                    query_embeddings=query_embeddings,  # pyright: ignore - ChromaDB supports batch queries natively
                    n_results=self.n_results,
                )
                return results
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.error(
                        f"ChromaDB query attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {self.initial_retry_delay * (1.5**attempt)}s..."
                    )
                    await asyncio.sleep(self.initial_retry_delay * (1.5**attempt))
                    continue
                raise e

        raise RuntimeError("All ChromaDB query attempts failed")

    async def invoke(self, tool_call: ToolCall) -> list[Message]:
        if tool_call.function.name != "search":
            raise ValueError(f"Invalid tool name: {tool_call.function.name}")

        # Parse arguments with error handling
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return [
                Message(
                    role="tool",
                    content=f"Error invoking search tool: Invalid JSON in arguments - {str(e)}",
                )
            ]

        query_list = args.get("query_list")
        if not isinstance(query_list, list):
            return [
                Message(role="tool", content="Error invoking search tool: query_list is required")
            ]
        if not query_list or not all(
            isinstance(query, str) and query.strip() for query in query_list
        ):
            return [
                Message(
                    role="tool",
                    content="Error invoking search tool: query_list must be a list of non-empty strings",
                )
            ]

        query_embeddings = await self._get_embeddings_with_retry(query_list)

        results = await self._query_chroma_with_retry(
            query_embeddings=query_embeddings,
        )
        assert results["documents"] is not None

        # assemble into a single tool call return message
        message_content = ""
        for query, documents in zip(query_list, results["documents"]):
            message_content += f"Query: {query}\n"
            for document_i, document in enumerate(documents):
                message_content += f"Document {document_i + 1}:\n"
                message_content += f"{document}\n"

        return [Message(role="tool", content=message_content)]
