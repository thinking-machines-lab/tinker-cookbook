"""
Shared utilities for Gemini embedding generation with retry logic
"""

import asyncio
from logging import getLogger
from os import environ
from typing import Any

import google.genai as genai
from google.genai import types

logger = getLogger(__name__)

# Retry configuration - using the more conservative setting from query_wiki.py
MAX_RETRIES = 10
RETRY_DELAY = 1.0


def get_gemini_client(
    *,
    vertexai: bool | None = None,
    project: str | None = None,
    location: str | None = None,
    http_options: types.HttpOptions | None = None,
    **kwargs: Any,
) -> genai.Client:
    import google.genai as genai
    from google.genai.types import HttpOptions

    project = project or environ.get("GCP_VERTEXAI_PROJECT_NUMBER")
    if project is None:
        raise ValueError("$GCP_VERTEXAI_PROJECT_NUMBER is not set")

    location = location or environ.get("GCP_VERTEXAI_REGION")
    if location is None:
        raise ValueError("$GCP_VERTEXAI_REGION is not set")

    return genai.Client(
        vertexai=(
            environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower().strip().startswith("t")
            if vertexai is None
            else vertexai
        ),
        project=project,
        location=location,
        http_options=http_options or HttpOptions(api_version="v1", timeout=10 * 1000),
        **kwargs,
    )


async def get_gemini_embedding(
    client: genai.Client,
    texts: list[str],
    model: str = "gemini-embedding-001",
    embedding_dim: int = 768,
    task_type: str = "RETRIEVAL_QUERY",
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> list[list[float]]:
    """
    Get embeddings from Gemini API with exponential backoff retry logic.

    Always takes a list of strings and returns a list of embeddings.

    Args:
        texts: List of texts to embed
        model: Gemini embedding model name (default: "gemini-embedding-001")
        embedding_dim: Desired embedding dimension (default: 768)
        task_type: Embedding task type (default: "RETRIEVAL_QUERY")
        max_retries: Maximum number of retries (default: 10)
        retry_delay: Delay between retries (default: 1.0)

    Returns:
        List of embeddings (list of list of floats) -- guaranteed to be the same length as the input texts

    Raises:
        Exception: If embedding generation fails after all retries
    """
    # Validate input
    if not texts:
        raise ValueError("No texts provided for embedding generation")

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Text at index {i} is not a string: {type(text)} = {text}")
        if not text.strip():
            raise ValueError(f"Text at index {i} is empty or whitespace only")

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(10):
                response = await client.aio.models.embed_content(
                    model=model,
                    contents=texts,  # pyright: ignore - Pass the list of texts directly works
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=embedding_dim
                    ),
                )

            if response.embeddings is None or len(response.embeddings) == 0:
                raise ValueError("No embeddings returned from Gemini API")

            if len(response.embeddings) != len(texts):
                raise ValueError(
                    f"Mismatch: expected {len(texts)} embeddings, got {len(response.embeddings)}"
                )

            # Extract embedding values
            embeddings: list[list[float]] = []
            for i, embedding in enumerate(response.embeddings):
                if embedding.values is None:
                    raise ValueError(f"No embedding values returned for text {i}")
                embeddings.append(embedding.values)

            return embeddings

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (1.5**attempt)  # Exponential backoff
                logger.error(
                    f"Attempt {attempt + 1}/{max_retries} failed for embedding ({len(texts)} texts): {repr(e)}. Retrying in {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"All {max_retries} attempts failed for embedding ({len(texts)} texts): {repr(e)}"
                )
                raise

    # This should never be reached due to the raise above, but satisfies type checker
    raise RuntimeError("Unexpected error in retry logic")
