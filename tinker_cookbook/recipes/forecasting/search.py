"""Dated web search for forecasting, using Linkup Rewind.

Rewind answers "as of the end of a UTC day": both the matching pages and their
ranking come only from what had been crawled by then. Each question searches as
of the last full UTC day before its snapshot, so nothing published on the
snapshot day or later, including the outcome, can reach the model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import aiohttp

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

REWIND_SEARCH_URL = "https://api.linkup.so/v1/rewind/search"
RATE_LIMIT_RETRIES = 4


def search_as_of_day(snapshot_time: datetime) -> str:
    """Return the last full UTC day before ``snapshot_time`` as ``YYYY-MM-DD``."""
    return (snapshot_time.astimezone(UTC).date() - timedelta(days=1)).isoformat()


def format_results(
    query: str, payload: dict[str, Any], num_results: int, content_chars: int
) -> str:
    results = payload.get("results") or []
    if not results:
        return f"Query: {query}\nNo results.\n"
    lines = [f"Query: {query}"]
    for i, result in enumerate(results[:num_results], start=1):
        content = (result.get("content") or "")[:content_chars]
        lines.append(
            f"Result {i}: {result.get('name', '')}\nURL: {result.get('url', '')}\n{content}"
        )
    return "\n\n".join(lines) + "\n"


@dataclass(frozen=True)
class SearchSettings:
    """How forecasting rollouts may search before answering.

    Without ``api_key``, requests use Linkup's shared, rate-limited keyless tier.
    """

    api_key: str | None = None
    max_turns: int = 4
    num_results: int = 5
    content_chars: int = 1500

    def tool_for(self, snapshot_time: datetime) -> RewindSearchTool:
        return RewindSearchTool(
            as_of=search_as_of_day(snapshot_time),
            api_key=self.api_key,
            num_results=self.num_results,
            content_chars=self.content_chars,
        )


@dataclass
class RewindSearchTool:
    """Web search pinned to one day, shared by every rollout of a question."""

    as_of: str
    api_key: str | None = None
    num_results: int = 5
    content_chars: int = 1500
    timeout_seconds: float = 30.0

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _fetch(self, session: aiohttp.ClientSession, query: str) -> dict[str, Any]:
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            async with session.post(
                REWIND_SEARCH_URL,
                json={"q": query, "asOf": self.as_of},
                headers=self.headers(),
            ) as response:
                if response.status == 429 and attempt < RATE_LIMIT_RETRIES:
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status != 200:
                    raise RuntimeError(
                        f"Rewind returned HTTP {response.status}: {await response.text()}"
                    )
                return await response.json()
        raise AssertionError("unreachable")

    @tool
    async def search(
        self,
        query_list: Annotated[
            list[str],
            "One or more web search queries. The tool returns results for each query.",
        ],
    ) -> ToolResult:
        """Search the web as it was before the forecast snapshot. Results never include pages published after it."""
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payloads = await asyncio.gather(*(self._fetch(session, query) for query in query_list))
        return simple_tool_result(
            "\n".join(
                format_results(query, payload, self.num_results, self.content_chars)
                for query, payload in zip(query_list, payloads)
            )
        )
