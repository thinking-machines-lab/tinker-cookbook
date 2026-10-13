from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tinker_cookbook import renderers
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.tool_use import ToolInput

from .data import ForecastExample
from .env import ForecastGroupBuilder, ForecastSearchEnv
from .search import RewindSearchTool, SearchSettings, format_results, search_as_of_day

PAYLOAD = {
    "asOf": "2025-07-07",
    "results": [
        {"name": "Preview", "url": "https://example.test/a", "content": "Pre-match preview text."},
        {"name": "Odds", "url": "https://example.test/b", "content": "Bookmaker odds."},
    ],
}


class FakeRenderer:
    def create_conversation_prefix_with_tools(
        self, tools: list[Any], system_prompt: str = ""
    ) -> list[Message]:
        return [
            {"role": "system", "content": f"{system_prompt}\nTools: {[t['name'] for t in tools]}"}
        ]

    def get_stop_sequences(self) -> list[str]:
        return []


def _example(outcome: int = 1) -> ForecastExample:
    return ForecastExample(
        submission_id="submission",
        event_ticker="event",
        event_title="Will the event happen?",
        market="Event happens",
        reference_material="Dated evidence available at the snapshot.",
        resolution_criteria="Resolves YES if the event happens.",
        snapshot_time=datetime(2025, 7, 8, 12, 1, tzinfo=UTC),
        close_time=datetime(2025, 7, 9, tzinfo=UTC),
        category="Other",
        outcome=outcome,
    )


def _patch_fetch(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []

    async def fake_fetch(self: RewindSearchTool, session: Any, query: str) -> dict[str, Any]:
        requests.append((query, self.as_of))
        return PAYLOAD

    monkeypatch.setattr(RewindSearchTool, "_fetch", fake_fetch)
    return requests


def _search_call(queries: list[str]) -> ToolCall:
    return ToolCall(
        id="call-1",
        function=ToolCall.FunctionBody(
            name="search", arguments=json.dumps({"query_list": queries})
        ),
    )


def test_search_day_is_last_full_utc_day_before_snapshot() -> None:
    assert search_as_of_day(datetime(2025, 7, 8, 12, 1, tzinfo=UTC)) == "2025-07-07"
    assert search_as_of_day(datetime(2025, 10, 20, tzinfo=UTC)) == "2025-10-19"


def test_format_results_caps_results_and_content() -> None:
    text = format_results("match preview", PAYLOAD, num_results=1, content_chars=7)

    assert "Query: match preview" in text
    assert "Result 1: Preview\nURL: https://example.test/a\nPre-mat" in text
    assert "Odds" not in text
    assert format_results("nothing", {"results": []}, 5, 100) == "Query: nothing\nNo results.\n"


def test_search_tool_sends_its_day_for_every_query(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _patch_fetch(monkeypatch)
    tool = RewindSearchTool(as_of="2025-07-07", api_key="key")

    result = asyncio.run(tool.search.run(ToolInput(arguments={"query_list": ["a", "b"]})))

    assert requests == [("a", "2025-07-07"), ("b", "2025-07-07")]
    content = result.messages[0]["content"]
    assert "Query: a" in content and "Query: b" in content


def test_search_sends_a_key_only_when_configured() -> None:
    assert RewindSearchTool(as_of="2025-07-07").headers() == {}
    assert RewindSearchTool(as_of="2025-07-07", api_key="key").headers() == {
        "Authorization": "Bearer key"
    }


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return PAYLOAD

    async def text(self) -> str:
        return "rate limited"


class FakeSession:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls = 0

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.statuses.pop(0))


def test_search_retries_when_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    tool = RewindSearchTool(as_of="2025-07-07")

    session = FakeSession([429, 429, 200])
    assert asyncio.run(tool._fetch(cast(Any, session), "q")) == PAYLOAD
    assert session.calls == 3

    with pytest.raises(RuntimeError, match="HTTP 429"):
        asyncio.run(tool._fetch(cast(Any, FakeSession([429] * 5)), "q"))


def test_search_env_scores_forecast_after_a_search(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _patch_fetch(monkeypatch)
    tool = SearchSettings(api_key="key").tool_for(_example().snapshot_time)
    env = ForecastSearchEnv(_example(), cast(renderers.Renderer, FakeRenderer()), tool, max_turns=3)

    messages = asyncio.run(env.initial_observation())
    assert messages[0]["role"] == "system" and "search" in messages[0]["content"]
    assert "Output only the probability of YES" in messages[1]["content"]

    searching = asyncio.run(
        env.step(
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [_search_call(["preview"])],
            }
        )
    )
    assert not searching.episode_done
    assert requests == [("preview", "2025-07-07")]
    assert "brier_reward" not in searching.logs

    final = asyncio.run(env.step({"role": "assistant", "content": "Reasoning.\n0.8"}))
    assert final.episode_done
    assert final.reward == pytest.approx(0.96)
    assert final.logs["brier_reward"] == pytest.approx(0.96)
    assert final.logs["forecast"] == "0.800000"
    assert final.logs["format_valid"] == 1.0


def test_search_env_without_a_final_answer_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch)
    tool = SearchSettings(api_key="key").tool_for(_example().snapshot_time)
    env = ForecastSearchEnv(_example(), cast(renderers.Renderer, FakeRenderer()), tool, max_turns=1)
    asyncio.run(env.initial_observation())

    result = asyncio.run(
        env.step({"role": "assistant", "content": "0.9", "tool_calls": [_search_call(["more"])]})
    )

    assert result.episode_done
    assert result.reward == 0.0
    assert result.logs["forecast"] == "invalid"
    assert result.logs["format_valid"] == 0.0


def test_group_builder_uses_search_env_only_when_configured() -> None:
    renderer = cast(renderers.Renderer, FakeRenderer())
    plain = asyncio.run(ForecastGroupBuilder(_example(), renderer, group_size=2).make_envs())
    searching = asyncio.run(
        ForecastGroupBuilder(
            _example(), renderer, group_size=2, search=SearchSettings(api_key="key")
        ).make_envs()
    )

    assert not any(
        isinstance(cast(EnvFromMessageEnv, e).message_env, ForecastSearchEnv) for e in plain
    )
    assert all(
        isinstance(cast(EnvFromMessageEnv, e).message_env, ForecastSearchEnv) for e in searching
    )
