from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest

from tinker_cookbook import renderers
from tinker_cookbook.renderers import Message
from tinker_cookbook.rl.types import Trajectory

from .data import ForecastExample
from .env import (
    ForecastEnv,
    ForecastGroupBuilder,
    ForecastRLDataset,
    brier_reward,
    parse_forecast,
    render_prompt,
)


def _example(outcome: int = 1) -> ForecastExample:
    return ForecastExample(
        submission_id="submission",
        event_ticker="event",
        event_title="Will the event happen?",
        market="Event happens",
        reference_material="Dated evidence available at the snapshot.",
        resolution_criteria="Resolves YES if the event happens.",
        snapshot_time=datetime(2025, 10, 20, tzinfo=UTC),
        close_time=datetime(2025, 10, 21, tzinfo=UTC),
        category="Other",
        outcome=outcome,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("0", 0.0),
        ("Brief reasoning.\n0.37\n", 0.37),
        ("The event is certain.\n1.000", 1.0),
        ("Reasoning.\n0.37", 0.37),
        ("Probability: 0.37.", 0.37),
        ("**0.37**", 0.37),
        ("37%", 0.37),
        ("Final answer: 0.37!", 0.37),
        ("Reasoning.\nProbability: **0.37**", 0.37),
        ("The market closes on 2025-10-21", None),
        ("The market should resolve by 2026-01-01", None),
        ("0.2 trailing", None),
        ("0.2.", 0.2),
        (".37", 0.37),
        ("7e-1", 0.7),
        ("-0.1", None),
        ("1.2", None),
    ],
)
def test_parse_forecast(response: str, expected: float | None) -> None:
    assert parse_forecast(response) == expected


def test_brier_reward_is_strictly_proper() -> None:
    belief = 0.3

    def expected_reward(report: float) -> float:
        return belief * brier_reward(report, 1) + (1 - belief) * brier_reward(report, 0)

    assert expected_reward(belief) > expected_reward(0.2)
    assert expected_reward(belief) > expected_reward(0.5)


def test_brier_reward_is_affine_negative_brier() -> None:
    assert brier_reward(1.0, 1) == 1.0
    assert brier_reward(0.5, 1) == 0.75
    assert brier_reward(0.2, 0) == pytest.approx(0.96)
    assert brier_reward(1.0, 0) == 0.0


def test_prompt_excludes_outcome_and_market_prices() -> None:
    prompt = render_prompt(_example())

    assert "Will the event happen?" in prompt
    assert "Event happens" in prompt
    assert "Dated evidence available at the snapshot." in prompt
    assert "2025-10-20T00:00:00+00:00" in prompt
    assert "yes_ask" not in prompt
    assert "outcome=1" not in prompt
    assert "Output only the probability of YES" in prompt


def test_valid_forecast_uses_brier_reward() -> None:
    message: Message = {"role": "assistant", "content": "Reasoning.\n0.8"}
    result = asyncio.run(ForecastEnv(_example()).step(message))

    assert result.reward == pytest.approx(0.96)
    assert result.logs["brier_reward"] == pytest.approx(0.96)
    assert result.logs["accuracy"] == 1.0
    assert result.logs["format_valid"] == 1.0


def test_invalid_forecast_receives_zero_reward() -> None:
    message: Message = {"role": "assistant", "content": "No numeric forecast."}
    result = asyncio.run(ForecastEnv(_example()).step(message))

    assert result.reward == 0.0
    assert result.logs["format_valid"] == 0.0


def test_training_dataset_repeats_complete_epochs() -> None:
    examples = [_example(0), _example(1), _example(0)]
    dataset = ForecastRLDataset(
        examples,
        batch_size=2,
        group_size=1,
        renderer=cast(renderers.Renderer, None),
        epochs=3,
    )

    assert len(dataset) == 6
    assert [
        cast(ForecastGroupBuilder, builder).example for builder in dataset.get_batch(0)
    ] == examples[:2]
    assert [
        cast(ForecastGroupBuilder, builder).example for builder in dataset.get_batch(2)
    ] == examples[:2]
    assert [
        cast(ForecastGroupBuilder, builder).example for builder in dataset.get_batch(4)
    ] == examples[:2]


def test_group_metrics_include_truncated_rollouts() -> None:
    valid = cast(
        Trajectory,
        SimpleNamespace(
            transitions=[
                SimpleNamespace(
                    logs={
                        "brier_reward": 0.96,
                        "accuracy": 1.0,
                        "format_valid": 1.0,
                    }
                )
            ]
        ),
    )
    truncated = cast(Trajectory, SimpleNamespace(transitions=[SimpleNamespace(logs={})]))
    builder = ForecastGroupBuilder(
        example=_example(),
        renderer=cast(renderers.Renderer, None),
        group_size=2,
    )

    results = asyncio.run(builder.compute_group_rewards([valid, truncated], []))

    assert results == [
        (
            0.0,
            {
                "brier_reward": 0.96,
                "accuracy": 1.0,
                "format_valid": 1.0,
            },
        ),
        (
            0.0,
            {
                "brier_reward": 0.0,
                "accuracy": 0.0,
                "format_valid": 0.0,
            },
        ),
    ]
