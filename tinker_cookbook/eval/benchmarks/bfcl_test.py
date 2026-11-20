"""BFCL simple-subset grading through the dataset builder and environment."""

import json
from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
from datasets import Dataset

from tinker_cookbook.eval.benchmarks import _bfcl
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
from tinker_cookbook.renderers.base import Renderer

# Grading fields from BFCL v3 simple at dataset revision
# 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda (gorilla-llm/Berkeley-Function-Calling-Leaderboard).
_ROWS = {
    0: {
        "id": "simple_0",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Find the area of a triangle with a base of 10 units and height of "
                    "5 units.",
                }
            ]
        ],
        "function": [
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                        "unit": {"type": "string"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
        "ground_truth": [
            {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
        ],
    },
    13: {
        "id": "simple_13",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Calculate the area under the curve y=x^2 from x=1 to x=3.",
                }
            ]
        ],
        "function": [
            {
                "name": "calculate_area_under_curve",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "function": {"type": "string"},
                        "interval": {"type": "array", "items": {"type": "float"}},
                        "method": {"type": "string"},
                    },
                    "required": ["function", "interval"],
                },
            }
        ],
        "ground_truth": [
            {
                "calculate_area_under_curve": {
                    "function": ["x**2", "lambda x: x**2", "y=x**2"],
                    "interval": [[1.0, 3.0]],
                    "method": ["", "trapezoidal"],
                }
            }
        ],
    },
    17: {
        "id": "simple_17",
        "question": [[{"role": "user", "content": "Find the prime factors of 450"}]],
        "function": [
            {
                "name": "get_prime_factors",
                "parameters": {
                    "type": "dict",
                    "properties": {"number": {"type": "integer"}, "formatted": {"type": "boolean"}},
                    "required": ["number", "formatted"],
                },
            }
        ],
        "ground_truth": [{"get_prime_factors": {"number": [450], "formatted": [True, ""]}}],
    },
    89: {
        "id": "simple_89",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Fetch all records for students studying Science in 'Bluebird High "
                    "School' from the StudentDB.",
                }
            ]
        ],
        "function": [
            {
                "name": "db_fetch_records",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "database_name": {"type": "string"},
                        "table_name": {"type": "string"},
                        "conditions": {
                            "type": "dict",
                            "properties": {
                                "department": {"type": "string"},
                                "school": {"type": "string"},
                            },
                        },
                        "fetch_limit": {"type": "integer"},
                    },
                    "required": ["database_name", "table_name", "conditions"],
                },
            }
        ],
        "ground_truth": [
            {
                "db_fetch_records": {
                    "database_name": ["StudentDB"],
                    "table_name": ["students"],
                    "conditions": [
                        {
                            "department": ["Science"],
                            "school": ["Bluebird High School", "Bluebird HS"],
                        }
                    ],
                    "fetch_limit": ["", 0],
                }
            }
        ],
    },
    363: {
        "id": "simple_363",
        "question": [
            [
                {
                    "role": "user",
                    "content": "Find the closest sushi restaurant with a patio in Boston.",
                }
            ]
        ],
        "function": [
            {
                "name": "restaurant_search.find_closest",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "location": {"type": "string"},
                        "cuisine": {"type": "string"},
                        "amenities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["location", "cuisine"],
                },
            }
        ],
        "ground_truth": [
            {
                "find_closest": {
                    "location": ["Boston", "Boston, MA"],
                    "cuisine": ["Sushi", "sushi"],
                    "amenities": [["Patio"]],
                }
            }
        ],
    },
}


def _make_env(monkeypatch: pytest.MonkeyPatch, row_id: int) -> _bfcl.BFCLEnv:
    row = deepcopy(_ROWS[row_id])
    ground_truth = row.pop("ground_truth")
    datasets = [
        Dataset.from_list([row]),
        Dataset.from_list([{"id": row["id"], "ground_truth": json.dumps(ground_truth)}]),
    ]
    monkeypatch.setattr(_bfcl, "load_dataset", Mock(side_effect=datasets))
    renderer = Mock(spec=Renderer)
    renderer.tokenizer = Mock()
    renderer.tokenizer.decode.side_effect = lambda action: bytes(action).decode()
    envs = _bfcl.BFCLBenchmarkBuilder().make_envs(cast(Renderer, renderer), BenchmarkConfig())
    assert len(envs) == 1
    return cast(_bfcl.BFCLEnv, envs[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row_id, arguments, reward",
    [
        pytest.param(0, {"base": 10, "height": 5}, 1, id="omitted-optional"),
        pytest.param(0, {"base": 10, "height": 5, "unit": "UNITS"}, 1, id="explicit-optional"),
        pytest.param(0, {"base": 10, "height": 5, "invented": 1}, 0, id="extra-argument"),
        pytest.param(0, {"base": 10}, 0, id="missing-required"),
        pytest.param(0, {"base": "10", "height": 5}, 0, id="wrong-type"),
        pytest.param(
            13,
            {"function": "lambda x: x**2", "interval": [1, 3]},
            1,
            id="alternative-and-int-to-float",
        ),
        pytest.param(13, {"function": "x**2", "interval": [3, 1]}, 0, id="list-order"),
        pytest.param(17, {"number": 450, "formatted": True}, 1, id="boolean"),
        pytest.param(17, {"number": 450}, 0, id="required-overrides-omission-marker"),
        pytest.param(17, {"number": 450, "formatted": 1}, 0, id="integer-is-not-boolean"),
        pytest.param(
            89,
            {
                "database_name": "StudentDB",
                "table_name": "students",
                "conditions": {"department": "Science", "school": "Bluebird HS"},
            },
            1,
            id="nested-alternative",
        ),
        pytest.param(
            89,
            {
                "database_name": "StudentDB",
                "table_name": "students",
                "conditions": {"department": "Science", "school": "Bluebird HS", "invented": True},
            },
            0,
            id="extra-nested-argument",
        ),
        pytest.param(
            363,
            {"location": "Boston, MA", "cuisine": "Sushi", "amenities": ["Patio"]},
            1,
            id="signature-supplies-qualified-name",
        ),
    ],
)
async def test_grading_real_simple_rows(
    monkeypatch: pytest.MonkeyPatch, row_id: int, arguments: dict, reward: float
):
    env = _make_env(monkeypatch, row_id)
    call = {"name": _ROWS[row_id]["function"][0]["name"], "arguments": arguments}
    result = await env.step(list(json.dumps(call).encode()))
    assert result.reward == reward
    assert result.metrics["correct"] == reward


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        {},
        {"arguments": {"base": 10, "height": 5}},
        {"name": None, "arguments": {"base": 10, "height": 5}},
        {"name": "different_function", "arguments": {"base": 10, "height": 5}},
        {"name": "calculate_triangle_area", "arguments": []},
    ],
)
async def test_invalid_calls_receive_no_credit(monkeypatch: pytest.MonkeyPatch, call: dict):
    env = _make_env(monkeypatch, 0)
    result = await env.step(list(json.dumps(call).encode()))
    assert result.reward == 0


def test_nested_question_is_rendered_as_the_user_query(monkeypatch: pytest.MonkeyPatch):
    env = _make_env(monkeypatch, 0)
    query = "Find the area of a triangle with a base of 10 units and height of 5 units."
    assert env.user_query == query
    assert f"User query: {query}\n\n" in env.prompt
