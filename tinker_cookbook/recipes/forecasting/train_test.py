from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from . import train as recipe
from .env import ProphetArenaRLDatasetBuilder


def test_final_checkpoint_is_evaluated_once(tmp_path: Path, monkeypatch) -> None:
    checkpoint = SimpleNamespace(batch=100, sampler_path="tinker://run/sampler_weights/final")
    monkeypatch.setattr(
        recipe.checkpoint_utils, "get_last_checkpoint", lambda *args, **kwargs: checkpoint
    )

    class DatasetBuilder:
        async def __call__(self):
            return object(), "validation"

    class ServiceClient:
        def __init__(self, *, base_url):
            assert base_url is None

        def create_sampling_client(self, *, base_model, model_path):
            assert base_model == "Qwen/Qwen3.8-27B"
            assert model_path == checkpoint.sampler_path
            return "sampling-client"

    monkeypatch.setattr(recipe.tinker, "ServiceClient", ServiceClient)
    monkeypatch.setattr(recipe.train, "RLTestSetEvaluator", lambda *args, **kwargs: "evaluator")

    calls = 0

    async def run_evaluations(evaluators, sampling_client, config, step, *, store):
        nonlocal calls
        calls += 1
        assert evaluators == ["evaluator"]
        assert sampling_client == "sampling-client"
        assert step == 100
        return {"test/env/all/brier_reward": 0.8}

    monkeypatch.setattr(recipe.train, "run_evaluations_parallel", run_evaluations)
    config = SimpleNamespace(
        eval_every=20,
        log_path=str(tmp_path),
        max_tokens=24_576,
        model_name="Qwen/Qwen3.8-27B",
        base_url=None,
        effective_rollout_strategy=lambda: "strategy",
    )

    train_config = cast(recipe.train.Config, config)
    for _ in range(2):
        asyncio.run(
            recipe._evaluate_final_checkpoint(
                train_config, cast(ProphetArenaRLDatasetBuilder, DatasetBuilder())
            )
        )

    assert calls == 1
