"""Tests for provenance and stable_reinforce support in the RL training loop."""

import asyncio
from typing import Any, cast

import chz
import pytest
import tinker
from tinker.types import LossFnType

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl import train as train_module
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.train import (
    AsyncConfig,
    Config,
    _check_provenance,
    main,
    train_step,
)
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder, Trajectory, Transition


def _datum(sequence_ids: list[str], include_provenance: bool = True) -> tinker.Datum:
    """A datum with one turn per sequence id, each turn extending the previous prompt."""
    transitions, prompt = [], [1]
    for sequence_id in sequence_ids:
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(prompt),
                ac=TokensWithLogprobs(tokens=[2], maybe_logprobs=[-1.0], sequence_id=sequence_id),
                reward=0.0,
                episode_done=False,
            )
        )
        prompt = [*prompt, 2, 1]
    trajectory = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
    (datum,) = trajectory_to_data(trajectory, 1.0, include_provenance=include_provenance)
    return datum


class _RecordingTrainingClient:
    def __init__(self) -> None:
        self.requests: list[list[tinker.Datum]] = []

    async def forward_backward_async(
        self, data: list[tinker.Datum], loss_fn: LossFnType, loss_fn_config: dict[str, Any] | None
    ) -> int:
        self.requests.append(data)
        return len(self.requests)


def test_missing_provenance_is_caught_before_any_request_is_sent():
    # The bad datum is in the second substep's batch; nothing may be sent for the first.
    client = _RecordingTrainingClient()
    data_D = [_datum(["a"]), _datum(["b"], include_provenance=False)]
    with pytest.raises(ValueError, match="provenance"):
        asyncio.run(
            train_step(
                data_D,
                cast(tinker.TrainingClient, client),
                learning_rate=1e-5,
                num_substeps=2,
                loss_fn="stable_reinforce",
            )
        )
    assert client.requests == []


def test_stable_reinforce_without_provenance_is_a_clear_error():
    with pytest.raises(ValueError, match="sequence_id"):
        _check_provenance(
            [_datum(["a"]), _datum(["b"], include_provenance=False)], loss_fn="stable_reinforce"
        )


@chz.chz
class _UnusedDatasetBuilder(RLDatasetBuilder):
    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        raise NotImplementedError


def _config(**kwargs: Any) -> Config:
    return Config(
        learning_rate=1e-5,
        dataset_builder=_UnusedDatasetBuilder(),
        model_name="fake-model",
        recipe_name="train_test",
        max_tokens=4,
        log_path="/unused",
        **kwargs,
    )


def test_kl_discount_is_rejected_with_stable_reinforce():
    config = _config(loss_fn="stable_reinforce", kl_penalty_coef=0.1, kl_discount_factor=0.9)
    with pytest.raises(ConfigurationError, match="kl_discount_factor"):
        asyncio.run(main(config))


def test_training_loss_fn_comes_from_the_active_config():
    async_config = AsyncConfig(max_steps_off_policy=1, groups_per_batch=2)
    assert _config().training_loss_fn() == "importance_sampling"
    assert not _config().records_stability_info()
    assert _config(loss_fn="stable_reinforce").training_loss_fn() == "stable_reinforce"
    assert _config(async_config=async_config).training_loss_fn() == "stable_reinforce"
    assert _config(async_config=async_config).records_stability_info()
    ppo_async = AsyncConfig(max_steps_off_policy=1, groups_per_batch=2, loss_fn="ppo")
    assert _config(async_config=ppo_async).training_loss_fn() == "ppo"


def test_conflicting_config_loss_fn_is_rejected_in_async_runs():
    """Config.loss_fn would be ignored in an async run, so a conflicting value is an error."""
    async_config = AsyncConfig(max_steps_off_policy=1, groups_per_batch=2)
    with pytest.raises(ConfigurationError, match=r"AsyncConfig\.loss_fn"):
        asyncio.run(main(_config(async_config=async_config, loss_fn="ppo")))


class _PastConfigChecks(Exception):
    pass


def test_matching_config_loss_fn_is_accepted_in_async_runs(monkeypatch):
    """Recipes pass their loss to both configs; a matching value isn't a conflict."""

    def stop(*args: Any, **kwargs: Any) -> None:
        raise _PastConfigChecks

    # main() sets up logging right after its config checks; stop there.
    monkeypatch.setattr(train_module.ml_log, "setup_logging", stop)
    async_config = AsyncConfig(max_steps_off_policy=1, groups_per_batch=2, loss_fn="ppo")
    config = _config(async_config=async_config, loss_fn="ppo")
    assert config.training_loss_fn() == "ppo"
    with pytest.raises(_PastConfigChecks):
        asyncio.run(main(config))
