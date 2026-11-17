"""Tests for provenance and stable_reinforce support in the RL training loop."""

import asyncio
import logging
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import chz
import pytest
import tinker
from tinker.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from tinker.types import LossFnType, SamplingClientKind

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
from tinker_cookbook.utils.cost_estimation import TokenUsage


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

    def forward_backward(
        self, data: list[tinker.Datum], loss_fn: LossFnType, loss_fn_config: dict[str, Any] | None
    ) -> AwaitableConcurrentFuture[int]:
        self.requests.append(data)
        future: Future[int] = Future()
        future.set_result(len(self.requests))
        return AwaitableConcurrentFuture(future)


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
    async_config = AsyncConfig(sampling_steps_ahead=1, groups_per_batch=2)
    assert _config().training_loss_fn() == "importance_sampling"
    assert not _config().records_stability_info()
    assert _config(loss_fn="stable_reinforce").training_loss_fn() == "stable_reinforce"
    assert _config(async_config=async_config).training_loss_fn() == "stable_reinforce"
    assert _config(async_config=async_config).records_stability_info()
    ppo_async = AsyncConfig(sampling_steps_ahead=1, groups_per_batch=2, loss_fn="ppo")
    assert _config(async_config=ppo_async).training_loss_fn() == "ppo"


def test_inflight_weight_updates_default_on_for_async_runs_only():
    assert _config().sampling_client_kind() == "snapshot"
    async_config = AsyncConfig(sampling_steps_ahead=1, groups_per_batch=2)
    assert _config(async_config=async_config).sampling_client_kind() == "inflight"
    snapshot = AsyncConfig(
        sampling_steps_ahead=1, groups_per_batch=2, inflight_weight_updates=False
    )
    assert _config(async_config=snapshot).sampling_client_kind() == "snapshot"
    # With an on-policy buffer no rollout overlaps a train step.
    on_policy = AsyncConfig(sampling_steps_ahead=0, groups_per_batch=2)
    assert _config(async_config=on_policy).sampling_client_kind() == "snapshot"


@pytest.mark.parametrize(
    "kind, checkpoint_step, expected_events",
    [
        ("snapshot", False, ["save_weights(snapshot)"]),
        ("snapshot", True, ["save_periodic", "create_sampling_client"]),
        ("inflight", False, ["save_weights(inflight)"]),
        ("inflight", True, ["save_weights(inflight)", "save_periodic"]),
    ],
)
def test_checkpoint_steps_publish_the_requested_client_kind(
    kind: SamplingClientKind, checkpoint_step: bool, expected_events: list[str]
):
    """A checkpoint's sampler weights are a snapshot, so an inflight client on a
    checkpoint step needs its own inflight save, published before the checkpoint."""
    events: list[str] = []

    class _TrainingClient:
        def create_sampling_client(self, path: str, **kwargs: Any) -> Future[str]:
            events.append("create_sampling_client")
            future: Future[str] = Future()
            future.set_result(path)
            return future

        async def save_weights_and_get_sampling_client(self, **kwargs: Any) -> str:
            events.append(f"save_weights({kwargs['kind']})")
            return kwargs["kind"]

    class _CheckpointManager:
        def should_save_periodic(self, step: int) -> bool:
            return checkpoint_step

        async def save_periodic_async(self, step: int, loop_state: dict[str, Any]):
            events.append("save_periodic")
            return {"sampler_path": f"sampler@{step}"}

    asyncio.run(
        train_module.save_checkpoint_and_get_sampling_client(
            cast(Any, _TrainingClient()),
            cast(Any, _CheckpointManager()),
            i_batch=2,
            record_stability_info=True,
            sampling_client_kind=kind,
        )
    )
    assert events == expected_events


@pytest.mark.parametrize(
    "kind, checkpoint_step, num_substeps, expected_events",
    [
        (
            "snapshot",
            False,
            1,
            [
                *["fwd_bwd", "optim", "publish(snapshot)"],
                *["fwd_bwd result", "optim result", "publish result"],
            ],
        ),
        # The checkpoint provides the snapshot weights, so there's no separate publish.
        (
            "snapshot",
            True,
            1,
            ["fwd_bwd", "optim", "fwd_bwd result", "optim result", "checkpoint", "load sampler"],
        ),
        (
            "inflight",
            True,
            1,
            [
                *["fwd_bwd", "optim", "publish(inflight)"],
                *["fwd_bwd result", "optim result", "publish result", "checkpoint"],
            ],
        ),
        (
            "inflight",
            False,
            2,
            [
                *["fwd_bwd", "optim", "fwd_bwd", "optim", "publish(inflight)"],
                *["fwd_bwd result", "optim result", "fwd_bwd result", "optim result"],
                "publish result",
            ],
        ),
        # Only after the third optim_step, not the second.
        (
            "snapshot",
            False,
            3,
            [
                *["fwd_bwd", "optim", "fwd_bwd", "optim", "fwd_bwd result", "optim result"],
                *["fwd_bwd", "optim", "publish(snapshot)", "fwd_bwd result", "optim result"],
                *["fwd_bwd result", "optim result", "publish result"],
            ],
        ),
    ],
)
def test_weights_are_published_once_the_final_optim_step_is_submitted(
    monkeypatch: pytest.MonkeyPatch,
    kind: SamplingClientKind,
    checkpoint_step: bool,
    num_substeps: int,
    expected_events: list[str],
):
    """The publish is submitted before any training result is awaited, so it
    lands on the same clock cycle as the final optim_step. A checkpoint is
    saved only after the published weights are ready."""
    events: list[str] = []

    class _Result:
        def __init__(self, name: str, value: Any) -> None:
            self.name, self.value = name, value

        async def _resolve(self) -> Any:
            events.append(f"{self.name} result")
            return self.value

        def __await__(self):
            return self._resolve().__await__()

    class _TrainingClient:
        def forward_backward(self, data: list[tinker.Datum], **kwargs: Any) -> _Result:
            events.append("fwd_bwd")
            return _Result("fwd_bwd", SimpleNamespace(loss_fn_outputs=[]))

        def optim_step(self, adam_params: tinker.AdamParams) -> _Result:
            events.append("optim")
            return _Result("optim", SimpleNamespace(metrics=None))

        def save_weights_and_get_sampling_client(self, **kwargs: Any) -> _Result:
            events.append(f"publish({kwargs['kind']})")
            return _Result("publish", kwargs["kind"])

        def create_sampling_client(self, path: str, **kwargs: Any) -> Future[str]:
            events.append("load sampler")
            future: Future[str] = Future()
            future.set_result(path)
            return future

    class _CheckpointManager:
        def should_save_periodic(self, step: int) -> bool:
            return checkpoint_step

        async def save_periodic_async(self, step: int, loop_state: dict[str, Any]):
            events.append("checkpoint")
            return {"sampler_path": "checkpoint-sampler"}

    data_D = [_datum(["a"]), _datum(["b"]), _datum(["c"])]
    monkeypatch.setattr(
        train_module, "prepare_minibatch", AsyncMock(return_value=(data_D, {}, TokenUsage()))
    )
    monkeypatch.setattr(train_module, "compute_kl_sample_train", lambda *args: {})
    config = _config(
        num_substeps=num_substeps,
        async_config=AsyncConfig(
            sampling_steps_ahead=1,
            groups_per_batch=2,
            inflight_weight_updates=kind == "inflight",
        ),
    )
    asyncio.run(
        train_module.do_train_step_and_get_sampling_client(
            config,
            0,
            cast(Any, _TrainingClient()),
            cast(Any, _CheckpointManager()),
            None,
            cast(Any, None),
            [],
            [],
        )
    )
    assert events == expected_events


def test_checkpoint_due_after_an_early_publish_is_saved_after_it():
    """A time-based checkpoint can come due while a step trains, after its
    snapshot was already published. The publish is used and the checkpoint
    follows it."""
    events: list[str] = []

    async def publish() -> str:
        events.append("publish")
        return "published"

    class _CheckpointManager:
        def should_save_periodic(self, step: int) -> bool:
            return True

        async def save_periodic_async(self, step: int, loop_state: dict[str, Any]):
            events.append("checkpoint")
            return {"sampler_path": "checkpoint-sampler"}

    sampling_client, _ = asyncio.run(
        train_module.save_checkpoint_and_get_sampling_client(
            cast(Any, None),
            cast(Any, _CheckpointManager()),
            i_batch=2,
            publish_future=publish(),
        )
    )
    assert sampling_client == "published"
    assert events == ["publish", "checkpoint"]


def test_conflicting_config_loss_fn_is_rejected_in_async_runs():
    """Config.loss_fn would be ignored in an async run, so a conflicting value is an error."""
    async_config = AsyncConfig(sampling_steps_ahead=1, groups_per_batch=2)
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
    async_config = AsyncConfig(sampling_steps_ahead=1, groups_per_batch=2, loss_fn="ppo")
    config = _config(async_config=async_config, loss_fn="ppo")
    assert config.training_loss_fn() == "ppo"
    with pytest.raises(_PastConfigChecks):
        asyncio.run(main(config))


# ---------------------------------------------------------------------------
# AsyncConfig.sampling_steps_ahead and the deprecated max_steps_off_policy
# ---------------------------------------------------------------------------


def _streaming(groups_per_batch: int) -> train_module.StreamMinibatchConfig:
    return train_module.StreamMinibatchConfig(groups_per_batch=groups_per_batch, num_minibatches=1)


def _steps_ahead_config(field: str, steps_ahead: int) -> AsyncConfig:
    """An AsyncConfig with groups_per_batch=2 and ``field`` (the new or the deprecated
    name) set to ``steps_ahead``."""
    if field == "sampling_steps_ahead":
        return AsyncConfig(sampling_steps_ahead=steps_ahead, groups_per_batch=2)
    return AsyncConfig(max_steps_off_policy=steps_ahead, groups_per_batch=2)


@pytest.mark.parametrize("field", ["sampling_steps_ahead", "max_steps_off_policy"])
@pytest.mark.parametrize(
    "steps_ahead, streaming_groups_per_batch, expected_buffer",
    [(0, None, 2), (2, None, 3 * 2), (2, 4, 3 * 4)],
    ids=["on_policy", "no_streaming", "streaming"],
)
def test_steps_ahead_sets_the_group_buffer(
    field: str, steps_ahead: int, streaming_groups_per_batch: int | None, expected_buffer: int
):
    """The buffer is groups_per_step * (steps_ahead + 1), where groups_per_step is the
    streaming batch size when streaming. The deprecated name means the same thing."""
    stream = (
        _streaming(streaming_groups_per_batch) if streaming_groups_per_batch is not None else None
    )
    async_config = _steps_ahead_config(field, steps_ahead)
    config = _config(async_config=async_config, stream_minibatch_config=stream)
    assert config.async_group_buffer_size() == expected_buffer


@pytest.mark.parametrize("deprecated", [True, False])
def test_main_logs_max_steps_off_policy_deprecation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, deprecated: bool
):
    """main() logs the deprecation once, naming the replacement."""

    def stop(*args: Any, **kwargs: Any) -> None:
        raise _PastConfigChecks

    monkeypatch.setattr(
        train_module.ml_log, "setup_logging", lambda **kwargs: SimpleNamespace(store=None)
    )
    monkeypatch.setattr(train_module.checkpoint_utils, "get_last_checkpoint", stop)
    async_config = (
        AsyncConfig(max_steps_off_policy=2, groups_per_batch=2)
        if deprecated
        else AsyncConfig(sampling_steps_ahead=2, groups_per_batch=2)
    )
    with caplog.at_level(logging.WARNING), pytest.raises(_PastConfigChecks):
        asyncio.run(main(_config(async_config=async_config)))
    messages = [r.getMessage() for r in caplog.records if "max_steps_off_policy" in r.getMessage()]
    assert len(messages) == deprecated
    if deprecated:
        assert "sampling_steps_ahead=2" in messages[0]


def test_setting_both_steps_ahead_fields_is_rejected():
    with pytest.raises(ConfigurationError, match="not both"):
        AsyncConfig(sampling_steps_ahead=1, max_steps_off_policy=1, groups_per_batch=2)


def test_setting_neither_steps_ahead_field_is_rejected():
    with pytest.raises(ConfigurationError, match="sampling_steps_ahead"):
        AsyncConfig(groups_per_batch=2)


@pytest.mark.parametrize("field", ["sampling_steps_ahead", "max_steps_off_policy"])
def test_negative_steps_ahead_is_rejected(field: str):
    with pytest.raises(ConfigurationError, match=f"{field} >= 0"):
        _steps_ahead_config(field, -1)


def test_zero_groups_per_batch_is_rejected():
    with pytest.raises(ConfigurationError, match="groups_per_batch >= 1"):
        AsyncConfig(sampling_steps_ahead=1, groups_per_batch=0)
