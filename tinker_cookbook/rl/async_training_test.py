"""
Latency torture tests for async RL training (``do_async_training``).

These tests run the full async training pipeline against in-process fakes of
the Tinker training/sampling clients, with heterogeneous artificial rollout
latencies designed to stress the staleness enforcement:

- **Nothing is discarded or regenerated**: each problem is sampled exactly once
  and trained on exactly once, even when its rollout takes many training steps'
  worth of wall-clock time.
- **The staleness bound holds end to end**: the fake sampling client stamps its
  weight version into the sampled tokens, and the fake training client records
  the step at which each datum is trained, so ``staleness = trained_step -
  sampled_version`` is measured from the data itself rather than from the
  training loop's own bookkeeping.
- **No deadlocks**: every end-to-end test runs under a hard ``asyncio.wait_for``
  timeout, including shutdown edge cases (constant-reward groups filtered to
  None, and datasets holding more groups than the training loop consumes).

The sleeps are tuned so the whole file runs in a few seconds.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import chz
import pytest
import tinker
import torch

from tinker_cookbook.rl.train import (
    AsyncConfig,
    Config,
    _InFlightGroupTracker,
    do_async_training,
)
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.utils import ml_log

# Sampled tokens encode the sampler's weight version as VERSION_BASE + version,
# so trained data can be attributed to the exact weights that generated it.
VERSION_BASE = 10_000

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Harness:
    """Shared recorder for everything the fakes observe."""

    # (problem_id, sampled_version, trained_step) per trained datum
    trained_records: list[tuple[int, int, int]] = field(default_factory=list)
    # number of sample_async calls per problem_id
    sample_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class _FakeFuture:
    value: Any
    delay_s: float = 0.0

    async def result_async(self) -> Any:
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        return self.value


@dataclass
class _FakePath:
    path: str


@dataclass
class _FakeSequence:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str = "stop"


@dataclass
class _FakeSampleResult:
    sequences: list[_FakeSequence]


@dataclass
class _FakeFwdBwdResult:
    loss_fn_outputs: list[dict[str, tinker.TensorData]]


@dataclass
class _FakeOptimResult:
    metrics: dict[str, float] = field(default_factory=dict)


class FakeSamplingClient:
    """Sampling client stub whose sampled tokens encode its weight version."""

    def __init__(self, version: int, harness: _Harness):
        self.version = version
        self.harness = harness

    async def sample_async(
        self, prompt: tinker.ModelInput, num_samples: int, sampling_params: Any
    ) -> _FakeSampleResult:
        problem_id = prompt.to_ints()[0]
        self.harness.sample_counts[problem_id] = self.harness.sample_counts.get(problem_id, 0) + 1
        return _FakeSampleResult(
            sequences=[
                _FakeSequence(tokens=[VERSION_BASE + self.version], logprobs=[-0.5])
                for _ in range(num_samples)
            ]
        )


class FakeTrainingClient:
    """Training client stub that records which (problem, version) pairs are trained.

    The weight version is the number of sampling-client publications so far
    (one per training iteration, regardless of num_substeps), matching the
    training loop's ``sampling_client_step`` accounting.
    """

    def __init__(self, harness: _Harness, train_time_s: float = 0.005):
        self.harness = harness
        self.train_time_s = train_time_s
        self.version = 0

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _FakeFuture:
        return _FakeFuture(_FakePath(path=f"mock://state/{name}"))

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> _FakeFuture:
        return _FakeFuture(_FakePath(path=f"mock://sampler/{name}"))

    async def save_weights_and_get_sampling_client_async(self) -> FakeSamplingClient:
        self.version += 1
        return FakeSamplingClient(self.version, self.harness)

    def create_sampling_client(self, path: str) -> FakeSamplingClient:
        # NOTE: does not bump `version` — only save_weights_and_get_sampling_client_async
        # publishes a new version. If a test enables periodic checkpoints
        # (save_every > 0), the training loop mints clients through
        # CheckpointManager.save_periodic_async instead, and this fake's version
        # accounting would need to model that path too.
        return FakeSamplingClient(self.version, self.harness)

    async def forward_backward_async(
        self,
        data: list[tinker.Datum],
        loss_fn: Any,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> _FakeFuture:
        for datum in data:
            problem_id = datum.model_input.to_ints()[0]
            version_token = int(datum.loss_fn_inputs["target_tokens"].to_torch().tolist()[-1])
            self.harness.trained_records.append(
                (problem_id, version_token - VERSION_BASE, self.version)
            )
        outputs = [
            {
                "logprobs": tinker.TensorData.from_torch(
                    torch.zeros(len(datum.loss_fn_inputs["target_tokens"].to_torch()))
                )
            }
            for datum in data
        ]
        return _FakeFuture(_FakeFwdBwdResult(loss_fn_outputs=outputs), delay_s=self.train_time_s)

    async def optim_step_async(self, adam_params: Any) -> _FakeFuture:
        return _FakeFuture(_FakeOptimResult())


class FakeTokenizer:
    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(str(t) for t in tokens)


class SleepEnv(Env):
    """Single-step env that takes ``latency_s`` wall-clock time to complete."""

    def __init__(self, problem_id: int, latency_s: float, reward: float):
        self.problem_id = problem_id
        self.latency_s = latency_s
        self.reward = reward

    async def initial_observation(self) -> tuple[Observation, list[int]]:
        return tinker.ModelInput.from_ints([self.problem_id]), []

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        await asyncio.sleep(self.latency_s)
        return StepResult(
            reward=self.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
        )


class SleepEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self, problem_id: int, latency_s: float, group_size: int, constant_reward: bool = False
    ):
        self.problem_id = problem_id
        self.latency_s = latency_s
        self.group_size = group_size
        self.constant_reward = constant_reward

    async def make_envs(self) -> Sequence[Env]:
        # Alternate rewards within the group (unless testing constant-reward
        # filtering) so that advantages are nonzero.
        return [
            SleepEnv(
                self.problem_id,
                self.latency_s,
                reward=0.0 if self.constant_reward else float(i % 2),
            )
            for i in range(self.group_size)
        ]


class ListDataset(RLDataset):
    def __init__(self, batches: list[list[EnvGroupBuilder]]):
        self.batches = batches

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.batches[index]

    def __len__(self) -> int:
        return len(self.batches)


@chz.chz
class _UnusedDatasetBuilder(RLDatasetBuilder):
    """Config placeholder; the tests pass the dataset to do_async_training directly."""

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    harness: _Harness
    elapsed_s: float
    num_problems: int
    group_size: int
    log_dir: str


async def _run_async_training(
    tmp_path: str,
    latencies: list[float],
    groups_per_batch: int,
    max_steps_off_policy: int,
    group_size: int = 2,
    train_time_s: float = 0.005,
    extra_groups_per_batch: int = 0,
    constant_reward_problem_ids: set[int] | None = None,
    num_substeps: int = 1,
    start_batch: int = 0,
    timeout_s: float = 60.0,
) -> RunResult:
    """Run do_async_training over fake clients with the given per-problem latencies.

    ``latencies[i]`` is the rollout duration of problem ``i``. Problems are
    grouped into batches of ``groups_per_batch + extra_groups_per_batch``
    builders; the training loop consumes exactly ``groups_per_batch`` per step,
    starting from batch ``start_batch`` (batches before it are never consumed,
    as when resuming from a checkpoint).
    """
    constant_reward_problem_ids = constant_reward_problem_ids or set()
    harness = _Harness()
    training_client = FakeTrainingClient(harness, train_time_s=train_time_s)
    # When resuming, the initial sampler weights correspond to iteration start_batch.
    training_client.version = start_batch

    dataset_batch_size = groups_per_batch + extra_groups_per_batch
    assert len(latencies) % dataset_batch_size == 0
    builders = [
        SleepEnvGroupBuilder(
            problem_id=i,
            latency_s=latency,
            group_size=group_size,
            constant_reward=i in constant_reward_problem_ids,
        )
        for i, latency in enumerate(latencies)
    ]
    batches = [
        cast(list[EnvGroupBuilder], builders[i : i + dataset_batch_size])
        for i in range(0, len(builders), dataset_batch_size)
    ]
    dataset = ListDataset(batches)

    config = Config(
        learning_rate=1e-5,
        dataset_builder=_UnusedDatasetBuilder(),
        model_name="fake-model",
        recipe_name="async_training_test",
        max_tokens=4,
        log_path=str(tmp_path),
        eval_every=0,
        save_every=0,
        num_groups_to_log=0,
        rollout_json_export=False,
        remove_constant_reward_groups=bool(constant_reward_problem_ids),
        num_substeps=num_substeps,
        async_config=AsyncConfig(
            max_steps_off_policy=max_steps_off_policy,
            groups_per_batch=groups_per_batch,
        ),
    )
    ml_logger = ml_log.setup_logging(log_dir=str(tmp_path), do_configure_logging_module=False)
    try:
        from tinker_cookbook import checkpoint_utils

        checkpoint_mgr = checkpoint_utils.CheckpointManager(
            training_client=cast(tinker.TrainingClient, training_client),
            service_client=cast(tinker.ServiceClient, None),
            log_path=str(tmp_path),
            save_every=0,
            store=ml_logger.store,
        )
        t_start = time.monotonic()
        await asyncio.wait_for(
            do_async_training(
                start_batch=start_batch,
                end_batch=len(batches),
                num_batches=len(batches),
                config=config,
                training_client=cast(tinker.TrainingClient, training_client),
                kl_reference_client=None,
                evaluators=[],
                dataset=dataset,
                ml_logger=ml_logger,
                tokenizer=cast(Any, FakeTokenizer()),
                error_counter=None,
                strategy=None,
                checkpoint_mgr=checkpoint_mgr,
            ),
            timeout=timeout_s,
        )
        elapsed_s = time.monotonic() - t_start
    finally:
        ml_logger.close()
    return RunResult(
        harness=harness,
        elapsed_s=elapsed_s,
        num_problems=len(latencies),
        group_size=group_size,
        log_dir=str(tmp_path),
    )


def _staleness_by_problem(result: RunResult) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for problem_id, sampled_version, trained_step in result.harness.trained_records:
        out.setdefault(problem_id, []).append(trained_step - sampled_version)
    return out


# ---------------------------------------------------------------------------
# Unit tests: _InFlightGroupTracker
# ---------------------------------------------------------------------------


class TestInFlightGroupTracker:
    def test_admission_control_blocks_at_capacity(self):
        async def _test():
            tracker = _InFlightGroupTracker(capacity=2)
            await tracker.acquire_slot()
            await tracker.acquire_slot()
            assert tracker.num_outstanding == 2

            third = asyncio.create_task(tracker.acquire_slot())
            await asyncio.sleep(0.01)
            assert not third.done(), "third acquire should block at capacity"

            tracker.release_slots(1)
            await asyncio.wait_for(third, timeout=1.0)
            assert tracker.num_outstanding == 2

        asyncio.run(_test())

    def test_stop_pacing_releases_blocked_acquirers(self):
        async def _test():
            tracker = _InFlightGroupTracker(capacity=1)
            await tracker.acquire_slot()
            blocked = asyncio.create_task(tracker.acquire_slot())
            await asyncio.sleep(0.01)
            assert not blocked.done()
            tracker.stop_pacing()
            await asyncio.wait_for(blocked, timeout=1.0)

        asyncio.run(_test())

    def test_in_flight_version_accounting(self):
        async def _test():
            queue: asyncio.Queue = asyncio.Queue()
            tracker = _InFlightGroupTracker(capacity=10)
            for _ in range(3):
                await tracker.acquire_slot()
            tracker.record_started(3)
            tracker.record_started(3)
            tracker.record_started(5)
            assert tracker.num_in_flight == 3
            assert tracker.num_in_flight_at_or_before(2) == 0
            assert tracker.num_in_flight_at_or_before(3) == 2
            assert tracker.num_in_flight_at_or_before(5) == 3
            # A None (filtered) completion frees its slot; real ones don't.
            tracker.record_completed_and_enqueue(3, None, queue)
            assert tracker.num_in_flight_at_or_before(3) == 1
            assert tracker.num_outstanding == 2
            tracker.record_completed_and_enqueue(3, cast(Any, "group"), queue)
            tracker.record_completed_and_enqueue(5, cast(Any, "group"), queue)
            assert tracker.num_in_flight == 0
            assert tracker.num_outstanding == 2
            assert [queue.get_nowait() for _ in range(3)] == [None, "group", "group"]

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# End-to-end torture tests
# ---------------------------------------------------------------------------


class TestAsyncStalenessTorture:
    def test_slow_rollouts_trained_exactly_once_within_staleness_bound(self, tmp_path):
        """The core torture test: heterogeneous latencies, including rollouts that
        take many training steps' worth of wall-clock time.

        Every 4th problem is 30x slower than the rest, and one problem is 150x
        slower. With the old discard-and-requeue behavior, the slow problems
        would be regenerated over and over and (mostly) never trained on. Now
        every problem must be sampled exactly once and trained exactly once,
        with staleness <= max_steps_off_policy measured from the data itself.
        """
        groups_per_batch = 4
        max_steps_off_policy = 2
        n_batches = 12
        latencies = [0.01] * (groups_per_batch * n_batches)
        for i in range(0, len(latencies), 4):
            latencies[i] = 0.3
        latencies[5] = 1.5  # monster straggler

        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=max_steps_off_policy,
            )
        )

        # Every problem is trained exactly once (group_size datums each), and
        # nothing is sampled more than once.
        staleness = _staleness_by_problem(result)
        assert sorted(staleness.keys()) == list(range(result.num_problems))
        for problem_id, values in staleness.items():
            assert len(values) == result.group_size, (
                f"problem {problem_id} trained {len(values)} times, "
                f"expected {result.group_size} datums (one group rollout)"
            )
        assert result.harness.sample_counts == dict.fromkeys(
            range(result.num_problems), result.group_size
        ), "each problem should be sampled exactly once per group member (no regeneration)"

        # The staleness bound holds for every trained datum.
        max_staleness = max(v for values in staleness.values() for v in values)
        assert max_staleness <= max_steps_off_policy, (
            f"staleness {max_staleness} exceeds bound {max_steps_off_policy}"
        )
        # And asynchrony actually happened (some data was trained off-policy).
        assert max_staleness > 0, "expected some off-policy data in async mode"

        # The logged async/staleness_* metrics must agree with the staleness
        # measured from the training data itself.
        expected_max_by_step: dict[int, int] = {}
        for _, sampled_version, trained_step in result.harness.trained_records:
            expected_max_by_step[trained_step] = max(
                expected_max_by_step.get(trained_step, 0), trained_step - sampled_version
            )
        logged_max_by_step = {}
        with open(f"{result.log_dir}/metrics.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if "async/staleness_max" in rec:
                    logged_max_by_step[rec["step"]] = rec["async/staleness_max"]
        assert logged_max_by_step == expected_max_by_step

    def test_zero_steps_off_policy_is_synchronous(self, tmp_path):
        """max_steps_off_policy=0 must reduce to fully on-policy training."""
        latencies = [0.01, 0.05, 0.01, 0.05] * 6
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                max_steps_off_policy=0,
            )
        )
        staleness = _staleness_by_problem(result)
        assert sorted(staleness.keys()) == list(range(len(latencies)))
        assert all(v == 0 for values in staleness.values() for v in values)

    def test_async_is_faster_than_synchronous_on_straggler_workload(self, tmp_path):
        """With one slow problem per batch, async training must overlap the slow
        rollouts with training instead of waiting for them every step."""
        groups_per_batch = 4
        n_batches = 10
        latencies = []
        for _ in range(n_batches):
            latencies.extend([0.2, 0.01, 0.01, 0.01])

        result_sync = asyncio.run(
            _run_async_training(
                str(tmp_path / "sync"),
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=0,
            )
        )
        result_async = asyncio.run(
            _run_async_training(
                str(tmp_path / "async"),
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=2,
            )
        )
        # Synchronous: every step waits for its slow rollout (~0.2s), so the run
        # takes >= n_batches * 0.2s. Async should pipeline the slow rollouts.
        assert result_sync.elapsed_s >= n_batches * 0.2
        assert result_async.elapsed_s < 0.75 * result_sync.elapsed_s, (
            f"async ({result_async.elapsed_s:.2f}s) should be well below "
            f"sync ({result_sync.elapsed_s:.2f}s)"
        )
        # Same data trained in both cases.
        assert sorted(_staleness_by_problem(result_async).keys()) == list(range(len(latencies)))

    def test_constant_reward_groups_do_not_stall_training(self, tmp_path):
        """Groups filtered to None (constant reward) release their slots; the run
        must terminate cleanly even though the trainer runs out of data early."""
        groups_per_batch = 4
        n_batches = 6
        latencies = [0.01] * (groups_per_batch * n_batches)
        latencies[2] = 0.3
        # More filtered groups than groups_per_batch * max_steps_off_policy, so
        # leaked admission slots (a regression in the None-completion path)
        # would exhaust capacity and deadlock this test.
        constant_ids = {1, 3, 6, 9, 13, 17}
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=1,
                constant_reward_problem_ids=constant_ids,
            )
        )
        staleness = _staleness_by_problem(result)
        # Constant-reward groups are never trained on.
        assert not (set(staleness.keys()) & constant_ids)
        # Everything that was trained respected the bound.
        assert all(0 <= v <= 1 for values in staleness.values() for v in values)
        # Each trained problem was trained exactly once.
        assert all(len(values) == result.group_size for values in staleness.values())

    def test_no_deadlock_when_dataset_outlives_training(self, tmp_path):
        """If the dataset holds more groups than end_batch * groups_per_batch, the
        workers left holding unconsumed problems must not deadlock on admission
        control after the training loop exits."""
        groups_per_batch = 3
        n_batches = 4
        extra = 2
        latencies = [0.01] * ((groups_per_batch + extra) * n_batches)
        latencies[0] = 0.2
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=1,
                extra_groups_per_batch=extra,
                timeout_s=30.0,
            )
        )
        staleness = _staleness_by_problem(result)
        # Exactly n_batches * groups_per_batch groups are trained on.
        assert len(staleness) == groups_per_batch * n_batches
        assert all(0 <= v <= 1 for values in staleness.values() for v in values)
        # Leftover problems are skipped once training is done, not rolled out:
        # at most one straggler per worker may already be mid-rollout.
        num_sampled_problems = len(result.harness.sample_counts)
        assert num_sampled_problems <= len(staleness) + groups_per_batch, (
            f"{num_sampled_problems - len(staleness)} leftover problems were sampled; "
            f"expected at most {groups_per_batch} (one per worker)"
        )

    def test_multiple_substeps_respect_staleness_bound(self, tmp_path):
        """num_substeps > 1 applies several optimizer updates per iteration, but
        staleness is defined in iterations (published sampler versions)."""
        latencies = [0.01, 0.2, 0.01, 0.01] * 8
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                max_steps_off_policy=1,
                num_substeps=2,
            )
        )
        staleness = _staleness_by_problem(result)
        assert sorted(staleness.keys()) == list(range(len(latencies)))
        assert all(0 <= v <= 1 for values in staleness.values() for v in values)

    def test_resume_from_nonzero_start_batch(self, tmp_path):
        """Resuming (start_batch > 0) keeps the staleness arithmetic aligned."""
        groups_per_batch = 3
        start_batch = 4
        n_batches = 8
        latencies = [0.01] * (groups_per_batch * n_batches)
        latencies[start_batch * groups_per_batch] = 0.25
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=1,
                start_batch=start_batch,
            )
        )
        staleness = _staleness_by_problem(result)
        # Only problems from batches start_batch..n_batches-1 are consumed.
        assert sorted(staleness.keys()) == list(
            range(start_batch * groups_per_batch, n_batches * groups_per_batch)
        )
        assert all(0 <= v <= 1 for values in staleness.values() for v in values)

    def test_async_config_incompatible_with_stream_minibatch(self, tmp_path):
        """async_config + stream_minibatch_config must raise, not silently ignore."""
        from tinker_cookbook.exceptions import ConfigurationError
        from tinker_cookbook.rl.train import StreamMinibatchConfig
        from tinker_cookbook.rl.train import main as train_main

        config = Config(
            learning_rate=1e-5,
            dataset_builder=_UnusedDatasetBuilder(),
            model_name="fake-model",
            recipe_name="async_training_test",
            max_tokens=4,
            log_path=str(tmp_path),
            async_config=AsyncConfig(max_steps_off_policy=1, groups_per_batch=4),
            stream_minibatch_config=StreamMinibatchConfig(groups_per_batch=4, num_minibatches=2),
        )
        with pytest.raises(ConfigurationError):
            asyncio.run(train_main(config))
        with pytest.raises(ConfigurationError):
            asyncio.run(
                do_async_training(
                    start_batch=0,
                    end_batch=1,
                    num_batches=1,
                    config=config,
                    training_client=cast(Any, None),
                    kl_reference_client=None,
                    evaluators=[],
                    dataset=cast(Any, None),
                    ml_logger=cast(Any, None),
                    tokenizer=cast(Any, None),
                    checkpoint_mgr=None,
                )
            )

    @pytest.mark.parametrize("max_steps_off_policy", [1, 3])
    def test_staleness_bound_respected_under_adversarial_latencies(
        self, tmp_path, max_steps_off_policy
    ):
        """Adversarial latency pattern: bursts of slow rollouts of one version,
        designed to pile up in-flight work at a single staleness deadline."""
        groups_per_batch = 3
        n_batches = 8
        latencies = []
        for i_batch in range(n_batches):
            if i_batch % 2 == 0:
                latencies.extend([0.25, 0.25, 0.01])
            else:
                latencies.extend([0.01, 0.01, 0.01])
        result = asyncio.run(
            _run_async_training(
                str(tmp_path / f"k{max_steps_off_policy}"),
                latencies,
                groups_per_batch=groups_per_batch,
                max_steps_off_policy=max_steps_off_policy,
            )
        )
        staleness = _staleness_by_problem(result)
        assert sorted(staleness.keys()) == list(range(len(latencies)))
        max_staleness = max(v for values in staleness.values() for v in values)
        assert max_staleness <= max_steps_off_policy
