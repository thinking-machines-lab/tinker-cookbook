"""
End-to-end tests for async RL training (``do_async_training``).

These tests run the full async training loop against in-process fakes of the
Tinker training and sampling clients, with artificial rollout latencies:

- The fake sampling client writes its weight version into the sampled tokens,
  and the fake training client records the step at which each datum is
  trained. Staleness (``trained_step - sampled_version``) is therefore measured
  from the trained data itself, not from the training loop's own bookkeeping.
- The fakes also record when each rollout starts and finishes, so the tests can
  check the admission cap and the oldest-first batch order from the outside.
- Every run is wrapped in a hard ``asyncio.wait_for`` timeout, so a deadlock
  fails the test instead of hanging it.

The sleeps are tuned so the whole file runs in about ten seconds.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import chz
import pytest
import tinker
import torch
from tinker.types import LossFnType, SampledProvenanceSpan, SamplingClientKind

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl import train as train_module
from tinker_cookbook.rl.rollout_strategy import MinViableGroup, RetryOnFailure, RolloutStrategy
from tinker_cookbook.rl.train import (
    AsyncConfig,
    Config,
    StreamMinibatchConfig,
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

# Latency of a rollout that never completes.
HANG = float("inf")

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _Harness:
    """Shared recorder for everything the fakes observe."""

    # (problem_id, sampled_version, trained_step) per trained datum
    trained_records: list[tuple[int, int, int]] = field(default_factory=list)
    # number of sample calls per problem_id
    sample_counts: dict[int, int] = field(default_factory=dict)
    # every sampler version that served at least one sample call
    sampled_versions: set[int] = field(default_factory=set)
    # problems whose rollout has started / whose training step has published new
    # weights (the point at which the training loop frees their slots)
    started_problems: set[int] = field(default_factory=set)
    trained_problems: set[int] = field(default_factory=set)
    # problems in the current training step, not yet published
    in_training_step: set[int] = field(default_factory=set)
    # problems whose group is filtered out (constant reward): their slot is freed
    # when the rollout finishes rather than when a step trains them
    filtered_problems: set[int] = field(default_factory=set)
    # largest number of outstanding problems seen when a rollout started
    max_outstanding: int = 0
    # envs currently inside step() per problem, and the most problems seen in step() at once
    running_envs: dict[int, int] = field(default_factory=dict)
    max_running: int = 0
    # monotonic time at which each problem's rollout finished
    finished_at: dict[int, float] = field(default_factory=dict)
    # monotonic time of the first forward_backward call of each training step
    step_started_at: dict[int, float] = field(default_factory=dict)
    # record_stability_info of every sampling client the training loop created
    record_stability_info: list[bool] = field(default_factory=list)
    # and how each was made: "checkpoint" (from a saved sampler path), or the
    # kind passed to save_weights_and_get_sampling_client ("snapshot"/"inflight")
    client_sources: list[str] = field(default_factory=list)
    # every sampling client the training loop created, in order
    clients: list[_FakeSamplingClient] = field(default_factory=list)
    # whether each sample_async call came from an inflight client
    sampled_inflight: set[bool] = field(default_factory=set)
    # (problem_id, client, version) of every sample_async call
    samples: list[tuple[int, _FakeSamplingClient, int]] = field(default_factory=list)
    # version of the weights last saved with kind="inflight"
    inflight_version: int | None = None
    # snapshot saves (kind="snapshot") in progress, and the version each rollout's
    # first sample used if it happened during one
    snapshot_saves_in_flight: int = 0
    # optim_step calls made while a snapshot save was in flight
    optim_steps_during_snapshot_save: int = 0
    first_sample_during_snapshot_save: dict[int, int] = field(default_factory=dict)
    # sequence ids named by the sampled provenance spans of each problem's trained data
    trained_sequence_ids: dict[int, set[str]] = field(default_factory=dict)
    # loss_fn of every forward_backward call
    loss_fns: list[str] = field(default_factory=list)
    # how long each problem's EnvGroupBuilder.cleanup takes, and whose cleanup finished
    cleanup_s: dict[int, float] = field(default_factory=dict)
    cleanups_finished: set[int] = field(default_factory=set)
    # problems with a forward_backward not yet followed by an optim_step, and
    # problems whose gradients an optim_step has applied
    pending_gradients: set[int] = field(default_factory=set)
    applied_problems: set[int] = field(default_factory=set)
    # optim_step calls so far, and how many of them were applied before the last
    # publication of new sampler weights
    optim_steps: int = 0
    # optim_steps behind each saved weight version (versions count saves of new weights)
    version_optim_steps: dict[int, int] = field(default_factory=lambda: {0: 0})
    published_optim_steps: int = 0


@dataclass
class _FakeFuture:
    value: Any
    delay_s: float = 0.0

    def result(self) -> Any:
        return self.value

    def __await__(self):
        return self.coroutine().__await__()

    async def coroutine(self) -> Any:
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
    sequence_id: str
    stop_reason: str = "stop"


@dataclass
class _FakeSampleResult:
    sequences: list[_FakeSequence]
    prompt_cache_hit_tokens: int = 0


@dataclass
class _FakeFwdBwdResult:
    loss_fn_outputs: list[dict[str, tinker.TensorData]]


@dataclass
class _FakeOptimResult:
    metrics: dict[str, float] = field(default_factory=dict)


class _FakeSamplingClient:
    """Sampling client stub whose sampled tokens encode its weight version.

    An inflight client samples from the latest inflight weights, like a real
    inflight client once an update has propagated.
    """

    def __init__(self, version: int, harness: _Harness, inflight: bool = False):
        self._version = version
        self.harness = harness
        self.inflight = inflight

    @property
    def version(self) -> int:
        if not self.inflight:
            return self._version
        assert self.harness.inflight_version is not None
        return self.harness.inflight_version

    def sample(
        self, prompt: tinker.ModelInput, num_samples: int, sampling_params: tinker.SamplingParams
    ) -> _FakeFuture:
        problem_id = prompt.to_ints()[0]
        h = self.harness
        h.sample_counts[problem_id] = h.sample_counts.get(problem_id, 0) + 1
        h.sampled_versions.add(self.version)
        h.sampled_inflight.add(self.inflight)
        h.samples.append((problem_id, self, self.version))
        if problem_id not in h.started_problems:
            h.started_problems.add(problem_id)
            if h.snapshot_saves_in_flight:
                h.first_sample_during_snapshot_save[problem_id] = self.version
            done = h.trained_problems | (h.filtered_problems & h.finished_at.keys())
            h.max_outstanding = max(h.max_outstanding, len(h.started_problems - done))
        return _FakeFuture(
            _FakeSampleResult(
                sequences=[
                    _FakeSequence(
                        tokens=[VERSION_BASE + self.version],
                        logprobs=[-0.5],
                        sequence_id=f"seq-{problem_id}-{h.sample_counts[problem_id]}-{i}",
                    )
                    for i in range(num_samples)
                ]
            )
        )


class _FakeTrainingClient:
    """Training client stub that records which (problem, version) pairs are trained.

    The weight version is the number of sampler saves that followed an
    optimizer step (one per training iteration, regardless of num_substeps),
    matching the training loop's ``sampling_client_step`` accounting. Saving
    again without an optimizer step in between saves the same version.
    """

    def __init__(self, harness: _Harness, train_time_s: float = 0.005, save_latency_s: float = 0.0):
        self.harness = harness
        self.train_time_s = train_time_s
        self.save_latency_s = save_latency_s
        self.version = 0
        self._optim_steps_at_save = 0

    def _save(self) -> int:
        if self.harness.optim_steps != self._optim_steps_at_save:
            self._optim_steps_at_save = self.harness.optim_steps
            self.version += 1
            self.harness.version_optim_steps[self.version] = self.harness.optim_steps
        return self.version

    def save_state(self, name: str, ttl_seconds: int | None = None) -> _FakeFuture:
        return _FakeFuture(_FakePath(path=f"mock://state/{name}"))

    def save_weights_for_sampler(self, name: str, ttl_seconds: int | None = None) -> _FakeFuture:
        self._save()
        return _FakeFuture(_FakePath(path=f"mock://sampler/{name}"))

    def _publish(
        self, version: int, record_stability_info: bool, source: str
    ) -> _FakeSamplingClient:
        h = self.harness
        h.record_stability_info.append(record_stability_info)
        h.client_sources.append(source)
        h.published_optim_steps = h.optim_steps
        h.trained_problems |= h.in_training_step
        h.in_training_step.clear()
        if source == "inflight":
            h.inflight_version = version
        client = _FakeSamplingClient(version, h, inflight=source == "inflight")
        h.clients.append(client)
        return client

    def save_weights_and_get_sampling_client(
        self, record_stability_info: bool = False, kind: SamplingClientKind = "snapshot"
    ) -> asyncio.Future[_FakeSamplingClient]:
        # Like the SDK, the save is submitted when requested: it captures the
        # weights then and takes a while, whether or not anyone awaits it yet.
        return asyncio.ensure_future(self._finish_save(self._save(), record_stability_info, kind))

    async def _finish_save(
        self, version: int, record_stability_info: bool, kind: SamplingClientKind
    ) -> _FakeSamplingClient:
        is_snapshot = kind == "snapshot"
        self.harness.snapshot_saves_in_flight += is_snapshot
        try:
            await asyncio.sleep(self.save_latency_s)
        finally:
            self.harness.snapshot_saves_in_flight -= is_snapshot
        return self._publish(version, record_stability_info, kind)

    def create_sampling_client(self, path: str, record_stability_info: bool = False) -> _FakeFuture:
        # Checkpoints are named by step (f"{step:06d}"), so the sampler path
        # identifies the weight version. Periodic checkpoints (save_every > 0)
        # publish new weights through this path instead of
        # save_weights_and_get_sampling_client.
        return _FakeFuture(
            self._publish(int(path.rsplit("/", 1)[-1]), record_stability_info, "checkpoint")
        )

    def forward_backward(
        self,
        data: list[tinker.Datum],
        loss_fn: LossFnType,
        loss_fn_config: dict[str, Any] | None = None,
    ) -> _FakeFuture:
        self.harness.step_started_at.setdefault(self.version, time.monotonic())
        self.harness.loss_fns.append(loss_fn)
        assert all(d.loss_fn_input_spans for d in data), "every datum should carry provenance"
        for datum in data:
            problem_id = datum.model_input.to_ints()[0]
            version_token = int(datum.loss_fn_inputs["target_tokens"].to_torch().tolist()[-1])
            self.harness.trained_records.append(
                (problem_id, version_token - VERSION_BASE, self.version)
            )
            self.harness.in_training_step.add(problem_id)
            self.harness.pending_gradients.add(problem_id)
            self.harness.trained_sequence_ids.setdefault(problem_id, set()).update(
                span.sequence_id
                for span in datum.loss_fn_input_spans or []
                if isinstance(span, SampledProvenanceSpan)
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

    def optim_step(self, adam_params: tinker.AdamParams) -> _FakeFuture:
        h = self.harness
        h.optim_steps += 1
        h.optim_steps_during_snapshot_save += h.snapshot_saves_in_flight > 0
        h.applied_problems |= h.pending_gradients
        h.pending_gradients.clear()
        return _FakeFuture(_FakeOptimResult())


class _FakeTokenizer:
    def decode(self, tokens: Sequence[int]) -> str:
        return " ".join(str(t) for t in tokens)


class _SleepEnv(Env):
    """Single-step env that takes ``latency_s`` wall-clock time to complete.

    ``latency_s=HANG`` never completes; ``fail=True`` raises after the latency.
    """

    def __init__(
        self,
        problem_id: int,
        latency_s: float,
        reward: float,
        harness: _Harness,
        fail: bool,
        two_turns: bool = False,
    ):
        self.problem_id = problem_id
        self.latency_s = latency_s
        self.reward = reward
        self.harness = harness
        self.fail = fail
        self.two_turns = two_turns
        self.turn = 0

    async def initial_observation(self) -> tuple[Observation, list[int]]:
        return tinker.ModelInput.from_ints([self.problem_id]), []

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        h = self.harness
        h.running_envs[self.problem_id] = h.running_envs.get(self.problem_id, 0) + 1
        h.max_running = max(h.max_running, len(h.running_envs))
        try:
            if self.latency_s == HANG:
                await asyncio.Event().wait()
            await asyncio.sleep(self.latency_s)
        finally:
            h.running_envs[self.problem_id] -= 1
            if h.running_envs[self.problem_id] == 0:
                del h.running_envs[self.problem_id]
        if self.fail:
            raise RuntimeError(f"env for problem {self.problem_id} failed")
        self.turn += 1
        if self.two_turns and self.turn == 1:
            # Wait for a newer inflight publication before the second turn, so the
            # rollout's second sample runs on newer weights than its first.
            first_version = int(action[-1]) - VERSION_BASE
            while (h.inflight_version or 0) <= first_version:
                await asyncio.sleep(0.005)
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=tinker.ModelInput.from_ints([self.problem_id, *action, 0]),
                next_stop_condition=[],
            )
        self.harness.finished_at[self.problem_id] = time.monotonic()
        return StepResult(
            reward=self.reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
        )


class _SleepEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        problem_id: int,
        latency_s: float,
        group_size: int,
        harness: _Harness,
        constant_reward: bool = False,
        fail: bool = False,
        two_turns: bool = False,
    ):
        self.problem_id = problem_id
        self.latency_s = latency_s
        self.group_size = group_size
        self.harness = harness
        self.constant_reward = constant_reward
        self.fail = fail
        self.two_turns = two_turns

    async def make_envs(self) -> Sequence[Env]:
        # Alternate rewards within the group (unless testing constant-reward
        # filtering) so that advantages are nonzero.
        return [
            _SleepEnv(
                self.problem_id,
                self.latency_s,
                reward=0.0 if self.constant_reward else float(i % 2),
                harness=self.harness,
                fail=self.fail,
                two_turns=self.two_turns,
            )
            for i in range(self.group_size)
        ]

    async def cleanup(self) -> None:
        await asyncio.sleep(self.harness.cleanup_s.get(self.problem_id, 0.0))
        self.harness.cleanups_finished.add(self.problem_id)


class _SlowEvaluator(SamplingClientEvaluator):
    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        await asyncio.sleep(0.4)
        return {}


class _FailingEvaluator(SamplingClientEvaluator):
    def __init__(self, failing_version: int) -> None:
        self.failing_version = failing_version

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        version = cast(_FakeSamplingClient, sampling_client).version
        await asyncio.sleep(0.01)
        if version == self.failing_version:
            raise RuntimeError(f"eval of version {version} failed")
        return {}


class _ListDataset(RLDataset):
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


class _RecordingEvaluator(SamplingClientEvaluator):
    def __init__(self, eval_time_s: float = 0.01) -> None:
        self.eval_time_s = eval_time_s
        self.versions: list[int] = []
        # weight versions at the end of each eval, and whether its client was inflight
        self.end_versions: list[int] = []
        self.inflight: list[bool] = []
        self.clients: list[_FakeSamplingClient] = []
        self.running = 0
        self.max_running = 0

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        client = cast(_FakeSamplingClient, sampling_client)
        version = client.version
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.versions.append(version)
        self.inflight.append(client.inflight)
        self.clients.append(client)
        await asyncio.sleep(self.eval_time_s)
        self.end_versions.append(client.version)
        self.running -= 1
        return {"eval/version": float(version)}


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@dataclass
class _RunResult:
    harness: _Harness
    num_problems: int
    group_size: int
    log_dir: str


async def _run_async_training(
    tmp_path: str,
    latencies: list[float],
    groups_per_batch: int,
    sampling_steps_ahead: int | None = None,
    max_steps_off_policy: int | None = None,
    group_size: int = 2,
    train_time_s: float = 0.005,
    extra_groups_per_batch: int = 0,
    constant_reward_problem_ids: set[int] | None = None,
    failing_problem_ids: set[int] | None = None,
    num_substeps: int = 1,
    start_batch: int = 0,
    strategy: RolloutStrategy | None = None,
    stream_minibatch_config: StreamMinibatchConfig | None = None,
    save_every: int = 0,
    eval_every: int = 0,
    evaluators: list[SamplingClientEvaluator] | None = None,
    loss_fn: LossFnType | None = None,
    inflight_weight_updates: bool = False,
    save_latency_s: float = 0.0,
    two_turn_problem_ids: set[int] | None = None,
    cleanup_s: dict[int, float] | None = None,
    timeout_s: float = 30.0,
) -> _RunResult:
    """Run do_async_training over fake clients with the given per-problem latencies.

    ``latencies[i]`` is the rollout duration of problem ``i``. Problems are
    grouped into dataset batches of ``groups_per_batch + extra_groups_per_batch``
    builders, starting from batch ``start_batch`` (earlier batches are never
    consumed, as when resuming from a checkpoint).
    """
    constant_reward_problem_ids = constant_reward_problem_ids or set()
    failing_problem_ids = failing_problem_ids or set()
    harness = _Harness(
        filtered_problems=set(constant_reward_problem_ids), cleanup_s=cleanup_s or {}
    )
    training_client = _FakeTrainingClient(
        harness, train_time_s=train_time_s, save_latency_s=save_latency_s
    )
    # When resuming, the initial sampler weights correspond to iteration start_batch.
    training_client.version = start_batch

    dataset_batch_size = groups_per_batch + extra_groups_per_batch
    assert len(latencies) % dataset_batch_size == 0
    builders = [
        _SleepEnvGroupBuilder(
            problem_id=i,
            latency_s=latency,
            group_size=group_size,
            harness=harness,
            constant_reward=i in constant_reward_problem_ids,
            fail=i in failing_problem_ids,
            two_turns=i in (two_turn_problem_ids or set()),
        )
        for i, latency in enumerate(latencies)
    ]
    batches = [
        cast(list[EnvGroupBuilder], builders[i : i + dataset_batch_size])
        for i in range(0, len(builders), dataset_batch_size)
    ]

    config = Config(
        learning_rate=1e-5,
        dataset_builder=_UnusedDatasetBuilder(),
        model_name="fake-model",
        recipe_name="async_training_test",
        max_tokens=4,
        log_path=str(tmp_path),
        eval_every=eval_every,
        save_every=save_every,
        num_groups_to_log=0,
        rollout_json_export=False,
        remove_constant_reward_groups=bool(constant_reward_problem_ids),
        num_substeps=num_substeps,
        stream_minibatch_config=stream_minibatch_config,
        async_config=(
            AsyncConfig(
                sampling_steps_ahead=sampling_steps_ahead,
                max_steps_off_policy=max_steps_off_policy,
                groups_per_batch=groups_per_batch,
                inflight_weight_updates=inflight_weight_updates,
            )
            if loss_fn is None
            else AsyncConfig(
                sampling_steps_ahead=sampling_steps_ahead,
                max_steps_off_policy=max_steps_off_policy,
                groups_per_batch=groups_per_batch,
                loss_fn=loss_fn,
                inflight_weight_updates=inflight_weight_updates,
            )
        ),
    )
    ml_logger = ml_log.setup_logging(log_dir=str(tmp_path), do_configure_logging_module=False)
    try:
        checkpoint_mgr = checkpoint_utils.CheckpointManager(
            training_client=cast(tinker.TrainingClient, training_client),
            service_client=cast(tinker.ServiceClient, None),
            log_path=str(tmp_path),
            save_every=save_every,
            store=ml_logger.store,
        )
        await asyncio.wait_for(
            do_async_training(
                start_batch=start_batch,
                end_batch=len(batches),
                num_batches=len(batches),
                config=config,
                training_client=cast(tinker.TrainingClient, training_client),
                kl_reference_client=None,
                evaluators=evaluators or [],
                dataset=_ListDataset(batches),
                ml_logger=ml_logger,
                tokenizer=cast(Any, _FakeTokenizer()),
                error_counter=None,
                strategy=strategy,
                checkpoint_mgr=checkpoint_mgr,
            ),
            timeout=timeout_s,
        )
    finally:
        ml_logger.close()
    return _RunResult(
        harness=harness,
        num_problems=len(latencies),
        group_size=group_size,
        log_dir=str(tmp_path),
    )


def _staleness_by_problem(result: _RunResult) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for problem_id, sampled_version, trained_step in result.harness.trained_records:
        out.setdefault(problem_id, []).append(trained_step - sampled_version)
    return out


def _assert_trained_exactly_once(
    result: _RunResult, problem_ids: Sequence[int] | None = None
) -> dict[int, list[int]]:
    """Each expected problem (default: all) was trained exactly once, as one full group.

    Returns the per-problem staleness values for further assertions.
    """
    staleness = _staleness_by_problem(result)
    expected = list(problem_ids) if problem_ids is not None else list(range(result.num_problems))
    assert sorted(staleness) == expected
    for problem_id, values in staleness.items():
        assert len(values) == result.group_size, (
            f"problem {problem_id} has {len(values)} trained datums, expected {result.group_size}"
        )
        assert all(v >= 0 for v in values)
    # Nothing was sampled twice (no discard-and-requeue).
    assert all(n == result.group_size for n in result.harness.sample_counts.values())
    return staleness


def _assert_every_update_applied_and_published(
    result: _RunResult, problem_ids: Sequence[int]
) -> None:
    """Exactly ``problem_ids`` were trained, every gradient was applied by an
    optim_step, and the weights after the last optim_step were published."""
    h = result.harness
    assert h.pending_gradients == set()
    assert h.applied_problems == set(problem_ids)
    assert h.trained_problems == set(problem_ids)
    assert h.published_optim_steps == h.optim_steps


def _assert_every_update_applied(result: _RunResult, problem_ids: Sequence[int]) -> None:
    """Exactly ``problem_ids`` were trained and every gradient was applied by an
    optim_step. Once every rollout worker has exited, steps nothing samples from
    or evaluates reuse the previous sampling client (#533), so the last updates
    need not be published."""
    h = result.harness
    assert h.pending_gradients == set()
    assert h.applied_problems == set(problem_ids)


def _metric_rows(result: _RunResult) -> list[dict[str, Any]]:
    """Training rows of metrics.jsonl (eval rows carry no staleness metrics)."""
    with open(f"{result.log_dir}/metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if "async/staleness_max" in r]


def _other_tasks() -> list[asyncio.Task[Any]]:
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


def _loop_tasks() -> list[asyncio.Task[Any]]:
    """Tasks created by do_async_training itself (it names them all ``*_loop*``)."""
    return [t for t in _other_tasks() if "_loop" in t.get_name()]


# ---------------------------------------------------------------------------
# No discarding, admission control, oldest first
# ---------------------------------------------------------------------------


class TestAsyncSemantics:
    def test_slow_rollouts_are_trained_exactly_once(self, tmp_path):
        """Rollouts spanning many training steps are trained, not discarded."""
        groups_per_batch = 4
        latencies = [0.01, 0.01, 0.01, 0.15] * 8
        latencies[1] = 1.0  # ~100x slower than the rest, spans many steps
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=1,
            )
        )
        staleness = _assert_trained_exactly_once(result)
        # The slow rollout was trained even though it is far more than one step stale.
        assert max(staleness[1]) > 1

    @pytest.mark.parametrize("steps_ahead", [0, 1, 3])
    def test_admission_cap_holds(self, tmp_path, steps_ahead):
        """At most groups_per_batch * (sampling_steps_ahead + 1) groups are ever
        outstanding."""
        groups_per_batch = 3
        # Fast sampling and slow training, so sampling would race ahead without the cap.
        latencies = [0.001] * (groups_per_batch * 8)
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=steps_ahead,
                train_time_s=0.03,
            )
        )
        _assert_trained_exactly_once(result)
        assert result.harness.max_outstanding == groups_per_batch * (steps_ahead + 1)

    @pytest.mark.parametrize("steps_ahead", [0, 1, 3])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_rollouts_run_concurrently_up_to_the_cap(self, tmp_path, steps_ahead, streaming):
        """Slow rollouts fill every buffer slot at once, not just groups_per_batch of them."""
        groups_per_batch = 2
        group_buffer_size = groups_per_batch * (steps_ahead + 1)
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.1] * (group_buffer_size * 2),
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=steps_ahead,
                stream_minibatch_config=(
                    StreamMinibatchConfig(groups_per_batch=groups_per_batch, num_minibatches=1)
                    if streaming
                    else None
                ),
            )
        )
        _assert_trained_exactly_once(result)
        assert result.harness.max_running == group_buffer_size

    @pytest.mark.parametrize("streaming", [False, True])
    def test_deprecated_max_steps_off_policy_sizes_the_buffer(self, tmp_path, streaming):
        """max_steps_off_policy=K still works like sampling_steps_ahead=K: the buffer holds
        groups_per_step * (K + 1) groups, where groups_per_step is the streaming batch size
        when streaming."""
        # With streaming, each step trains 4 groups rather than groups_per_batch=2.
        groups_per_step = 4 if streaming else 2
        expected_buffer = groups_per_step * 2
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.1] * (expected_buffer * 2),
                groups_per_batch=2,
                max_steps_off_policy=1,
                stream_minibatch_config=(
                    StreamMinibatchConfig(groups_per_batch=4, num_minibatches=1)
                    if streaming
                    else None
                ),
            )
        )
        _assert_trained_exactly_once(result)
        assert result.harness.max_running == expected_buffer

    @pytest.mark.parametrize("groups_per_batch", [1, 2])
    def test_streaming_trains_groups_in_the_order_they_finish(self, tmp_path, groups_per_batch):
        """With streaming minibatches every group is trained, none is pushed behind
        later-finishing groups, and none is lost at the end of training."""
        latencies = [0.01, 0.15, 0.03, 0.09, 0.05, 0.21, 0.07, 0.11, 0.13, 0.17, 0.19, 0.23]
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=1,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=groups_per_batch, num_minibatches=1
                ),
            )
        )
        _assert_trained_exactly_once(result)
        trained_step = {p: step for p, _, step in result.harness.trained_records}
        by_finish = sorted(trained_step, key=lambda p: result.harness.finished_at[p])
        steps = [trained_step[p] for p in by_finish]
        assert steps == sorted(steps)

    def test_buffer_of_one_step_is_on_policy(self, tmp_path):
        latencies = [0.02, 0.1] * 6
        result = asyncio.run(
            _run_async_training(tmp_path, latencies, groups_per_batch=2, sampling_steps_ahead=0)
        )
        staleness = _assert_trained_exactly_once(result)
        assert all(v == 0 for values in staleness.values() for v in values)

    @pytest.mark.parametrize("steps_ahead", [1, 2])
    def test_typical_staleness_tracks_the_buffer(self, tmp_path, steps_ahead):
        """With fast, uniform rollouts the pipeline fills up: staleness settles at
        sampling_steps_ahead steps."""
        groups_per_batch = 3
        n_batches = 10
        latencies = [0.001] * (groups_per_batch * n_batches)
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=steps_ahead,
                train_time_s=0.02,
            )
        )
        _assert_trained_exactly_once(result)
        rows = _metric_rows(result)
        assert all(r["async/staleness_max"] <= steps_ahead for r in rows)
        steady = [r for r in rows if r["training_client/step"] >= steps_ahead + 1]
        assert steady and all(r["async/staleness_mean"] >= steps_ahead - 0.5 for r in steady)

    @pytest.mark.parametrize(
        "streaming,inflight,deprecated",
        [
            (False, False, False),
            (False, True, False),
            (True, False, False),
            (True, True, False),
            (False, False, True),
            (True, False, True),
        ],
    )
    def test_buffer_utilization_metrics(self, tmp_path, streaming, inflight, deprecated):
        """Fast rollouts and slow training keep the buffer full, so every step but the
        last logs groups_in_flight == buffer size; the last step holds only its batch.
        Covers streaming, inflight updates and the max_steps_off_policy mapping."""
        groups_per_batch = 3
        buffer_size = groups_per_batch * 2
        n_batches = 6
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.001] * (groups_per_batch * n_batches),
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=None if deprecated else 1,
                max_steps_off_policy=1 if deprecated else None,
                stream_minibatch_config=(
                    StreamMinibatchConfig(groups_per_batch=groups_per_batch, num_minibatches=1)
                    if streaming
                    else None
                ),
                inflight_weight_updates=inflight,
                train_time_s=0.03,
            )
        )
        _assert_trained_exactly_once(result)
        rows = sorted(_metric_rows(result), key=lambda r: r["training_client/step"])
        assert len(rows) == n_batches
        for r in rows:
            assert r["async/buffer_utilization"] == r["async/groups_in_flight"] / buffer_size
        assert [r["async/groups_in_flight"] for r in rows] == [buffer_size] * (n_batches - 1) + [
            groups_per_batch
        ]

    def test_filtered_groups_leave_the_buffer(self, tmp_path):
        """A group filtered out for constant reward frees its slot right away and is
        never counted as in flight."""
        groups_per_batch = 2
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.001] * 6,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=0,
                constant_reward_problem_ids={0, 1},
                train_time_s=0.03,
            )
        )
        rows = _metric_rows(result)
        assert rows
        assert all(r["async/groups_in_flight"] == groups_per_batch for r in rows)
        assert all(r["async/buffer_utilization"] == 1.0 for r in rows)

    def test_each_step_logs_its_train_step_time(self, tmp_path):
        result = asyncio.run(
            _run_async_training(tmp_path, [0.001] * 6, groups_per_batch=3, sampling_steps_ahead=1)
        )
        rows = _metric_rows(result)
        assert rows and all("time/train_step" in r for r in rows)

    def test_batches_take_the_oldest_completed_groups_first(self, tmp_path):
        """Whenever a step trains, no older group that had already finished is left waiting.

        Uses a deep pipeline and slow training so that groups from several
        versions pile up between steps, with stragglers mixed in.
        """
        groups_per_batch = 2
        latencies = [0.001, 0.001, 0.06, 0.001] * 6
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=3,
                train_time_s=0.03,
            )
        )
        _assert_trained_exactly_once(result)
        h = result.harness
        version_of = {p: v for p, v, _ in h.trained_records}
        trained_step_of = {p: s for p, _, s in h.trained_records}
        steps = sorted(h.step_started_at)
        for prev_step, step in itertools.pairwise(steps):
            batch_max_version = max(v for p, v in version_of.items() if trained_step_of[p] == step)
            # Groups that finished before the previous step's train call were
            # certainly available when this step's batch was chosen.
            waiting = [
                p
                for p, t in h.finished_at.items()
                if t < h.step_started_at[prev_step] and trained_step_of[p] > step
            ]
            assert all(version_of[p] >= batch_max_version for p in waiting), (
                f"step {step}: trained version {batch_max_version} while older groups waited"
            )

    def test_filtered_groups_do_not_leak_slots(self, tmp_path):
        """More constant-reward (filtered) groups than the capacity still let training finish."""
        groups_per_batch = 2
        n_batches = 8
        latencies = [0.01] * (groups_per_batch * n_batches)
        # capacity = 2 * (1 + 1) = 4; filter 6 groups so a leaked slot per group deadlocks.
        filtered = {0, 1, 2, 3, 4, 5}
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=1,
                constant_reward_problem_ids=filtered,
            )
        )
        trained = _staleness_by_problem(result)
        assert not filtered & set(trained)
        # The 10 unfiltered groups fill 5 full batches.
        assert len(trained) == 10

    def test_logged_staleness_matches_trained_data(self, tmp_path):
        latencies = [0.01, 0.3, 0.01, 0.05] * 5
        result = asyncio.run(
            _run_async_training(tmp_path, latencies, groups_per_batch=4, sampling_steps_ahead=1)
        )
        _assert_trained_exactly_once(result)
        by_step: dict[int, list[int]] = {}
        for _, version, step in result.harness.trained_records:
            by_step.setdefault(step, []).append(step - version)
        for row in _metric_rows(result):
            values = by_step[row["training_client/step"]]
            assert row["async/staleness_max"] == max(values)
            assert row["async/staleness_min"] == min(values)
            assert row["async/staleness_mean"] == pytest.approx(sum(values) / len(values))


# ---------------------------------------------------------------------------
# Liveness: hung and failing rollouts, shutdown
# ---------------------------------------------------------------------------


class TestAsyncLiveness:
    def test_hung_rollout_does_not_block_training(self, tmp_path):
        """A rollout that never finishes only holds one slot: training goes on and the run ends."""
        # One spare group per dataset batch, so the training budget (4 steps of 4
        # groups) can be met without the hung one. (If the dataset runs out, the
        # run waits for the missing group, as synchronous training would.)
        latencies = [0.01] * 20
        latencies[1] = HANG

        async def _test() -> _RunResult:
            result = await _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                extra_groups_per_batch=1,
                sampling_steps_ahead=1,
                timeout_s=5.0,
            )
            assert _other_tasks() == []
            return result

        result = asyncio.run(_test())
        trained = _staleness_by_problem(result)
        assert 1 not in trained
        assert len(trained) == 16

    def test_failing_rollout_raises_promptly_and_cancels_everything(self, tmp_path):
        latencies = [0.01] * 16
        latencies[2] = HANG  # something still running when the failure lands

        async def _test() -> None:
            with pytest.raises(RuntimeError, match="problem 5 failed"):
                await _run_async_training(
                    tmp_path,
                    latencies,
                    groups_per_batch=4,
                    sampling_steps_ahead=2,
                    failing_problem_ids={5},
                    timeout_s=5.0,
                )
            # FailFast leaves the failed group's other rollouts to finish on their
            # own; the loop's own tasks must all be gone.
            assert _loop_tasks() == []

        asyncio.run(_test())

    def test_cancelling_the_run_cancels_everything(self, tmp_path):
        latencies = [0.01] * 16
        latencies[1] = HANG
        latencies[6] = HANG

        async def _test() -> None:
            run = asyncio.create_task(
                _run_async_training(tmp_path, latencies, groups_per_batch=4, sampling_steps_ahead=1)
            )
            await asyncio.sleep(0.2)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            assert _other_tasks() == []

        asyncio.run(_test())

    @pytest.mark.parametrize(
        "strategy",
        [None, RetryOnFailure(max_retries=1), MinViableGroup(min_successful=1)],
        ids=["fail_fast", "retry_on_failure", "min_viable_group"],
    )
    def test_rollouts_in_flight_when_training_ends_are_cancelled(self, tmp_path, strategy):
        """Rollouts that can never be trained don't keep the run (or their trajectories) alive."""
        # Dataset batches of 4 with groups_per_batch=2: training uses 6 of the 12
        # groups; problems 6 and 7 start early and hang.
        latencies = [0.01] * 12
        latencies[6] = HANG
        latencies[7] = HANG

        async def _test() -> _RunResult:
            result = await _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=2,
                extra_groups_per_batch=2,
                sampling_steps_ahead=3,
                strategy=strategy,
                timeout_s=5.0,
            )
            assert _other_tasks() == []
            return result

        result = asyncio.run(_test())
        trained = _staleness_by_problem(result)
        assert len(trained) == 6 and not {6, 7} & set(trained)

    def test_per_rollout_timeout_turns_a_hang_into_a_filtered_group(self, tmp_path):
        """The recommended guard for hanging envs: the group is dropped and its slot freed."""
        latencies = [0.01] * 16
        latencies[1] = HANG
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                strategy=RetryOnFailure(max_retries=0, per_rollout_timeout=0.2),
                timeout_s=10.0,
            )
        )
        trained = _staleness_by_problem(result)
        # Every other group is trained, the last three as a final smaller step.
        assert sorted(trained) == [p for p in range(16) if p != 1]

    def test_rollout_failing_after_training_ends_does_not_fail_the_run(self, tmp_path):
        """Once training is done, leftover rollouts are cancelled right away, even
        while the final evaluation is still running."""
        # One training step uses problems 0-1; problem 2 would fail mid-evaluation.
        asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.01, 0.15, 0.15],
                groups_per_batch=2,
                extra_groups_per_batch=2,
                sampling_steps_ahead=1,
                failing_problem_ids={2},
                eval_every=1,
                evaluators=[_SlowEvaluator()],
            )
        )

    def test_cancelled_rollouts_finish_their_cleanup(self, tmp_path):
        """Leftover rollouts are cancelled once; a second cancellation would abort
        EnvGroupBuilder.cleanup (e.g. leaking a sandbox)."""
        # Problems 2 and 3 hang and are cancelled after the only training step;
        # problem 2's quick cleanup ends while problem 3's is still running.
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.01, HANG, HANG],
                groups_per_batch=2,
                extra_groups_per_batch=2,
                sampling_steps_ahead=1,
                eval_every=1,
                evaluators=[_SlowEvaluator()],
                cleanup_s={2: 0.01, 3: 0.1},
            )
        )
        assert {2, 3} <= result.harness.cleanups_finished

    def test_cleanup_outlasting_the_final_evaluation_is_not_interrupted(self, tmp_path):
        """Shutdown after the final evaluation must not cancel leftover rollouts again:
        problem 3's cleanup is still running when the evaluation ends."""
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.01, HANG, HANG],
                groups_per_batch=2,
                extra_groups_per_batch=2,
                sampling_steps_ahead=1,
                eval_every=1,
                evaluators=[_SlowEvaluator()],
                cleanup_s={2: 0.01, 3: 1.0},
            )
        )
        assert {2, 3} <= result.harness.cleanups_finished


# ---------------------------------------------------------------------------
# Other configurations: substeps, resume, streaming, checkpoints, evaluation
# ---------------------------------------------------------------------------


class TestAsyncIntegrationPaths:
    def test_multiple_substeps(self, tmp_path):
        latencies = [0.01, 0.01, 0.2, 0.01] * 5
        result = asyncio.run(
            _run_async_training(
                tmp_path, latencies, groups_per_batch=4, sampling_steps_ahead=1, num_substeps=2
            )
        )
        _assert_trained_exactly_once(result)

    def test_resume_from_nonzero_start_batch(self, tmp_path):
        groups_per_batch = 2
        start_batch = 3
        latencies = [0.01, 0.1] * 8
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=0,
                start_batch=start_batch,
            )
        )
        resumed = range(start_batch * groups_per_batch, len(latencies))
        staleness = _assert_trained_exactly_once(result, problem_ids=resumed)
        assert all(v == 0 for values in staleness.values() for v in values)
        assert min(s for _, _, s in result.harness.trained_records) == start_batch

    def test_streaming_minibatches(self, tmp_path):
        latencies = [0.01, 0.05, 0.01, 0.2] * 5
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=2
                ),
            )
        )
        _assert_trained_exactly_once(result)
        assert result.harness.max_outstanding <= 4 * 2

    def test_streaming_with_a_different_batch_size(self, tmp_path):
        """The admission cap follows the batch size the training loop consumes."""
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.0] * 8,
                groups_per_batch=2,
                extra_groups_per_batch=2,
                sampling_steps_ahead=1,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=1
                ),
                timeout_s=10.0,
            )
        )
        assert len(_staleness_by_problem(result)) == 8

    def test_streaming_skips_filtered_groups(self, tmp_path):
        """Each streamed step trains a full batch of unfiltered groups."""
        # Spare groups per dataset batch, so the run never reaches the end of the data.
        latencies = [0.01] * 24
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                extra_groups_per_batch=2,
                sampling_steps_ahead=1,
                # Interleaved with unfiltered groups, so they land mid-minibatch.
                constant_reward_problem_ids={0, 2, 4, 6, 8, 10},
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=2
                ),
            )
        )
        trained_per_step: dict[int, set[int]] = {}
        for problem_id, _, step in result.harness.trained_records:
            trained_per_step.setdefault(step, set()).add(problem_id)
        assert [len(p) for _, p in sorted(trained_per_step.items())] == [4, 4, 4, 4]

    @pytest.mark.parametrize("inflight", [False, True], ids=["snapshot", "inflight"])
    def test_streaming_step_cut_short_by_shutdown_is_published(self, tmp_path, inflight):
        """If the data runs out after a step's first optimizer substep, that update is
        still published (new weights, metrics) rather than silently dropped."""
        # 6 groups, 4 per streamed step in 2 substeps: step 1 applies its first
        # substep (problems 4-5), then the dataset runs out.
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 6,
                groups_per_batch=2,
                sampling_steps_ahead=3,
                num_substeps=2,
                inflight_weight_updates=inflight,
                save_latency_s=0.01 if inflight else 0.0,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=1
                ),
            )
        )
        _assert_trained_exactly_once(result)
        h = result.harness
        # Every trained problem ended up in published weights.
        assert h.in_training_step == set()
        assert h.trained_problems == set(range(6))
        assert [r["training_client/step"] for r in _metric_rows(result)] == [0, 1]
        assert result.harness.sampled_inflight == {inflight}

    @pytest.mark.parametrize("inflight", [False, True], ids=["snapshot", "inflight"])
    def test_streaming_step_cut_short_mid_substep_is_published(self, tmp_path, inflight):
        """Two steps of two substeps with one filtered group: the last substep is
        short a group when the data runs out, and is still applied and published."""
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 8,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                num_substeps=2,
                constant_reward_problem_ids={5},
                inflight_weight_updates=inflight,
                save_latency_s=0.01 if inflight else 0.0,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=1
                ),
            )
        )
        trained = [p for p in range(8) if p != 5]
        _assert_every_update_applied_and_published(result, trained)
        assert result.harness.optim_steps == 4
        assert [r["training_client/step"] for r in _metric_rows(result)] == [0, 1]
        assert result.harness.sampled_inflight == {inflight}

    @pytest.mark.parametrize("inflight", [False, True], ids=["snapshot", "inflight"])
    def test_streaming_partial_minibatch_at_shutdown_is_trained(self, tmp_path, inflight):
        """A minibatch the data runs out before filling is trained, not dropped."""
        # Two dataset batches of 4 with three groups filtered: step 0 trains 4 in two
        # minibatches of 2; step 1 has a single group when the data runs out.
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 8,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                constant_reward_problem_ids={2, 5, 6},
                inflight_weight_updates=inflight,
                save_latency_s=0.01 if inflight else 0.0,
                stream_minibatch_config=StreamMinibatchConfig(
                    groups_per_batch=4, num_minibatches=2
                ),
            )
        )
        _assert_every_update_applied_and_published(result, [0, 1, 3, 4, 7])
        assert result.harness.optim_steps == 2
        assert [r["training_client/step"] for r in _metric_rows(result)] == [0, 1]
        assert result.harness.sampled_inflight == {inflight}

    def test_groups_left_when_the_data_runs_out_are_trained(self, tmp_path):
        """Four groups offered two per step with one filtered: the leftover group is
        trained as a final, smaller step rather than dropped."""
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 4,
                groups_per_batch=2,
                sampling_steps_ahead=1,
                constant_reward_problem_ids={1},
                eval_every=1,
                evaluators=[_RecordingEvaluator()],
            )
        )
        _assert_every_update_applied_and_published(result, [0, 2, 3])
        assert result.harness.optim_steps == 2
        assert [r["training_client/step"] for r in _metric_rows(result)] == [0, 1]

    def test_requested_steps_match_applied_and_published_updates(self, tmp_path):
        """With an evaluator consuming every step, N requested steps over N batches
        give exactly N optimizer updates, N publications and N logged steps."""
        n_steps, groups_per_batch, num_substeps = 5, 2, 2
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * (n_steps * groups_per_batch),
                groups_per_batch=groups_per_batch,
                sampling_steps_ahead=1,
                num_substeps=num_substeps,
                eval_every=1,
                evaluators=[_RecordingEvaluator()],
            )
        )
        _assert_every_update_applied_and_published(result, range(n_steps * groups_per_batch))
        assert result.harness.optim_steps == n_steps * num_substeps
        # One sampler publication per step, plus the initial client.
        assert len(result.harness.record_stability_info) == n_steps + 1
        assert [r["training_client/step"] for r in _metric_rows(result)] == list(range(n_steps))

    def test_single_group_per_batch(self, tmp_path):
        latencies = [0.01, 0.2, 0.01, 0.01, 0.15, 0.01] * 2
        result = asyncio.run(
            _run_async_training(tmp_path, latencies, groups_per_batch=1, sampling_steps_ahead=2)
        )
        _assert_trained_exactly_once(result)
        assert result.harness.max_outstanding <= 3

    def test_periodic_checkpoints(self, tmp_path):
        """With save_every > 0 some steps publish weights through the checkpoint path."""
        latencies = [0.01, 0.01, 0.01, 0.2] * 6
        result = asyncio.run(
            _run_async_training(
                tmp_path, latencies, groups_per_batch=4, sampling_steps_ahead=1, save_every=2
            )
        )
        _assert_trained_exactly_once(result)
        assert result.harness.sampled_versions >= {0, 1, 2}

    def test_evaluation_loop(self, tmp_path):
        evaluator = _RecordingEvaluator()
        latencies = [0.01, 0.01, 0.05, 0.01] * 5
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                latencies,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                eval_every=1,
                evaluators=[evaluator],
            )
        )
        _assert_trained_exactly_once(result)
        assert evaluator.versions, "evaluation loop never ran"
        assert evaluator.versions == sorted(evaluator.versions)


# ---------------------------------------------------------------------------
# Loss function: stable_reinforce by default
# ---------------------------------------------------------------------------


class TestAsyncDefaultLoss:
    @pytest.mark.parametrize(
        "loss_fn, streaming, save_every",
        [
            (None, False, 0),
            (None, True, 0),
            (None, False, 2),
            ("ppo", False, 0),
        ],
        ids=[
            "default_batched",
            "default_streaming",
            "default_periodic_checkpoints",
            "explicit_ppo",
        ],
    )
    def test_loss_fn_and_stability_info_follow_the_config(
        self, tmp_path, loss_fn, streaming, save_every
    ):
        """By default async training uses stable_reinforce on every step and every
        sampling client records stability info; another AsyncConfig.loss_fn turns
        stability info off. Datums carry provenance either way (the fake trainer
        asserts it)."""
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.05, 0.01, 0.1] * 4,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                loss_fn=loss_fn,
                save_every=save_every,
                stream_minibatch_config=(
                    StreamMinibatchConfig(groups_per_batch=4, num_minibatches=2)
                    if streaming
                    else None
                ),
            )
        )
        _assert_trained_exactly_once(result)
        h = result.harness
        expected_loss_fn = loss_fn or "stable_reinforce"
        assert set(h.loss_fns) == {expected_loss_fn}
        assert h.record_stability_info
        assert all(r == (expected_loss_fn == "stable_reinforce") for r in h.record_stability_info)


# ---------------------------------------------------------------------------
# Inflight weight updates
# ---------------------------------------------------------------------------


def _eval_rows(result: _RunResult) -> list[dict[str, Any]]:
    with open(f"{result.log_dir}/metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if "eval/version" in r]


_STREAMING = StreamMinibatchConfig(groups_per_batch=4, num_minibatches=2)
# Gives the fake's sampler saves real latency, so other loops run during them.
_SAVE_LATENCY_S = 0.02


class TestAsyncInflightWeights:
    @pytest.mark.parametrize("save_every", [0, 2])
    def test_flag_off_publishes_snapshots(self, tmp_path, save_every):
        """With the flag off, every client is a snapshot, one per step plus the
        initial one, and evals use those same clients (no extra saves)."""
        n_steps = 4
        evaluator = _RecordingEvaluator()
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.05, 0.01, 0.02] * n_steps,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                save_every=save_every,
                eval_every=2,
                evaluators=[evaluator],
                save_latency_s=_SAVE_LATENCY_S,
                inflight_weight_updates=False,
            )
        )
        _assert_trained_exactly_once(result)
        h = result.harness
        assert h.sampled_inflight == {False}
        expected_sources = ["checkpoint"] + [
            "checkpoint" if save_every and step % save_every == 0 else "snapshot"
            for step in range(1, n_steps + 1)
        ]
        assert h.client_sources == expected_sources
        # Each eval ran on the very client published for its step's rollouts.
        assert evaluator.versions == [0, 2, 4]
        assert all(any(c is e for c in h.clients) for e in evaluator.clients)

    @pytest.mark.parametrize(
        "streaming, save_every, start_batch, loss_fn",
        [
            (False, 0, 0, None),
            (False, 2, 0, None),
            (True, 0, 0, None),
            (True, 2, 0, None),
            (False, 2, 2, None),
            (False, 2, 0, "ppo"),
        ],
        ids=["batched", "checkpoints", "streaming", "streaming_checkpoints", "resume", "ppo"],
    )
    def test_rollouts_sample_from_inflight_weights(
        self, tmp_path, streaming, save_every, start_batch, loss_fn
    ):
        """With the flag, the initial client and the client published by every
        step (checkpoint steps included) are inflight, recording stability info
        exactly when the loss needs it."""
        n_steps = 5
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.05, 0.01, 0.1] * (n_steps + start_batch),
                groups_per_batch=4,
                sampling_steps_ahead=1,
                save_every=save_every,
                start_batch=start_batch,
                stream_minibatch_config=_STREAMING if streaming else None,
                loss_fn=loss_fn,
                inflight_weight_updates=True,
                save_latency_s=_SAVE_LATENCY_S,
            )
        )
        trained = range(4 * start_batch, 4 * (n_steps + start_batch))
        _assert_trained_exactly_once(result, trained)
        _assert_every_update_applied(result, trained)
        h = result.harness
        assert h.sampled_inflight == {True}
        # No evaluators, so no snapshot clients, not even from checkpoints: the
        # initial client, then one per step until the rollout workers have exited.
        assert set(h.client_sources) == {"inflight"}
        assert 1 + n_steps - 2 <= len(h.client_sources) <= n_steps + 1
        assert set(h.record_stability_info) == {loss_fn is None}

    def test_multi_turn_rollout_moves_to_newer_weights(self, tmp_path):
        """A rollout whose second turn starts after new weights are published
        samples both turns through the same client (one session), the second on
        the newer weights, and its trained datum names both sampled sequences."""
        two_turn = {4, 5}
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 8,
                groups_per_batch=2,
                sampling_steps_ahead=2,
                two_turn_problem_ids=two_turn,
                inflight_weight_updates=True,
                save_latency_s=_SAVE_LATENCY_S,
            )
        )
        h = result.harness
        _assert_every_update_applied(result, range(8))
        for problem_id in two_turn:
            samples = [(c, v) for p, c, v in h.samples if p == problem_id]
            # group_size samples per turn
            assert len(samples) == 2 * result.group_size
            first, second = samples[: result.group_size], samples[result.group_size :]
            assert len({id(c) for c, _ in samples}) == 1
            assert max(v for _, v in first) < min(v for _, v in second)
            # The trained data ends with the second turn's sampled token...
            trained_versions = {v for p, v, _ in h.trained_records if p == problem_id}
            assert trained_versions == {v for _, v in second}
            # ...and its provenance names both turns' sequences of every trajectory.
            assert len(h.trained_sequence_ids[problem_id]) == 2 * result.group_size

    @pytest.mark.parametrize(
        "streaming, save_every, start_batch",
        [(False, 0, 0), (False, 3, 0), (True, 0, 0), (True, 3, 0), (False, 0, 2)],
        ids=["batched", "checkpoints", "streaming", "streaming_checkpoints", "resume"],
    )
    def test_evals_use_snapshots_of_the_reported_step(
        self, tmp_path, streaming, save_every, start_batch
    ):
        """Every eval runs on fixed weights of exactly the step it is logged at,
        including the final step off the eval cadence. With evals faster than
        training no cadence step is skipped, and each snapshot saved is evaluated."""
        n_steps, eval_every = 5, 2
        # Slow enough that an inflight client would move to newer weights mid-eval.
        evaluator = _RecordingEvaluator(eval_time_s=0.05)
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.03, 0.01, 0.02] * (n_steps + start_batch),
                groups_per_batch=4,
                sampling_steps_ahead=1,
                train_time_s=0.05,
                save_every=save_every,
                start_batch=start_batch,
                eval_every=eval_every,
                evaluators=[evaluator],
                stream_minibatch_config=_STREAMING if streaming else None,
                inflight_weight_updates=True,
                save_latency_s=_SAVE_LATENCY_S,
            )
        )
        h = result.harness
        end = start_batch + n_steps
        assert h.sampled_inflight == {True}
        assert not any(evaluator.inflight)
        assert evaluator.end_versions == evaluator.versions
        rows = _eval_rows(result)
        assert [r["eval/version"] for r in rows] == [float(r["step"]) for r in rows]
        expected = sorted({s for s in range(start_batch, end + 1) if s % eval_every == 0} | {end})
        assert sorted(r["step"] for r in rows) == expected
        # One snapshot save per evaluated step, the starting step included.
        assert h.client_sources.count("snapshot") == len(expected)
        assert h.client_sources.count("checkpoint") == 0
        assert h.client_sources.count("inflight") == n_steps + 1
        # Only the rollout clients record stability info.
        assert h.record_stability_info == [src == "inflight" for src in h.client_sources]
        # Each eval logs its snapshot save's time with its results.
        assert all("eval/timing/time/save_eval_snapshot" in r for r in rows)

    def test_slow_evals_overlap_and_none_are_skipped(self, tmp_path):
        """With evals slower than training, evals of different steps run at the
        same time instead of skipping cadence steps, and each evaluates its own
        step's snapshot."""
        n_steps = 8
        evaluator = _RecordingEvaluator(eval_time_s=0.3)
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * (4 * n_steps),
                groups_per_batch=4,
                sampling_steps_ahead=1,
                eval_every=1,
                evaluators=[evaluator],
                inflight_weight_updates=True,
                save_latency_s=_SAVE_LATENCY_S,
            )
        )
        h = result.harness
        rows = _eval_rows(result)
        assert sorted(int(r["step"]) for r in rows) == list(range(n_steps + 1))
        assert [r["eval/version"] for r in rows] == [float(r["step"]) for r in rows]
        # One snapshot save per step, each evaluated on weights that stay fixed.
        assert h.client_sources.count("snapshot") == n_steps + 1
        assert not any(evaluator.inflight)
        assert evaluator.end_versions == evaluator.versions
        assert evaluator.max_running > 1

    def test_training_continues_while_an_eval_snapshot_saves(self, tmp_path):
        """An eval's snapshot save is submitted before the next optim_step, so it
        holds that step's weights, but the eval task waits for it: training keeps
        going in the meantime."""
        n_steps = 4
        evaluator = _RecordingEvaluator()
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * (4 * n_steps),
                groups_per_batch=4,
                sampling_steps_ahead=1,
                eval_every=1,
                evaluators=[evaluator],
                inflight_weight_updates=True,
                save_latency_s=0.2,
            )
        )
        h = result.harness
        assert h.optim_steps_during_snapshot_save > 0
        rows = _eval_rows(result)
        assert [r["eval/version"] for r in rows] == [float(r["step"]) for r in rows]
        assert evaluator.end_versions == evaluator.versions

    def test_rollouts_started_during_an_eval_snapshot_save_get_the_new_step(
        self, tmp_path, monkeypatch
    ):
        """The training loop publishes a step's client and step and frees its slots
        before saving that step's eval snapshot. Rollouts that start during the
        save, including on slots freed by filtered groups finishing then, record
        the step whose weights their inflight client already serves."""
        labels: dict[int, int] = {}
        export = train_module._maybe_export_rollout_summary_jsonl

        def record_labels(**kwargs: Any) -> None:
            for group in kwargs["groups_P"]:
                first_ob = group.trajectory_group.trajectories_G[0].transitions[0].ob
                labels[first_ob.to_ints()[0]] = group.sampling_client_step
            export(**kwargs)

        monkeypatch.setattr(train_module, "_maybe_export_rollout_summary_jsonl", record_labels)
        n_problems = 36
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01, 0.03, 0.06] * (n_problems // 3),
                groups_per_batch=2,
                sampling_steps_ahead=1,
                extra_groups_per_batch=1,
                constant_reward_problem_ids=set(range(2, n_problems, 3)),
                eval_every=1,
                evaluators=[_RecordingEvaluator()],
                inflight_weight_updates=True,
                save_latency_s=0.1,
            )
        )
        started = result.harness.first_sample_during_snapshot_save
        trained_started = {p: v for p, v in started.items() if p in labels}
        assert trained_started, "no trained rollout started during an eval snapshot save"
        assert {p: labels[p] for p in trained_started} == trained_started

    def test_final_off_cadence_step_is_evaluated_after_a_slow_save(self, tmp_path):
        """The final step is evaluated even when saving its snapshot takes a while."""
        evaluator = _RecordingEvaluator()
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 12,
                groups_per_batch=4,
                sampling_steps_ahead=1,
                eval_every=2,
                evaluators=[evaluator],
                inflight_weight_updates=True,
                save_latency_s=0.2,
            )
        )
        assert [int(r["step"]) for r in _eval_rows(result)] == [0, 2, 3]
        assert evaluator.versions == [0, 2, 3]

    def test_drain_steps_reuse_clients_and_early_final_is_evaluated(self, tmp_path):
        """Once the rollout workers have exited, off-cadence steps reuse the previous
        inflight client and start no eval; when data runs out early, the last trained
        step is still evaluated on a snapshot of exactly its weights."""
        evaluator = _RecordingEvaluator()
        result = asyncio.run(
            _run_async_training(
                tmp_path,
                [0.01] * 12,
                groups_per_batch=2,
                sampling_steps_ahead=2,
                # Two filtered groups leave 5 trainable steps of the 6 planned.
                constant_reward_problem_ids={10, 11},
                eval_every=2,
                evaluators=[evaluator],
                inflight_weight_updates=True,
                save_latency_s=_SAVE_LATENCY_S,
            )
        )
        rows = _eval_rows(result)
        assert [int(r["step"]) for r in rows] == [0, 2, 4, 5]
        # Drain steps skip saves, so weight versions lag steps; map them back.
        steps_of = result.harness.version_optim_steps
        assert [steps_of[v] for v in evaluator.versions] == [0, 2, 4, 5]
        assert evaluator.end_versions == evaluator.versions
        assert not any(evaluator.inflight)
        h = result.harness
        # Fewer inflight publishes than steps: the drain steps reused the client.
        assert h.client_sources.count("inflight") < 1 + 5

    @pytest.mark.parametrize("failing_step", [0, 8], ids=["during_training", "final"])
    def test_eval_failure_fails_the_run(self, tmp_path, failing_step):
        """An eval that fails while training runs stops the run right away; a
        failing final eval fails the run once training is done."""
        n_steps = 8
        with pytest.raises(RuntimeError, match=f"eval of version {failing_step} failed"):
            asyncio.run(
                _run_async_training(
                    tmp_path,
                    [0.01] * (4 * n_steps),
                    groups_per_batch=4,
                    sampling_steps_ahead=1,
                    train_time_s=0.05,
                    eval_every=1,
                    evaluators=[_FailingEvaluator(failing_version=failing_step)],
                    inflight_weight_updates=True,
                    save_latency_s=_SAVE_LATENCY_S,
                )
            )
        metrics_path = tmp_path / "metrics.jsonl"
        rows = (
            map(json.loads, metrics_path.read_text().splitlines()) if metrics_path.exists() else []
        )
        logged_steps = [r["step"] for r in rows if "training_client/step" in r]
        assert (len(logged_steps) < n_steps) == (failing_step < n_steps)
