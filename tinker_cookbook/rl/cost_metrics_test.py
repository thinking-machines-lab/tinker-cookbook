"""The RL loops record each step's token usage and log ``tokens/`` and ``cost/`` metrics."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl import train
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.sync_final_eval_test import (  # noqa: F401  (evaluated is a fixture)
    LOOPS,
    _CheckpointManager,
    _config,
    _Dataset,
    _Logger,
    evaluated,
)
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.utils.cost_estimation import CostTracker, TokenCostRates, TokenUsage

RATES = TokenCostRates(
    prefill_usd_per_million=1.0,
    cached_prefill_usd_per_million=0.2,
    sample_usd_per_million=2.0,
    train_usd_per_million=3.0,
)


def transition(ob_len: int, ac_len: int, cache_hits: int) -> Transition:
    return Transition(
        ob=tinker.ModelInput.from_ints(list(range(ob_len))),
        ac=TokensWithLogprobs(
            tokens=[0] * ac_len,
            maybe_logprobs=[0.0] * ac_len,
            prompt_cache_hit_tokens=cache_hits,
        ),
        reward=0.0,
        episode_done=False,
    )


def group(*transitions: Transition, reward: float = 1.0) -> TrajectoryGroup:
    traj = Trajectory(transitions=list(transitions), final_ob=tinker.ModelInput.from_ints([]))
    return TrajectoryGroup(trajectories_G=[traj], final_rewards_G=[reward], metrics_G=[{}])


# One fake step: two turns of 100 sampled tokens, 500 prompt tokens prefilled per
# turn of which 400 hit the prefix cache on the second, and 700 tokens in the
# datum sent to forward_backward (the shared prefix is only trained once).
GROUP = group(transition(500, 100, 0), transition(500, 100, 400))
ROLLOUT_USAGE = TokenUsage(sampled=200, prefill_cached=400, prefill_uncached=600)
DATA_D = [
    tinker.Datum(model_input=tinker.ModelInput.from_ints(list(range(700))), loss_fn_inputs={}),
]
STEP_USAGE = ROLLOUT_USAGE + TokenUsage(trained=700)
STEP_USD = (600 * 1.0 + 400 * 0.2 + 200 * 2.0 + 700 * 3.0) / 1e6
# What the KL scoring calls report for the datum: one 701-token sequence scored
# by the reference model (300 cache hits) and by the post-update model (100).
REFERENCE_KL_USAGE = TokenUsage(sampled=1, prefill_cached=300, prefill_uncached=401)
POST_KL_USAGE = TokenUsage(sampled=1, prefill_cached=100, prefill_uncached=601)


def test_rollout_usage_counts_sampled_and_prefilled_tokens_from_the_trajectories():
    assert train.rollout_usage([GROUP]) == ROLLOUT_USAGE
    assert train.rollout_usage([GROUP, GROUP]) == ROLLOUT_USAGE + ROLLOUT_USAGE
    assert train.rollout_usage([]) == TokenUsage()


def test_training_usage_comes_from_the_datums():
    data_D = [
        tinker.Datum(model_input=tinker.ModelInput.from_ints([1, 2, 3]), loss_fn_inputs={}),
        tinker.Datum(model_input=tinker.ModelInput.from_ints([4, 5]), loss_fn_inputs={}),
    ]
    assert train.training_usage(data_D) == TokenUsage(trained=5)


def test_trajectory_metrics_count_prefix_cache_hits():
    metrics = compute_trajectory_metrics([group(transition(10, 4, 0), transition(14, 2, 14))], [[]])
    assert metrics["env/all/total_ob_tokens"] == 24
    assert metrics["env/all/total_cached_ob_tokens"] == 14
    assert metrics["env/all/total_ac_tokens"] == 6


def test_prepare_minibatch_returns_the_rollout_usage_plus_the_kl_pass(monkeypatch):
    monkeypatch.setattr(train, "print_group", lambda *a, **k: None)
    monkeypatch.setattr(train, "assemble_training_data", lambda *a, **k: (DATA_D, []))

    async def kl_penalty(data_D, *args):
        return {"kl_policy_base": 0.0}, REFERENCE_KL_USAGE

    monkeypatch.setattr(train, "incorporate_kl_penalty", kl_penalty)
    _, _, usage = asyncio.run(
        train.prepare_minibatch(
            [], [GROUP], cast(Any, None), None, kl_penalty_coef=0.0, kl_discount_factor=0.0
        )
    )
    assert usage == ROLLOUT_USAGE
    _, metrics, usage = asyncio.run(
        train.prepare_minibatch(
            [],
            [GROUP],
            cast(Any, None),
            cast(Any, object()),
            kl_penalty_coef=0.1,
            kl_discount_factor=0.0,
        )
    )
    assert metrics["kl_policy_base"] == 0.0
    assert usage == ROLLOUT_USAGE + REFERENCE_KL_USAGE


def test_full_batch_metrics_add_the_trained_tokens(monkeypatch):
    async def save_checkpoint(training_client, checkpoint_mgr, step, **kwargs):
        return object(), {}

    monkeypatch.setattr(train, "save_checkpoint_and_get_sampling_client", save_checkpoint)
    monkeypatch.setattr(train, "compute_kl_sample_train", lambda data_D, logprobs_D: {})
    tracker = CostTracker(RATES)
    _, metrics = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            cast(Any, None),
            cast(Any, None),
            1,
            DATA_D,
            [],
            do_compute_post_kl=False,
            usage=ROLLOUT_USAGE,
            cost_tracker=tracker,
        )
    )
    assert tracker.cumulative == STEP_USAGE
    assert metrics["tokens/trained"] == 700
    assert metrics["cost/step_usd"] == pytest.approx(STEP_USD)

    _, metrics = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            cast(Any, None), cast(Any, None), 1, DATA_D, [], do_compute_post_kl=False
        )
    )
    assert not any(key.startswith(("tokens/", "cost/")) for key in metrics)


class _RecordingCheckpointManager:
    """Periodic save on every step, remembering the loop_state it was given."""

    def __init__(self) -> None:
        self.loop_states: list[dict[str, Any]] = []

    def should_save_periodic(self, step: int) -> bool:
        return True

    async def save_periodic_async(self, step: int, loop_state: dict[str, Any]) -> dict[str, str]:
        self.loop_states.append(loop_state)
        return {"sampler_path": f"sampler@{step}"}

    async def maybe_save_rolling_async(self, step: int, loop_state: dict[str, Any]) -> None:
        self.loop_states.append(loop_state)


def test_full_batch_metrics_record_cost_before_the_checkpoint(monkeypatch):
    class _TrainingClient:
        def create_sampling_client(self, path: str, **kwargs: Any) -> str:
            return path

    monkeypatch.setattr(train, "compute_kl_sample_train", lambda data_D, logprobs_D: {})
    checkpoint_mgr = _RecordingCheckpointManager()
    tracker = CostTracker(RATES)
    for i_batch in (1, 2):
        _, metrics = asyncio.run(
            train.compute_full_batch_metrics_and_get_sampling_client(
                cast(Any, _TrainingClient()),
                cast(Any, checkpoint_mgr),
                i_batch,
                DATA_D,
                [],
                do_compute_post_kl=False,
                usage=ROLLOUT_USAGE,
                cost_tracker=tracker,
            )
        )
        assert metrics["tokens/trained"] == 700
        assert metrics["tokens/cumulative/sampled"] == 200 * i_batch
        assert metrics["cost/cumulative_usd"] == pytest.approx(STEP_USD * i_batch)
    # The checkpoint written at step i carries the totals through step i, so a
    # resumed run does not lose the step that produced the checkpoint.
    assert [ls["cumulative_tokens"]["trained"] for ls in checkpoint_mgr.loop_states] == [700, 1400]
    assert checkpoint_mgr.loop_states[-1] == {
        "batch": 2,
        "cumulative_tokens": {
            "sampled": 400,
            "prefill_cached": 800,
            "prefill_uncached": 1200,
            "trained": 1400,
        },
        "discarded_tokens": asdict(TokenUsage()),
    }


def test_post_kl_usage_is_recorded_after_the_checkpoint_it_follows(monkeypatch):
    """Order: record measured work, save checkpoint, make sampler, run post-KL, record it."""
    events: list[str] = []

    class _TrainingClient:
        def create_sampling_client(self, path: str, **kwargs: Any) -> str:
            events.append("create_sampling_client")
            return path

    class _EventCheckpointManager(_RecordingCheckpointManager):
        async def save_periodic_async(self, step: int, loop_state: dict[str, Any]):
            events.append("save_periodic")
            return await super().save_periodic_async(step, loop_state)

    async def post_kl(data_D, sampling_client):
        events.append(f"post_kl@{sampling_client}")
        return {"kl_pre_post_v1": 0.0}, POST_KL_USAGE

    monkeypatch.setattr(train, "compute_kl_sample_train", lambda data_D, logprobs_D: {})
    monkeypatch.setattr(train, "compute_post_kl", post_kl)
    checkpoint_mgr = _EventCheckpointManager()
    tracker = CostTracker(RATES)

    _, metrics = asyncio.run(
        train.compute_full_batch_metrics_and_get_sampling_client(
            cast(Any, _TrainingClient()),
            cast(Any, checkpoint_mgr),
            1,
            DATA_D,
            [],
            do_compute_post_kl=True,
            usage=ROLLOUT_USAGE + REFERENCE_KL_USAGE,
            cost_tracker=tracker,
        )
    )

    assert events == ["save_periodic", "create_sampling_client", "post_kl@sampler@1"]
    # One sampler and one checkpoint; the checkpoint holds exactly the work measured
    # before it (rollouts, reference KL, training), nothing from the post-KL calls.
    before_post_kl = STEP_USAGE + REFERENCE_KL_USAGE
    assert checkpoint_mgr.loop_states == [
        {
            "batch": 1,
            "cumulative_tokens": asdict(before_post_kl),
            "discarded_tokens": asdict(TokenUsage()),
        }
    ]
    # The step's metrics include the post-KL calls exactly once.
    step = before_post_kl + POST_KL_USAGE
    assert metrics["kl_pre_post_v1"] == 0.0
    assert metrics["tokens/sampled"] == step.sampled == 202
    assert metrics["tokens/prefill_cached"] == step.prefill_cached == 800
    assert metrics["tokens/prefill_uncached"] == step.prefill_uncached == 1602
    assert metrics["tokens/trained"] == 700
    assert metrics["cost/step_usd"] == pytest.approx(RATES.estimate_usd(step))
    assert metrics["cost/cumulative_usd"] == pytest.approx(metrics["cost/step_usd"])
    assert tracker.step == step
    assert tracker.cumulative == step

    # A later save (the loop's rolling checkpoint, or the next periodic one) includes it,
    # and a tracker resumed from either checkpoint continues from exactly those totals.
    assert train._loop_state(2, tracker)["cumulative_tokens"] == asdict(step)
    resumed = CostTracker(
        RATES, cumulative=TokenUsage(**checkpoint_mgr.loop_states[0]["cumulative_tokens"])
    )
    out = resumed.record_step(STEP_USAGE)
    assert out["tokens/cumulative/trained"] == 1400
    assert out["tokens/cumulative/prefill_uncached"] == 1001 + 600


def test_streaming_step_sums_the_usage_of_its_minibatches(monkeypatch):
    """Two minibatches each report usage; the step hands the sum to the cost accounting."""

    class _Future:
        def __init__(self, result: Any) -> None:
            self._result = result

        async def result_async(self) -> Any:
            return self._result

    class _TrainingClient:
        async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
            return _Future(SimpleNamespace(loss_fn_outputs=[]))

        async def optim_step_async(self, adam_params):
            return _Future(SimpleNamespace(metrics={}))

    async def prepare_minibatch(env_group_builders, trajectory_groups, tokenizer, kl_client, **kw):
        return [DATA_D[0]], {"kl_policy_base": 0.5}, ROLLOUT_USAGE + REFERENCE_KL_USAGE

    seen_usage: list[TokenUsage] = []

    async def full_batch_metrics(training_client, checkpoint_mgr, step, *args, **kwargs):
        seen_usage.append(kwargs["usage"])
        return "sampler", {}

    monkeypatch.setattr(train, "prepare_minibatch", prepare_minibatch)
    monkeypatch.setattr(train, "_training_logprobs_from_fwd_bwd", lambda result: [])
    monkeypatch.setattr(train, "_check_provenance", lambda data_D, loss_fn: None)
    monkeypatch.setattr(train, "compute_trajectory_metrics", lambda *a, **k: {})
    monkeypatch.setattr(train, "compute_sampling_client_metrics", lambda *a, **k: {})
    monkeypatch.setattr(
        train, "compute_full_batch_metrics_and_get_sampling_client", full_batch_metrics
    )

    config = SimpleNamespace(
        stream_minibatch_config=train.StreamMinibatchConfig(groups_per_batch=4, num_minibatches=2),
        num_substeps=1,
        learning_rate=1e-4,
        kl_penalty_coef=0.1,
        kl_discount_factor=0.0,
        compute_post_kl=False,
        loss_fn_config=None,
        training_loss_fn=lambda: "importance_sampling",
        records_stability_info=lambda: False,
    )
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for i in range(4):
        queue.put_nowait(
            SimpleNamespace(
                trajectory_group=f"group{i}",
                env_group_builder=SimpleNamespace(logging_tags=lambda: []),
                sampling_client_step=0,
                metrics={},
            )
        )

    result = asyncio.run(
        train.do_train_step_streaming_and_get_sampling_client(
            cast(Any, config),
            0,
            queue,
            cast(Any, _TrainingClient()),
            cast(Any, None),
            cast(Any, object()),
            cast(Any, None),
        )
    )

    assert result is not None
    _, metrics, _ = result
    minibatch = ROLLOUT_USAGE + REFERENCE_KL_USAGE
    assert seen_usage == [minibatch + minibatch]
    assert metrics["kl_policy_base"] == 0.5


def test_resumed_tracker_continues_checkpointed_totals():
    first = CostTracker(RATES)
    first.record_step(STEP_USAGE)
    saved = train._loop_state(1, first)["cumulative_tokens"]

    resumed = CostTracker(RATES, cumulative=TokenUsage(**saved))
    metrics = resumed.record_step(STEP_USAGE)
    assert metrics["tokens/cumulative/sampled"] == 400
    assert metrics["tokens/cumulative/prefill_cached"] == 800
    assert metrics["tokens/cumulative/prefill_uncached"] == 1200
    assert metrics["tokens/cumulative/trained"] == 1400
    assert metrics["cost/cumulative_usd"] == pytest.approx(2 * STEP_USD)
    # Checkpoints from before the totals were saved resume from zero.
    assert TokenUsage(**{}) == TokenUsage()
    assert train._loop_state(0, None) == {"batch": 0}


@pytest.mark.parametrize("loop", LOOPS)
def test_loops_checkpoint_cost_totals(loop, evaluated, monkeypatch, tmp_path):  # noqa: F811
    """The loops pass the tracker into the train step and persist its totals in rolling saves."""
    seen: list[CostTracker | None] = []

    async def train_step(config, i_batch, *args, cost_tracker=None):
        seen.append(cost_tracker)
        return f"weights@{i_batch + 1}", {}

    async def train_step_streaming(config, i_batch, *args, cost_tracker=None):
        seen.append(cost_tracker)
        return f"weights@{i_batch + 1}", {}, []

    monkeypatch.setattr(train, "do_train_step_and_get_sampling_client", train_step)
    monkeypatch.setattr(
        train, "do_train_step_streaming_and_get_sampling_client", train_step_streaming
    )
    tracker = CostTracker(RATES, cumulative=TokenUsage(trained=5))
    checkpoint_mgr = _RecordingCheckpointManager()
    asyncio.run(
        loop(
            start_batch=0,
            end_batch=2,
            num_batches=2,
            config=_config(0, tmp_path),
            training_client=cast(Any, None),
            kl_reference_client=None,
            evaluators=[],
            dataset=cast(Any, _Dataset()),
            ml_logger=cast(Any, _Logger()),
            tokenizer=cast(Any, None),
            checkpoint_mgr=cast(Any, checkpoint_mgr),
            cost_tracker=tracker,
        )
    )
    assert seen == [tracker, tracker]
    totals = {"sampled": 0, "prefill_cached": 0, "prefill_uncached": 0, "trained": 5}
    none = asdict(TokenUsage())
    assert checkpoint_mgr.loop_states == [
        {"batch": 1, "cumulative_tokens": totals, "discarded_tokens": none},
        {"batch": 2, "cumulative_tokens": totals, "discarded_tokens": none},
    ]
