import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tinker_cookbook.rl import train


class _Logger:
    store = None

    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, Any]]] = []

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        self.logged.append((step, dict(metrics)))


class _CheckpointManager:
    async def maybe_save_rolling_async(self, step: int, loop_state: dict[str, Any]) -> None:
        return None


class _Builder:
    def logging_tags(self) -> list[str]:
        return []


class _Dataset:
    def get_batch(self, i_batch: int) -> list[_Builder]:
        return [_Builder()]


def _config(eval_every: int, tmp_path) -> train.Config:
    return cast(
        train.Config,
        SimpleNamespace(
            learning_rate=1e-4,
            eval_every=eval_every,
            log_path=str(tmp_path),
            num_groups_to_log=0,
            max_tokens=8,
            temperature=1.0,
            effective_termination=lambda: None,
            remove_constant_reward_groups=False,
            span_chart_every=0,
        ),
    )


@pytest.fixture
def evaluated(monkeypatch) -> list[tuple[str, int]]:
    """Patch the Tinker-facing pieces; record (weights, step) for each evaluation."""
    calls: list[tuple[str, int]] = []

    async def save_checkpoint(training_client, checkpoint_mgr, step, batch, **kwargs):
        return f"weights@{step}", {}

    async def rollout(sampling_client, builder, **kwargs):
        return object()

    async def train_step(config, i_batch, *args):
        return f"weights@{i_batch + 1}", {}

    async def train_step_streaming(config, i_batch, *args):
        return f"weights@{i_batch + 1}", {}, []

    async def run_evals(evaluators, sampling_client, config, step, store=None):
        calls.append((sampling_client, step))
        return {"eval/score": 1.0}

    monkeypatch.setattr(train, "save_checkpoint_and_get_sampling_client", save_checkpoint)
    monkeypatch.setattr(train, "do_group_rollout_and_filter_constant_reward", rollout)
    monkeypatch.setattr(train, "do_train_step_and_get_sampling_client", train_step)
    monkeypatch.setattr(
        train, "do_train_step_streaming_and_get_sampling_client", train_step_streaming
    )
    monkeypatch.setattr(train, "run_evaluations_parallel", run_evals)
    monkeypatch.setattr(train, "_maybe_export_rollout_summary_jsonl", lambda **kwargs: None)
    monkeypatch.setattr(train, "_get_logtree_scope", lambda *a, **k: contextlib.nullcontext())
    return calls


LOOPS = [train.do_sync_training, train.do_sync_training_with_stream_minibatch]


def _run(loop, config: train.Config, logger: _Logger, evaluators: list[Any], end_batch: int):
    asyncio.run(
        loop(
            start_batch=0,
            end_batch=end_batch,
            num_batches=end_batch,
            config=config,
            training_client=cast(Any, None),
            kl_reference_client=None,
            evaluators=evaluators,
            dataset=cast(Any, _Dataset()),
            ml_logger=cast(Any, logger),
            tokenizer=cast(Any, None),
            checkpoint_mgr=cast(Any, _CheckpointManager()),
        )
    )


@pytest.mark.parametrize("loop", LOOPS)
@pytest.mark.parametrize("end_batch", [4, 5])
def test_final_weights_are_evaluated_once(loop, end_batch, evaluated, tmp_path):
    logger = _Logger()
    _run(loop, _config(2, tmp_path), logger, [object()], end_batch)

    in_loop = [(f"weights@{s}", s) for s in range(0, end_batch, 2)]
    assert evaluated == in_loop + [(f"weights@{end_batch}", end_batch)]
    final_logs = [m for step, m in logger.logged if step == end_batch]
    assert final_logs and "eval/score" in final_logs[0]


@pytest.mark.parametrize("loop", LOOPS)
@pytest.mark.parametrize("eval_every, evaluators", [(0, [object()]), (2, [])])
def test_final_eval_skipped_when_disabled(loop, eval_every, evaluators, evaluated, tmp_path):
    logger = _Logger()
    _run(loop, _config(eval_every, tmp_path), logger, evaluators, 4)

    assert all(step < 4 for _, step in evaluated)
    assert all(step < 4 for step, _ in logger.logged)
