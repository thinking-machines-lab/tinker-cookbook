"""Integration tests for the token DB hooks into the RL training loop.

Covers:

- Hook 1: ``rl/train._maybe_export_rollout_summary_jsonl`` teeing summary
  groups into the active token DB writer while keeping the rollout-summary
  JSONL byte-identical, and the disabled (``token_db=None``) path leaving no
  ``tokens/`` directory.
- Hook 2: the filtered-group sink firing from
  ``rl/rollouts._do_group_rollout_and_filter_constant_reward_impl`` for the
  three drop reasons, end-to-end into parquet for constant-reward groups.

A full train-loop smoke (``rl/train.main`` with ``token_db`` enabled) is not
included: ``main()`` requires a real Tinker backend (``ServiceClient`` +
training client) and there is no mock-client harness in the unit test suite.
"""

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

pytest.importorskip("pyarrow")

import chz
import tinker

from tinker_cookbook.exceptions import AllTrajectoriesFailedError, ConfigurationError
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.rollouts import _do_group_rollout_and_filter_constant_reward_impl
from tinker_cookbook.rl.train import Config, _maybe_export_rollout_summary_jsonl
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDatasetBuilder, StepResult
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore
from tinker_cookbook.tokendb.capture import (
    ActiveCapture,
    CaptureContext,
    active_capture_filtered_sink,
    set_active_capture,
    set_capture_context,
    set_filtered_group_sink,
)
from tinker_cookbook.tokendb.capture_test import FakeTokenizer, make_group, single_step_trajectory
from tinker_cookbook.tokendb.config import TokenDbConfig
from tinker_cookbook.tokendb.writer import TOKENS_DIR, TokenDbWriter
from tinker_cookbook.tokendb.writer_test import read_all_segments

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Ensure capture registries never leak between tests."""
    yield
    set_active_capture(None)
    set_filtered_group_sink(None)


@chz.chz
class _FakeDatasetBuilder(RLDatasetBuilder):
    async def __call__(self):
        raise NotImplementedError


def make_config(log_path: Path, *, token_db: TokenDbConfig | None) -> Config:
    return Config(
        learning_rate=1e-4,
        dataset_builder=_FakeDatasetBuilder(),
        model_name="test-model",
        recipe_name="test-recipe",
        max_tokens=8,
        log_path=str(log_path),
        token_db=token_db,
    )


def make_summary_groups() -> list[RolloutSummaryGroup]:
    group = make_group(
        [
            single_step_trajectory([1, 2, 3], [4, 5], reward=1.0),
            single_step_trajectory([1, 2, 3], [6, 7], reward=0.0),
        ],
        final_rewards=[1.0, 0.0],
    )
    return [RolloutSummaryGroup(trajectory_group=group, tags=["fake_env"], sampling_client_step=3)]


def summary_jsonl_path(log_path: Path, iteration: int) -> Path:
    return log_path / f"iteration_{iteration:06d}" / "train_rollout_summaries.jsonl"


class _FakeSamplingClient:
    """Minimal stand-in for tinker.SamplingClient (only sample_async is used)."""

    async def sample_async(self, prompt, num_samples, sampling_params):
        class _Seq:
            tokens = [4, 5]
            logprobs = [-0.1, -0.2]
            stop_reason = "stop"

        class _Result:
            sequences = [_Seq()]

        return _Result()


class _ConstantRewardEnv(Env):
    async def initial_observation(self):
        return tinker.ModelInput.from_ints([1, 2, 3]), [0]

    async def step(self, action, *, extra=None):
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[0],
        )


class _ConstantRewardEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, n_envs: int = 2):
        self.n_envs = n_envs

    async def make_envs(self):
        return [_ConstantRewardEnv() for _ in range(self.n_envs)]

    def logging_tags(self) -> list[str]:
        return ["constant_env"]


def sampling_client() -> tinker.SamplingClient:
    return cast(tinker.SamplingClient, _FakeSamplingClient())


# ---------------------------------------------------------------------------
# Hook 1: export funnel tee
# ---------------------------------------------------------------------------


class TestExportFunnelTee:
    def test_tee_writes_parquet_and_keeps_jsonl(self, tmp_path: Path):
        config = make_config(tmp_path, token_db=TokenDbConfig())
        store = TrainingRunStore(LocalStorage(tmp_path))
        writer = TokenDbWriter(tmp_path, flush_interval_s=3600)
        set_active_capture(ActiveCapture(writer=writer, tokenizer=FakeTokenizer()))

        _maybe_export_rollout_summary_jsonl(
            config=config,
            base_name="train",
            split="train",
            iteration=7,
            groups_P=make_summary_groups(),
            store=store,
        )
        writer.close()

        # Rows landed in parquet with the funnel-provided identity.
        table = read_all_segments(tmp_path)
        assert table.num_rows == 2
        assert table.column("split").to_pylist() == ["train", "train"]
        assert table.column("iteration").to_pylist() == [7, 7]
        assert table.column("source").to_pylist() == ["rollout", "rollout"]
        assert table.column("tags").to_pylist() == [["fake_env"], ["fake_env"]]
        assert table.column("sampling_client_step").to_pylist() == [3, 3]
        assert table.column("ac_text").to_pylist() == ["t4 t5", "t6 t7"]

        # ... and the summary JSONL is still written as before.
        jsonl = summary_jsonl_path(tmp_path, 7)
        assert jsonl.exists()
        records = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert len(records) == 2

    def test_disabled_path_leaves_no_tokens_dir(self, tmp_path: Path):
        config = make_config(tmp_path, token_db=None)
        store = TrainingRunStore(LocalStorage(tmp_path))

        _maybe_export_rollout_summary_jsonl(
            config=config,
            base_name="train",
            split="train",
            iteration=0,
            groups_P=make_summary_groups(),
            store=store,
        )

        assert not (tmp_path / TOKENS_DIR).exists()
        assert summary_jsonl_path(tmp_path, 0).exists()

    def test_capture_failure_does_not_break_jsonl_export(self, tmp_path: Path):
        class _BrokenWriter:
            def append_rows(self, rows):
                raise RuntimeError("boom")

            def flush(self):
                pass

            def close(self):
                pass

        config = make_config(tmp_path, token_db=TokenDbConfig())
        store = TrainingRunStore(LocalStorage(tmp_path))
        set_active_capture(ActiveCapture(writer=_BrokenWriter()))

        _maybe_export_rollout_summary_jsonl(
            config=config,
            base_name="train",
            split="train",
            iteration=1,
            groups_P=make_summary_groups(),
            store=store,
        )

        assert summary_jsonl_path(tmp_path, 1).exists()

    def test_jsonl_bytes_identical_with_and_without_token_db(self, tmp_path: Path):
        groups = make_summary_groups()
        outputs: dict[str, bytes] = {}
        for name, token_db in [("off", None), ("on", TokenDbConfig())]:
            log_path = tmp_path / name
            log_path.mkdir()
            config = make_config(log_path, token_db=token_db)
            store = TrainingRunStore(LocalStorage(log_path))
            if token_db is not None:
                writer = TokenDbWriter(log_path, flush_interval_s=3600)
                set_active_capture(ActiveCapture(writer=writer, tokenizer=FakeTokenizer()))
            _maybe_export_rollout_summary_jsonl(
                config=config,
                base_name="train",
                split="train",
                iteration=2,
                groups_P=groups,
                store=store,
            )
            if token_db is not None:
                writer.close()
                set_active_capture(None)
            outputs[name] = summary_jsonl_path(log_path, 2).read_bytes()
        assert outputs["off"] == outputs["on"]


# ---------------------------------------------------------------------------
# Hook 2: filtered-group sink
# ---------------------------------------------------------------------------


class TestFilteredGroupSink:
    def test_constant_reward_group_lands_in_parquet(self, tmp_path: Path):
        writer = TokenDbWriter(tmp_path, flush_interval_s=3600)
        set_active_capture(ActiveCapture(writer=writer, tokenizer=FakeTokenizer()))
        set_filtered_group_sink(active_capture_filtered_sink)

        async def run():
            with set_capture_context(
                CaptureContext(split="train", iteration=5, sampling_client_step=5)
            ):
                return await _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client(),
                    _ConstantRewardEnvGroupBuilder(),
                    max_tokens=8,
                    temperature=1.0,
                    do_remove_constant_reward_groups=True,
                )

        result = asyncio.run(run())
        writer.close()

        assert result is None
        table = read_all_segments(tmp_path)
        assert table.num_rows == 2
        assert set(table.column("source").to_pylist()) == {"filtered"}
        assert set(table.column("filtered_reason").to_pylist()) == {"constant_reward"}
        assert table.column("iteration").to_pylist() == [5, 5]
        assert table.column("sampling_client_step").to_pylist() == [5, 5]
        assert table.column("tags").to_pylist() == [["constant_env"], ["constant_env"]]

    def test_all_failed_reason(self):
        calls: list[tuple[object, list[str], str]] = []
        set_filtered_group_sink(lambda group, tags, reason: calls.append((group, tags, reason)))

        async def run():
            with patch(
                "tinker_cookbook.rl.rollouts.do_group_rollout",
                side_effect=AllTrajectoriesFailedError("all failed"),
            ):
                return await _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client(),
                    _ConstantRewardEnvGroupBuilder(),
                    max_tokens=8,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                )

        assert asyncio.run(run()) is None
        assert calls == [(None, ["constant_env"], "all_failed")]

    def test_group_error_reason(self):
        calls: list[tuple[object, list[str], str]] = []
        set_filtered_group_sink(lambda group, tags, reason: calls.append((group, tags, reason)))

        async def run():
            with patch(
                "tinker_cookbook.rl.rollouts.do_group_rollout",
                side_effect=RuntimeError("sandbox flake"),
            ):
                return await _do_group_rollout_and_filter_constant_reward_impl(
                    sampling_client(),
                    _ConstantRewardEnvGroupBuilder(),
                    max_tokens=8,
                    temperature=1.0,
                    do_remove_constant_reward_groups=False,
                    strategy=RetryOnFailure(max_retries=0),
                )

        assert asyncio.run(run()) is None
        assert calls == [(None, ["constant_env"], "group_error")]

    def test_sink_exception_never_breaks_rollout(self):
        def broken_sink(group, tags, reason):
            raise RuntimeError("sink boom")

        set_filtered_group_sink(broken_sink)

        async def run():
            return await _do_group_rollout_and_filter_constant_reward_impl(
                sampling_client(),
                _ConstantRewardEnvGroupBuilder(),
                max_tokens=8,
                temperature=1.0,
                do_remove_constant_reward_groups=True,
            )

        assert asyncio.run(run()) is None  # dropped, but no exception escapes

    def test_no_sink_registered_is_a_noop(self):
        async def run():
            return await _do_group_rollout_and_filter_constant_reward_impl(
                sampling_client(),
                _ConstantRewardEnvGroupBuilder(),
                max_tokens=8,
                temperature=1.0,
                do_remove_constant_reward_groups=True,
            )

        assert asyncio.run(run()) is None


# ---------------------------------------------------------------------------
# Config-time dependency check
# ---------------------------------------------------------------------------


class TestDependencyCheck:
    def test_check_passes_with_pyarrow_installed(self):
        from tinker_cookbook.tokendb.config import check_token_db_dependencies

        check_token_db_dependencies()

    def test_check_raises_configuration_error_without_pyarrow(self):
        import builtins

        from tinker_cookbook.tokendb.config import check_token_db_dependencies

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ConfigurationError, match=r"tinker-cookbook\[tokendb\]"):
                check_token_db_dependencies()
