"""Tests for the token DB capture layer (TrajectoryGroup -> TokenRow)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.tokendb.capture import (
    CaptureContext,
    extract_ob_tokens,
    get_capture_context,
    get_filtered_group_sink,
    record_groups,
    set_capture_context,
    set_filtered_group_sink,
)
from tinker_cookbook.tokendb.schema import TokenRow
from tinker_cookbook.tokendb.writer import TokenDbWriter
from tinker_cookbook.tokendb.writer_test import read_all_segments


class FakeTokenizer:
    """Stub tokenizer: decodes token IDs to a deterministic string."""

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"t{t}" for t in token_ids)


class ListWriter:
    """In-memory TokenWriter capturing appended rows."""

    def __init__(self) -> None:
        self.rows: list[TokenRow] = []

    def append_rows(self, rows) -> None:
        self.rows.extend(rows)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_transition(
    ob_tokens: list[int],
    ac_tokens: list[int],
    *,
    logprobs: list[float] | None = None,
    reward: float = 0.0,
    episode_done: bool = False,
    metrics: dict | None = None,
    logs: dict | None = None,
    ob: tinker.ModelInput | None = None,
) -> Transition:
    if logprobs is None:
        logprobs = [-0.1] * len(ac_tokens)
    return Transition(
        ob=ob if ob is not None else tinker.ModelInput.from_ints(ob_tokens),
        ac=TokensWithLogprobs(tokens=ac_tokens, maybe_logprobs=logprobs),
        reward=reward,
        episode_done=episode_done,
        metrics=metrics or {},
        logs=logs or {},
    )


def make_group(
    trajectories: list[Trajectory],
    final_rewards: list[float] | None = None,
    metrics_G: list[dict] | None = None,
) -> TrajectoryGroup:
    if final_rewards is None:
        final_rewards = [0.0] * len(trajectories)
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
        metrics_G=metrics_G if metrics_G is not None else [{} for _ in trajectories],
    )


def single_step_trajectory(
    ob_tokens: list[int], ac_tokens: list[int], *, reward: float = 0.0, **kwargs
) -> Trajectory:
    return Trajectory(
        transitions=[
            make_transition(ob_tokens, ac_tokens, reward=reward, episode_done=True, **kwargs)
        ],
        final_ob=tinker.ModelInput.from_ints([]),
    )


class TestEnumeration:
    def test_row_count_and_keys_match_group_structure(self):
        groups = [
            make_group(
                [
                    single_step_trajectory([1, 2], [3]),
                    Trajectory(
                        transitions=[
                            make_transition([1, 2], [3, 4]),
                            make_transition([1, 2, 3, 4, 5], [6], episode_done=True),
                        ],
                        final_ob=tinker.ModelInput.from_ints([]),
                    ),
                ]
            ),
            make_group([single_step_trajectory([9], [8])]),
        ]
        writer = ListWriter()
        rows = record_groups(writer, groups, split="train", iteration=3)
        assert rows is not None and writer.rows == rows
        keys = [(r.group_idx, r.traj_idx, r.step_idx) for r in rows]
        assert keys == [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0)]
        assert all(r.split == "train" and r.iteration == 3 for r in rows)
        assert all(r.source == "rollout" and r.filtered_reason is None for r in rows)

    def test_summary_group_tags_and_step_take_precedence(self):
        summary_groups = [
            RolloutSummaryGroup(
                trajectory_group=make_group([single_step_trajectory([1], [2])]),
                tags=["gsm", "math"],
                sampling_client_step=7,
            )
        ]
        rows = record_groups(
            ListWriter(),
            summary_groups,
            split="train",
            iteration=0,
            tags=["fallback"],
            sampling_client_step=99,
        )
        assert rows[0].tags == ["gsm", "math"]
        assert rows[0].sampling_client_step == 7

    def test_plain_group_uses_kwarg_tags_and_step(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="eval/gsm8k",
            iteration=1,
            tags=["gsm"],
            sampling_client_step=12,
        )
        assert rows[0].tags == ["gsm"]
        assert rows[0].sampling_client_step == 12
        assert rows[0].split == "eval/gsm8k"


class TestDeltaOb:
    def test_prefix_extension_stores_delta(self):
        # Turn 2 observation extends turn 1's ob+ac by [30, 31].
        traj = Trajectory(
            transitions=[
                make_transition([1, 2], [10, 11]),
                make_transition([1, 2, 10, 11, 30, 31], [40], episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].ob_tokens == [1, 2]
        assert rows[0].ob_is_delta is False
        assert rows[1].ob_tokens == [30, 31]
        assert rows[1].ob_is_delta is True

    def test_non_prefix_reset_stores_full_ob(self):
        traj = Trajectory(
            transitions=[
                make_transition([1, 2], [10]),
                make_transition([99, 98], [40], episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[1].ob_tokens == [99, 98]
        assert rows[1].ob_is_delta is False

    def test_delta_resets_across_trajectories(self):
        # Two trajectories with identical obs: the second must not delta
        # against the first.
        groups = [
            make_group([single_step_trajectory([1, 2], [3]), single_step_trajectory([1, 2], [3])])
        ]
        rows = record_groups(ListWriter(), groups, split="train", iteration=0)
        assert rows[1].ob_tokens == [1, 2]
        assert rows[1].ob_is_delta is False


class TestRewards:
    def test_reward_and_denormalized_totals(self):
        traj_a = Trajectory(
            transitions=[
                make_transition([1], [2], reward=0.25),
                make_transition([1, 2, 3], [4], reward=0.5, episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        traj_b = single_step_trajectory([1], [2], reward=1.0)
        group = make_group([traj_a, traj_b], final_rewards=[2.0, -1.0])
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)

        assert [r.reward for r in rows] == [0.25, 0.5, 1.0]
        assert [r.episode_done for r in rows] == [False, True, True]
        # total = sum of step rewards + final group reward, same on every row
        # of the trajectory.
        assert rows[0].total_reward == rows[1].total_reward == pytest.approx(2.75)
        assert rows[0].final_reward == rows[1].final_reward == pytest.approx(2.0)
        assert rows[2].total_reward == pytest.approx(0.0)
        assert rows[2].final_reward == pytest.approx(-1.0)

    def test_ac_fields_come_from_transition(self):
        traj = single_step_trajectory([1], [5, 6], logprobs=[-0.5, -1.5])
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].ac_tokens == [5, 6]
        assert rows[0].ac_logprobs == [-0.5, -1.5]
        assert rows[0].stop_reason == "stop"

    def test_missing_logprobs_stored_as_none(self):
        traj = Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.from_ints([1]),
                    ac=TokensWithLogprobs(tokens=[2], maybe_logprobs=None),
                    reward=0.0,
                    episode_done=True,
                )
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].ac_logprobs is None


class TestLogsAndMetrics:
    def test_env_row_id_promotion(self):
        traj = single_step_trajectory([1], [2], logs={"env/row_id": "gsm8k-42", "other": 1})
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].env_row_id == "gsm8k-42"
        assert rows[0].logs == {"env/row_id": "gsm8k-42", "other": 1}

    def test_no_env_row_id(self):
        traj = single_step_trajectory([1], [2], logs={"other": 1})
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].env_row_id is None

    def test_metrics_map_and_logs_json_roundtrip(self, tmp_path: Path):
        class Weird:
            def __repr__(self) -> str:
                return "<weird>"

        traj = single_step_trajectory(
            [1],
            [2],
            metrics={"format_ok": 1},
            logs={"env/row_id": "r-1", "obj": Weird()},
        )
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            record_groups(writer, [make_group([traj])], split="train", iteration=0)
        got = read_all_segments(tmp_path).to_pylist()[0]
        assert dict(got["metrics"]) == {"format_ok": 1.0}
        logs = json.loads(got["logs"])
        assert logs["env/row_id"] == "r-1"
        assert logs["obj"] == "<weird>"  # default=str fallback


class TestGroupMetrics:
    def test_metrics_g_merged_under_group_prefix_on_every_row(self):
        # Two-step trajectory + one-step trajectory, each with its own
        # group-level metrics dict.
        traj_a = Trajectory(
            transitions=[
                make_transition([1], [2], metrics={"turn_acc": 1.0}),
                make_transition([1, 2, 3], [4], episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        traj_b = single_step_trajectory([9], [8])
        group = make_group(
            [traj_a, traj_b],
            metrics_G=[{"rubric/score": 0.75}, {"rubric/score": 0.25}],
        )
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        # Denormalized onto EVERY row of the trajectory, like final_reward.
        assert rows[0].metrics == {"turn_acc": 1.0, "group/rubric/score": 0.75}
        assert rows[1].metrics == {"group/rubric/score": 0.75}
        assert rows[2].metrics == {"group/rubric/score": 0.25}

    def test_transition_metrics_win_nothing_without_group_metrics(self):
        traj = single_step_trajectory([1], [2], metrics={"acc": 1})
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].metrics == {"acc": 1.0}
        assert rows[0].attrs == {}
        assert rows[0].tool_calls is None

    def test_non_coercible_group_metric_dropped(self):
        traj = single_step_trajectory([1], [2])
        group = make_group([traj], metrics_G=[{"note": "not-a-number", "score": 1}])
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].metrics == {"group/score": 1.0}


class TestTextDecode:
    def test_store_text_decodes_whole_turn_and_delta(self):
        traj = Trajectory(
            transitions=[
                make_transition([1, 2], [10]),
                make_transition([1, 2, 10, 30], [40], episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        rows = record_groups(
            ListWriter(),
            [make_group([traj])],
            split="train",
            iteration=0,
            tokenizer=FakeTokenizer(),
        )
        assert rows[0].ob_text == "t1 t2"
        assert rows[0].ac_text == "t10"
        # Delta row decodes only the delta portion.
        assert rows[1].ob_tokens == [30]
        assert rows[1].ob_text == "t30"
        assert rows[1].ac_text == "t40"

    def test_store_text_false_skips_decode(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="train",
            iteration=0,
            tokenizer=FakeTokenizer(),
            store_text=False,
        )
        assert rows[0].ob_text is None
        assert rows[0].ac_text is None

    def test_no_tokenizer_skips_decode(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="train",
            iteration=0,
        )
        assert rows[0].ob_text is None
        assert rows[0].ac_text is None


class TestImageChunks:
    def test_extract_ob_tokens_with_image_chunk(self):
        ob = tinker.ModelInput(
            chunks=[
                tinker.EncodedTextChunk(tokens=[1, 2]),
                tinker.types.ImageChunk(data=b"fake", format="jpeg", expected_tokens=5),
                tinker.EncodedTextChunk(tokens=[3]),
            ]
        )
        tokens, has_images = extract_ob_tokens(ob)
        assert tokens == [1, 2, 3]
        assert has_images is True

    def test_extract_ob_tokens_text_only(self):
        tokens, has_images = extract_ob_tokens(tinker.ModelInput.from_ints([1, 2]))
        assert tokens == [1, 2]
        assert has_images is False

    def test_record_groups_with_image_ob_never_raises(self):
        ob = tinker.ModelInput(
            chunks=[
                tinker.EncodedTextChunk(tokens=[1, 2]),
                tinker.types.ImageChunk(data=b"fake", format="jpeg", expected_tokens=5),
            ]
        )
        traj = Trajectory(
            transitions=[make_transition([], [7], ob=ob, episode_done=True)],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        rows = record_groups(
            ListWriter(),
            [make_group([traj])],
            split="train",
            iteration=0,
            tokenizer=FakeTokenizer(),
        )
        assert rows[0].has_images is True
        assert rows[0].ob_tokens == [1, 2]  # text-chunk tokens only
        assert rows[0].ob_text == "t1 t2"


class TestSourceAndFiltered:
    def test_filtered_source_and_reason(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="train",
            iteration=0,
            source="filtered",
            filtered_reason="constant_reward",
        )
        assert rows[0].source == "filtered"
        assert rows[0].filtered_reason == "constant_reward"


class TestWriterIntegration:
    def test_rows_reach_parquet_via_token_db_writer(self, tmp_path: Path):
        groups = [
            make_group([single_step_trajectory([1, 2], [3], reward=1.0)], final_rewards=[0.5]),
            make_group([single_step_trajectory([4], [5, 6])]),
        ]
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            record_groups(
                writer,
                groups,
                split="train",
                iteration=2,
                tags=["unit"],
                tokenizer=FakeTokenizer(),
            )
        table = read_all_segments(tmp_path)
        assert table.num_rows == 2
        got = sorted(table.to_pylist(), key=lambda r: r["group_idx"])
        assert got[0]["ob_tokens"] == [1, 2]
        assert got[0]["ac_tokens"] == [3]
        assert got[0]["total_reward"] == pytest.approx(1.5)
        assert got[0]["final_reward"] == pytest.approx(0.5)
        assert got[0]["ob_text"] == "t1 t2"
        assert got[0]["tags"] == ["unit"]
        assert got[0]["iteration"] == 2
        assert got[1]["ac_tokens"] == [5, 6]
        assert got[0]["run_id"] == got[1]["run_id"] != ""


class TestCaptureContext:
    def test_set_and_reset(self):
        assert get_capture_context() is None
        ctx = CaptureContext(split="train", iteration=5, sampling_client_step=3, tags=("gsm",))
        with set_capture_context(ctx) as active:
            assert active is ctx
            assert get_capture_context() is ctx
        assert get_capture_context() is None

    def test_nesting(self):
        outer = CaptureContext(split="train", iteration=1)
        inner = CaptureContext(split="test", iteration=2)
        with set_capture_context(outer):
            with set_capture_context(inner):
                assert get_capture_context() is inner
            assert get_capture_context() is outer


class TestFilteredGroupSink:
    def test_registry_set_get_clear(self):
        assert get_filtered_group_sink() is None
        calls = []

        def sink(group, tags, reason):
            calls.append((group, tags, reason))

        set_filtered_group_sink(sink)
        try:
            assert get_filtered_group_sink() is sink
            get_filtered_group_sink()(None, ["gsm"], "rollout_error")  # type: ignore[misc]
            assert calls == [(None, ["gsm"], "rollout_error")]
        finally:
            set_filtered_group_sink(None)
        assert get_filtered_group_sink() is None
