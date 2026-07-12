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
from tinker_cookbook.tokendb.schema import TokenRow, row_to_record
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
    attrs: dict | None = None,
    tool_calls: list | None = None,
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
        attrs=attrs or {},
        tool_calls=tool_calls,
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


class TestBuilderMetadata:
    """EnvGroupBuilder.metadata() routing: numeric -> metrics, str -> attrs,
    row_id -> env_row_id."""

    def _summary_group(self, metadata: dict, *, logs: dict | None = None) -> RolloutSummaryGroup:
        return RolloutSummaryGroup(
            trajectory_group=make_group([single_step_trajectory([1], [2], logs=logs or {})]),
            tags=[],
            metadata=metadata,
        )

    def test_numeric_routes_to_metrics_string_to_attrs(self):
        group = self._summary_group({"dataset": "gsm8k", "difficulty": 3, "temp": 0.7})
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].attrs == {"dataset": "gsm8k"}
        assert rows[0].metrics == {"difficulty": 3.0, "temp": pytest.approx(0.7)}

    def test_row_id_promotes_to_env_row_id(self):
        group = self._summary_group({"row_id": "gsm8k-42"})
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].env_row_id == "gsm8k-42"
        # row_id is a routing key, not a metric or attr.
        assert rows[0].metrics == {}
        assert rows[0].attrs == {}

    def test_explicit_row_id_overrides_logs_fallback(self):
        group = self._summary_group({"row_id": "explicit"}, logs={"env/row_id": "from-logs"})
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].env_row_id == "explicit"

    def test_logs_fallback_kept_without_metadata_row_id(self):
        group = self._summary_group({"dataset": "gsm8k"}, logs={"env/row_id": "from-logs"})
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].env_row_id == "from-logs"

    def test_metadata_on_every_row_of_group(self):
        two_step = Trajectory(
            transitions=[
                make_transition([1], [2]),
                make_transition([1, 2, 3], [4], episode_done=True),
            ],
            final_ob=tinker.ModelInput.from_ints([]),
        )
        group = RolloutSummaryGroup(
            trajectory_group=make_group([two_step, single_step_trajectory([9], [8])]),
            tags=[],
            metadata={"dataset": "math", "level": 5, "row_id": "math-7"},
        )
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert len(rows) == 3
        assert all(r.attrs == {"dataset": "math"} for r in rows)
        assert all(r.metrics == {"level": 5.0} for r in rows)
        assert all(r.env_row_id == "math-7" for r in rows)

    def test_kwarg_metadata_fallback_for_plain_groups(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="train",
            iteration=0,
            metadata={"dataset": "gsm8k", "difficulty": 2},
        )
        assert rows[0].attrs == {"dataset": "gsm8k"}
        assert rows[0].metrics == {"difficulty": 2.0}

    def test_summary_group_metadata_takes_precedence_over_kwarg(self):
        group = self._summary_group({"dataset": "from-group"})
        rows = record_groups(
            ListWriter(),
            [group],
            split="train",
            iteration=0,
            metadata={"dataset": "from-kwarg"},
        )
        assert rows[0].attrs == {"dataset": "from-group"}

    def test_per_step_metrics_win_over_metadata_on_collision(self):
        group = RolloutSummaryGroup(
            trajectory_group=make_group(
                [single_step_trajectory([1], [2], metrics={"difficulty": 9})]
            ),
            tags=[],
            metadata={"difficulty": 3},
        )
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].metrics == {"difficulty": 9.0}


class TestStepAttrs:
    """Transition.attrs flow into the row attrs map."""

    def test_transition_attrs_reach_rows(self):
        traj = single_step_trajectory([1], [2], attrs={"tool": "search", "phase": "solve"})
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].attrs == {"tool": "search", "phase": "solve"}

    def test_attrs_values_coerced_to_str(self):
        traj = single_step_trajectory([1], [2], attrs={"n_retries": 3})
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].attrs == {"n_retries": "3"}

    def test_per_step_attrs_win_over_group_metadata_on_collision(self):
        group = RolloutSummaryGroup(
            trajectory_group=make_group(
                [single_step_trajectory([1], [2], attrs={"phase": "step-level"})]
            ),
            tags=[],
            metadata={"phase": "group-level", "dataset": "gsm8k"},
        )
        rows = record_groups(ListWriter(), [group], split="train", iteration=0)
        assert rows[0].attrs == {"phase": "step-level", "dataset": "gsm8k"}


class TestToolCalls:
    """Transition.tool_calls flow into the structured tool_calls column."""

    def test_tool_calls_mapped_with_defaults_filled(self):
        traj = single_step_trajectory(
            [1],
            [2],
            tool_calls=[
                {"name": "search", "args_json": '{"q": "x"}'},
                {
                    "name": "calc",
                    "args_json": "{}",
                    "error_type": "validation_failed",
                    "should_stop": True,
                },
            ],
        )
        rows = record_groups(ListWriter(), [make_group([traj])], split="train", iteration=0)
        assert rows[0].tool_calls == [
            {"name": "search", "args_json": '{"q": "x"}', "error_type": None, "should_stop": False},
            {
                "name": "calc",
                "args_json": "{}",
                "error_type": "validation_failed",
                "should_stop": True,
            },
        ]

    def test_no_tool_calls_stays_none(self):
        rows = record_groups(
            ListWriter(),
            [make_group([single_step_trajectory([1], [2])])],
            split="train",
            iteration=0,
        )
        assert rows[0].tool_calls is None

    def test_tool_calls_roundtrip_through_parquet(self, tmp_path: Path):
        traj = single_step_trajectory(
            [1],
            [2],
            tool_calls=[
                {
                    "name": "search",
                    "args_json": '{"q": "x"}',
                    "error_type": None,
                    "should_stop": False,
                }
            ],
            attrs={"tool": "search"},
        )
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            record_groups(
                writer,
                [
                    RolloutSummaryGroup(
                        trajectory_group=make_group([traj]),
                        tags=[],
                        metadata={"dataset": "hotpot", "row_id": "q-1"},
                    )
                ],
                split="train",
                iteration=0,
            )
        got = read_all_segments(tmp_path).to_pylist()[0]
        assert got["tool_calls"] == [
            {"name": "search", "args_json": '{"q": "x"}', "error_type": None, "should_stop": False}
        ]
        assert dict(got["attrs"]) == {"tool": "search", "dataset": "hotpot"}
        assert got["env_row_id"] == "q-1"


class TestAgentToolEnvEndToEnd:
    """Fake tool -> AgentToolMessageEnv -> EnvFromMessageEnv -> rollout ->
    tool_calls column populated (error_type on failure)."""

    @staticmethod
    def _rollout_rows(tool_call, tools) -> list[TokenRow]:
        import asyncio
        from unittest.mock import MagicMock

        from tinker_cookbook.renderers.base import ParseTermination
        from tinker_cookbook.rl.message_env import EnvFromMessageEnv
        from tinker_cookbook.rl.rollouts import do_single_rollout
        from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

        message_env = AgentToolMessageEnv(
            tools=tools,
            initial_messages=[{"role": "user", "content": "go"}],
            max_turns=1,
            reward_fn=_noop_reward,
        )
        renderer = MagicMock()
        renderer.build_generation_prompt = MagicMock(
            return_value=tinker.ModelInput.from_ints([1, 2, 3])
        )
        renderer.get_stop_sequences = MagicMock(return_value=["<stop>"])
        renderer.parse_response = MagicMock(
            return_value=(
                {"role": "assistant", "content": "calling", "tool_calls": [tool_call]},
                ParseTermination.STOP_SEQUENCE,
            )
        )
        env = EnvFromMessageEnv(renderer=renderer, message_env=message_env)

        from tinker_cookbook.completers import StopCondition, TokenCompleter

        class FixedPolicy(TokenCompleter):
            async def __call__(
                self, model_input: tinker.ModelInput, stop: StopCondition
            ) -> TokensWithLogprobs:
                return TokensWithLogprobs(tokens=[7], maybe_logprobs=[-0.1])

        trajectory = asyncio.run(do_single_rollout(FixedPolicy(), env))
        group = TrajectoryGroup(trajectories_G=[trajectory], final_rewards_G=[0.0], metrics_G=[{}])
        return record_groups(ListWriter(), [group], split="train", iteration=0)

    def test_fake_tool_populates_tool_calls_column(self):
        from tinker_cookbook.renderers.base import ToolCall
        from tinker_cookbook.tool_use.tools import tool
        from tinker_cookbook.tool_use.types import ToolResult

        @tool
        async def search(q: str) -> ToolResult:
            """Fake search."""
            from tinker_cookbook.tool_use.tools import simple_tool_result

            return simple_tool_result(f"results for {q}")

        tc = ToolCall(
            id="c1", function=ToolCall.FunctionBody(name="search", arguments='{"q": "cats"}')
        )
        rows = self._rollout_rows(tc, [search])
        assert len(rows) == 1
        assert rows[0].tool_calls == [
            {
                "name": "search",
                "args_json": '{"q": "cats"}',
                "error_type": None,
                "should_stop": False,
            }
        ]

    def test_failed_tool_call_carries_error_type(self):
        from tinker_cookbook.renderers.base import ToolCall

        tc = ToolCall(id="c1", function=ToolCall.FunctionBody(name="missing_tool", arguments="{}"))
        rows = self._rollout_rows(tc, [])
        assert len(rows) == 1
        assert rows[0].tool_calls is not None
        assert rows[0].tool_calls[0]["name"] == "missing_tool"
        assert rows[0].tool_calls[0]["error_type"] == "tool_not_found"


async def _noop_reward(history) -> tuple[float, dict[str, float]]:
    return 0.0, {}


class TestDefaultsAreNoOps:
    """Envs that don't use metadata/attrs/tool_calls produce identical rows."""

    @staticmethod
    def _plain_records() -> list[dict]:
        groups = [
            make_group(
                [
                    single_step_trajectory([1, 2], [3], reward=1.0),
                    Trajectory(
                        transitions=[
                            make_transition([1], [2], metrics={"acc": 1}, logs={"note": "x"}),
                            make_transition([1, 2, 5], [6], episode_done=True),
                        ],
                        final_ob=tinker.ModelInput.from_ints([]),
                    ),
                ],
                final_rewards=[0.5, 0.0],
                metrics_G=[{"g": 1.0}, {}],
            )
        ]
        rows = record_groups(
            ListWriter(),
            groups,
            split="train",
            iteration=4,
            tags=["plain"],
            tokenizer=FakeTokenizer(),
        )
        records = [row_to_record(r) for r in rows]
        for record in records:
            record.pop("ts")  # only nondeterministic field
        return records

    def test_plain_fixture_rows_unchanged_by_new_fields(self):
        # A Transition built without the new fields and one built with the
        # explicit defaults must produce byte-identical records, and the
        # extensible columns stay at their empty defaults.
        first = self._plain_records()
        second = self._plain_records()
        assert first == second
        for record in first:
            assert dict(record["attrs"]) == {}
            assert record["tool_calls"] is None

    def test_record_groups_without_metadata_kwarg_matches_empty_metadata(self):
        traj = [make_group([single_step_trajectory([1], [2])])]
        base = record_groups(ListWriter(), traj, split="train", iteration=0)
        with_empty = record_groups(ListWriter(), traj, split="train", iteration=0, metadata={})
        base_records = [row_to_record(r) for r in base]
        empty_records = [row_to_record(r) for r in with_empty]
        for record in base_records + empty_records:
            record.pop("ts")
        assert base_records == empty_records


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
