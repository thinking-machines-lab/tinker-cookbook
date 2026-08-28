"""Tests for TerminationRewardPolicy application in do_group_rollout.

Covers the grade-then-clamp semantics (limit-stopped trajectories get
``min(reward, 0.0)``), the skip-grading-on-timeout triple condition, and the
grader timeout, at both application points: the group-level
``compute_group_rewards`` call in ``do_group_rollout`` and the env-level
``reward_fn`` call in ``AgentToolMessageEnv``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    StepResult,
    StopReason,
    Trajectory,
    Transition,
)
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

LIMIT_REASONS = [
    StopReason.MAX_TOKENS,
    StopReason.MAX_SAMPLED_TOKENS,
    StopReason.MAX_TURNS,
    StopReason.MAX_TOOL_CALLS,
    StopReason.ROLLOUT_TIMEOUT,
]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubEnv(Env):
    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str]]:
        return tinker.ModelInput.from_ints([1]), ["<stop>"]

    async def step(self, action: list[int], *, extra: Any = None) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=["<stop>"],
        )


class _UnusedPolicy(TokenCompleter):
    async def __call__(
        self, model_input: tinker.ModelInput, stop: Any, *, max_tokens: int | None = None
    ) -> TokensWithLogprobs:
        raise AssertionError("policy should not be sampled from in these tests")


def _traj(stop_reason: str | None, step_reward: float = 0.0) -> Trajectory:
    transition = Transition(
        ob=tinker.ModelInput.empty(),
        ac=TokensWithLogprobs(tokens=[1], maybe_logprobs=[0.0]),
        reward=step_reward,
        episode_done=True,
    )
    return Trajectory(
        transitions=[transition], final_ob=tinker.ModelInput.empty(), stop_reason=stop_reason
    )


class _PrebuiltStrategy(RolloutStrategy):
    """Returns fixed trajectories, bypassing envs and sampling."""

    def __init__(self, trajectories: list[Trajectory]):
        self._trajectories = trajectories

    async def execute(
        self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> RolloutResult:
        return RolloutResult(
            trajectories=self._trajectories,
            envs=[_StubEnv() for _ in self._trajectories],
            errors=[],
        )


class _CountingGraderBuilder(EnvGroupBuilder):
    """EnvGroupBuilder whose group grader counts calls and replays rewards."""

    def __init__(self, rewards: list[float], delay_seconds: float = 0.0):
        self.rewards = rewards
        self.delay_seconds = delay_seconds
        self.calls = 0

    async def make_envs(self) -> Sequence[Env]:
        return []

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        self.calls += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return [(r, {}) for r in self.rewards]


def _run_group(
    trajectories: list[Trajectory],
    rewards: list[float],
    termination: TerminationRewardPolicy | None,
    delay_seconds: float = 0.0,
) -> tuple[Any, _CountingGraderBuilder]:
    builder = _CountingGraderBuilder(rewards, delay_seconds=delay_seconds)
    group = asyncio.run(
        do_group_rollout(
            builder,
            _UnusedPolicy(),
            strategy=_PrebuiltStrategy(trajectories),
            termination=termination,
        )
    )
    return group, builder


# ---------------------------------------------------------------------------
# Zero-on-limit clamp
# ---------------------------------------------------------------------------


class TestZeroRewardOnLimit:
    @pytest.mark.parametrize("reason", LIMIT_REASONS)
    def test_positive_graded_reward_clamped_to_zero(self, reason: StopReason):
        """Grade normally, then min(reward, 0): a positive grade on a
        limit-stopped trajectory becomes 0.0; other members are untouched."""
        policy = TerminationRewardPolicy(zero_reward_on_limit=True)
        group, builder = _run_group(
            [_traj(reason), _traj(StopReason.COMPLETED)], [0.7, 0.5], policy
        )
        assert builder.calls == 1  # grading ran (grade-then-clamp)
        assert group.get_total_rewards() == [0.0, 0.5]
        assert group.metrics_G[0]["zero_reward_on_limit"] == 1.0
        assert "zero_reward_on_limit" not in group.metrics_G[1]

    @pytest.mark.parametrize("reason", LIMIT_REASONS)
    def test_negative_graded_reward_stays(self, reason: StopReason):
        policy = TerminationRewardPolicy(zero_reward_on_limit=True)
        group, _ = _run_group([_traj(reason)], [-0.3], policy)
        assert group.get_total_rewards() == [-0.3]
        assert "zero_reward_on_limit" not in group.metrics_G[0]

    def test_step_rewards_count_toward_the_clamped_total(self):
        """The clamp applies to the trajectory *total* (step rewards + group
        reward), so a positive step-reward sum is zeroed out too."""
        policy = TerminationRewardPolicy(zero_reward_on_limit=True)
        group, _ = _run_group([_traj(StopReason.MAX_TURNS, step_reward=0.4)], [0.3], policy)
        assert group.get_total_rewards() == [0.0]

    @pytest.mark.parametrize(
        "reason",
        [StopReason.COMPLETED, StopReason.CONTEXT_OVERFLOW, StopReason.PARSE_ERROR, None],
    )
    def test_non_limit_stops_not_clamped(self, reason: str | None):
        policy = TerminationRewardPolicy(zero_reward_on_limit=True)
        group, _ = _run_group([_traj(reason)], [0.7], policy)
        assert group.get_total_rewards() == [0.7]

    def test_no_policy_leaves_rewards_untouched(self):
        group, _ = _run_group([_traj(StopReason.MAX_TURNS)], [0.7], None)
        assert group.get_total_rewards() == [0.7]

    def test_disabled_policy_leaves_rewards_untouched(self):
        policy = TerminationRewardPolicy(zero_reward_on_limit=False)
        group, _ = _run_group([_traj(StopReason.MAX_TURNS)], [0.7], policy)
        assert group.get_total_rewards() == [0.7]

    def test_custom_limit_stop_reasons(self):
        policy = TerminationRewardPolicy(
            zero_reward_on_limit=True, limit_stop_reasons=(StopReason.MAX_TURNS,)
        )
        group, _ = _run_group(
            [_traj(StopReason.MAX_TURNS), _traj(StopReason.MAX_TOKENS)], [0.7, 0.7], policy
        )
        assert group.get_total_rewards() == [0.0, 0.7]


# ---------------------------------------------------------------------------
# Skip-grading-on-timeout triple condition
# ---------------------------------------------------------------------------


class TestSkipGradingOnTimeout:
    def test_all_timed_out_skips_grader_and_uses_zero(self):
        """Triple condition met (rollout_timeout stop + zero_reward_on_limit +
        rollout_timeout in limit_stop_reasons) for every member: the group
        grader is never called and rewards are 0.0."""
        policy = TerminationRewardPolicy(zero_reward_on_limit=True, skip_grading_on_timeout=True)
        trajectories = [_traj(StopReason.ROLLOUT_TIMEOUT), _traj(StopReason.ROLLOUT_TIMEOUT)]
        group, builder = _run_group(trajectories, [0.7, 0.7], policy)
        assert builder.calls == 0
        assert group.get_total_rewards() == [0.0, 0.0]

    def test_mixed_group_still_grades_everyone(self):
        """compute_group_rewards is group-simultaneous: if any member needs
        real grading, the whole group is graded (the timed-out member is
        handled by the clamp instead)."""
        policy = TerminationRewardPolicy(zero_reward_on_limit=True, skip_grading_on_timeout=True)
        trajectories = [_traj(StopReason.ROLLOUT_TIMEOUT), _traj(StopReason.COMPLETED)]
        group, builder = _run_group(trajectories, [0.7, 0.5], policy)
        assert builder.calls == 1
        assert group.get_total_rewards() == [0.0, 0.5]  # clamp still zeroes the timeout

    def test_without_zero_reward_on_limit_grading_runs(self):
        policy = TerminationRewardPolicy(zero_reward_on_limit=False, skip_grading_on_timeout=True)
        group, builder = _run_group([_traj(StopReason.ROLLOUT_TIMEOUT)], [0.7], policy)
        assert builder.calls == 1
        assert group.get_total_rewards() == [0.7]

    def test_without_skip_flag_grading_runs(self):
        policy = TerminationRewardPolicy(zero_reward_on_limit=True, skip_grading_on_timeout=False)
        _, builder = _run_group([_traj(StopReason.ROLLOUT_TIMEOUT)], [0.7], policy)
        assert builder.calls == 1

    def test_without_rollout_timeout_in_limit_reasons_grading_runs(self):
        """Third leg of the triple condition: if rollout_timeout is not a
        limit reason, the grade would stand, so it must not be skipped."""
        policy = TerminationRewardPolicy(
            zero_reward_on_limit=True,
            skip_grading_on_timeout=True,
            limit_stop_reasons=(StopReason.MAX_TURNS,),
        )
        group, builder = _run_group([_traj(StopReason.ROLLOUT_TIMEOUT)], [0.7], policy)
        assert builder.calls == 1
        assert group.get_total_rewards() == [0.7]


# ---------------------------------------------------------------------------
# Grader timeout
# ---------------------------------------------------------------------------


class TestGraderTimeout:
    def test_group_grader_timeout_raises(self):
        policy = TerminationRewardPolicy(grader_timeout_seconds=0.05)
        with pytest.raises(TimeoutError):
            _run_group([_traj(StopReason.COMPLETED)], [0.7], policy, delay_seconds=5.0)

    def test_group_grader_within_timeout_ok(self):
        policy = TerminationRewardPolicy(grader_timeout_seconds=5.0)
        group, _ = _run_group([_traj(StopReason.COMPLETED)], [0.7], policy, delay_seconds=0.0)
        assert group.get_total_rewards() == [0.7]

    def test_env_reward_fn_timeout_raises(self):
        """The env-level application point: AgentToolMessageEnv bounds its
        reward_fn call when a termination policy is installed on the env."""

        async def slow_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
            await asyncio.sleep(5.0)
            return 1.0, {}

        env = AgentToolMessageEnv(
            tools=[],
            initial_messages=[{"role": "user", "content": "hi"}],
            max_turns=5,
            reward_fn=slow_reward,
            termination_policy=TerminationRewardPolicy(grader_timeout_seconds=0.05),
        )

        async def run() -> None:
            await env.initial_observation()
            await env.step({"role": "assistant", "content": "done"})

        with pytest.raises(TimeoutError):
            asyncio.run(run())


# ---------------------------------------------------------------------------
# Grader message scope (pass_all_messages_to_grader)
# ---------------------------------------------------------------------------


class _RecordingReward:
    def __init__(self) -> None:
        self.received: list[Message] | None = None

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.received = list(history)
        return 1.0, {}


def _grade_once(termination_policy: TerminationRewardPolicy | None) -> list[Message]:
    reward_fn = _RecordingReward()
    env = AgentToolMessageEnv(
        tools=[],
        initial_messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        max_turns=5,
        reward_fn=reward_fn,
        termination_policy=termination_policy,
    )

    async def run() -> None:
        await env.initial_observation()
        await env.step({"role": "assistant", "content": "done"})

    asyncio.run(run())
    assert reward_fn.received is not None
    return reward_fn.received


class TestGraderMessageScope:
    def test_no_policy_passes_full_history(self):
        received = _grade_once(None)
        assert [m["role"] for m in received] == ["system", "user", "assistant"]

    def test_policy_default_passes_completion_suffix_only(self):
        received = _grade_once(TerminationRewardPolicy())
        assert [m["role"] for m in received] == ["assistant"]

    def test_pass_all_messages_restores_full_history(self):
        received = _grade_once(TerminationRewardPolicy(pass_all_messages_to_grader=True))
        assert [m["role"] for m in received] == ["system", "user", "assistant"]


class TestTrainerWiring:
    """train.Config.termination must reach every group-rollout call in train.py."""

    def test_config_accepts_termination_policy(self):
        from tinker_cookbook.rl import train

        assert "termination" in dir(train.Config)

    def test_all_train_call_sites_thread_termination(self):
        """Every do_group_rollout_and_filter_constant_reward call in train.py
        passes termination=config.termination, so a configured policy (e.g.
        agentic().termination) is live in all execution modes (sync, async
        off-policy, stream-minibatch)."""
        import ast
        import inspect

        from tinker_cookbook.rl import train

        tree = ast.parse(inspect.getsource(train))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id.endswith("do_group_rollout_and_filter_constant_reward")
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr.endswith("do_group_rollout_and_filter_constant_reward")
                )
            )
        ]
        assert calls, "expected at least one group-rollout call site in train.py"
        for call in calls:
            kwarg_names = {kw.arg for kw in call.keywords}
            assert "termination" in kwarg_names, (
                f"train.py line {call.lineno}: do_group_rollout_and_filter_constant_reward "
                "call is missing termination=config.termination"
            )
