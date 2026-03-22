"""Tests for trajectory_hook on Env.

The trajectory_hook is an optional callback on Env that do_single_rollout()
invokes after the Trajectory is fully assembled. It runs for every episode
termination — including parse_error and context_overflow cases that bypass
MessageEnv.step() — making it the right place for trajectory-level logging,
persistence, or analysis.

Example use case: an RL training system that logs every trajectory to a
database for post-hoc analysis. Without the hook, trajectories that end
via parse errors or context overflow would be silently dropped from the
log, because the inner MessageEnv never sees those termination events.
"""

import asyncio
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.trajectory_hook import TrajectoryHook
from tinker_cookbook.rl.types import Action, Env, StepResult, Trajectory

# ---------------------------------------------------------------------------
# A simple counting env with a trajectory_hook
# ---------------------------------------------------------------------------


@dataclass
class CountingEnv(Env):
    """Env that counts down from max_steps, then terminates.

    Demonstrates trajectory_hook usage: the hook receives the complete
    Trajectory after the rollout loop finishes, regardless of how the
    episode ended.
    """

    max_steps: int = 3
    trajectory_hook: TrajectoryHook | None = None

    _step: int = field(default=0, init=False)

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return tinker.ModelInput.from_ints([0]), ["<stop>"]

    async def step(self, action: Action) -> StepResult:
        self._step += 1
        done = self._step >= self.max_steps
        return StepResult(
            reward=1.0 if done else 0.0,
            episode_done=done,
            next_observation=tinker.ModelInput.from_ints([self._step]),
            next_stop_condition=["<stop>"],
            metrics={"step": float(self._step)},
            logs={"action_tokens": len(action)},
        )


class StubPolicy(TokenCompleter):
    """Policy that always returns the same tokens."""

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        return TokensWithLogprobs(tokens=[42, 43], maybe_logprobs=[0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrajectoryHook:
    def test_hook_receives_complete_trajectory(self):
        """The trajectory_hook is called with the full Trajectory after the episode."""
        captured: list[Trajectory] = []

        async def my_hook(trajectory: Trajectory) -> None:
            captured.append(trajectory)

        env = CountingEnv(max_steps=3, trajectory_hook=my_hook)
        policy = StubPolicy()
        trajectory = asyncio.run(do_single_rollout(policy, env))

        # Hook was called exactly once
        assert len(captured) == 1
        # Hook received the same trajectory that was returned
        assert captured[0] is trajectory
        # Trajectory has all transitions
        assert len(trajectory.transitions) == 3
        assert trajectory.transitions[-1].episode_done is True
        assert trajectory.transitions[-1].reward == 1.0

    def test_no_hook_is_fine(self):
        """Envs without a trajectory_hook work normally."""
        env = CountingEnv(max_steps=2)
        policy = StubPolicy()
        trajectory = asyncio.run(do_single_rollout(policy, env))

        assert len(trajectory.transitions) == 2
        assert trajectory.transitions[-1].episode_done is True

    def test_hook_sees_single_step_episode(self):
        """Hook fires even for 1-step episodes (e.g. immediate termination)."""
        captured: list[Trajectory] = []

        async def my_hook(trajectory: Trajectory) -> None:
            captured.append(trajectory)

        env = CountingEnv(max_steps=1, trajectory_hook=my_hook)
        policy = StubPolicy()
        asyncio.run(do_single_rollout(policy, env))

        assert len(captured) == 1
        assert len(captured[0].transitions) == 1

    def test_hook_can_be_any_callable(self):
        """The hook just needs to be an async callable — no base class required."""
        call_count = 0

        class MyObserver:
            async def __call__(self, trajectory: Trajectory) -> None:
                nonlocal call_count
                call_count += 1
                # Could persist to DB, log to W&B, etc.
                assert len(trajectory.transitions) > 0

        env = CountingEnv(max_steps=2, trajectory_hook=MyObserver())
        policy = StubPolicy()
        asyncio.run(do_single_rollout(policy, env))

        assert call_count == 1

    def test_hook_gets_metrics_and_logs(self):
        """The hook can inspect per-transition metrics and logs."""
        captured: list[Trajectory] = []

        async def my_hook(trajectory: Trajectory) -> None:
            captured.append(trajectory)

        env = CountingEnv(max_steps=2, trajectory_hook=my_hook)
        policy = StubPolicy()
        asyncio.run(do_single_rollout(policy, env))

        traj = captured[0]
        # Each transition should have the metrics and logs from CountingEnv
        for i, t in enumerate(traj.transitions):
            assert t.metrics["step"] == float(i + 1)
            assert t.logs["action_tokens"] == 2  # StubPolicy returns 2 tokens
