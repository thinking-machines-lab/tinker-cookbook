"""Retry must re-create the env for the problem that failed.

Both error-tolerant strategies (RetryOnFailure, MinViableGroup) relaunch a
failed rollout by calling ``make_envs()`` again and taking ``new_envs[0]``.
When a group is heterogeneous — one builder spanning several problems, as
the verifiers integration and mixed datasets produce — that retrains a
duplicate of the FIRST problem and the failed problem never retrains at all.
These tests pin the launch-index fix: the retry env is the one at the failed
task's index in the (deterministic) group.
"""

from __future__ import annotations

import collections

import pytest
import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl.rollout_strategy import MinViableGroup, RetryOnFailure
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

PROBLEMS = list(range(8))  # heterogeneous group: one env per problem
FAILING_PROBLEM = 5


class _ProblemEnv(Env):
    """Env for one problem id; counts how many rollouts were launched on it."""

    def __init__(self, problem_id: int, attempts: collections.Counter):
        self.problem_id = problem_id
        self.attempts = attempts

    async def initial_observation(self):
        self.attempts[self.problem_id] += 1
        return tinker.ModelInput.from_ints([self.problem_id]), [0]

    async def step(self, action, *, extra=None):
        return StepResult(
            reward=1.0,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[0],
        )


class _HeterogeneousBuilder(EnvGroupBuilder):
    """Deterministic make_envs: the same problems, in the same order, every call."""

    def __init__(self, problem_ids: list[int]):
        self.problem_ids = problem_ids
        self.attempts: collections.Counter = collections.Counter()
        self.make_envs_calls = 0

    async def make_envs(self):
        self.make_envs_calls += 1
        return [_ProblemEnv(pid, self.attempts) for pid in self.problem_ids]

    async def compute_group_rewards(self, trajectory_group, env_group):
        return [(0.0, {}) for _ in trajectory_group]


class _FailsOnProblem(TokenCompleter):
    """Policy that crashes on the observation of one specific problem."""

    def __init__(self, failing_problem: int):
        self.failing_problem = failing_problem

    async def __call__(self, model_input, stop, *, max_tokens=None):
        problem_id = model_input.to_ints()[0]
        if problem_id == self.failing_problem:
            raise RuntimeError(f"boom on problem {problem_id}")
        return TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2])


@pytest.mark.asyncio
async def test_retry_on_failure_retries_the_failed_problem():
    builder = _HeterogeneousBuilder(PROBLEMS)
    strategy = RetryOnFailure(max_retries=2)

    with pytest.raises(RuntimeError, match="boom on problem 5"):
        await strategy.execute(builder, _FailsOnProblem(FAILING_PROBLEM))

    # Initial launch: every problem attempted once. Two retries: both must
    # land on problem 5 again — not on a duplicate of problem 0.
    expected = collections.Counter(dict.fromkeys(PROBLEMS, 1))
    expected[FAILING_PROBLEM] = 3  # 1 initial + 2 retries
    assert builder.attempts == expected


@pytest.mark.asyncio
async def test_min_viable_group_retries_the_failed_problem():
    builder = _HeterogeneousBuilder(PROBLEMS)
    strategy = MinViableGroup(min_successful=len(PROBLEMS), max_retries=2)

    with pytest.raises(AllTrajectoriesFailedError):
        await strategy.execute(builder, _FailsOnProblem(FAILING_PROBLEM))

    expected = collections.Counter(dict.fromkeys(PROBLEMS, 1))
    expected[FAILING_PROBLEM] = 3
    assert builder.attempts == expected


@pytest.mark.asyncio
async def test_successful_group_is_untouched_by_the_index_mapping():
    """No failures: one attempt per problem, full group returned."""
    builder = _HeterogeneousBuilder(PROBLEMS)

    class _AlwaysSucceeds(_FailsOnProblem):
        async def __call__(self, model_input, stop, *, max_tokens=None):
            return TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[0.1, 0.2])

    result = await RetryOnFailure(max_retries=2).execute(builder, _AlwaysSucceeds(-1))
    assert len(result.trajectories) == len(PROBLEMS)
    assert builder.attempts == collections.Counter(dict.fromkeys(PROBLEMS, 1))
