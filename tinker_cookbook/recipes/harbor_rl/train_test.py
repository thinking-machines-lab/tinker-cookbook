from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import tinker

from tinker_cookbook.capture import capture
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.harbor_rl.train import RelaySession
from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.types import TrajectoryGroup


@pytest.mark.asyncio
async def test_out_of_order_rewards_reach_their_sampled_tokens() -> None:
    sampler = Mock(sample=AsyncMock())
    session = RelaySession(TinkerTokenCompleter(sampler, max_tokens=8))
    success = Mock(id=uuid4(), rewards={"reward": 1}, error_type=None)
    failure = Mock(id=uuid4(), rewards={"reward": 0}, error_type="NonZeroAgentExitCodeError")

    for rollout, token in [(success, 101), (failure, 202)]:
        sampler.sample.return_value = tinker.SampleResponse(
            sequences=[
                tinker.SampledSequence(
                    sequence_id=str(token),
                    stop_reason="stop",
                    _tokens_list=[token],
                    _logprobs_list=[-0.5],
                )
            ]
        )
        with capture(rollout_id=str(rollout.id)):
            await session.sample(tinker.ModelInput.from_ints([1]), 1, tinker.SamplingParams())

    trajectories = [session.finish_rollout(rollout) for rollout in (failure, success)]
    group = TrajectoryGroup(trajectories, final_rewards_G=[0, 0], metrics_G=[{}, {}])
    (advantages,) = compute_advantages([group])
    assert [
        (trajectory.transitions[-1].ac.tokens, advantage)
        for trajectory, advantage in zip(trajectories, advantages.tolist(), strict=True)
    ] == [([202], -0.5), ([101], 0.5)]


@pytest.mark.parametrize(
    ("error_type", "reward", "turn_count"),
    [("SandboxStartupError", None, 0), (None, None, 1), (None, 0, 0)],
)
def test_incomplete_rollout_reports_available_evidence(
    error_type: str | None, reward: int | None, turn_count: int
) -> None:
    session = RelaySession(Mock())
    rollout = Mock(id=uuid4(), error_type=error_type, rewards={"reward": reward})
    session.turns[str(rollout.id)] = [Mock() for _ in range(turn_count)]
    with pytest.raises(ValueError) as exc:
        session.finish_rollout(rollout)
    assert str(exc.value) == (
        f"Incomplete Harbor rollout {rollout.id}: error_type={error_type}, "
        f"reward_present={reward is not None}, captured_turns={turn_count}"
    )
