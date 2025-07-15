import asyncio
from typing import Sequence

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
