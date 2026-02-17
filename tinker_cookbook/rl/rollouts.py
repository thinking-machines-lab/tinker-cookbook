import asyncio
from typing import Any, Sequence

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree

def _log_transition_logs(logs: dict[str, Any]) -> None:
    """Render transition logs in a readable structure without truncating table cells."""
    if not logs:
        return
    with logtree.scope_header("Diagnostics"):
        for key, value in logs.items():
            text = str(value)
            if "\n" in text or len(text) > 120:
                logtree.details(text, summary=key, pre=True)
            else:
                logtree.log_text(f"{key}: {text}")


def _log_single_trajectory_details(traj: Trajectory, final_reward: float) -> None:
    with logtree.scope_header("Rollout"):
        for turn_idx, transition in enumerate(traj.transitions, start=1):
            with logtree.scope_header(f"Turn {turn_idx}"):
                logtree.table_from_dict(
                    {
                        "ob_len": transition.ob.length,
                        "ac_len": len(transition.ac.tokens),
                        "step_reward": f"{transition.reward:.3f}",
                    },
                    caption="Step stats",
                )
                _log_transition_logs(transition.logs)

        logtree.table_from_dict(
            {
                "num_turns": len(traj.transitions),
                "final_ob_len": traj.final_ob.length,
                "sum_step_rewards": f"{sum(t.reward for t in traj.transitions):.3f}",
                "final_group_reward": f"{final_reward:.3f}",
                "total_return": f"{sum(t.reward for t in traj.transitions) + final_reward:.3f}",
            },
            caption="Episode totals",
        )


@logtree.scope_header_decorator
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
            logs=step_result.logs,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Log compact group summary plus per-trajectory sections.
    with logtree.scope_header("Trajectory Summary"):
        summary_rows: list[dict[str, Any]] = []
        for traj_idx, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            step_reward_sum = sum(t.reward for t in traj.transitions)
            total_return = step_reward_sum + final_reward
            summary_rows.append(
                {
                    "trajectory": traj_idx,
                    "turns": len(traj.transitions),
                    "step_reward_sum": f"{step_reward_sum:.3f}",
                    "final_group_reward": f"{final_reward:.3f}",
                    "total_return": f"{total_return:.3f}",
                }
            )
        logtree.table(summary_rows, caption="Per-trajectory totals")

        for traj_idx, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            with logtree.scope_header(f"Trajectory {traj_idx}"):
                _log_single_trajectory_details(traj, final_reward)

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
