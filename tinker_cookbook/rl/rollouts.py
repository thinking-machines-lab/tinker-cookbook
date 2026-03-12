import asyncio
import numbers
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


def _log_transition_metrics(metrics: dict[str, Any] | None) -> None:
    """Render transition metrics in a compact, always-visible table."""
    if not metrics:
        return
    formatted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, numbers.Real):
            formatted_metrics[key] = f"{float(value):.3f}"
        else:
            formatted_metrics[key] = str(value)
    with logtree.scope_header("Step Metrics"):
        logtree.table_from_dict(
            formatted_metrics,
            caption="Metrics emitted by env.step",
        )


def _log_single_trajectory_details(traj: Trajectory, final_reward: float) -> None:
    with logtree.scope_header("Episode Details"):
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
                _log_transition_metrics(transition.metrics)
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


async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """Run a single rollout (env episode). Env logging (if any) goes into
    whatever logtree scope the caller has set up."""
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


@logtree.scope_header_decorator("Group Rollout")
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    envs_G: Sequence[Env] = await env_group_builder.make_envs()

    async def _scoped_rollout(idx: int, env: Env) -> Trajectory:
        with logtree.scope_header_deferred(f"Rollout {idx}"):
            return await do_single_rollout(policy, env)

    trajectories_G = await asyncio.gather(
        *[_scoped_rollout(i, env) for i, env in enumerate(envs_G)]
    )

    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    with logtree.scope_header("Trajectory Details"):
        for traj_idx, (traj, final_reward) in enumerate(
            zip(trajectories_G, rewards_G, strict=True)
        ):
            with logtree.scope_header(f"Trajectory {traj_idx} Episode"):
                _log_single_trajectory_details(traj, final_reward)

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))
