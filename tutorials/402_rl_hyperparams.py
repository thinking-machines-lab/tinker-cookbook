import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 10: RL Hyperparameters

    Tune KL penalty, group size, and advantage normalization.

    RL training has several hyperparameters beyond learning rate that critically affect stability and performance. This tutorial covers the most important ones.
    """)
    return


@app.cell
def _():
    import torch

    from tinker_cookbook.rl.data_processing import compute_advantages
    from tinker_cookbook.rl.train import Config, KLReferenceConfig

    return Config, KLReferenceConfig, compute_advantages, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KL divergence in RL fine-tuning

    Without regularization, RL can push the model far from its pretrained distribution, causing:
    - **Reward hacking** -- exploiting artifacts in the reward function
    - **Catastrophic forgetting** -- losing general capabilities
    - **Mode collapse** -- generating repetitive outputs

    The **KL penalty** adds a term to the advantage that penalizes divergence from a reference model:

    ```
    adjusted_advantage = original_advantage + kl_coef * (avg_kl - per_token_kl)
    ```

    This keeps the policy close to the reference while still improving on the reward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KLReferenceConfig

    To enable KL penalty in `rl.train.Config`, set `kl_penalty_coef > 0` and provide a `kl_reference_config`:
    """)
    return


@app.cell
def _(KLReferenceConfig):
    # Example: KL penalty against the base model
    kl_config = KLReferenceConfig(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        load_checkpoint_path=None,  # Use base model weights as reference
    )

    print("KL reference config:")
    print(f"  base_model: {kl_config.base_model}")
    print(f"  checkpoint:  {kl_config.load_checkpoint_path}")
    print()

    # In rl.train.Config, you would set:
    # kl_penalty_coef=0.05,        # Strength of KL regularization
    # kl_discount_factor=0.0,       # 0 = no discounting
    # kl_reference_config=kl_config,
    print("Typical kl_penalty_coef values:")
    print("  0.0   -- no KL penalty (default)")
    print("  0.01  -- light regularization")
    print("  0.05  -- moderate (good starting point)")
    print("  0.1+  -- strong (may slow reward improvement)")
    return (kl_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Group size and advantage computation

    In GRPO, each problem is solved by a *group* of rollouts. Advantages are centered within each group:

    ```
    advantage_i = reward_i - mean(rewards in group)
    ```

    **Group size** controls the variance of the advantage estimate:
    - **Small groups (2-4)**: High variance, but every problem gets gradient signal even if most rollouts fail
    - **Large groups (8-16)**: Lower variance, better advantage estimates, but more compute per problem

    Let's see how group size affects the advantage distribution.
    """)
    return


@app.cell
def _(compute_advantages, torch):
    from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup

    def make_mock_group(rewards):
        """Create a TrajectoryGroup with the given rewards (no actual trajectories needed for advantage computation)."""
        trajs = [Trajectory(transitions=[], final_ob=None) for _ in rewards]
        return TrajectoryGroup(
            trajectories_G=trajs,
            final_rewards_G=rewards,
            metrics_G=[{} for _ in rewards],
        )

    # Compare group sizes: same total reward distribution, different grouping
    all_rewards = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    # Group size 2: 4 groups
    groups_2 = [make_mock_group(all_rewards[i : i + 2]) for i in range(0, 8, 2)]
    advs_2 = compute_advantages(groups_2)
    print("Group size 2:")
    for i, adv in enumerate(advs_2):
        print(f"  Group {i}: rewards={all_rewards[i * 2 : i * 2 + 2]}, advantages={adv.tolist()}")

    print()

    # Group size 4: 2 groups
    groups_4 = [make_mock_group(all_rewards[i : i + 4]) for i in range(0, 8, 4)]
    advs_4 = compute_advantages(groups_4)
    print("Group size 4:")
    for i, adv in enumerate(advs_4):
        print(f"  Group {i}: rewards={all_rewards[i * 4 : i * 4 + 4]}, advantages={adv.tolist()}")

    print()

    # Group size 8: 1 group
    groups_8 = [make_mock_group(all_rewards)]
    advs_8 = compute_advantages(groups_8)
    print("Group size 8:")
    print(f"  Group 0: rewards={all_rewards}, advantages={advs_8[0].tolist()}")
    return (Trajectory, TrajectoryGroup, make_mock_group)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Constant-reward filtering

    When all rollouts in a group get the same reward, advantages are all zero -- no gradient signal. The `remove_constant_reward_groups` option filters these out:
    """)
    return


@app.cell
def _(make_mock_group):
    from tinker_cookbook.rl.data_processing import remove_constant_reward_groups

    groups = [
        make_mock_group([1.0, 1.0, 1.0, 1.0]),  # All correct -- no signal
        make_mock_group([0.0, 1.0, 0.0, 1.0]),  # Mixed -- has signal
        make_mock_group([0.0, 0.0, 0.0, 0.0]),  # All wrong -- no signal
    ]

    filtered = remove_constant_reward_groups(groups)
    print(f"Before filtering: {len(groups)} groups")
    print(f"After filtering:  {len(filtered)} groups")
    print(f"Kept rewards: {[g.get_total_rewards() for g in filtered]}")
    return (filtered, groups)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary of RL hyperparameters

    | Parameter | Default | Range | Effect |
    |-----------|---------|-------|--------|
    | `learning_rate` | -- | 1e-6 to 1e-4 | Step size; too high causes instability |
    | `kl_penalty_coef` | 0.0 | 0.01 to 0.1 | Regularization toward reference |
    | `group_size` | 4 | 2 to 16 | Advantage estimation quality |
    | `num_substeps` | 1 | 1 to 4 | Gradient accumulation |
    | `loss_fn` | importance_sampling | IS, PPO | Policy gradient estimator |
    | `temperature` | 1.0 | 0.7 to 1.0 | Exploration vs exploitation |
    | `remove_constant_reward_groups` | False | True/False | Filter zero-signal groups |

    Start with `learning_rate=1e-5`, `group_size=4`, `kl_penalty_coef=0.05`, then adjust based on reward curves.
    """)
    return


if __name__ == "__main__":
    app.run()
