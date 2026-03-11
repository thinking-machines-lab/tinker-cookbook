---
name: grpo
description: Set up and run reinforcement learning with verifiable rewards (RLVR/GRPO) for math, code, or custom environments using the Tinker API. Use when the user wants to do RL training, GRPO, reward-based optimization, or train with verifiable rewards.
argument-hint: "[model-name] [environment]"
---

# Group Relative Policy Optimization (GRPO / RL)

Help the user set up and run RL training with verifiable rewards using the Tinker API.

## Step 1: Understand the request

Ask the user (if not already specified):
- **Model**: Which model to train (e.g., `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`)
- **Environment/Task**: What type of reward signal — math (GSM8K, DeepMath, arithmetic), code (DeepCoder), instruction following (IFBench), or custom
- **Reward type**: Verifiable (programmatic correctness) or learned (preference model)

## Step 2: Reference existing recipes

Read these files for patterns:
- `tinker_cookbook/recipes/rl_basic.py` — Minimal RL example (GSM8K)
- `tinker_cookbook/recipes/math_rl/train.py` — Full math RL with multiple environments and loss functions
- `tinker_cookbook/recipes/code_rl/train.py` — Code generation RL with sandbox execution
- `tinker_cookbook/recipes/if_rl/train.py` — Instruction-following RL
- `tinker_cookbook/rl/train.py` — Core RL training loop
- `tinker_cookbook/rl/types.py` — Env, EnvGroupBuilder, RLDatasetBuilder
- `docs/rl/rl-basic.mdx` — Getting started
- `docs/rl/rl-envs.mdx` — Custom environments
- `docs/rl/rl-hyperparams.mdx` — Hyperparameter guidance

## Step 3: Configure the training run

### Environment Setup
RL requires an environment that produces rewards. Key patterns:

**Built-in environments:**
- `Gsm8kDatasetBuilder` — Grade-school math (from `recipes/math_rl/math_env.py`)
- `ArithmeticDatasetBuilder` — Simple arithmetic
- `DeepMathDatasetBuilder`, `PolarisDatasetBuilder` — Advanced math
- `DeepCoderDatasetBuilder` — Code generation with sandbox
- `IFBenchDatasetBuilder` — Instruction following

**Custom environments:**
Implement the `Env` protocol from `tinker_cookbook/rl/types.py`. Key points:
- `Env` objects are **single-use** (no reset method)
- Create new envs via `EnvGroupBuilder` each batch
- Each env returns a `float` reward

### Key Hyperparameters

- `group_size`: Number of rollouts per prompt (typically 4-16). Advantages are centered within each group.
- `groups_per_batch` (or `batch_size`): Number of problems per batch
- `max_tokens`: Maximum generation length
- `learning_rate`: Typically 1e-5 to 4e-5 for RL
- `kl_penalty_coef`: KL penalty against reference model (0.0 = no penalty)
- `temperature`: Sampling temperature (default 1.0)

### Loss Functions
- `importance_sampling` — Default, on-policy
- `ppo` — Proximal Policy Optimization (clipped)
- `cispo` — Conservative Importance Sampling PPO
- `dro` — Distributionally Robust Optimization
- Configure via `loss_fn` and `loss_fn_config` parameters

### Async Training (Off-Policy)
For overlapping sampling and training:
```python
async_config=AsyncConfig(
    max_steps_off_policy=cli_config.max_steps_off_policy,
    groups_per_batch=cli_config.groups_per_batch,
)
```

## Step 4: Write the training script

Follow the pattern from `rl_basic.py` / `math_rl/train.py`:

```python
import asyncio
import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    # Configure your dataset builder with environment
    builder = ...  # e.g., Gsm8kDatasetBuilder(...)

    return chz.Blueprint(train.Config).apply({
        "model_name": model_name,
        "renderer_name": renderer_name,
        "log_path": "/tmp/tinker-examples/my_rl_run",
        "dataset_builder": builder,
        "learning_rate": 4e-5,
        "max_tokens": 256,
        "eval_every": 20,
    })

def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))
```

For the full CLI pattern with `@chz.chz` config class, see `recipes/math_rl/train.py`.

## Step 5: Run

```bash
python -m tinker_cookbook.recipes.<recipe_name>
```

Override: `python -m tinker_cookbook.recipes.<recipe_name> env=gsm8k group_size=16 learning_rate=4e-5`

## Common pitfalls
- `Env` objects are single-use — always create fresh envs via builder
- Advantages are centered within each group — `group_size` matters for variance reduction
- `max_tokens` too small truncates reasoning; too large wastes compute
- Start with small `groups_per_batch` for debugging, scale up for real runs
- Use `num_substeps > 1` for very large batches to split optimizer steps
