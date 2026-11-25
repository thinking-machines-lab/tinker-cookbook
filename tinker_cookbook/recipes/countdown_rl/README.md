# Countdown Number Game — RL with Reward Shaping

Train a language model to solve Countdown-style arithmetic puzzles using GRPO with verifiable rewards. Given 3–4 numbers and a target, the model must combine the numbers with `+`, `-`, `*`, `/` to reach the target.

This recipe demonstrates how **reward function design** directly affects RL training — specifically, how partial credit rewards improve sample efficiency over binary pass/fail grading.

## Quick start

```bash
# Quick experiment (~20 min, 20 steps)
python -m tinker_cookbook.recipes.countdown_rl.train \
    n_train=1600 n_test=100 eval_every=5 max_steps=20

# Full training (~2.5 hours, best config)
python -m tinker_cookbook.recipes.countdown_rl.train \
    max_tokens=2048 n_train=3200 n_test=200 max_steps=40
```

## Motivation

The Countdown task is a clean testbed for studying RL reward design:

- **Verifiable**: we can check answers programmatically (does the expression evaluate to the target using only the given numbers?)
- **Graded difficulty**: some problems are trivially solvable (`44 + 19 + 35 = 98`) while others require creative use of division and parentheses
- **Observable failure modes**: by reading rollouts, we can see exactly *why* the model fails (truncation? wrong answer? bad format?)

The central question: **when binary rewards leave most GRPO groups with zero learning signal, can partial credit fix that?**

## Key findings

### 1. Partial credit converts dead groups into useful signal

With binary rewards at step 0, **50% of GRPO groups are "all-bad"** — every completion in the group is wrong, so all advantages are zero and the group contributes nothing to the gradient. Partial credit (grading proximity to the target) converts many of these into "mixed" groups with reward variance, adding ~4% test accuracy.

```
Binary reward:  correct=1.0, wrong=0.0, invalid=0.0
Partial credit: correct=1.0, close=0.54, far=0.30, invalid=0.0
```

### 2. Token budget is the single biggest lever

Rollout analysis revealed that **100% of remaining failures at step 15+ are token truncations** — the model never gets the wrong answer when it has enough tokens. Going from 512 to 2048 tokens added 17 percentage points.

| Token budget | Best test accuracy |
|---|---|
| 512 | 68% |
| 1024 | 76% |
| 2048 | **85%** |

### 3. The model learns conciseness naturally

Average response length drops from **~1100 tokens to ~500 tokens** over 40 training steps, without any explicit length penalty. Within each GRPO group, a 300-token correct response gets reward 1.0, while a 2048-token truncated attempt gets 0.0 — the advantage signal naturally pushes the policy toward brevity.

### 4. Look at your rollouts

The decision to increase token budget from 512 to 2048 came entirely from reading actual model responses. Metrics showed accuracy plateauing; rollouts showed that every failure was a truncation. Without looking at the rollouts, the natural next step would have been to tune the learning rate or reward function — neither of which would have helped.

## Full hyperparameter sweep

All experiments use Qwen3-4B-Instruct-2507 with LoRA rank 32 and lr=1e-4 unless noted.

| Config | Best Test Acc | Finding |
|---|---|---|
| binary, 512tok, g8, lr=5e-4 | 68% | Baseline (unstable) |
| binary, 512tok, g16, lr=1e-4 | 68% | Lower LR stabilized training |
| binary, 1024tok | 72% | Token increase alone helps |
| **partial, 1024tok** | **76%** | **Partial rewards +4%** |
| partial, g32 | 76% | Eliminates all-bad groups, same peak |
| partial, KL=0.02 | 70% | KL penalty too conservative |
| partial, no fewshot | 72% | Fewshot prefix is critical for cold start |
| partial, 1024tok, 40 steps | 80% | More training teaches conciseness |
| partial, temp=0.7 | 78.5% | Lower temperature hurts exploration |
| **partial, 2048tok, 40 steps** | **85%** | **Best config** |

## Recipe structure

```
tinker_cookbook/recipes/countdown_rl/
├── __init__.py
├── countdown_env.py      # CountdownEnv (ProblemEnv subclass) + reward logic
├── countdown_env_test.py  # 14 unit tests for reward verification
└── train.py               # CLI entrypoint with chz config
```

## Configuration

All parameters can be set via the command line using `chz` syntax (`key=value`):

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-4B-Instruct-2507` | HuggingFace model identifier |
| `reward_mode` | `partial` | `binary` or `partial` (with proximity bonus) |
| `max_tokens` | `2048` | Max generation tokens per response |
| `group_size` | `16` | GRPO completions per problem |
| `groups_per_batch` | `16` | Problems per training step |
| `learning_rate` | `1e-4` | Adam learning rate |
| `kl_penalty_coef` | `0.0` | KL penalty (0 = disabled) |
| `include_fewshot` | `True` | Include 1-shot demonstration in prompt |
| `max_steps` | `None` | Stop after N steps (None = train on all data) |

## Reproducing the experiments

```bash
# Binary reward baseline (512 tokens)
python -m tinker_cookbook.recipes.countdown_rl.train \
    reward_mode=binary max_tokens=512 n_train=1600 n_test=100 \
    eval_every=5 max_steps=20 behavior_if_log_dir_exists=delete

# Partial reward (1024 tokens)
python -m tinker_cookbook.recipes.countdown_rl.train \
    reward_mode=partial max_tokens=1024 n_train=1600 n_test=100 \
    eval_every=5 max_steps=20 behavior_if_log_dir_exists=delete

# Best config (2048 tokens, 40 steps)
python -m tinker_cookbook.recipes.countdown_rl.train \
    reward_mode=partial max_tokens=2048 n_train=3200 n_test=200 \
    eval_every=10 max_steps=40 behavior_if_log_dir_exists=delete

# Compare with KL penalty
python -m tinker_cookbook.recipes.countdown_rl.train \
    kl_penalty_coef=0.02 max_steps=20 n_train=1600 n_test=100 \
    eval_every=5 behavior_if_log_dir_exists=delete
```

Training logs (metrics, rollout transcripts, HTML reports) are written to `~/tinker-experiments/countdown_rl/` by default. Metrics are in `metrics.jsonl`; per-trajectory data is in `iteration_*/train_rollout_summaries.jsonl`.
