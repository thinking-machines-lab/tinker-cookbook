# Self-Distilled Policy Optimization (SDPO)

SDPO [1] augments on-policy RL by distilling from the model's own successful trajectories. Instead of assigning a single scalar reward per sequence (like GRPO), SDPO provides **dense, token-level credit assignment** by measuring how much each token's probability diverges from a solution-conditioned teacher.

## How it works

1. **Rollout**: Generate multiple responses per problem using the current policy.
2. **Identify successes**: Find trajectories that solve the problem correctly.
3. **Build teacher prompts**: Prepend the successful solution to the original question and ask the model to "solve again."
4. **Compute teacher logprobs**: A frozen reference model scores each response under the solution-conditioned teacher prompt.
5. **SDPO loss**: Minimize the token-level reverse KL between the student (current policy, normal prompt) and the teacher (reference model, solution-conditioned prompt). From Proposition 2.1 in the paper, this is equivalent to a policy gradient with per-token advantages equal to the log-ratio of teacher to student probabilities.

The frozen reference teacher (theta_ref) is used for regularization. Table 4 in the paper shows this achieves 48.8 accuracy vs 36.1 for the unregularized variant, while being simpler than EMA.

## Quick start

```bash
uv run python -m tinker_cookbook.recipes.sdpo.train \
    model_name="Qwen/Qwen3-8B" \
    env=math \
    group_size=8 \
    groups_per_batch=64 \
    learning_rate=1e-5 \
    max_tokens=2048
```

## Evaluation

Evaluation on the MATH-500 test set runs automatically during training:

- **Step 0** (before any training): provides the baseline accuracy.
- **Every `eval_every` steps** (default 10): tracks accuracy as training progresses.

The key metric is `test/env/all/correct` (fraction of MATH-500 problems solved). Results are logged to the metrics file:

```bash
# View eval results (path is printed at startup)
cat /tmp/tinker-examples/sdpo/<run-name>/metrics.jsonl | grep "test/env"
```

To track with Weights & Biases:

```bash
uv run python -m tinker_cookbook.recipes.sdpo.train \
    wandb_project="sdpo-math" \
    model_name="Qwen/Qwen3-8B" \
    env=math \
    group_size=8 \
    groups_per_batch=64 \
    learning_rate=1e-5 \
    max_tokens=2048
```

## Configuration

Key SDPO-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Number of responses sampled per problem |
| `success_reward_threshold` | 0.5 | Minimum reward to count a trajectory as successful |
| `reprompt_template` | "The above is a correct solution..." | Text appended after the solution in the teacher prompt |
| `eval_every` | 10 | Evaluate on test set every N steps (0 to disable) |
| `save_every` | 10 | Save checkpoint every N steps (0 to disable) |

Other parameters (model, dataset, logging, checkpointing) follow the same conventions as the [math_rl](../math_rl/) recipe.

### Debugging with a smaller run

```bash
uv run python -m tinker_cookbook.recipes.sdpo.train \
    groups_per_batch=8 \
    group_size=4 \
    max_tokens=512 \
    eval_every=5
```

## Supported environments

This recipe reuses the math environments from the `math_rl` recipe:

- `math` — Hendrycks MATH (12k train / MATH-500 test)
- `gsm8k` — Grade School Math 8K
- `polaris` — Polaris-53K
- `deepmath` — DeepMath-103K

## References

1. Huebotter, J., Luebeck, F., Behric, A., et al. (2026). Self-Distilled Policy Optimization. arXiv. https://arxiv.org/abs/2601.20802
