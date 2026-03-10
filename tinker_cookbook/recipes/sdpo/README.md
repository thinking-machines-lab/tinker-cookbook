# Self-Distilled Policy Optimization (SDPO)

SDPO [1] augments on-policy RL by distilling from the model's own successful trajectories. Instead of assigning a single scalar reward per sequence (like GRPO), SDPO provides **dense, token-level credit assignment** by measuring how much each token's probability diverges from a solution-conditioned teacher.

## How it works

1. **Rollout**: Generate multiple responses per problem using the current policy.
2. **Identify successes**: Find trajectories that solve the problem correctly.
3. **Build teacher prompts**: Prepend the successful solution to the original question, then append "Correctly solve the original question."
4. **Compute teacher logprobs**: A frozen reference model scores each response under the solution-conditioned teacher prompt.
5. **SDPO loss**: Minimize the token-level reverse KL between the student (current policy, normal prompt) and the teacher (reference model, solution-conditioned prompt). From Proposition 2.1 in the paper, this is equivalent to a policy gradient with per-token advantages equal to the log-ratio of teacher to student probabilities.

The frozen reference teacher (theta_ref) is used for regularization. Table 4 in the paper shows this achieves 48.8 accuracy vs 36.1 for the unregularized variant, while being simpler than EMA.

## Quick start

### SciKnowEval (paper's benchmark)

```bash
uv run python -m tinker_cookbook.recipes.sdpo.train \
    model_name="Qwen/Qwen3-8B" \
    env=sciknoweval \
    sciknoweval_domain=chemistry \
    group_size=8 \
    groups_per_batch=32 \
    learning_rate=1e-5 \
    max_tokens=8192
```

### MATH

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

Evaluation on the test set runs automatically during training:

- **Step 0** (before any training): provides the baseline accuracy.
- **Every `eval_every` steps** (default 10): tracks accuracy as training progresses.

The key metric is `test/env/all/correct` (fraction of problems solved). Results are logged to the metrics file:

```bash
# View eval results (path is printed at startup)
cat /tmp/tinker-examples/sdpo/<run-name>/metrics.jsonl | grep "test/env"
```

To track with Weights & Biases, add `wandb_project="sdpo"`.

## Configuration

Key SDPO-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Number of responses sampled per problem (paper uses 8) |
| `success_reward_threshold` | 0.5 | Minimum reward to count a trajectory as successful |
| `reprompt_suffix` | "Correctly solve the original question." | Text appended after the solution in the teacher prompt |
| `dont_reprompt_on_self_success` | True | Exclude a trajectory's own output from being its teacher solution |
| `remove_thinking_from_demonstration` | True | Strip `<think>...</think>` blocks from the demonstration solution |
| `eval_every` | 10 | Evaluate on test set every N steps (0 to disable) |
| `save_every` | 10 | Save checkpoint every N steps (0 to disable) |

### Debugging with a smaller run

```bash
uv run python -m tinker_cookbook.recipes.sdpo.train \
    groups_per_batch=8 \
    group_size=4 \
    max_tokens=512 \
    eval_every=5
```

## Supported environments

- `sciknoweval` — SciKnowEval MCQ (paper's primary benchmark; domains: chemistry, physics, biology, material)
- `math` — Hendrycks MATH (12k train / MATH-500 test)
- `gsm8k` — Grade School Math 8K
- `polaris` — Polaris-53K
- `deepmath` — DeepMath-103K

## Code structure

Reusable SDPO logic lives in `tinker_cookbook/sdpo/`:

| Module | Contents |
|--------|----------|
| `sdpo/data.py` | Datum construction with SDPO advantages |
| `sdpo/teacher.py` | Teacher prompt construction and logprob computation |
| `sdpo/loss.py` | Standalone loss function (for reference/debugging) |
| `sdpo/train.py` | `Config` + `main()` — core training loop |

The recipe CLI (`recipes/sdpo/train.py`) is a thin wrapper that builds the config and calls `sdpo.train.main()`.

## Implementation approach

The SDPO gradient (Proposition 2.1) is a policy gradient with per-token advantages:

```
advantages_t = log pi_teacher(y_t) - log pi_student(y_t)
```

This maps directly to tinker's `importance_sampling` loss. We encode the SDPO signal as advantages in the datum and use standard `forward_backward(..., loss_fn="importance_sampling")`, following the same pattern as tinker's online distillation. This avoids the 1.5-3x overhead of `forward_backward_custom`.

## Differences from the paper

- **Full-logit vs token-level distillation**: The paper uses full-logit JSD (alpha=0.5) with top-k=100 approximation. Our implementation uses token-level reverse KL via the `importance_sampling` loss, which only has access to per-token logprobs rather than the full vocabulary distribution. This is a simpler approximation that still captures the key idea of dense token-level credit assignment.
- **Full fine-tuning vs LoRA**: The paper uses full fine-tuning; this recipe uses LoRA (rank 32) for efficiency.

## References

1. Huebotter, J., Luebeck, F., Behric, A., et al. (2026). Self-Distilled Policy Optimization. arXiv. https://arxiv.org/abs/2601.20802
