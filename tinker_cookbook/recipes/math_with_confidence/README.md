# Math RL With Confidence

This recipe trains a model to output both a final answer and its confidence:

```text
<answer>3</answer><confidence>0.7</confidence>
```

The environment reward is:

```text
reward = correctness + alpha * brier_term + consistency_coef * consistency + consistency_v2_coef * consistency_v2
```

where `correctness` is `1.0` for a correct answer and `0.0` otherwise.
`consistency` is an LLM-judged score in `[0, 1]` for how well the verbal confidence in the
response matches the numeric confidence in the `<confidence>` tag.
`consistency_v2` is computed from inferred communicated certainty in the response body, with
reward term `1 - (inferred_confidence - tagged_confidence)^2`.

## Quickstart

```bash
python -m tinker_cookbook.recipes.math_with_confidence.train \
  model_name="Qwen/Qwen3-30B-A3B-Base" \
  dataset_name=math \
  alpha=0.5 \
  group_size=8 \
  groups_per_batch=64 \
  learning_rate=2e-5 \
  max_tokens=768
```

Interactive inspection:

```bash
python -m tinker_cookbook.recipes.math_with_confidence.interactive \
  model_name="Qwen/Qwen3-30B-A3B-Base" \
  dataset_name=math \
  split=test \
  num_examples=5
```

## Documented Choices

- `ProblemEnv` subclass: We subclass `ProblemEnv` and override `step()` so we stay compatible with existing RL rollout/logging machinery while customizing reward logic.
- Default Brier term: `brier_reward_mode="one_minus_squared_error"` so the calibration term stays in `[0, 1]`.
- Consistency grader: one rubric-like item only; default grader model is `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Default consistency coefficients are `0.0` for both `consistency_coef` and `consistency_v2_coef`.
- Instruction placement: this recipe now uses user-message instructions only (no system prompt), to avoid duplicate instruction channels.
- One-shot example: We include one one-shot demonstration by default (`include_fewshot=True`) to stabilize output format early in training.
- Model/renderer flexibility: `model_name` and `renderer_name` are configurable; renderer defaults are derived from `model_info.get_recommended_renderer_name(model_name)`.

## Notes

- Supported datasets: `math`, `polaris`, `deepmath`, `gsm8k`.
- For `polaris` and `deepmath`, test split is not provided in this recipe builder (same behavior as the existing `math_rl` recipe).
