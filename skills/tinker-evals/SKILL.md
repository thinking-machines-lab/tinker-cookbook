---
name: tinker-evals
description: Guide for evaluation — inline evaluators, Inspect AI integration, and custom evaluators for measuring training progress. Use when the user asks about evaluation, metrics, benchmarks, or how to measure model quality during training.
---

# Evaluation

Training scripts support inline evaluation at configurable intervals. The cookbook provides several evaluator patterns.

## Reference

Read these for details:
- `docs/evals.mdx` — Evaluation guide
- `tinker_cookbook/supervised/train.py` — SL evaluator integration (search for `evaluator_builders`)
- `tinker_cookbook/rl/train.py` — RL evaluator integration
- `tinker_cookbook/recipes/chat_sl/train.py` — Example with Inspect AI evaluators

## Evaluator types

### SL evaluators
SL training supports two evaluator tiers:

```python
config = supervised_train.Config(
    evaluator_builders=[...],              # Run every eval_every steps
    infrequent_evaluator_builders=[...],   # Run every infrequent_eval_every steps
    eval_every=8,
    infrequent_eval_every=50,
)
```

### RL evaluators
RL training uses `SamplingClientEvaluator`:

```python
async def my_evaluator(sampling_client: SamplingClient) -> dict[str, float]:
    # Generate samples, compute metrics
    return {"accuracy": 0.85, "avg_length": 150}

config = rl_train.Config(
    evaluator_builders=[my_evaluator],
    eval_every=20,
)
```

### RL test set evaluator
Evaluates the policy on a held-out test set of environments:

```python
# Built into rl/train.py via test_dataset from RLDatasetBuilder
# RLDatasetBuilder.__call__() returns (train_dataset, test_dataset)
```

## Inspect AI integration

The cookbook integrates with [Inspect AI](https://inspect.ai) for standard benchmarks:

```python
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

# Create an Inspect evaluator that uses Tinker sampling
evaluator = InspectAPIFromTinkerSampling(
    task="gsm8k",          # Inspect task name
    renderer_name=renderer_name,
    model_name=model_name,
    include_reasoning=True,  # Include reasoning traces
)
```

See `tinker_cookbook/recipes/chat_sl/train.py` for a working example with GSM8K and IFEval.

## Custom evaluators

### Pattern 1: Sampling-based evaluation

```python
async def eval_math(sampling_client: SamplingClient) -> dict[str, float]:
    correct = 0
    total = 100
    for problem in test_problems:
        response = sampling_client.sample(
            prompt=problem.prompt,
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=256, temperature=0.0),
        )
        answer = parse_answer(response.sequences[0].tokens)
        if answer == problem.expected:
            correct += 1
    return {"math_accuracy": correct / total}
```

### Pattern 2: NLL-based evaluation

Compute negative log-likelihood on a held-out dataset without generating text. See `tinker_cookbook/supervised/train.py` for the built-in NLL evaluator.

## Metrics logging

```python
from tinker_cookbook.utils.ml_log import log_metrics

log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

## Common pitfalls
- Evaluators run inline during training — keep them fast to avoid stalling the training loop
- Use `infrequent_evaluator_builders` for expensive evals (large benchmarks)
- RL evaluators receive a SamplingClient — create completers from it if needed
- For Inspect AI, set `include_reasoning=True` to capture thinking traces
