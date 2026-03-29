# Evaluation

Complete reference for evaluator patterns.

## Reference

- `tinker_cookbook/eval/evaluators.py` — Evaluator types
- `tinker_cookbook/supervised/train.py` — SL evaluator integration
- `tinker_cookbook/rl/train.py` — RL evaluator integration

## SL evaluators

Two tiers:
```python
config = supervised_train.Config(
    evaluator_builders=[...],              # Every eval_every steps
    infrequent_evaluator_builders=[...],   # Every infrequent_eval_every steps
    eval_every=8,
    infrequent_eval_every=50,
)
```

## RL evaluators

Uses `SamplingClientEvaluator`:
```python
async def my_evaluator(sampling_client: SamplingClient) -> dict[str, float]:
    return {"accuracy": 0.85, "avg_length": 150}

config = rl_train.Config(evaluator_builders=[my_evaluator], eval_every=20)
```

## RL test set evaluator

Built into `rl/train.py` via the test dataset from `RLDatasetBuilder.__call__()`:
```python
# RLDatasetBuilder.__call__() returns (train_dataset, test_dataset)
```

## Inspect AI integration

```python
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

evaluator = InspectAPIFromTinkerSampling(
    task="gsm8k", renderer_name=renderer_name,
    model_name=model_name, include_reasoning=True,
)
```

See `tinker_cookbook/recipes/chat_sl/train.py` for a working example with GSM8K and IFEval.

## Custom evaluators

### Sampling-based

```python
async def eval_math(sampling_client: SamplingClient) -> dict[str, float]:
    async def evaluate_one(problem):
        response = await sampling_client.sample_async(
            prompt=problem.prompt, num_samples=1,
            sampling_params=SamplingParams(max_tokens=256, temperature=0.0),
        )
        return parse_answer(response.sequences[0].tokens) == problem.expected

    # Evaluate all problems concurrently — sequential loops waste throughput
    results = await asyncio.gather(*[evaluate_one(p) for p in test_problems])
    return {"math_accuracy": sum(results) / len(results)}
```

### NLL-based

Compute NLL on a held-out dataset without generating text. See the built-in evaluator in `tinker_cookbook/supervised/train.py`.

## Metrics logging

```python
from tinker_cookbook.utils.ml_log import log_metrics
log_metrics({"train/loss": 0.5, "eval/accuracy": 0.85}, step=100)
```

## Code references

- `tinker_cookbook/eval/evaluators.py` — TrainingClientEvaluator, SamplingClientEvaluator
- `tinker_cookbook/eval/inspect_evaluators.py` — Inspect-based evaluators
- `tinker_cookbook/eval/custom_evaluators.py` — Custom evaluator implementations
- `tinker_cookbook/supervised/nll_evaluator.py` — NLL evaluator
