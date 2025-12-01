# GEPA: Prompt Optimization via LLM Reflection

Evolve system prompts through reflection - optimizer-free prompt optimization. GEPA evaluates prompts, has a teacher LLM reflect on failures, mutates the prompt, and keeps improvements.

**Paper**: [arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457) | **Library**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)

## Running This Recipe

```bash
pip install tinker_cookbook[gepa]
```

### GSM8K

```bash
python -m tinker_cookbook.recipes.gepa.train \
    task_name=gsm8k \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=50
```

After optimization, expect `final/best_score` around 0.91.

### HotpotQA

```bash
python -m tinker_cookbook.recipes.gepa.train \
    task_name=hotpotqa \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=100
```

### AIME

```bash
python -m tinker_cookbook.recipes.gepa.train \
    task_name=aime \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=150 \
    eval_test=true
```

## Custom Tasks

Register via `TASK_REGISTRY`:

```python
from tinker_cookbook.recipes.gepa.tasks import GEPATask, register_task

@register_task("my_benchmark")
class MyBenchmarkTask(GEPATask):
    @property
    def name(self) -> str:
        return "my_benchmark"

    @property
    def seed_prompt(self) -> str:
        return "You are a helpful assistant..."

    def load_data(self, seed: int = 0):
        return train, val, test  # GEPADataInstance lists

    def score(self, response: str, answer: str, metadata: dict | None = None) -> float:
        return 1.0 if answer.strip() in response else 0.0
```

Then: `python -m tinker_cookbook.recipes.gepa.train task_name=my_benchmark`
