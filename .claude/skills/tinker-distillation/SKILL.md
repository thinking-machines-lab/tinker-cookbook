---
name: tinker-distillation
description: Set up and run knowledge distillation (on-policy, off-policy, or multi-teacher) from a teacher model to a student model using the Tinker API. Use when the user wants to distill knowledge, compress models, or train a student from a teacher.
argument-hint: "[student-model] [teacher-model]"
---

# Knowledge Distillation

Help the user set up and run distillation from teacher to student models using the Tinker API.

## Step 1: Understand the request

Ask the user (if not already specified):
- **Student model**: Which model to train (e.g., `Qwen/Qwen3-8B-Base`)
- **Teacher model**: Which model to distill from (e.g., `Qwen/Qwen3-8B`, or a checkpoint path)
- **Distillation type**:
  - **On-policy**: Student generates, teacher scores via KL — best for reasoning/chat
  - **Off-policy reasoning**: SFT on teacher-generated reasoning traces (e.g., OpenThoughts3)
  - **Multi-teacher**: Combine multiple teachers on different datasets

## Step 2: Reference existing recipes

Read these files for patterns:
- `tinker_cookbook/recipes/distillation/on_policy_distillation.py` — On-policy distillation CLI
- `tinker_cookbook/recipes/distillation/off_policy_reasoning.py` — SFT on OpenThoughts3 traces
- `tinker_cookbook/recipes/distillation/on_policy_multi_teacher.py` — Multi-teacher setup
- `tinker_cookbook/distillation/train_on_policy.py` — Core on-policy training loop
- `tinker_cookbook/distillation/datasets.py` — TeacherConfig, PromptOnlyDatasetBuilder, DistillationDatasetConfig

## Step 3: Choose distillation approach

### On-Policy Distillation (Recommended)
Student generates samples, teacher provides KL penalty supervision. No correctness rewards needed.

Key config:
- `TeacherConfig(base_model="Qwen/Qwen3-8B", load_checkpoint_path=None)`
- `PromptOnlyDatasetBuilder(dataset_name="deepmath"|"tulu3", ...)`
- `DistillationDatasetConfig(dataset_builder=..., teacher_config=..., groups_per_batch=...)`
- `kl_penalty_coef`: Weight of KL penalty (default 1.0)
- `kl_discount_factor`: Discount for future KL (0.0 = no discount)

### Off-Policy Reasoning (SFT on Traces)
Standard SFT on pre-generated reasoning traces (e.g., OpenThoughts3). Simpler but less effective than on-policy.

See `recipes/distillation/off_policy_reasoning.py` — uses the standard SL pipeline from `supervised/train.py`.

### Multi-Teacher Distillation
Combine multiple teacher models on different datasets. Each dataset can have its own teacher.

See `recipes/distillation/on_policy_multi_teacher.py` — passes multiple `DistillationDatasetConfig` objects.

## Step 4: Write the training script

Follow the on-policy distillation pattern:

```python
import asyncio
import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B-Base"       # Student
    teacher_model: str = "Qwen/Qwen3-8B"          # Teacher
    dataset: str = "deepmath"                       # deepmath or tulu3
    group_size: int = 4
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    lora_rank: int = 128
    loss_fn: str = "importance_sampling"

async def cli_main(cli_config: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name, ...)

    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name=cli_config.dataset,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )
    teacher_config = TeacherConfig(base_model=cli_config.teacher_model)
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )
    config = train_on_policy.Config(
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        loss_fn=cli_config.loss_fn,
        log_path="/tmp/tinker-examples/distillation/my_run",
    )
    await train_on_policy.main(config)
```

## Step 5: Run

```bash
# On-policy distillation (reasoning)
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base dataset=deepmath learning_rate=1e-4

# Off-policy reasoning (SFT on traces)
python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
    model_name=Qwen/Qwen3-8B-Base learning_rate=2e-4

# Multi-teacher
python -m tinker_cookbook.recipes.distillation.on_policy_multi_teacher \
    model_name=Qwen/Qwen3-8B-Base learning_rate=1e-4
```

## Step 6: Add tests

If you created a new distillation recipe, add a smoke test:

```python
# tests/recipes/test_recipe_<name>.py
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        ["behavior_if_log_dir_exists=delete", "groups_per_batch=16"],
    )
```

`run_recipe()` automatically passes `max_steps=2` so the recipe runs 2 training steps and exits. See `tests/recipes/test_recipe_on_policy_distillation.py` and `tests/recipes/test_recipe_on_policy_multi_teacher.py` for existing examples.

## Step 7: Export weights (optional)

After distillation, export the student model using the `tinker_cookbook.weights` API:

```python
from tinker_cookbook import weights

adapter_dir = weights.download(tinker_path="tinker://run-id/sampler_weights/final", output_dir="./adapter")
weights.build_hf_model(base_model="Qwen/Qwen3-8B-Base", adapter_path=adapter_dir, output_path="./model")
weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-distilled-model")
```

## Common pitfalls
- Teacher model must be compatible with student's tokenizer/renderer
- On-policy is generally better than off-policy but more compute-intensive
- `kl_discount_factor=0.0` means no discounting — increase for longer sequences
- High `kl_penalty_coef` can make training too conservative
- For multi-teacher, ensure `groups_per_batch` is balanced across datasets
