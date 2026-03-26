---
name: tinker-distillation
description: Set up and run knowledge distillation (on-policy, off-policy, or multi-teacher) from a teacher model to a student model using the Tinker API. Use when the user wants to distill knowledge, compress models, or train a student from a teacher.
---

# Knowledge Distillation

Help the user set up and run distillation from teacher to student models using the Tinker API.

## Key concepts

**Distillation types:**
- **On-policy** (recommended): Student generates, teacher scores via KL divergence. No correctness rewards needed.
- **Off-policy reasoning**: SFT on teacher-generated reasoning traces (e.g., OpenThoughts3). Simpler but less effective.
- **Multi-teacher**: Different teachers for different datasets. Each dataset gets its own `DistillationDatasetConfig`.

**Core abstractions:**
- `TeacherConfig(base_model, load_checkpoint_path)` — identifies the teacher model
- `PromptOnlyDatasetBuilder(dataset_name, ...)` — loads prompts (built-in: `"deepmath"`, `"tulu3"`)
- `DistillationDatasetConfig(dataset_builder, teacher_config, groups_per_batch)` — binds a dataset to a teacher

**Key parameters:**
- `kl_penalty_coef`: Weight of KL penalty (default 1.0). The only supervision signal — no reward function needed.
- `kl_discount_factor`: Discount for future KL (0.0 = no discount). Increase for longer sequences.
- `group_size`: Rollouts per prompt (default 4)
- `groups_per_batch`: Prompts per batch (default 1024)

## Minimal working example

This is a complete, runnable on-policy distillation script:

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
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )
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
    log_path = cli_config.log_path or f"/tmp/tinker-examples/distillation/{cli_config.dataset}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train_on_policy.Config(
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        learning_rate=cli_config.learning_rate,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        log_path=log_path,
        max_steps=cli_config.max_steps,
    )
    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
```

Run it: `python my_distill.py` or `python my_distill.py dataset=tulu3 teacher_model=Qwen/Qwen3-32B`

## Customization

**Multi-teacher:** Pass multiple `DistillationDatasetConfig` objects with different teachers:
```python
config = train_on_policy.Config(
    dataset_configs=[math_dataset_config, chat_dataset_config],
    ...
)
```
See the [multi-teacher recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation/on_policy_multi_teacher.py) for a full example.

**Off-policy reasoning:** Use standard SFT (see `/tinker-sft`) on teacher-generated traces like OpenThoughts3. See the [off-policy recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation/off_policy_reasoning.py).

**Custom prompt dataset:** Subclass `PromptOnlyDatasetBuilder` or create a custom `RLDatasetBuilder` that returns `EnvGroupBuilder` objects.

For testing and weight export patterns, see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook).

## Common pitfalls
- Teacher model must be compatible with student's tokenizer/renderer
- On-policy is generally better than off-policy but more compute-intensive
- High `kl_penalty_coef` can make training too conservative; start with 1.0
- For multi-teacher, balance `groups_per_batch` across datasets
