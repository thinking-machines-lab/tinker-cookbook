# Knowledge Distillation

Complete reference for on-policy, off-policy, and multi-teacher distillation.

## Key concepts

**Distillation types:**
- **On-policy** (recommended): Student generates, teacher scores via KL divergence
- **Off-policy reasoning**: SFT on teacher-generated traces (e.g., OpenThoughts3)
- **Multi-teacher**: Different teachers for different datasets

**Core abstractions:**
- `TeacherConfig(base_model, load_checkpoint_path)` — identifies the teacher model
- `PromptOnlyDatasetBuilder(dataset_name, ...)` — loads prompts (built-in: `"deepmath"`, `"tulu3"`)
- `DistillationDatasetConfig(dataset_builder, teacher_config, groups_per_batch)` — binds dataset to teacher

**Key parameters:**
- `kl_penalty_coef`: Weight of KL penalty (default 1.0). The only supervision signal.
- `kl_discount_factor`: Discount for future KL (0.0 = no discount). Increase for longer sequences.
- `group_size`: Rollouts per prompt (default 4)
- `groups_per_batch`: Prompts per batch (default 1024)

## Complete on-policy example

```python
import asyncio
import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig, PromptOnlyDatasetBuilder, TeacherConfig,
)

@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-9B-Base"
    teacher_model: str = "Qwen/Qwen3-8B"
    dataset: str = "deepmath"
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

## Multi-teacher

Pass multiple `DistillationDatasetConfig` objects:
```python
config = train_on_policy.Config(
    dataset_configs=[math_dataset_config, chat_dataset_config],
    ...
)
```

See `tinker_cookbook/recipes/distillation/on_policy_multi_teacher.py`.

## Off-policy reasoning

Use standard SFT on teacher-generated traces. See `tinker_cookbook/recipes/distillation/off_policy_reasoning.py`.

## Code references

- `tinker_cookbook/distillation/train_on_policy.py` — On-policy distillation training
- `tinker_cookbook/distillation/datasets.py` — Distillation datasets
- `tinker_cookbook/recipes/distillation/` — All distillation recipes
