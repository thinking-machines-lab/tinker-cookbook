---
name: preferences
description: Set up and run preference-based training — DPO (Direct Preference Optimization) and RLHF (RL from Human Feedback) pipelines using the Tinker API. Use when the user wants to train with preference data, chosen/rejected pairs, DPO, reward models, RLHF, or human feedback alignment.
---

# Preference-Based Training

DPO and RLHF for aligning models with human preferences.

## DPO (Direct Preference Optimization)

DPO directly optimizes a policy from preference pairs without a separate reward model.

### Key parameters

- `dpo_beta`: Controls deviation from reference model. **Start with 0.1**.
- `learning_rate`: Typically **1e-5** (lower than SFT's ~2e-4)
- `reference_model_name`: Defaults to base model

### Built-in comparison datasets

- `HHHComparisonBuilder` — Anthropic HHH (Helpful, Harmless, Honest)
- `HelpSteer3ComparisonBuilder` — NVIDIA HelpSteer3
- `UltraFeedbackComparisonBuilder` — UltraFeedback

All from `tinker_cookbook.recipes.preference.datasets`.

### Minimal DPO example

```python
import chz

from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import (
    HelpSteer3ComparisonBuilder, HHHComparisonBuilder, UltraFeedbackComparisonBuilder,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "hhh"  # hhh, helpsteer3, or ultrafeedback
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    dpo_beta: float = 0.1
    max_length: int | None = 8192
    batch_size: int = 256
    log_path: str | None = None
    reference_model_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


COMPARISON_BUILDERS = {
    "hhh": HHHComparisonBuilder,
    "helpsteer3": HelpSteer3ComparisonBuilder,
    "ultrafeedback": UltraFeedbackComparisonBuilder,
}


def cli_main(cli_config: CLIConfig):
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
    )
    dataset = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=COMPARISON_BUILDERS[cli_config.dataset](),
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/dpo/{cli_config.dataset}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train_dpo.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        dataset_builder=dataset,
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        dpo_beta=cli_config.dpo_beta,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        reference_model_name=cli_config.reference_model_name,
        max_steps=cli_config.max_steps,
    )
    train_dpo.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
```

### Custom preference data

```python
from tinker_cookbook.preference.dpo_datasets import ComparisonBuilder
from tinker_cookbook.renderers.base import Message

class MyComparisonBuilder(ComparisonBuilder):
    async def __call__(self) -> list[tuple[list[Message], list[Message]]]:
        return [
            (
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good answer"}],
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bad answer"}],
            ),
        ]
```

## RLHF Pipeline

Full 3-stage pipeline: SFT -> Reward Model -> RL.

### How it works

1. **SFT** — Fine-tune base model on instruction data (e.g., NoRobots)
2. **Reward Model** — Train on preference comparisons (e.g., HHH) using supervised learning
3. **RL** — Optimize the SFT policy using the RM as reward signal

Stages are chained by checkpoints:
```python
from tinker_cookbook import checkpoint_utils
sft_ckpt = checkpoint_utils.get_last_checkpoint(sft_log_path)
rm_ckpt = checkpoint_utils.get_last_checkpoint(rm_log_path)
```

### Key abstractions

- `ChatDatasetBuilderFromComparisons` — Wraps preference data for RM training
- `PreferenceModelBuilderFromChatRenderer` — Wraps trained RM for use as RL reward
- `PairwisePreferenceRLDatasetBuilder` — Creates RL environment from preference data + RM

### Minimal RLHF example

```python
import asyncio
from pathlib import Path

import chz

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference.preference_datasets import ChatDatasetBuilderFromComparisons
from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer
from tinker_cookbook.recipes.chat_sl.chat_datasets import NoRobotsBuilder
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.rl import preference_envs, train
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


@chz.chz
class CLIConfig:
    base_model: str = "meta-llama/Llama-3.2-3B"
    lora_rank: int = 64
    batch_size: int = 256
    max_length: int = 16384
    sft_learning_rate: float = 2e-4
    rm_learning_rate: float = 3e-4
    rl_learning_rate: float = 1e-5
    rl_max_tokens: int = 1024
    rl_group_size: int = 4
    max_steps: int | None = None


def sft_stage(cli: CLIConfig, log_path: str):
    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.base_model, renderer_name=renderer_name,
        max_length=cli.max_length, batch_size=cli.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    config = supervised_train.Config(
        log_path=log_path, model_name=cli.base_model, renderer_name=renderer_name,
        dataset_builder=NoRobotsBuilder(common_config=common),
        learning_rate=cli.sft_learning_rate, lr_schedule="linear",
        num_epochs=1, lora_rank=cli.lora_rank, max_steps=cli.max_steps,
    )
    asyncio.run(supervised_train.main(config))


def train_rm(cli: CLIConfig, log_path: str):
    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.base_model, renderer_name=renderer_name,
        max_length=cli.max_length, batch_size=cli.batch_size,
    )
    config = supervised_train.Config(
        log_path=log_path, model_name=cli.base_model, renderer_name=renderer_name,
        dataset_builder=ChatDatasetBuilderFromComparisons(
            common_config=common, comparison_builder=HHHComparisonBuilder()
        ),
        learning_rate=cli.rm_learning_rate, lr_schedule="linear",
        num_epochs=1, lora_rank=cli.lora_rank, max_steps=cli.max_steps,
    )
    asyncio.run(supervised_train.main(config))


async def train_rl(cli: CLIConfig, log_path: str, sft_log: str, rm_log: str):
    sft_ckpt = checkpoint_utils.get_last_checkpoint(sft_log)
    rm_ckpt = checkpoint_utils.get_last_checkpoint(rm_log)
    assert sft_ckpt and rm_ckpt, "SFT and RM checkpoints required"

    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    preference_model = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name, model_name=cli.base_model,
        rm_weights_path=rm_ckpt.sampler_path,
    )
    config = train.Config(
        model_name=cli.base_model, renderer_name=renderer_name,
        dataset_builder=preference_envs.PairwisePreferenceRLDatasetBuilder(
            comparison_builder=HHHComparisonBuilder(),
            policy_renderer_name=renderer_name, policy_model_name=cli.base_model,
            preference_model_builder=preference_model,
            batch_size=cli.batch_size, group_size=cli.rl_group_size,
            tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        ),
        load_checkpoint_path=sft_ckpt.state_path,
        learning_rate=cli.rl_learning_rate, max_tokens=cli.rl_max_tokens,
        lora_rank=cli.lora_rank, log_path=log_path, max_steps=cli.max_steps,
    )
    await train.main(config)


def cli_main(cli_config: CLIConfig):
    root = Path(f"/tmp/tinker-examples/rlhf")
    sft_stage(cli_config, str(root / "sft"))
    train_rm(cli_config, str(root / "rm"))
    asyncio.run(train_rl(cli_config, str(root / "rl"), str(root / "sft"), str(root / "rm")))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
```

## Common pitfalls

- **Sequential API calls**: In custom DPO/RLHF loops, always use `_async` variants and overlap GPU work with data prep. Sequential `.result()` chains waste GPU cycles. The built-in training loops already handle this, but custom evaluation or reward scoring code should use `asyncio.gather()` for concurrent execution.
- **Sampler desync**: After saving weights between RLHF stages, create a **new** SamplingClient. A stale client silently uses old weights — especially dangerous when chaining SFT -> RM -> RL.
- **DPO beta**: Start with 0.1 — well-tested default
- **DPO LR**: Should be lower than SFT (1e-5 vs 2e-4)
- **DPO from SFT**: Works best from an SFT checkpoint, not raw base model
- **RLHF RL LR**: Must be much lower than SFT (1e-5 vs 2e-4)
- **RLHF checkpoints**: Flow between stages — SFT state -> RL init, RM sampler -> RL reward
- **RM quality**: Validate RM before running Stage 3
- **Preference data quality**: Matters more than quantity

## Code references

- `tinker_cookbook/preference/train_dpo.py` — DPO training Config and loop
- `tinker_cookbook/preference/types.py` — Comparison, LabeledComparison, PreferenceModel
- `tinker_cookbook/preference/dpo_datasets.py` — DPODatasetBuilderFromComparisons
- `tinker_cookbook/preference/preference_datasets.py` — ChatDatasetBuilderFromComparisons
- `tinker_cookbook/rl/preference_envs.py` — PairwisePreferenceRLDatasetBuilder
- `tinker_cookbook/recipes/preference/` — All preference recipes
- `tinker_cookbook/recipes/preference/datasets.py` — Built-in comparison builders
