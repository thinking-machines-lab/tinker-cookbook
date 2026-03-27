---
name: tinker-dpo
description: Set up and run Direct Preference Optimization (DPO) training on preference datasets using the Tinker API. Use when the user wants to train with preference data, chosen/rejected pairs, or DPO.
---

# Direct Preference Optimization (DPO)

Help the user set up and run DPO training using the Tinker API.

## Key concepts

**DPO parameters:**
- `dpo_beta`: Controls deviation from reference model. **Start with 0.1** (recommended).
  - Lower beta → more aggressive optimization; higher beta → stays closer to reference
- `learning_rate`: Typically **1e-5** for DPO (lower than SFT's ~2e-4)
- `reference_model_name`: Defaults to the base model; set explicitly if different

**Preference datasets** use `DPODatasetBuilderFromComparisons` wrapping a `ComparisonBuilder`:

```python
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=renderer_name,
    max_length=8192,
    batch_size=256,
)
dataset = DPODatasetBuilderFromComparisons(
    common_config=common_config,
    comparison_builder=HHHComparisonBuilder(),
)
```

**Built-in comparison builders:**
- `HHHComparisonBuilder` — Anthropic HHH (Helpful, Harmless, Honest)
- `HelpSteer3ComparisonBuilder` — NVIDIA HelpSteer3
- `UltraFeedbackComparisonBuilder` — UltraFeedback

All from `tinker_cookbook.recipes.preference.datasets`.

## Minimal working example

This is a complete, runnable DPO script:

```python
import chz

from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import (
    HelpSteer3ComparisonBuilder,
    HHHComparisonBuilder,
    UltraFeedbackComparisonBuilder,
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

Run it: `python my_dpo.py` or `python my_dpo.py dataset=ultrafeedback dpo_beta=0.05`

## Customization

**Custom preference data:** Create a `ComparisonBuilder` that yields `(chosen, rejected)` conversation pairs:

```python
from tinker_cookbook.preference.dpo_datasets import ComparisonBuilder
from tinker_cookbook.renderers.base import Message

class MyComparisonBuilder(ComparisonBuilder):
    async def __call__(self) -> list[tuple[list[Message], list[Message]]]:
        # Return list of (chosen_messages, rejected_messages) pairs
        return [
            (
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good answer"}],
                [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bad answer"}],
            ),
        ]
```

**From SFT checkpoint:** Set `load_checkpoint_path` to your SFT checkpoint path. DPO works best starting from an SFT model rather than a raw base model.

For testing and weight export patterns, see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook).

## Common pitfalls
- **Start with `dpo_beta=0.1`** — well-tested default. Tune from there.
- DPO LR should be **lower than SFT** (1e-5 vs 2e-4)
- DPO works best from an SFT checkpoint, not a raw base model
- Preference data quality matters more than quantity — ensure clear quality differences between chosen/rejected
