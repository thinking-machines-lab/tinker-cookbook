---
name: tinker-rlhf
description: Set up and run the full RLHF pipeline (SFT, reward model training, RL from reward model) using the Tinker API. Use when the user wants to do RLHF, train a reward model, or run the full preference-based RL pipeline.
---

# RL from Human Feedback (RLHF) Pipeline

Help the user set up and run the full 3-stage RLHF pipeline using the Tinker API.

## Key concepts

**Three stages, chained by checkpoints:**
1. **SFT** — Fine-tune base model on instruction data (e.g., NoRobots)
2. **Reward Model (RM)** — Train on preference comparisons (e.g., HHH)
3. **RL** — Optimize the SFT policy using the RM as reward signal

**Key abstractions:**
- `ChatDatasetBuilderFromComparisons` — Wraps preference data for RM training (supervised)
- `PreferenceModelBuilderFromChatRenderer` — Wraps trained RM for use as reward in RL
- `PairwisePreferenceRLDatasetBuilder` — Creates RL environment from preference data + RM
- `checkpoint_utils.get_last_checkpoint(log_path)` — Chains stages by loading previous checkpoints

**Typical hyperparameters:**
- SFT LR: 2e-4, RM LR: 3e-4, RL LR: 1e-5 (RL must be much lower)
- `lora_rank`: 64, `batch_size`: 256, `rl_group_size`: 4

## Minimal working example

This is a complete, runnable 3-stage RLHF pipeline:

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
    short_name: str = "llama3b"
    run_sft: bool = True
    run_rm: bool = True
    run_rl: bool = True
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
    """Stage 1: Fine-tune base model on instruction data."""
    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.base_model,
        renderer_name=renderer_name,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    config = supervised_train.Config(
        log_path=log_path,
        model_name=cli.base_model,
        renderer_name=renderer_name,
        dataset_builder=NoRobotsBuilder(common_config=common),
        learning_rate=cli.sft_learning_rate,
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=cli.lora_rank,
        max_steps=cli.max_steps,
    )
    asyncio.run(supervised_train.main(config))


def train_rm(cli: CLIConfig, log_path: str):
    """Stage 2: Train reward model on preference comparisons."""
    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.base_model,
        renderer_name=renderer_name,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
    )
    config = supervised_train.Config(
        log_path=log_path,
        model_name=cli.base_model,
        renderer_name=renderer_name,
        dataset_builder=ChatDatasetBuilderFromComparisons(
            common_config=common, comparison_builder=HHHComparisonBuilder()
        ),
        learning_rate=cli.rm_learning_rate,
        lr_schedule="linear",
        num_epochs=1,
        lora_rank=cli.lora_rank,
        max_steps=cli.max_steps,
    )
    asyncio.run(supervised_train.main(config))


async def train_rl(cli: CLIConfig, log_path: str, sft_log: str, rm_log: str):
    """Stage 3: RL from reward model."""
    sft_ckpt = checkpoint_utils.get_last_checkpoint(sft_log)
    rm_ckpt = checkpoint_utils.get_last_checkpoint(rm_log)
    assert sft_ckpt and rm_ckpt, "SFT and RM checkpoints required"

    renderer_name = model_info.get_recommended_renderer_name(cli.base_model)
    preference_model = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name,
        model_name=cli.base_model,
        rm_weights_path=rm_ckpt.sampler_path,
    )
    config = train.Config(
        model_name=cli.base_model,
        renderer_name=renderer_name,
        dataset_builder=preference_envs.PairwisePreferenceRLDatasetBuilder(
            comparison_builder=HHHComparisonBuilder(),
            policy_renderer_name=renderer_name,
            policy_model_name=cli.base_model,
            preference_model_builder=preference_model,
            batch_size=cli.batch_size,
            group_size=cli.rl_group_size,
            tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        ),
        load_checkpoint_path=sft_ckpt.state_path,
        learning_rate=cli.rl_learning_rate,
        max_tokens=cli.rl_max_tokens,
        lora_rank=cli.lora_rank,
        log_path=log_path,
        max_steps=cli.max_steps,
    )
    await train.main(config)


def cli_main(cli_config: CLIConfig):
    root = Path(f"/tmp/tinker-examples/rlhf-{cli_config.short_name}")
    if cli_config.run_sft:
        sft_stage(cli_config, str(root / "sft"))
    if cli_config.run_rm:
        train_rm(cli_config, str(root / "rm"))
    if cli_config.run_rl:
        asyncio.run(train_rl(cli_config, str(root / "rl"), str(root / "sft"), str(root / "rm")))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
```

Run it: `python my_rlhf.py` or skip stages: `python my_rlhf.py run_sft=False run_rm=False`

## Customization

**Different preference data:** Replace `HHHComparisonBuilder` with `HelpSteer3ComparisonBuilder` or `UltraFeedbackComparisonBuilder` (from `tinker_cookbook.recipes.preference.datasets`), or write a custom `ComparisonBuilder`.

**Add RM evaluation:** Use `ComparisonEvaluator` to evaluate RM quality on a held-out test set. See the [full RLHF recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/preference/rlhf) for evaluator setup.

For testing and weight export patterns, see the [tinker-cookbook repo](https://github.com/thinking-machines-lab/tinker-cookbook).

## Common pitfalls
- RL learning rate must be **much lower** than SFT (1e-5 vs 2e-4)
- Checkpoints flow between stages: SFT → RL policy init, RM → RL reward scoring
- Use `checkpoint_utils.get_last_checkpoint()` to find checkpoints from previous stages
- RM quality directly impacts RL — validate RM before running Stage 3
