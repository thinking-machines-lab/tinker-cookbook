---
name: dpo
description: Set up and run Direct Preference Optimization (DPO) training on preference datasets using the Tinker API. Use when the user wants to train with preference data, chosen/rejected pairs, or DPO.
argument-hint: "[model-name] [dataset]"
---

# Direct Preference Optimization (DPO)

Help the user set up and run DPO training using the Tinker API.

## Step 1: Understand the request

Ask the user (if not already specified):
- **Model**: Which model to train (e.g., `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-8B`)
- **Dataset**: Which preference dataset — built-in (HHH, HelpSteer3, UltraFeedback) or custom
- **Starting checkpoint**: Train from base model or from an SFT checkpoint

## Step 2: Reference existing recipes

Read these files for patterns:
- `tinker_cookbook/recipes/preference/dpo/train.py` — DPO CLI with built-in datasets
- `tinker_cookbook/preference/train_dpo.py` — Core DPO training loop
- `tinker_cookbook/preference/dpo_datasets.py` — DPO dataset builders
- `tinker_cookbook/recipes/preference/datasets.py` — HHH, HelpSteer3, UltraFeedback builders
- `docs/preferences/dpo-guide.mdx` — DPO guide

## Step 3: Configure the training run

### Key Parameters

- `dpo_beta`: Controls how much the model deviates from reference. **Start with 0.1** (recommended default).
  - Lower beta = more deviation from reference (more aggressive optimization)
  - Higher beta = stays closer to reference (more conservative)
- `learning_rate`: Typically **1e-5** for DPO (lower than SFT)
- `lr_schedule`: `"linear"` decay is standard
- `batch_size`: Number of tokens per batch (default: 256)
- `max_length`: Maximum sequence length (default: 8192)
- `reference_model_name`: Explicit reference model (defaults to the base model)

### Preference Datasets

**Built-in:**
- `"hhh"` — Anthropic HHH (Helpful, Harmless, Honest) comparisons
- `"helpsteer3"` — NVIDIA HelpSteer3 preference data
- `"ultrafeedback"` — UltraFeedback preference data

**Custom:** Create a `ComparisonBuilder` that yields `(chosen, rejected)` conversation pairs. See `recipes/preference/datasets.py` for examples.

### Dataset Construction
```python
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder

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

## Step 4: Write the training script

Follow the pattern from `recipes/preference/dpo/train.py`:

```python
import chz
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

config = train_dpo.Config(
    log_path="/tmp/tinker-examples/dpo/my_run",
    model_name="meta-llama/Llama-3.2-1B",
    renderer_name=renderer_name,
    dataset_builder=dataset,
    learning_rate=1e-5,
    lr_schedule="linear",
    dpo_beta=0.1,
    reference_model_name=None,  # Uses base model as reference
    load_checkpoint_path=None,  # Or path to SFT checkpoint
)

train_dpo.main(config)
```

## Step 5: Run

```bash
# Basic DPO with HHH dataset
python -m tinker_cookbook.recipes.preference.dpo.train dataset=hhh

# With different model and dataset
python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name=meta-llama/Llama-3.1-8B \
    dataset=ultrafeedback \
    dpo_beta=0.1 \
    learning_rate=1e-5

# From an SFT checkpoint
python -m tinker_cookbook.recipes.preference.dpo.train \
    load_checkpoint_path=/tmp/tinker-examples/sft/checkpoint_100
```

## Step 6: Add tests

If you created a new DPO recipe, add a smoke test:

```python
# tests/recipes/test_recipe_<name>.py
import pytest
from tests.helpers import run_recipe

@pytest.mark.integration
def test_<recipe_name>():
    run_recipe(
        "tinker_cookbook.recipes.<recipe_name>.train",
        ["behavior_if_log_dir_exists=delete"],
    )
```

`run_recipe()` automatically passes `max_steps=2` so the recipe runs 2 training steps and exits. See `tests/recipes/test_recipe_dpo.py` for the existing example.

## Step 7: Export weights (optional)

After DPO training, export weights using the `tinker_cookbook.weights` API:

```python
from tinker_cookbook import weights

adapter_dir = weights.download(tinker_path="tinker://run-id/sampler_weights/final", output_dir="./adapter")
weights.build_hf_model(base_model="meta-llama/Llama-3.2-1B", adapter_path=adapter_dir, output_path="./model")
weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-dpo-model")
```

## Common pitfalls
- **Start with `dpo_beta=0.1`** — this is well-tested. Tune from there.
- DPO LR should be **lower than SFT** (1e-5 vs 2e-4)
- DPO works best when starting from an SFT checkpoint, not a raw base model
- Reference model defaults to the base model — set `reference_model_name` explicitly if you want a different reference
- Preference data quality matters more than quantity — ensure chosen/rejected pairs have clear quality differences
