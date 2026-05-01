# Tinker Cookbook Fireworks Fork

This fork keeps the Tinker Cookbook training abstractions and recipes, with extra support for running them against Firetitan / Fireworks training infrastructure.

The original upstream project is [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

## What This Fork Adds

- Firetitan service client wiring for SFT, RL, and distillation recipes.
- Full-parameter training and LoRA training.
- Support for more model families, including Qwen, GLM5-class models, Gemma4, and MiniMax M2-class models.
- Long-context training shapes, up to 256k context.

## Basic Client Example

```python
import tinker
from fireworks.training.sdk import FiretitanServiceClient

service_client = FiretitanServiceClient(base_url="https://api.fireworks.ai/training/v1/...")
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-4B-Instruct-2507",
    rank=0,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)
```

Use `rank=0` for full-parameter fine-tuning, or a positive rank for LoRA fine-tuning.

## Setup

Detailed setup instructions live in [`tinker_cookbook/fireworks/README.md`](tinker_cookbook/fireworks/README.md).

The short version:

```bash
export FIREWORKS_API_KEY=...

# Provision RL infrastructure.
python -m tinker_cookbook.fireworks.setup_for_rl

# Or provision SFT infrastructure.
python -m tinker_cookbook.fireworks.setup_for_sft
```

After setup, pass the provisioned trainer endpoint and deployment identifiers into the recipe you want to run:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>" \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model_name=accounts/fireworks/models/<model>
```

