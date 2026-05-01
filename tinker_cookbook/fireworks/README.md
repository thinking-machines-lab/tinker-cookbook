# Fireworks Backend

This package wires `tinker-cookbook` training loops to the Fireworks training and inference infrastructure. The setup scripts here provision the trainer job and inference deployment that your RL / SFT / distillation run will then connect to.

## Prerequisites

1. Set your API key:
   ```bash
   export FIREWORKS_API_KEY=...
   ```
2. Pick a training shape. List available shapes with:
   ```bash
   firectl list training-shapes
   ```
   The training shape encodes the model, accelerator type, GPU count, and max context length. The setup scripts auto-derive most other config values from it.

## One-time setup: provision trainer + deployment

Configuration lives in [`fireworks.yaml`](./fireworks.yaml) (RL / SFT). Edit the relevant fields — at minimum `model.name` and `training_infra.training_shape_id` — then run **one** of:

```bash
# RL: creates a policy trainer job, optional reference trainer, and a deployment
python -m tinker_cookbook.fireworks.setup

# SFT: trainer job only, no deployment
python -m tinker_cookbook.fireworks.setup_for_sft


Each script logs the provisioned resources you'll need for the next step:
- The **trainer job ID** (printed as the policy / student endpoint `base_url`).
- The **deployment ID** (printed by the deployment manager).
- The **base model name** (echoed from your YAML).

### Overriding config from the command line

The setup scripts use Hydra, so you can override any field in `fireworks.yaml` with dotted keys. The fields below are the ones you'll typically change when switching models or shapes — change them together to stay consistent.

```bash
python -m tinker_cookbook.fireworks.setup_for_rl \
    model.name=accounts/fireworks/models/qwen3-8b \
    model.lora_rank=0 \
    training_infra.training_shape_id=accounts/fireworks/trainingShapes/qwen3-8b-128k \
    deployment.tokenizer_model=Qwen/Qwen3-8B \
    deployment.replica_count=2 \
    display_name="qwen3-8b-policy"

python -m tinker_cookbook.fireworks.setup_for_sft \
    model.name=accounts/fireworks/models/qwen3-8b \
    model.lora_rank=0 \
    training_infra.training_shape_id=accounts/fireworks/trainingShapes/qwen3-8b-128k \
    display_name="qwen3-8b-policy"
```

Other useful patterns:

```bash
# Use a different config file in the same directory (e.g. fireworks_8b.yaml)
python -m tinker_cookbook.fireworks.setup_for_rl --config-name fireworks_8b

# Point at a config file outside this directory
python -m tinker_cookbook.fireworks.setup_for_rl \
    --config-path /abs/path/to/configs --config-name my_run

```

For repeatable runs, prefer copying `fireworks.yaml` to e.g. `fireworks_my_run.yaml`, edit it, then pass `--config-name fireworks_my_run`.

## Running a training job against the provisioned infra

Pass the three identifiers above to your training entry point. For example, an RL run:

```bash
python -m tinker_cookbook.recipes.math_rl.train \
    base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/<account>/<job-id>" \
    fireworks_deployment_id=<deployment-id> \
    fireworks_base_model_name=accounts/fireworks/models/<model>
```

The same three flags work across the RL / SFT / distillation recipes in `tinker_cookbook/recipes/`.

## Promoting a checkpoint to a deployable model

After training, promote a sampler checkpoint into a Fireworks model artifact you can deploy. The script queries the control plane for the trainer job's checkpoints and promotes one — by default the newest promotable row, or a specific one if you pass `--checkpoint-name`. No temporary trainer is needed; promotion works even after the trainer job has been deleted.

```bash
# Promote the newest promotable checkpoint on a job
python -m tinker_cookbook.fireworks.tools.promote_checkpoint \
    --job-id <trainer-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b

# Promote a specific checkpoint (matches both 'step-50' and 'step-50-a1b2c3d4')
python -m tinker_cookbook.fireworks.tools.promote_checkpoint \
    --job-id <trainer-job-id> \
    --checkpoint-name step-50 \
    --base-model accounts/fireworks/models/qwen3-8b

# Override the auto-generated output model ID
python -m tinker_cookbook.fireworks.tools.promote_checkpoint \
    --job-id <trainer-job-id> \
    --base-model accounts/fireworks/models/qwen3-8b \
    --output-model-id my-fine-tuned-qwen3-8b
```

`--output-model-id` is optional — the script auto-generates a sanitized ID from the base model and checkpoint name if you omit it. `--hot-load-deployment-id` is only needed for legacy jobs that predate the stored-bucket-URL migration; modern runs (cookbook >= 0.3.0) should omit it.

To list what's promotable on a job before promoting, use the sibling tool:

```bash
python -m tinker_cookbook.fireworks.tools.list_checkpoints --job-id <trainer-job-id>
```

## Files

| File | Purpose |
| --- | --- |
| `setup_for_rl.py` | Provision RL infra (policy trainer + optional reference + deployment). |
| `setup_for_sft.py` | Provision SFT infra (trainer job only). |
| `fireworks.yaml` | RL / SFT config. |
| `utils/` | Shared helpers: `ReconnectableClient`, `create_trainer_job`, `setup_deployment`, config dataclasses. |
