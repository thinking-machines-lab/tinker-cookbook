---
name: tinker-cli
description: Guide for the Tinker CLI — managing training runs, checkpoints, downloading weights, and publishing to HuggingFace. Use when the user asks about CLI commands, listing runs, managing checkpoints from the terminal, or uploading to HF Hub.
---

# Tinker CLI

The `tinker` CLI is installed with the Tinker Python SDK. It provides commands for managing training runs and checkpoints from the terminal.

Requires `TINKER_API_KEY` environment variable to be set.

## Global options

```bash
tinker --format table   # Rich table output (default)
tinker --format json    # JSON output (for scripting)
```

## Training runs

```bash
# List recent training runs
tinker run list
tinker run list --limit 50

# Show details for a specific run
tinker run info <RUN_ID>

# Custom columns
tinker run list --columns id,model,lora,updated,status,checkpoint
```

Available columns: `id`, `model`, `owner`, `lora`, `updated`, `status`, `checkpoint`, `checkpoint_time`.

## Checkpoints

### List and inspect

```bash
# List checkpoints for a specific run
tinker checkpoint list --run-id <RUN_ID>

# List all your checkpoints across runs
tinker checkpoint list
tinker checkpoint list --limit 50

# Show checkpoint details
tinker checkpoint info <TINKER_PATH>
```

### Download

```bash
# Download and extract a checkpoint
tinker checkpoint download <TINKER_PATH>
tinker checkpoint download <TINKER_PATH> --output ./my-adapter
tinker checkpoint download <TINKER_PATH> --force  # Overwrite existing
```

### Visibility

```bash
# Make a checkpoint publicly accessible
tinker checkpoint publish <TINKER_PATH>

# Make a checkpoint private
tinker checkpoint unpublish <TINKER_PATH>
```

### TTL (expiration)

```bash
# Set checkpoint to expire in 24 hours
tinker checkpoint set-ttl <TINKER_PATH> --ttl 86400

# Remove expiration (keep indefinitely)
tinker checkpoint set-ttl <TINKER_PATH> --remove
```

### Delete

```bash
# Delete checkpoints (with confirmation prompt)
tinker checkpoint delete <TINKER_PATH>

# Delete without confirmation
tinker checkpoint delete <TINKER_PATH> -y

# Delete multiple
tinker checkpoint delete <PATH1> <PATH2> <PATH3>
```

### Upload to HuggingFace Hub

```bash
# Push checkpoint to HuggingFace
tinker checkpoint push-hf <TINKER_PATH> --repo user/my-model

# Push as public repo
tinker checkpoint push-hf <TINKER_PATH> --repo user/my-model --public

# Advanced options
tinker checkpoint push-hf <TINKER_PATH> \
    --repo user/my-model \
    --revision main \
    --commit-message "Upload fine-tuned model" \
    --create-pr \
    --no-model-card
```

Options: `--repo`, `--public`, `--revision`, `--commit-message`, `--create-pr`, `--allow-pattern`, `--ignore-pattern`, `--no-model-card`.

## Version

```bash
tinker version   # e.g. "tinker 0.15.0"
```

## Common patterns

### Script-friendly output
```bash
# Get checkpoint paths as JSON for scripting
tinker checkpoint list --format json | jq '.[].tinker_path'

# Get run IDs
tinker run list --format json | jq '.[].id'
```

### Typical workflow
```bash
# 1. Find your training run
tinker run list

# 2. List checkpoints for that run
tinker checkpoint list --run-id <RUN_ID>

# 3. Download the final checkpoint
tinker checkpoint download tinker://<RUN_ID>/sampler_weights/final -o ./adapter

# 4. Or push directly to HuggingFace
tinker checkpoint push-hf tinker://<RUN_ID>/sampler_weights/final --repo user/my-model
```

## Common pitfalls
- `TINKER_API_KEY` must be set — the CLI reads it from the environment
- Checkpoint paths use the format `tinker://<run-id>/<type>/<checkpoint-id>`
- `push-hf` uploads the raw checkpoint — for merged HF models, use `weights.build_hf_model()` in Python first (see `/tinker-weights` skill)
- `delete` is permanent and irreversible — use `-y` flag carefully
