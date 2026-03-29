# Tinker CLI

The `tinker` CLI is installed with the Tinker Python SDK. Requires `TINKER_API_KEY`.

## Global options

```bash
tinker --format table   # Rich table output (default)
tinker --format json    # JSON output (for scripting)
```

## Training runs

```bash
tinker run list
tinker run list --limit 50
tinker run info <RUN_ID>
tinker run list --columns id,model,lora,updated,status,checkpoint
```

Available columns: `id`, `model`, `owner`, `lora`, `updated`, `status`, `checkpoint`, `checkpoint_time`.

## Checkpoints

### List and inspect

```bash
tinker checkpoint list --run-id <RUN_ID>
tinker checkpoint list              # All your checkpoints
tinker checkpoint info <TINKER_PATH>
```

### Download

```bash
tinker checkpoint download <TINKER_PATH>
tinker checkpoint download <TINKER_PATH> --output ./my-adapter
tinker checkpoint download <TINKER_PATH> --force
```

### Visibility

```bash
tinker checkpoint publish <TINKER_PATH>
tinker checkpoint unpublish <TINKER_PATH>
```

### TTL (expiration)

```bash
tinker checkpoint set-ttl <TINKER_PATH> --ttl 86400
tinker checkpoint set-ttl <TINKER_PATH> --remove
```

### Delete

```bash
tinker checkpoint delete <TINKER_PATH>
tinker checkpoint delete <TINKER_PATH> -y          # No confirmation
tinker checkpoint delete <PATH1> <PATH2> <PATH3>   # Multiple
```

### Upload to HuggingFace Hub

```bash
tinker checkpoint push-hf <TINKER_PATH> --repo user/my-model
tinker checkpoint push-hf <TINKER_PATH> --repo user/my-model --public
tinker checkpoint push-hf <TINKER_PATH> \
    --repo user/my-model --revision main \
    --commit-message "Upload fine-tuned model" --create-pr --no-model-card
```

## Version

```bash
tinker version
```

## Common patterns

### Script-friendly output

```bash
tinker checkpoint list --format json | jq '.[].tinker_path'
tinker run list --format json | jq '.[].id'
```

### Typical workflow

```bash
tinker run list
tinker checkpoint list --run-id <RUN_ID>
tinker checkpoint download tinker://<RUN_ID>/sampler_weights/final -o ./adapter
tinker checkpoint push-hf tinker://<RUN_ID>/sampler_weights/final --repo user/my-model
```

## Pitfalls

- `push-hf` uploads raw checkpoint — for merged HF models, use `weights.build_hf_model()` in Python first
- `delete` is permanent and irreversible
- Checkpoint paths: `tinker://<run-id>/<type>/<checkpoint-id>`
