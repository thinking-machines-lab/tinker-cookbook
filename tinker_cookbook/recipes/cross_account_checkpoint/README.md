# Cross-Account Checkpoint Copy

Copy trainable Tinker weights from one account into another without making the
source checkpoint public. This is useful when one account owns a checkpoint and
another account needs its own `tinker://` path for sampling or continued
training.

The script creates a fresh LoRA training run under the destination account,
loads the source `weights/...` checkpoint with `weights_access_token`, then
saves a new destination-owned checkpoint.

## Security model

- **`source-api-key`** is sent to the Tinker service as `weights_access_token`
  so the destination run can load the source checkpoint.
- **`destination-api-key`** authenticates the new training run and saved
  checkpoint.
- Pass each key via an env var so it does not land in shell history.
- No checkpoint bytes flow through the client running the script.

## Usage

```bash
export SRC_TINKER_API_KEY=...   # owns the source checkpoint
export DST_TINKER_API_KEY=...   # owns the new checkpoint after the copy

python -m tinker_cookbook.scripts.copy_checkpoint \
    --source-path tinker://<run-id>:train:0/weights/<name> \
    --source-api-key "$SRC_TINKER_API_KEY" \
    --destination-api-key "$DST_TINKER_API_KEY"
# sampler_path: tinker://<new-run-id>:train:0/sampler_weights/<name>
```

By default, the script saves a sampler checkpoint. Use
`--output-kind training` to save a trainable `weights/...` checkpoint instead.

| `--output-kind` | Saves                    | Use case                                |
|-----------------|--------------------------|-----------------------------------------|
| `sampler`       | `sampler_weights/<name>` | Default. Feed `create_sampling_client`. |
| `training`      | `weights/<name>`         | Resume training in the dest account.    |

`--output-name` overrides the auto-derived checkpoint name (the trailing
component of `--source-path`).

The script attaches `copied_from_path=<source-path>` to the new training
run's metadata so you can trace provenance later.

## Caveats

- **Source path must be `weights/...`, not `sampler_weights/...`.** If the
  source account only saved sampler weights, ask them to also call
  `save_state(name)` on the same training run and pass that path.
- Each call creates a **new training run** in the destination account, so the
  destination `tinker://` paths have a fresh run id (different from source).
- Only LoRA checkpoints are supported.
- `--output-kind training` saves model weights only; optimizer state is fresh.
- The destination account needs billing enabled (and capacity) to spin up a
  training client for the source's base model.
