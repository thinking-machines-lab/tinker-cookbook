# Cross-Account Checkpoint Copy

Self-serve copy of a Tinker checkpoint from one account into another. Useful
when a checkpoint is produced under account A (e.g. a research org) and a
sampler / evaluator runs under account B (e.g. a separate dev org), and you
do not want B's runtime code to carry A's API key.

## How it works

The Tinker SDK supports passing a `weights_access_token` to weight loads
(`TrainingClient.load_state(path, weights_access_token=...)`). The destination
service uses the forwarded token to read the source weights, then creates a
fresh LoRA training run owned by the destination account. Re-saving the loaded
weights produces new `tinker://` paths that live entirely in the destination
account.

## Security model

- **`source-api-key`** is sent (over TLS) to the Tinker service in the
  load-weights request body. Use the narrowest-scoped key you can; rotate it
  after the copy if it grants more than this single read.
- **`destination-api-key`** stays client-side and authenticates as the destination
  account. The source account never sees it.
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

`--output-kind` controls what gets materialized in the destination account:

| `--output-kind` | Saves                    | Use case                                |
|-----------------|--------------------------|-----------------------------------------|
| `sampler`       | `sampler_weights/<name>` | Default. Feed `create_sampling_client`. |
| `training`      | `weights/<name>`         | Resume training in the dest account.    |

`--output-name` overrides the auto-derived checkpoint name (the trailing
component of `--source-path`).

The script attaches `copied_from_path=<source-path>` to the new training
run's metadata so you can trace provenance later.

## Caveats

- **Source path must be `weights/...`, not `sampler_weights/...`.** The
  cross-account `load_state` API only consumes trainable state (produced by
  `save_state`), not sampler-only checkpoints (produced by
  `save_weights_for_sampler`). If the source account only saved sampler
  weights, ask them to also call `save_state(name)` on the same training run
  and pass that path. There is no public-SDK workaround today —
  `create_sampling_client` does not accept `weights_access_token`.
- Each call creates a **new training run** in the destination account, so the
  destination `tinker://` paths have a fresh run id (different from source).
- The SDK currently assumes LoRA training runs (`assert weights_info.is_lora`).
  Non-LoRA sources will fail at load time.
- `--output-kind training` from this script loads weights only — the resulting
  destination `weights/...` checkpoint has fresh (zeroed) optimizer state, not
  the source's optimizer moments.
- The destination account needs billing enabled (and capacity) to spin up a
  training client for the source's base model.
