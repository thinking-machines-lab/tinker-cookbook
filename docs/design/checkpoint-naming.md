# Design: Consistent Checkpoint Naming

**Status:** Draft
**Date:** 2026-03-12
**Triggered by:** Customer feedback on `rl/train.py` final checkpoint naming

---

## Problem

Today, tinker_cookbook uses two different naming conventions for checkpoints:

- **Periodic checkpoints** (mid-training): zero-padded step number, e.g. `000010`, `000020`, `000030`
- **Final checkpoint** (end of training): the literal string `"final"`

This creates several issues for users:

1. **Lost step information.** Given a checkpoint named `final`, there's no way to know how many batches/steps produced it without opening `checkpoints.jsonl` and reading the `loop_state`.
2. **Ambiguous on resume.** When a run is resumed and completes, the new `"final"` overwrites the old `"final"` on the Tinker service. Users can't distinguish between the original and resumed final checkpoint from the name alone.
3. **Inconsistency.** Periodic and final checkpoints follow different naming conventions, making it harder to reason about checkpoint sequences programmatically.

### Customer's suggestion

Use the numeric batch number consistently for all checkpoints, including the final one:

```
tinker://44754624-2e95-5b79-85ae-6ce5ea1097ac:train:0/sampler_weights/000030
```

---

## Current State

### Code: 6 call sites save `name="final"`

| # | File | Line | Sync/Async | `ttl_seconds` |
|---|------|------|------------|---------------|
| 1 | `rl/train.py` | 1399 | async | `None` |
| 2 | `supervised/train.py` | 374 | async | `None` |
| 3 | `preference/train_dpo.py` | 393 | sync | `config.ttl_seconds` * |
| 4 | `recipes/sl_loop.py` | 156 | sync | `None` |
| 5 | `recipes/rl_loop.py` | 244 | sync | `None` |
| 6 | `distillation/train_on_policy.py` | 468 | async | _(not passed)_ * |

\* DPO (#3) and distillation (#6) have TTL inconsistencies — see "Pre-existing issues" below.

### Code: Periodic checkpoints use `f"{step:06d}"`

All periodic saves already use zero-padded 6-digit step numbers. The step variable differs per file (`submitted.step`, `step`, `batch_idx`, `i_batch`) but the format is always `f"{var:06d}"`.

### Resume logic: Not affected by name

`get_last_checkpoint()` in `checkpoint_utils.py` reads the last line of `checkpoints.jsonl` and uses `loop_state["batch"]` to determine resume position. It does **not** inspect the `name` field. Changing the name won't break resume.

### Documentation and user-facing references to `"final"`

These locations reference `sampler_weights/final` or `weights/final` as a tinker path:

| Location | Context |
|----------|---------|
| `docs/download-weights.mdx:23` | SDK download example: `"tinker://<unique_id>/sampler_weights/final"` |
| `tinker_cookbook/chat_app/README.md:13,19,41` | Chat app CLI usage examples |
| `tinker_cookbook/recipes/distillation/README.md:35,51,139` | Distillation recipe examples with real run IDs |

Users may also have existing scripts, notebooks, or external tooling that hardcode `sampler_weights/final` paths from past training runs.

---

## Pre-existing Issues (out of scope, but worth noting)

1. **DPO TTL bug** (`preference/train_dpo.py:397`): Final checkpoint uses `config.ttl_seconds` instead of `None`. This means the final DPO checkpoint can expire, unlike all other final checkpoints (which were fixed in commit `83a678f`).

2. **Distillation missing explicit TTL** (`distillation/train_on_policy.py:466-472`): Final checkpoint doesn't pass `ttl_seconds` at all. Falls back to the function default of `None`, so it happens to work, but is inconsistent with the explicit `ttl_seconds=None` used elsewhere.

---

## Recommendation: Numeric name + `"final"` backward-compat save

Save the final checkpoint under its numeric step name (consistent with periodic checkpoints), then also save under `"final"` for backward compatibility.

```python
# Primary: numeric name, consistent with periodic checkpoints
await save_checkpoint_async(
    training_client=training_client,
    name=f"{num_batches:06d}",
    log_path=cfg.log_path,
    kind="both",
    loop_state={"batch": num_batches},
    ttl_seconds=None,
)

# Backward-compat alias: keep "final" so existing paths still resolve
await save_checkpoint_async(
    training_client=training_client,
    name="final",
    log_path=cfg.log_path,
    kind="both",
    loop_state={"batch": num_batches},
    ttl_seconds=None,
)
```

### Why this approach

- **Addresses the feedback.** The primary checkpoint name now carries the step count, visible directly in the tinker:// path.
- **Backward compatible.** Existing `sampler_weights/final` references in user scripts, docs, and tooling continue to work.
- **Low cost.** The extra save is one additional API call at the very end of training — negligible compared to the full training run. This is not a hot path.
- **Gradual migration.** We can deprecate and eventually remove the `"final"` save once users have migrated.

### Duplicate name edge case

If `save_every` divides total steps evenly, the last periodic checkpoint already saved under the same numeric name. The final save would be a no-op overwrite on the Tinker service and add a duplicate line to `checkpoints.jsonl`. This is harmless — but we can add a guard if we prefer clean JSONL:

```python
# Only save numeric if it wasn't already saved as a periodic checkpoint
if save_every == 0 or num_batches % save_every != 0:
    await save_checkpoint_async(..., name=f"{num_batches:06d}", ...)
```

### Implementation scope

**Must change (6 files):**

| File | Final step expression |
|------|-----------------------|
| `rl/train.py` | `num_batches` |
| `supervised/train.py` | `config.num_epochs * n_batches` |
| `preference/train_dpo.py` | `config.num_epochs * n_batches` |
| `recipes/sl_loop.py` | `n_train_batches` |
| `recipes/rl_loop.py` | `n_train_batches` |
| `distillation/train_on_policy.py` | `num_batches` |

**Optional — bundle TTL fixes:**
- `preference/train_dpo.py:397`: Change `ttl_seconds=config.ttl_seconds` → `ttl_seconds=None`
- `distillation/train_on_policy.py`: Add explicit `ttl_seconds=None`

**No doc changes required** — `"final"` paths continue to work.

### Considered and rejected

| Alternative | Why not |
|-------------|---------|
| **Numeric only (no `"final"`)** | Breaks existing `sampler_weights/final` path references in user scripts and docs |
| **Symbolic alias on Tinker service** | Requires backend feature work disproportionate to the problem |
| **Metadata-only (`is_final` flag)** | Still breaks `sampler_weights/final` paths; adds new schema to `checkpoints.jsonl` |
| **Keep `"final"` as-is** | Doesn't address the customer feedback |

---

## Open Questions

1. **Should we add the duplicate-name guard**, or just let the final numeric save be idempotent when it overlaps with the last periodic save?

2. **Should we bundle the DPO/distillation TTL fixes** in the same PR?

3. **Deprecation timeline for `"final"` save** — do we want to log a deprecation warning, or just silently keep it until a future major version?
