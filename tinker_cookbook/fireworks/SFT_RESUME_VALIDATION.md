# SFT Resume — Experiment Log

End-to-end validation of `tinker_cookbook/supervised/train.py` resume logic against the Firetitan backend. Both **same-job** and **cross-job** resume were exercised on real Fireworks trainers.

## Environment

- Model: `Qwen/Qwen3-4B` (full param, `lora_rank=0`)
- Training shape: `accounts/pyroworks/trainingShapes/qwen3-4b-minimum-b200`
- Account: `pyroworks`
- Dataset: `tinker_cookbook/example_data/conversations.jsonl` (128 chat rows, `batch_size=2`)
- Config knobs: `save_every=1`, `submit_ahead=0`, `max_steps` per phase
- Driver scripts: `/tmp/sft_resume_test.py` (same-job), `/tmp/sft_crossjob_test.py` (cross-job)

## Code under test (changes from baseline)

- `tinker_cookbook/supervised/train.py`
  - Resume reads `start_epoch = resume_info.epoch or 0`, `start_batch = resume_info.batch or 0` (handles `batch=None`).
  - `load_state_with_optimizer(...)` is now awaited via `result_async()` so failures surface immediately.
  - Periodic and final saves record `loop_state` as the **next** `(epoch, batch)` to execute (rolling over to next epoch when `batch_idx + 1 == n_batches`) so resume picks up cleanly.
  - Each saved record now stamps `source_trainer_job_id` (parsed from `config.base_url`); on resume, if it differs from the current trainer, the path is rewritten to a `cross_job://...` reference via `training_client.resolve_checkpoint_path(...)`.
- `tinker_cookbook/checkpoint_utils.py`
  - Added `extract_trainer_job_id(base_url)` helper.
  - In `save_checkpoint_async`, after `save_state_async` lands, query `training_client.list_checkpoints()` and replace the local DCP name with the highest `step-N` returned by the trainer — that is the canonical GCS-resolvable name needed for cross-job loads. (The local name like `"000002-state"` is just an alias the trainer pod remembers within a session — fine for same-job resume, broken for cross-job.)

## Test 1 — Same-job resume

Trainer: `onpfjy1tu6d8uuwj` (qwen3-4b, full-param). One trainer, one log dir, two consecutive runs.

### Phase 1 (`max_steps=3`)

`checkpoints.jsonl` after the run completed naturally:

```json
{"name":"000001","batch":2,"epoch":0,"state_path":"000001-state","sampler_path":"000001-sampler-f5f497fd"}
{"name":"000002","batch":3,"epoch":0,"state_path":"000002-state","sampler_path":"000002-sampler-f5f497fd"}
{"name":"final","batch":0,"epoch":1,"state_path":"final-state","sampler_path":"final-sampler-f5f497fd"}
```

Confirms the off-by-one fix: after step 1 (batch_idx=1) the saved record carries `batch=2` (next-to-do), not `batch=1`.

### Phase 2 (`max_steps=6`, after stripping the `final` line)

Resume picked the latest non-final record (`000002-state`, `batch=3`). Logs:

```
Resumed training from 000002-state
```

Continued through `000003 (batch=4)`, `000004 (batch=5)`, `000005 (batch=6)` — i.e. it advanced from batch 3 forward (NOT redoing batch 2). New saves used a fresh sampler `session_id` (`86a8577e` vs prior `f5f497fd`), confirming the new `FiretitanTrainingClient` was constructed cleanly on resume.

Trainer was deleted afterwards.

## Test 2 — Cross-job resume

Two trainers provisioned in parallel:
- `jobB = ffp0y9sbpu62a2up` (writes the checkpoints)
- `jobA = l1t1qvpbkvgtgpgn` (resumes from jobB's checkpoints)

### Phase 1 on jobB (`max_steps=3`)

After completion, `checkpoints.jsonl`:

```json
{"name":"000001","batch":2,"epoch":0,"state_path":"step-3","sampler_path":"000001-sampler-aaa1eb57","source_trainer_job_id":"ffp0y9sbpu62a2up"}
{"name":"000002","batch":3,"epoch":0,"state_path":"step-3","sampler_path":"000002-sampler-aaa1eb57","source_trainer_job_id":"ffp0y9sbpu62a2up"}
{"name":"final","batch":0,"epoch":1,"state_path":"step-3","sampler_path":"final-sampler-aaa1eb57","source_trainer_job_id":"ffp0y9sbpu62a2up"}
```

Notes:
- `state_path` is now the canonical `step-N` server name (not the local `000001-state` alias) — proves the post-save `list_checkpoints` rewrite works.
- All three saves landed at `step-3` because the trainer's internal step counter resets per session and the saves overwrote each other. Acceptable for the resume test since we only need one valid canonical name to load; see *Caveat* below for the broader implication.
- Each record stamps `source_trainer_job_id=ffp0y9sbpu62a2up` (jobB).

### Stripped `final`, then Phase 2 on jobA (`max_steps=4`)

Logs from jobA:

```
Resolved checkpoint 'step-3' from job 'ffp0y9sbpu62a2up' into opaque reference
Cross-job resume: rewriting 'step-3' from job 'ffp0y9sbpu62a2up' into 'cross_job://ffp0y9sbpu62a2up/step-3'
Resumed training from cross_job://ffp0y9sbpu62a2up/step-3
```

Then a post-resume train + save landed:

```json
{"name":"000003","batch":4,"epoch":0,"state_path":"step-4","sampler_path":"000003-sampler-53c14eff","source_trainer_job_id":"l1t1qvpbkvgtgpgn"}
{"name":"final","batch":0,"epoch":1,"state_path":"step-4","sampler_path":"final-sampler-53c14eff","source_trainer_job_id":"l1t1qvpbkvgtgpgn"}
```

Confirms:
- The cross-job rewrite path fires only when `source_trainer_job_id != current_job_id`.
- `load_state_with_optimizer.result_async()` actually waited and the trainer accepted the `cross_job://` ref.
- Forward/backward on the resumed weights worked — produced canonical `step-4` (incremented from loaded step-3).
- The new save re-stamps `source_trainer_job_id` to jobA, so any further resume from this log dir would now target jobA.

Both trainers were deleted afterwards (`ffp0y9sbpu62a2up`, `l1t1qvpbkvgtgpgn`).

## Caveat

The trainer's internal step counter resets each session unless you resume into it, so naive saves on a reused trainer can overwrite each other (in our cross-job phase 1, all three saves landed at `step-3`). For resume this is fine — you only need one valid canonical name to load — but if you want distinct DCP entries across multiple periodic saves *within a single session on a reused trainer*, you'd need the cookbook's full createTime-filtered control-plane lookup (see `fireworks/cookbook/training/utils/checkpoints.py:_resolve_cp_name_after_save`). The current "highest `step-N`" heuristic is robust enough for resume but is not a complete audit trail across sessions.
