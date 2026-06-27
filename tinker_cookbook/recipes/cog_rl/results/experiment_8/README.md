# Experiment 8 — self-distillation (rejection sampling) + GRPO

**Date:** 2026-06-27
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**SFT checkpoint:** `tinker://6ab8ed3d-3529-5b4c-892e-6c936243e203:train:0/sampler_weights/exp8sft-self`
**Final (SFT+GRPO) checkpoint:** `tinker://76ae26df-105a-5fcc-9108-cd2182263f90:train:0/sampler_weights/final` (project `8a4e2d8a-…`, label `exp8`)

## Why this experiment

Experiment 7 showed the corpus task source scales, but pure GRPO from base plateaued at
0.364 held-out: sparse reward makes the model spend many steps just discovering the OOD
surface syntax. The fix is an SFT warm-start so RL starts from a model that already writes
valid Cog. The question is **where the SFT data comes from**. Distilling from a frontier
model (gpt-5.5) is not self-contained and bumps against its terms of use, so the training
pipeline must not depend on it.

## Method: self-distillation (no external model)

The verifier, not a teacher, is what makes SFT data trustworthy. So the *open model we are
training* generates the candidate solutions and the interpreter + hidden tests decide which
to keep (rejection-sampling fine-tuning / STaR / expert iteration):

1. **Harvest.** Serve the best open checkpoint so far (exp7) and sample each of the 561
   corpus train tasks 6 times through the harness (`curate.py`). Keep the first program that
   passes the hidden tests. Yield: **316/561 (56.3%) verified gold solutions**, all written
   by the model itself (`self_distilled_gold.jsonl`).
2. **SFT.** Fine-tune the base on those 316 gold solutions (`sft.py`, cross-entropy on the
   assistant program, 3 epochs, lr 1e-4). mean NLL 0.153 -> 0.044.
3. **GRPO.** Warm-start `train.py` from the SFT checkpoint (`init_state_path=...`) and run
   100 batches of GRPO on the corpus.

No frontier model anywhere in the training path. gpt-5.5 appears only as the frontier eval
baseline below.

## Result (held-out, 99 distinct MBPP problems)

| stage | pass@1 | mean reward |
|---|---|---|
| base + handbook | 0.283 | 0.467 |
| exp7 (GRPO only, from base) | 0.364 | 0.665 |
| self-distilled SFT only | 0.394 | 0.655 |
| **exp8 (self-distilled SFT + GRPO)** | **0.495** | **0.708** |
| gpt-5.5 + handbook (frontier reference) | 0.751 | 0.845 |

Every completion is a real `solve(...)` function; 0 constants.

## Verdict

- **Distill-then-RL compounds, and it's fully self-contained.** exp8 at 0.495 is +0.21 over
  base, and beats both pieces alone (GRPO-only 0.364, SFT-only 0.394). The SFT warm-start
  gives RL a model that already writes valid Cog, so the GRPO steps buy correctness instead
  of syntax discovery.
- **Expert iteration works.** SFT on exp7's *own* verified outputs (0.394) already beats the
  exp7 checkpoint it was harvested from (0.364). The model improves off its own correct
  solutions, with the interpreter as the only judge. This is a loop: each RL round raises the
  solve rate, which yields more gold for the next round.
- **The gap to the frontier is real but closing.** From base, exp8 closes ~45% of the
  distance to gpt-5.5 ((0.495-0.283)/(0.751-0.283)) at ~100x lower serving cost, with a
  pipeline that needs no frontier model to train.

## Takeaways → next

- **Iterate the loop.** Re-harvest gold from exp8 (it solves more than exp7), SFT, GRPO
  again. Expert iteration should keep lifting the solve rate until it saturates.
- More harvest attempts and `task_source=both` (corpus + the guaranteed-solvable families)
  would grow and de-noise the gold set.
- Per-task pass@1 here is single-sample; `--repeat 5` would tighten the headline number.

## Note: environment recovery

The devbox was recycled mid-experiment; the uv-managed Python under `/tmp` was wiped, which
broke the venv (its `bin/python` symlink dangled). The cp312 site-packages survived on
persistent disk, so the fix was to repoint `.venv/bin/python*` at the system
`/usr/bin/python3.12` (ABI-compatible) and set `home = /usr/bin` in `pyvenv.cfg` — no
reinstall. Tinker checkpoints are server-side and were unaffected.
