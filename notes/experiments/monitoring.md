# Experiment Monitoring Log

## 2026-04-08 06:27 UTC — Experiments Launched

**FIPO** (PID 2175419):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=4096, lr=1e-6, 50 steps
- FIPO config: tau=32, clip=[0.8, 1.2], ppo_clip=0.2
- Log: /tmp/tinker-fipo-math-qwen3-8b

**GRPO Baseline** (PID 2175565):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=4096, lr=1e-6, 50 steps, loss_fn=importance_sampling
- Log: /tmp/tinker-grpo-baseline-qwen3-8b

Both processes started successfully. Waiting for first metrics.
Commit: d65aaf3 (research/fipo branch)

## 2026-04-08 06:40 UTC — Restarted with improved config (v2)

Killed v1 experiments (max_tokens=4096 too small for math reasoning).

**FIPO v2** (PID 2182775):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=16384 (using Qwen3-8B's 32k context window), lr=1e-6, 50 steps
- FIPO config: tau=32, clip=[0.8, 1.2], ppo_clip=0.2
- TINKER_SUBPROCESS_SAMPLING=1 enabled
- Log: /tmp/tinker-fipo-math-v2

**GRPO Baseline v2** (PID 2182853):
- Same config but loss_fn=importance_sampling (standard GRPO)
- Log: /tmp/tinker-grpo-baseline-v2

Monitoring cron set up (every 30 min).

## 2026-04-08 06:55 UTC — First Metrics

**FIPO** (step 2/50): reward=0.613, correct=62.5%, avg_tokens=10425, time=336s/step
- influence_weight=0.987±0.161, future_kl_abs=0.1228
- Generating much longer responses than baseline (10k vs 3k tokens)

**GRPO** (step 0/50): reward=1.0, correct=100%, avg_tokens=3314, time=1040s/step
- First batch only; likely easy problems at start

Both running well. FIPO producing significantly longer reasoning chains.

## 2026-04-08 07:10 UTC — Switched to DeepMath-103K

Killed v2 experiments (Hendrycks MATH too easy — 91%+ base accuracy).
Aligned FIPO loss with reference implementation (asymmetric clip, dual-clip, sequence filtering).

**FIPO v3** (PID 2218588):
- Model: Qwen/Qwen3-8B, env=deepmath (DeepMath-103K), group_size=8, groups_per_batch=16
- max_tokens=16384, lr=1e-6, 50 steps
- FIPO: tau=32, influence_clip=[0.8, 1.2], ppo_clip=[0.2, 0.28], dual_clip_c=10.0
- TINKER_SUBPROCESS_SAMPLING=1
- Log: /tmp/tinker-fipo-deepmath

**GRPO Baseline v3** (PID 2218666):
- Same config, loss_fn=importance_sampling
- Log: /tmp/tinker-grpo-deepmath

## 2026-04-08 07:45 UTC — First DeepMath Results (step 0-3)

DeepMath is significantly harder than Hendrycks MATH (~70% vs 91% base accuracy).

| Step | FIPO Correct | FIPO Tokens/ep | GRPO Correct | GRPO Tokens/ep |
|------|-------------|----------------|-------------|----------------|
| 0 | 70.8% | 11521 | 88.3% | 10125 |
| 1 | 62.5% | 12011 | 84.4% | 7928 |
| 2 | 67.5% | 11044 | 92.2% | 7023 |
| 3 | 80.4% | 11132 | 94.5% | 8769 |

Notes:
- Accuracy differences likely due to different random training batches
- FIPO generates 30-50% longer responses consistently
- Influence weights stable: ~0.985±0.165
- DeepMath doesn't have a test split — no eval checkpoint available

## 2026-04-08 08:30 UTC — Steps 0-10 Analysis

FIPO (11 steps), GRPO (12 steps). Key findings:

1. **FIPO generates 30-60% longer responses**: avg ~11k tokens vs ~8k for GRPO
   - This matches the paper's main finding of extending reasoning chains
2. **More accuracy variance with FIPO**: 43.8%-87.5% vs 78.9%-98.4% for GRPO
   - Likely due to different random batches + longer exploration
3. **KL divergence very small** for both (~0.0006) — stable training
4. **Influence weights stable**: mean ~0.986, std ~0.165, future_kl_abs ~0.13
5. **No test eval** available — DeepMath doesn't return a test split

## 2026-04-08 08:49 UTC — Steps 0-12, Cron Check

Both alive. FIPO step 11, GRPO step 12.

Averages over all steps:
- **FIPO**: correct=67.1%, tokens/ep=11535
- **GRPO**: correct=88.8%, tokens/ep=8257

GRPO's higher train accuracy likely reflects different random batches (no shared eval).
FIPO consistently generates 40% longer responses.
Influence weights stable at 0.986±0.17, confirming non-trivial reweighting.

Note: participation mask fix committed (negative-advantage only). Current running
experiments use the older code; next run will incorporate the fix.

## 2026-04-08 09:19 UTC — Steps 14-15, Cron Check

Both alive, zero tracebacks.
- **FIPO step 14**: correct=62.5%, tokens/ep=11962, infl=0.982±0.175
- **GRPO step 15**: correct=72.7%, tokens/ep=8598

~30% complete. ETA ~3h for 50 steps.

## 2026-04-08 09:49 UTC — Steps 18-20, Cron Check

Both alive, zero tracebacks.
- **FIPO step 18**: correct=55.0%, tokens/ep=11322, infl=0.992±0.165
- **GRPO step 20**: correct=86.7%, tokens/ep=7978

~40% complete. GRPO running slightly faster (~7min/step vs ~8min/step for FIPO).
FIPO still generating ~42% longer responses consistently.

## 2026-04-08 10:19 UTC — Steps 23-25, Cron Check

Both alive, zero tracebacks.
- **FIPO step 23**: correct=50.0%, tokens/ep=14006, infl=0.986±0.166
- **GRPO step 25**: correct=88.3%, tokens/ep=8629

~50% complete. FIPO responses growing even longer (14k tokens this step).
GRPO ~2 steps ahead due to shorter response generation.

## 2026-04-08 10:49 UTC — Steps 28-29, Cron Check

Both alive, zero tracebacks.
- **FIPO step 28**: correct=70.8%, tokens/ep=7623, infl=0.982±0.174
- **GRPO step 29**: correct=84.4%, tokens/ep=7490

~58% complete. FIPO and GRPO nearly caught up in step count.
Interesting: this step FIPO tokens/ep dropped to ~7.6k (closer to GRPO).
Gap closing — may be batch-dependent variance.

## 2026-04-08 12:19 UTC — All experiments check

All 4 alive, zero tracebacks.

**8B-DeepMath** (nearing completion):
- FIPO step 41: correct=60.4%, tokens/ep=11362, infl=0.988±0.161
- GRPO step 43: correct=86.7%, tokens/ep=9082

**8B-Base-DAPO** (just started, paper-matching setup):
- FIPO step 0: correct=0.0%, tokens/ep=1633, infl=0.981±0.094
- GRPO step 1: correct=0.0%, tokens/ep=1470
- Base model starts at 0% — expected. 8 substeps per iteration now active.

## 2026-04-08 12:32 UTC — 10min check

8B-DeepMath: FIPO step 44, GRPO step 47 (nearly done).
8B-Base-DAPO: both step 5, still 0% correct. FIPO tokens DROPPING (873, was 1633).
**FLAG**: Base model can't solve DAPO-17K → zero advantages → no learning signal.
Need model with existing math capability, or easier dataset.

## 2026-04-08 12:42 UTC — 10min check

8B-DeepMath: FIPO step 46, GRPO step 48 (almost done).
30B-Base-DAPO: both initializing (loading DAPO-17K + 30B model). No metrics yet.
All 4 processes alive, zero tracebacks.

## 2026-04-08 12:52 UTC — 10min check

3 processes (GRPO-8B-DeepMath at step 49, about to finish).
30B-Base-DAPO: still in first rollout (81% for FIPO). 30B + 512 trajectories + 20k tokens = slow.
FIPO-8B step 47, GRPO-8B step 49. Zero tracebacks.

## 2026-04-08 13:02 UTC — 10min check *** FIRST 30B EVAL ***

8B-DeepMath: both at step 49, GRPO finished. 2 processes running.

**30B-Base-DAPO step 0 (baseline, before any training):**
- FIPO: train=0.0%, **EVAL=70.0%**, tokens=803
- GRPO: train=0.0%, **EVAL=69.2%**, tokens=1297

30B-A3B-Base has real math capability (70% MATH-500). Train is 0% because
DAPO-17K competition math is much harder. Now watching for eval improvement
at step 10.

## 2026-04-08 13:22 UTC — 10min check

Restarted 30B with AIME 2024 eval (was using MATH-500 which was too easy).
Both 30B processes alive, zero tracebacks. Still in first batch (AIME eval + 512 rollouts at 20k tokens).
Grading logs show model producing answers — saw "540 against 540" (correct match).

## 2026-04-08 13:32 UTC — 10min check

Both 30B alive, zero tracebacks. First batch rollouts at 94% (FIPO) and 94% (GRPO).
~45s per group for 30B. First metrics with AIME eval imminent.

## 2026-04-08 13:42 UTC — 10min check *** 30B FIRST METRICS ***

**AIME 2024 baseline: 0.0% for both FIPO and GRPO.**
Train correct: 0.0% on DAPO-17K.
FIPO tokens collapsing: 1378→737 (same failure as 8B-Base).

**Problem**: Qwen3-30B-A3B-Base has only 3B active params (MoE).
Not enough math capability for AIME or DAPO-17K competition math.
Need a dense model or an instruct model with existing math ability.

## 2026-04-08 13:52 UTC — 10min check

30B-Base: FIPO step 2, GRPO step 1. Both still 0% correct. FIPO tokens collapsed to ~800.
**Experiment is unproductive** — no learning signal possible with 0% accuracy.
Should kill and switch to a capable model (Qwen3-32B or Qwen3-8B instruct).

## 2026-04-08 14:02 UTC — Killed 30B-Base, launched Qwen3-32B

30B-A3B-Base stuck at 0% (3B active params too weak). Killed.
Launched Qwen3-32B (dense, math-capable) with:
- Train: DAPO-17K, Eval: AIME 2024
- G=16, 32 groups/batch, 8 substeps, 20480 max tokens
- FIPO PID 2512995, GRPO PID 2512997

## 2026-04-08 14:22 UTC — Fixed grading bug, relaunched 30B-Base

**Root cause of 0% correct**: MathEnv used extract_boxed() but DAPO prompts
use "Answer: $Answer" format. Added answer_format="answer_line" extractor.
Relaunched 30B-A3B-Base with sympy grader + correct answer extraction.
FIPO PID 2517511, GRPO PID 2517513. Still in first eval+rollout.

## 2026-04-08 14:32 UTC — 10min check

Both 30B-Base alive, zero tracebacks. In batch 0 rollouts:
FIPO 66% (21/32 groups), GRPO 94% (30/32). First metrics imminent.

## 2026-04-08 14:42 UTC — *** FIRST REAL RESULTS (fixed grading) ***

Grading fix confirmed working — non-zero accuracy on DAPO-17K!

**FIPO step 0**: train=12.8%, AIME=0.0%, tokens=995, infl=1.000±**0.233**
**GRPO step 0**: train=9.0%, AIME=0.0%, tokens=1151
**GRPO step 1**: train=12.3%, tokens=1253

Model solving ~10% of DAPO-17K → gradient signal exists.
AIME=0% expected for base model (paper's 32B gets ~50% after full training).
FIPO infl_std=0.233 — much higher than 8B runs (0.17), 8 substeps working.

## 2026-04-08 14:52 UTC — 10min check

Both alive, zero tracebacks. Accuracy trending up:
- FIPO: 12.8%→15.1% (step 0→1)
- GRPO: 9.0%→12.3%→13.5% (step 0→1→2)
Tokens stable ~1000-1250. Learning signal confirmed.

## 2026-04-08 15:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO: step 2, correct=15.3% (↑ from 12.8%), tokens=1225 (↑ from 995)
- GRPO: step 3, correct=10.4% (fluctuating), tokens=1189
FIPO showing steady improvement in both accuracy and token length.
GRPO more volatile. ~6% complete.

## 2026-04-08 15:12 UTC — 10min check

Both alive, zero tracebacks.
- FIPO: step 3, correct=15.2%, tokens=1121, infl_std=0.253 (growing)
- GRPO: step 5, correct=10.4% (volatile, range 9-14.5%), tokens=1082
FIPO ~2x slower per step (forward_backward_custom overhead).
FIPO accuracy more stable; GRPO fluctuating.

## 2026-04-08 15:22 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 5: correct=15.2%, tokens=1014, infl_std=0.246
- GRPO step 7: correct=12.1%, tokens=975
Both accuracy ~12-15%, plateaued. Tokens ~1000. GRPO step 10 (AIME eval) approaching.

## 2026-04-08 15:32 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 6: correct=15.1%, tokens=1102, infl_std=0.247
- GRPO step 8: correct=6.8%, tokens=1333
GRPO dipped to 6.8% this batch. FIPO more stable at ~15%.
GRPO 2 steps from AIME eval.

## 2026-04-08 15:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 7: correct=**16.7%** (new high!), tokens=1023, infl_std=0.219
- GRPO step 9: correct=10.4%, tokens=1144
FIPO consistently outperforming GRPO on train accuracy (~15-17% vs ~7-14%).
GRPO one step from AIME eval.

## 2026-04-08 15:52 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 9: correct=19.9%, tokens=925, kl=0.00170
- GRPO step 10: correct=13.9%, tokens=1086, **AIME=0.0%**
GRPO AIME eval at step 10 = 0%. FIPO approaching step 10 for its AIME eval.
Train accuracy gap: FIPO ~20% vs GRPO ~14%.

## 2026-04-08 16:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO: in batch 10 rollouts (step 9 metrics last written, step 10 AIME eval done)
- GRPO step 11: correct=14.1%, tokens=1047
FIPO step 10 metrics (with AIME eval) should appear soon.

## 2026-04-08 16:12 UTC — 10min check *** FIPO AIME step 10 ***

Both alive, zero tracebacks.
- FIPO step 10: train=18.0%, **AIME=0.0%**, tokens=1109
- GRPO step 13: train=12.9%, tokens=1325
Both AIME=0% at step 10. Model too small for AIME (3B active params).
Train accuracy gap persists: FIPO ~18% vs GRPO ~13%.
Next AIME eval at step 20.

## 2026-04-08 16:22 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 11: correct=19.0%, tokens=1143
- GRPO step 14: correct=6.8%, tokens=1171
Train accuracy gap widening: FIPO ~19% vs GRPO ~7% this step.
Note: AIME eval via benchmark suite returns 0% due to parse failures — base model
doesn't produce chat-formatted output, so EnvFromMessageEnv.parse_response fails.
The RL training loop handles raw tokens differently and works fine.
Train accuracy on DAPO-17K is the reliable signal for base models.

## 2026-04-08 16:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 12: correct=17.3%, tokens=994
- GRPO step 16: correct=14.6%, tokens=1130
FIPO still higher train accuracy. GRPO ~4 steps ahead (2x speed difference).

## 2026-04-08 16:52 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 13: correct=17.1%, tokens=1212
- GRPO step 17: correct=11.1%, tokens=1030
Consistent pattern: FIPO ~17% vs GRPO ~11% train accuracy.

## 2026-04-08 17:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 14: correct=16.5%, tokens=1166, infl_std=0.230
- GRPO step 18: correct=10.7%, tokens=1140
Pattern holds. ~36% complete for GRPO, ~28% for FIPO.

## 2026-04-08 17:12 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 15: correct=15.4%, tokens=883
- GRPO step 20: correct=5.5%, tokens=909 (GRPO dipped low this batch)
AIME 2025 baseline running separately (concurrency=1, 30 problems).
Confirmed base model CAN solve AIME problems (1/3 in quick test).
Previous 0% was parse failures, not model inability.

## 2026-04-08 17:32 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 17: correct=14.8%, tokens=1074, infl_std=0.230
- GRPO step 22: correct=13.1%, tokens=1234
AIME 2025 baseline confirmed: 1/30 (3.3%) on Qwen3-30B-A3B-Base.
Fixed eval: example_id propagation + grade_fn with concurrency=30.
GRPO approaching step 30 (60% done).

## 2026-04-08 17:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 18: correct=13.1%, tokens=1153
- GRPO step 22: correct=13.1%, tokens=1234
Accuracy converging this step — both at 13.1%. GRPO ~44%, FIPO ~36% done.

## 2026-04-08 18:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 19: correct=20.7% (new high!), tokens=941
- GRPO step 24: correct=14.5%, tokens=1398
FIPO hit 20.7% — highest train accuracy yet. ~48% done for GRPO.
Note: train accuracy not directly comparable (different batches despite same seed).

## 2026-04-08 18:12 UTC — 10min check

Both alive, zero tracebacks.
- FIPO: in batch 20 rollouts (step 20 AIME eval done, waiting for metrics)
- GRPO step 26: correct=14.5%, tokens=1034
GRPO ~52% done. FIPO step 20 metrics (with AIME) imminent.

## 2026-04-08 18:22 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 20: correct=12.5%, tokens=917, AIME=0.0% (parse failures), infl_std=0.263
- GRPO step 26: correct=14.5%, tokens=1034, AIME@20=0.0% (parse failures)
In-training AIME eval still broken (grade_fn fix not wired into training loop).
Standalone eval confirmed 3.3% baseline. ~52% GRPO, ~40% FIPO done.

## 2026-04-08 18:32 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 21: correct=17.9%, tokens=1084
- GRPO step 28: correct=10.9%, tokens=1135
GRPO ~56% done, FIPO ~42% done.

## 2026-04-08 18:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 23: correct=18.8%, tokens=1006
- GRPO step 29: correct=10.0%, tokens=1114
GRPO about to hit step 30 (AIME eval). GRPO ~58%, FIPO ~46% done.

## 2026-04-08 18:52 UTC — 10min check

Both alive, zero tracebacks. No progress since last check (both in slow steps).
- FIPO still step 23, GRPO still step 29 (likely in step 30 AIME eval).

## 2026-04-08 19:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 24: correct=13.4%, tokens=1001, infl_std=0.242
- GRPO step 30: correct=16.6%, tokens=1429
GRPO 60% done, FIPO 48% done.

## 2026-04-08 19:12 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 26: correct=17.4%, tokens=955
- GRPO step 32: correct=14.3%, tokens=1283
GRPO 64% done, FIPO 52% done.

## 2026-04-08 19:22 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 26: correct=17.4%, tokens=955 (no change — in slow step)
- GRPO step 33: correct=11.3%, tokens=1281
GRPO 66% done, FIPO 52% done.

## 2026-04-08 19:32 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 28: correct=15.9%, tokens=978, infl_std=0.246
- GRPO step 35: correct=15.6%, tokens=1066
Very close this check. GRPO 70% done, FIPO 56% done.

## 2026-04-08 19:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 29: correct=20.2%, tokens=917, infl_std=0.244
- GRPO step 37: correct=13.3%, tokens=1104
FIPO back to ~20%. GRPO 74% done, FIPO 58% done.

## 2026-04-08 19:52 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 30: correct=**22.8%** (new high!), tokens=1088, infl_std=0.245
- GRPO step 38: correct=**21.1%** (new high!), tokens=1248
Both hit new highs this batch. GRPO 76% done, FIPO 60% done.

## 2026-04-08 20:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 31: correct=19.8%, tokens=1138
- GRPO step 39: correct=11.3%, tokens=1302
GRPO 78% done, FIPO 62% done. GRPO approaching step 40 (AIME eval).

## 2026-04-08 20:12 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 32: correct=**24.7%** (new high!), tokens=975
- GRPO step 40: correct=13.5%, tokens=1202, AIME=0.0% (parse failures)
GRPO 80% done, FIPO 64% done. FIPO train accuracy trending up.

## 2026-04-08 20:22 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 34: correct=24.7%, tokens=1181
- GRPO step 42: correct=19.3%, tokens=1417
Both trending up. GRPO 84% done, FIPO 68% done.

## 2026-04-08 20:32 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 35: correct=22.2%, tokens=1162
- GRPO step 43: correct=19.7%, tokens=1122
Both sustaining higher accuracy in the second half. GRPO 86%, FIPO 70% done.

## 2026-04-08 20:42 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 36: correct=17.4%, tokens=1127
- GRPO step 44: correct=17.0%, tokens=1398
GRPO 88% done, FIPO 72% done. GRPO should finish within ~30min.

## 2026-04-08 20:52 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 37: correct=13.5%, tokens=1162
- GRPO step 46: correct=15.0%, tokens=1325
GRPO 92% done (4 steps left), FIPO 74% done.

## 2026-04-08 21:02 UTC — 10min check

Both alive, zero tracebacks.
- FIPO step 38: correct=18.3%, tokens=1040
- GRPO step 47: correct=19.9%, tokens=1481
GRPO 94% done (3 steps left!), FIPO 76% done.

## 2026-04-08 21:12 UTC — *** GRPO FINISHED ***

GRPO completed 50 steps. FIPO at step 39 (78% done).

GRPO final: train_correct=13.5% overall (11.5% 1st half → 15.5% 2nd half)
FIPO so far: train_correct=17.3% overall (16.2% 1st half → 18.3% 2nd half)

Both show learning (second half > first half).
FIPO ~10 steps remaining.

## 2026-04-08 21:22 UTC — 10min check

FIPO step 40: correct=17.0%, tokens=1212, AIME=0.0% (parse failures).
1 process (FIPO only, GRPO done). Zero tracebacks. ~10 steps left.

## 2026-04-08 21:32 UTC — 10min check

FIPO step 41: correct=16.3%, tokens=1070. 1 process, zero tracebacks. ~9 steps left.

## 2026-04-08 21:42 UTC — 10min check

FIPO step 42: correct=**26.8%** (new all-time high!), tokens=1132.
In batch 43. 1 process, zero tracebacks. ~7 steps left.

## 2026-04-08 21:52 UTC — 10min check

FIPO step 44: correct=18.5%, tokens=1224. 45/50 steps logged. 1 process, zero tracebacks.
~5 steps left — should finish within ~30min.

## 2026-04-08 22:02 UTC — 10min check

FIPO step 45: correct=19.3%, tokens=1092. 46/50 logged. 1 process, zero tracebacks.
~4 steps left.

## 2026-04-08 22:12 UTC — 10min check

FIPO step 46: correct=23.3%, tokens=1227. 47/50 logged. 1 process, zero tracebacks.
~3 steps left — finishing soon.

## 2026-04-08 22:22 UTC — 10min check

FIPO step 47: correct=19.6%, tokens=1129. 48/50 logged. 1 process, zero tracebacks.
~2 steps left.

## 2026-04-08 22:32 UTC — 10min check

FIPO step 48: correct=**28.3%** (highest single step ever!), tokens=1074.
49/50 logged. 1 process, zero tracebacks. Last step running.

## 2026-04-08 22:42 UTC — *** BOTH EXPERIMENTS COMPLETE ***

FIPO finished at step 49. Final results written to 02_30b_base_dapo_fipo_vs_grpo.md.

FIPO: train_correct=18.2% (1st_half=16.1%, 2nd_half=20.2%), best=28.3%
GRPO: train_correct=13.5% (1st_half=11.5%, 2nd_half=15.5%), best=21.1%

## 2026-04-08 11:19 UTC — Steps 32-34, Cron Check

Both alive, zero tracebacks.
- **FIPO step 32**: correct=68.8%, tokens/ep=10201, infl=0.985±0.164
- **GRPO step 34**: correct=69.5%, tokens/ep=10062

~66% complete. This step shows very similar accuracy and token lengths —
the batch difficulty matters more than the algorithm at this scale.
ETA ~1h to completion.

## 2026-04-08 11:49 UTC — Steps 37-38, Cron Check

Both alive, zero tracebacks.
- **FIPO step 37**: correct=42.5%, tokens/ep=12446, infl=0.986±0.167
- **GRPO step 38**: correct=72.7%, tokens/ep=8595

~75% complete. FIPO back to generating longer responses (~12.4k) on this harder batch.
ETA ~40min.
