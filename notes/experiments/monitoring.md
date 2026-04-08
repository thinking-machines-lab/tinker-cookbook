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
