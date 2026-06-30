# Experiment 9 — expert-iteration sweep (how far does the RL loop go?)

**Date:** 2026-06-30
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Checkpoints:** exp9 `tinker://a6a75dc4-…`, exp10 `tinker://d078e04a-…` (project `8a4e2d8a-…`)

## What we ran

Three more expert-iteration rounds on top of exp8, to find where the self-distillation + RL
loop saturates. Each round: serve the current best checkpoint, harvest verified gold from it
(`attempts=10`), union with all prior gold, SFT from base on the accumulated set, GRPO from
that warm-start, eval. The new checkpoint generates the next round's gold.

The harvest coverage kept climbing — the loop's core mechanism works:

| generator | tasks solved (of 561) | accumulated gold |
|---|---|---|
| exp7 | 316 (56%) | 316 |
| exp8 | 380 (68%) | 385 |
| exp9 | 395 (70%) | 402 |
| exp10 | (round 4 not completed) | 408 |

The model keeps getting better at *generating* verified Cog. The question is whether that
translates to held-out accuracy.

## Result — noise-resolved (repeat=5, n=495 per checkpoint)

The single-sample (n=99) per-round numbers wobbled (0.495 / 0.475 / 0.495) inside the noise
band, so we re-evaluated every checkpoint at 5 samples/task:

| checkpoint | pass@1 | mean reward |
|---|---|---|
| exp7 (GRPO only, from base) | 0.374 | 0.654 |
| **exp8 (self-distilled SFT + GRPO)** | **0.503** | 0.727 |
| exp9 (expert-iteration round 2) | 0.511 | 0.720 |
| exp10 (expert-iteration round 3) | 0.513 | 0.720 |

At n=495 the standard error on a ~0.5 rate is ±0.022, so **exp8 / exp9 / exp10 are
statistically indistinguishable**. The loop converged at ~0.51.

## Verdict

- **The SFT warm-start was the win, not the RL iterations.** exp7 → exp8 (add a
  self-distilled SFT warm-start) is +0.13 (many SE, real). Rounds 2–3 on top add nothing
  significant. Once RL starts from a model that already writes valid Cog, more
  harvest-SFT-GRPO rounds don't move held-out accuracy, even though harvest coverage keeps
  rising (the gains are on tasks the model was already near, not new capability).
- **This 4B is capacity-bound at ~0.51 held-out.** The remaining gap to gpt-5.5 (0.751) is
  model size, not training technique. The recipe flags `Qwen/Qwen3.5-9B` as the lever for
  absolute quality.
- **Loop bug, logged for honesty:** the round-4 (exp11) GRPO app failed to bind its port
  (the prior round didn't release it), so every rollout failed to the parse-only floor
  (reward 0.150, zero advantage, no datums) and it learned nothing. A port-cleanup defect in
  the throwaway orchestrator, not a model collapse. exp11's harvest+SFT still succeeded
  (gold 408), but its GRPO was a no-op, so it's excluded.

## Takeaways

- The recipe's headline lands here: a verifier-only, fully self-contained pipeline (corpus
  tasks + self-distillation + RL, no frontier model in training) takes a 4B from 0.283 (base)
  to ~0.51 held-out on an OOD DSL. The single highest-leverage step is the self-distilled SFT
  warm-start.
- Further RL on a 4B is into the noise. To push toward the frontier, scale the base model;
  to push the loop itself, the lever is harder/larger task coverage, not more rounds.
