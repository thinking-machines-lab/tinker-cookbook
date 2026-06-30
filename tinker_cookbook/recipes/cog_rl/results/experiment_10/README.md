# Experiment 10 — scale the base model to 9B

**Date:** 2026-06-30
**Model:** `Qwen/Qwen3.5-9B` (LoRA rank 32)
**SFT checkpoint:** `tinker://b1f4ac59-…:train:0/sampler_weights/9b-sft`
**Final (SFT+GRPO) checkpoint:** `tinker://73207c2b-113b-515f-8e00-84111ff07fe4:train:0/sampler_weights/final` (project `8a4e2d8a-…`, label `9b`)

## Why

Experiment 9 showed the 4B plateaued at ~0.51 held-out: the self-distilled SFT warm-start was
the win, and further GRPO rounds added nothing significant. The diagnosis was a capacity
ceiling, not a method ceiling. This experiment tests that by running the *same recipe* on
`Qwen/Qwen3.5-9B`: SFT on the same 408 verified self-distilled solutions (verifier-correct Cog,
valid for any model), then GRPO. No new data, no method change, only a bigger base.

## Result (held-out, repeat=5, n=495)

| stage | pass@1 | mean reward |
|---|---|---|
| base 4B + handbook | 0.283 | 0.467 |
| 4B best (SFT+GRPO, Exp 9 exp10) | 0.513 | 0.720 |
| base 9B + handbook | 0.347 | 0.559 |
| 9B SFT (on the 408 gold) | 0.519 | 0.728 |
| **9B SFT + GRPO** | **0.562** | 0.748 |
| gpt-5.5 + handbook (frontier ref) | 0.751 | 0.845 |

## Verdict

- **Capacity was the lever.** The 9B with SFT *alone* (0.519) already matches the 4B's fully
  trained ceiling (0.513), on the exact same gold. Scaling the base, not more training tricks,
  is what moved the needle.
- **GRPO has juice again on the bigger model.** On the 4B, GRPO-after-SFT was flat
  (0.503 -> 0.513, within noise). On the 9B it adds +0.043 (0.519 -> 0.562), a real gain. The
  Exp-9 plateau was the 4B running out of room to exploit RL, not a limit of the method. A
  larger model has the headroom for RL to keep paying.
- **New best, ~60% of the way to the frontier.** 9B SFT+GRPO at 0.562 closes
  (0.562-0.283)/(0.751-0.283) ≈ 60% of the base-4B-to-gpt-5.5 gap, with mean reward (0.748)
  approaching the frontier's 0.845, and still no frontier model anywhere in the training path.

## Note: infra hang on the first GRPO attempt

The first 9B GRPO run hung after batch 5 (healthy batches, then a ~12h stall on a Tinker/app
call) and never saved a checkpoint, so its "eval" (0.352) was an empty-checkpoint artifact and
is discarded. The re-run from the same `9b-sft` warm-start completed all 80 batches normally
and produced the 0.562 above. A stall detector now flags this within ~18 min instead of losing
hours.

## Takeaways

- The recipe's arc is complete: verifier-only, fully self-contained (corpus tasks +
  self-distillation + RL, no frontier model in training) takes an open model from 0.283 (base
  4B) to 0.562 (9B), with the two highest-leverage steps being the self-distilled SFT warm-start
  and base-model scale.
- Remaining gap to gpt-5.5 (0.562 -> 0.751) is further scale (35B/397B variants are available)
  and, partly, the Cog-infeasible tail of the corpus that caps any model.
