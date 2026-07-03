# Experiment 11 — autonomous hill-climb loop on the 9B (final)

**Date:** 2026-07-02/03
**Model:** `Qwen/Qwen3.5-9B` (LoRA rank 32)
**Best checkpoint:** `tinker://ec49a425-e3c1-59ca-a6d0-d3c03373ff0f:train:0/sampler_weights/final` (label `9b-i3`)

## What we ran

An autonomous experiment loop to push the 9B's held-out corpus pass@1 as far as it would
go, guided by the paired failure analysis (the 9B's failures are wrong *reasoning* on valid
programs, never syntax). Four iterations, each chosen from the results of the last:

| iter | experiment | result | verdict |
|---|---|---|---|
| 1 | CoT/reasoning-trace distillation (SFT on the 397B teacher's reasoning+code) | 0.414 vs 0.465 program-only A/B | **negative** — off-policy imitation of teacher *reasoning text* hurts more than program SFT; also found `get_lr` (4.7e-4) too hot for SFT (0.465 < 0.519 @1e-4) |
| 2a | OPD v2: better init (0.519 SFT) + 160 batches | 0.600 vs 0.598 | **null** — fwd_kl bottoms ~0.11 and flattens; OPD saturates at ~0.60 regardless of init/duration |
| 2b | test-time: visible example I/O + best-of-4 self-verified | 0.626 → 0.717 → **0.768** (hidden-only protocol) | **big positive**, but a *product* win: the same harness lifts gpt-5.5 equally (0.768 → 0.889) |
| 3 | OPD → **GRPO@8e-5 chain** on corpus+HumanEval | **0.618** (new best; OPD stage alone 0.574) | **positive** — verifier-RL adds +0.044 on top of saturated imitation, same-size boost as after SFT |
| 4 | extend the chain's GRPO 120 more batches | 0.616 | **null** — converged |

Supporting work: HumanEval task extension (+116 verified train tasks, eval benchmark
unchanged), `distill.py` trainable-state saves (enables OPD→GRPO chaining), app-health
watchdogs (the agent app died silently under load several times; 3-strike auto-restart
plus client retry-with-backoff fixed it).

## Final standings (held-out corpus, repeat=5, n=495)

| model | pass@1 | reward |
|---|---|---|
| 9B SFT (program gold, lr 1e-4) | 0.519 | — |
| 9B OPD (any variant) | 0.598–0.600 | 0.76 |
| **9B OPD→GRPO chain (`9b-i3`)** | **0.618** | 0.777 |
| Kimi K2.6 (untrained, ref) | 0.616 | 0.738 |
| Qwen 397B base (untrained, ref) | 0.620 | 0.763 |
| 397B trained teacher | 0.673 | 0.809 |
| gpt-5.5 (frontier ref) | 0.751 | 0.845 |

With the test-time harness (visible example + best-of-4), the deployed 9B agent scores
**0.768** on the hidden-only protocol — equal to plain gpt-5.5's corpus number, at ~100x
lower model cost (gpt-5.5 with the same harness: 0.889).

## Conclusions

1. **The 9B's trained ceiling on this benchmark is ~0.62**, reached by imitation (OPD, to
   0.60) plus one round of verifier-RL on top (+0.02–0.04). Further RL, more/better
   imitation, richer gold, and reasoning-trace SFT all return zero or negative.
2. **A trained 9B matches untrained 400B-class open models** (397B 0.620, Kimi K2.6 0.616)
   on the specialized task — the recipe's compression story in one line.
3. **Imitation and verifier-RL are complementary, in that order**: OPD covers most of the
   distance fast; GRPO@8e-5 (not `get_lr`; GRPO collapses at 4.7e-4) adds a consistent
   final increment; repeating either alone is flat.
4. **Test-time technique is the cheapest big win** (+0.14) but lifts every model, so it's a
   product improvement, not a competitive-gap improvement.
5. What remains between 0.618 and the teacher's 0.673 is student capacity (the failure
   analysis: wrong derivations on tasks the bigger model reasons through), and between
   0.673 and ~1.0 is largely task ambiguity / Cog-infeasibility that caps every model.
   The next real lever is a bigger student, not another training technique.
