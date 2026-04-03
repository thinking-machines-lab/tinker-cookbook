# Experiment 03: Hypothesis Verification

**Commit:** f5987b2a
**Date:** 2026-04-03

## H1: Per-token training cost (VERIFIED)

Controlling for sequence length (first 5 steps, all ~520K tokens):

| Loss | µs/token | Relative to CISPO |
|------|---------|-------------------|
| IS | 30.5 | 0.95x |
| **CISPO** | **32.1** | **1.00x** |
| DRO | 33.3 | 1.04x |
| **PPO** | **80.2** | **2.50x** |

**Conclusion:** PPO is **2.5x more expensive per token** than IS/CISPO/DRO at equal
sequence lengths. IS, CISPO, and DRO have essentially identical per-token cost.
The late-training speed advantage of CISPO over IS (8.8s vs 10.8s per step) is
explained by CISPO generating shorter responses (203K vs 234K tokens) because it
learned faster.

## H2: CISPO preserves rare correction tokens (INCONCLUSIVE)

Measured correction word frequency in logtree transcripts at convergence (steps 85, 90, 95):
- IS: 457, PPO: 458, CISPO: 423

No meaningful difference. The theoretical claim from MiniMax-M1 may be true but
this measurement is too noisy to detect it. Would need per-token gradient magnitude
tracking (server-side instrumentation) to verify properly.

## H3: PPO prevents overshooting → early lead (REFUTED)

Regression analysis (steps 0-99):
- IS: 41 regressions, max drop 8.6%
- PPO: 39 regressions, max drop 7.3%
- CISPO: 46 regressions, max drop 9.0%

Step-to-step std (steps 0-30): IS=0.058, PPO=0.058, CISPO=0.061

**PPO does NOT have fewer regressions or lower variance than IS/CISPO.**

## H3 (revised): PPO's early lead is run variance, not algorithmic (VERIFIED)

Seed reproducibility at step 20 test accuracy:

| Loss | seed=0 | seed=1 | |Δ| |
|------|--------|--------|-----|
| IS | 42.9% | 52.6% | **9.7%** |
| PPO | 51.3% | 51.1% | **0.2%** |
| CISPO | 46.1% | 48.7% | **2.6%** |

PPO's "51% lead at step 20" (seed=0) was noise: IS seed=1 also reaches 53%.
But PPO IS remarkably reproducible (|Δ|=0.2% across seeds).

**Revised finding:** PPO doesn't converge faster. It converges MORE CONSISTENTLY.
IS and CISPO have higher seed-to-seed variance in mid-training, but converge to
the same accuracy (all ~93-94% at step 40, |Δ| < 1% across seeds).

## H4: DRO benefits from off-policy conditions (PARTIALLY VERIFIED)

Test accuracy comparison with num_substeps=4 (4 gradient steps per rollout batch):

| Step | DRO (substeps=1) | DRO (substeps=4) | IS (substeps=1) | IS (substeps=4) |
|------|-----------------|-----------------|-----------------|-----------------|
| 10 | 6.6% | 9.5% | 11.3% | 79.9% |
| 20 | 7.4% | 15.7% | 42.9% | 93.0% |
| 30 | 8.7% | 37.5% | 88.7% | 93.6% |
| 40 | 10.1% | 75.9% | 93.3% | 93.6% |

DRO_substeps4 reaches 75.9% at step 40 (vs 10.1% without substeps). The multi-step
updates provide a mild off-policy setting where DRO's conservatism becomes helpful.
However, IS_substeps4 still vastly outperforms, reaching 93% at step 20.

**Conclusion:** DRO benefits from multi-step updates but IS benefits MORE from them.
DRO's conservatism helps avoid overshooting on stale data, but the effect isn't
enough to match unconstrained methods on this task.
