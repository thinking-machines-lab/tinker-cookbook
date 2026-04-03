# Hypothesis Verification Plan

## Claims in the README that need experimental verification

### H1: "CISPO is 33% faster per training step than PPO"
**Status:** CONFOUNDED. CISPO learns faster → shorter sequences → faster steps.
**Test:** Compare early-step training times when all methods have similar sequence lengths (~520K tokens).
**Data we have:** early_train (steps 0-9): IS=19.0s, PPO=32.7s, CISPO=18.5s, DRO=21.9s
**Result:** At equal sequence lengths, PPO is genuinely 1.7x slower than CISPO/IS.
Late-step advantage is partly sequence length (CISPO 203K vs IS 234K tokens).
**Action needed:** Separate the claims. PPO is inherently slower. CISPO's late advantage is compounded by learning faster.

### H2: "CISPO preserves gradients for rare correction tokens"
**Status:** UNVERIFIED theoretical claim.
**Test:** Compare CoT diversity/length at convergence between PPO and CISPO.
If CISPO preserves rare tokens, we'd expect more varied reasoning strategies.
**How to measure:** Response length distribution, unique reasoning patterns, entropy.

### H3: "PPO leads early because clipping prevents overshooting"
**Status:** UNVERIFIED hypothesis.
**Test:** Check step-to-step variance in early training. If IS/CISPO overshoot,
they should show more accuracy regression (accuracy drops between consecutive steps).

### H4: "DRO needs off-policy data to shine"
**Status:** Partially verified (beta=0 → converges like IS).
**Test:** Run with num_substeps > 1 (multiple gradient steps per rollout batch).
This creates mild off-policy conditions where DRO's conservatism should help.

### H5: Are the IS/PPO/CISPO accuracy differences statistically significant?
**Status:** UNKNOWN. Single run per method.
**Test:** Run a second seed for at least IS and CISPO to check variance.

## Priority order
1. H1 (timing) — just analysis, no new experiments needed
2. H3 (PPO early lead) — just analysis on existing data
3. H2 (rare tokens) — analysis on rollout transcripts
4. H5 (reproducibility) — needs new experiments (second seed)
5. H4 (DRO off-policy) — needs new experiments (num_substeps)
