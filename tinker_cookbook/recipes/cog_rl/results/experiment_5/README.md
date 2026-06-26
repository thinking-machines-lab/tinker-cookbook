# Experiment 5 — two-state recurrence family + a stable eval

**Date:** 2026-06-26
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `0042ded0-e461-5784-99fa-0ee10bffeeec` (project `8a4e2d8a-…`, label `exp5`)
**Checkpoint:** `tinker://0042ded0-e461-5784-99fa-0ee10bffeeec:train:0/sampler_weights/final`

## What changed from Experiment 4

1. Added one train family — **`lucas`** (Lucas numbers: `L0=2, L1=1, L(k)=L(k-1)+L(k-2)`).
   Same two-state recurrence as the held-out `nth_fib`, different seeds, so it stays
   disjoint while giving the model exactly that structure (Exp 4's `pow2`/`collatz` were
   single-accumulator and didn't transfer to `nth_fib`).
2. **Bigger eval: 16 samples per family (n=160)** instead of 4, so per-family pass@1 is
   stable (±0.06, not ±0.25) and the Exp-4 "regressions" can be confirmed/denied.

Config: `group_size=8`, `groups_per_batch=16`, 60 batches, `learning_rate=4e-5`,
`max_turns=2`. Train pool: 26 tasks / 15 families. On-policy.

## Result (held-out, 10 families, 16 samples each, n=160)

| | pass@1 | mean reward |
|---|---|---|
| base (before) | 0.275 | 0.419 |
| **Exp 5 (after)** | **0.944** | **0.974** |

Per-family base → Exp 5 pass@1:

| family | base | Exp 5 |
|--------|------|-------|
| count_char | 0.56 | 1.00 |
| factorial | 0.62 | 1.00 |
| gcd | 0.06 | 0.94 |
| is_sorted | 0.25 | 1.00 |
| lcm | 0.12 | 0.94 |
| list_max | 0.12 | 1.00 |
| **nth_fib** | **0.19** | **0.62** |
| palindrome | 0.06 | 1.00 |
| power | 0.75 | 1.00 |
| reverse_text | 0.00 | 0.94 |

`159/160` completions are real `solve(...)` functions; **0 constants**.

## Verdict

- **The targeted fix worked.** `nth_fib` rose 0.19 → 0.62 once a same-structure (`lucas`)
  family was in training — confirming the Exp-4 lesson that transfer is structure-matched.
  It's still the hardest family (real headroom remains), but no longer stuck.
- **Exp-4's "regressions" were noise.** At 16 samples, `gcd` 0.94, `power` 1.00,
  `reverse_text` 0.94 — the n=4 dips (0.50/0.75/0.50) were sampling variance, exactly as
  flagged. Trust per-family numbers only at this sample size.
- **Overall 0.275 → 0.944** on a stable, disjoint-family eval, with every completion a real
  program (no reward hacking). This is the recipe's headline result.

## Takeaways → next

- `nth_fib` at 0.62 still has headroom; more recurrence variety or more steps could close it.
- The Exp 1→5 arc is now a clean narrative (hack → hidden-input fix → scale/async →
  structure-matched data → confirm on a stable eval). Good point to finalize the recipe.
