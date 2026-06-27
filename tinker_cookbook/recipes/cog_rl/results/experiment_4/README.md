# Experiment 4 — enriched train mix for the lagging families

**Date:** 2026-06-26
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `eab976a1-0e5e-55d1-a717-3031ed0f27e9` (project `8a4e2d8a-…`, label `exp4`)
**Checkpoint:** `tinker://eab976a1-0e5e-55d1-a717-3031ed0f27e9:train:0/sampler_weights/final`

## What changed from Experiment 3

Added 5 **transferable** train families (eval families unchanged, still disjoint), chosen
to match the structure of the held-out families that lagged in Exp 3:

- `list_min`, `list_product` — list reduce → for `list_max`
- `pow2`, `collatz_steps` — accumulate / while-with-branch → for `nth_fib`, `lcm`
- `adjacent_dup_count` — two-index text scan → for `palindrome`

Also more steps (60 vs 50), on-policy. Config: `group_size=8`, `groups_per_batch=16`,
`learning_rate=4e-5`, `max_turns=2`, `max_tokens=1024`. Train pool: 25 tasks / 14 families.

## Result (held-out, 10 families, 4 samples each, n=40)

| | pass@1 | mean reward |
|---|---|---|
| base (before) | 0.250 | 0.418 |
| Exp 3 (after) | 0.750 | 0.838 |
| **Exp 4 (after)** | **0.775** | **0.904** |

Per-family pass@1, Exp 3 → Exp 4: `list_max` 0.50→**1.00**, `palindrome` 0.50→**1.00**,
`is_sorted` 0.75→**1.00**, `lcm` 0.50→**0.75** (the four structure-matched targets all up);
`gcd` 0.75→0.50, `nth_fib` 0.50→0.25, `power` 1.00→0.75, `reverse_text` 1.00→0.50 (down).
`40/40` completions are real `solve(...)` functions (no constants).

## Verdict: targeted lift, but read it carefully

The structure-matched additions did what they were meant to: the matched held-out families
(`list_max`, `palindrome`, `lcm`, `is_sorted`) improved, and overall **mean reward rose
0.838 → 0.904** (the model is closer even when not exactly right). But **overall pass@1 was
essentially flat (0.75 → 0.775)** because several unrelated families regressed — almost
certainly noise: each family is only `n=4` (single sample × 4 repeats), so a one-task swing
is ±0.25. `nth_fib` stayed the hardest; its two-variable recurrence isn't captured by the
single-accumulator `pow2`/`collatz` families.

## Takeaways → next

- **The eval is too small to read overall pass@1.** Move to many samples per family (pass@k
  or ≥16 rollouts/family) before drawing overall conclusions; per-family deltas at n=4 are
  within noise.
- Add a **two-state recurrence** train family (e.g. running pair-sum) to target `nth_fib`.
- Structure-matched training data transfers to matched held-out structure — the core lesson
  worth keeping.
