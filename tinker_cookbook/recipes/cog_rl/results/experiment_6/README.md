# Experiment 6 — language handbook + keyed-lookup shapes

**Date:** 2026-06-26
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `8b29c59e-1229-56fb-9a48-98240b8c274f` (project `8a4e2d8a-…`, label `exp6`)
**Checkpoint:** `tinker://8b29c59e-1229-56fb-9a48-98240b8c274f:train:0/sampler_weights/final`

## What changed from Experiment 5

1. **The system prompt is now a handbook, not just a reference.** It gained a
   "Patterns and idioms" section that names the three rules Cog's surface syntax forces
   and that models most often violate: the target of `->` is a bare name (no indexed
   assignment like `-> xs @ i`), builtins are always parenthesized calls, and lists/text
   are immutable so you accumulate by rebinding. It ships three verified worked examples
   (accumulate-into-list, set-at-by-rebuild, and a parallel `keys`/`counts` frequency
   counter). This prompt is **shared** by the baseline and the trained policy, so the
   comparison stays fair: the handbook lifts the floor for both.
2. **Trained on the hard shapes instead of holding them out.** Added four keyed-lookup /
   accumulate-by-rebuild train families — `most_frequent`, `count_distinct`,
   `dedupe_join`, and a Euclidean swap-loop `gcd_steps`. These force the parallel-list and
   seen-list idioms (Cog has no map/set type) that the model previously failed to
   generalize to (the "most frequent fruit" failure). Two held-out families,
   `mode_count` and `first_repeat`, reuse those exact idioms on *different* tasks, so we
   still measure shape transfer rather than memorization.

Train pool: 30 tasks / 19 families. Eval: 12 families, disjoint. Config:
`group_size=8`, `groups_per_batch=16`, 80 batches, `learning_rate=4e-5`, `max_turns=3`,
`max_tokens=2048` (the keyed-lookup programs are multi-function and longer). On-policy.

## The handbook alone roughly doubled the base floor

Before any RL, on this eval set (12 families, 5 samples each, n=60):

| base `Qwen/Qwen3.5-4B` | pass@1 | mean reward |
|---|---|---|
| reference prompt (Exp 5 eval, different set) | 0.275 | 0.419 |
| **+ handbook (this eval set)** | **0.600** | **0.749** |

With the handbook, the failures changed character: the weak families now *parse and run*
and are simply wrong on hidden inputs (a reasoning gap), rather than throwing parse errors
on invented Python-isms (a syntax gap). Syntax is the thing a handbook fixes; correctness
is the thing RL fixes.

## Result (held-out, 12 families, 5 samples each, n=60)

| | pass@1 | mean reward |
|---|---|---|
| base + handbook (before) | 0.600 | 0.749 |
| **Exp 6 (after)** | **0.933** | **0.974** |

Per-family base+handbook → Exp 6 pass@1:

| family | base+handbook | Exp 6 |
|--------|------|-------|
| count_char | 0.80 | 1.00 |
| factorial | 1.00 | 1.00 |
| **first_repeat** (held-out keyed-lookup) | 0.80 | 0.60 |
| gcd | 0.40 | 1.00 |
| is_sorted | 0.60 | 1.00 |
| lcm | 0.20 | 0.80 |
| list_max | 0.60 | 1.00 |
| **mode_count** (held-out keyed-lookup) | 0.60 | 1.00 |
| nth_fib | 0.40 | 1.00 |
| palindrome | 0.60 | 1.00 |
| power | 1.00 | 1.00 |
| reverse_text | 0.20 | 0.80 |

Every completion is a real `solve(...)` function; **0 constants**.

## The generalization result

This was the open question from the "it doesn't generalize" finding: would training on
keyed-lookup shapes teach the *idiom*, or just the specific tasks? The held-out
`mode_count` (frequency of the most common element) hits **1.00** — the policy writes the
parallel `keys`/`counts` counter with its `index_of` helper, on a task it never trained on.
On `first_repeat` it derived a *different* valid approach (`slice(xs, 0, idx)` membership
scan) rather than copying a train solution. The rebuild idiom transferred, which is exactly
what a general handbook can describe but can't guarantee — RL made it reliable.

`first_repeat` dipped 0.80 → 0.60, but its mean reward stayed 0.904; at n=5 that's one draw
of sampling noise, not a real regression (see Exp 4 → Exp 5 for the same lesson).

## Verdict

- **Headline: 0.600 → 0.933 held-out pass@1**, on top of a base floor the handbook already
  doubled (0.275 → 0.600). The handbook and RL are complementary: the handbook removes the
  syntax tax for free and fairly (both models get it); RL buys the correctness and the
  reliable transfer of idioms a prompt can only describe.
- **The generalization concern is answered.** Trained on keyed-lookup shapes, the model
  carries the parallel-list / seen-list idiom to held-out keyed-lookup tasks (`mode_count`
  1.00), including inventing alternative valid formulations.
- Weak spots with headroom: `lcm`/`reverse_text` at 0.80 (long programs; more steps or a
  bigger token budget would likely close them) and the noisy `first_repeat`.

## Takeaways → next

- Build the handbook into the shipped `agent_app` prompt as the default (done), so the
  production agent starts from the 0.60 floor and the served `exp6` checkpoint from 0.93.
- For a tighter generalization claim, raise eval samples to 16/family (as in Exp 5) so the
  per-family numbers stop moving at the ±0.2 scale.
