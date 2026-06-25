# Experiment 2 — hidden-input grading (reward-hack fix)

**Date:** 2026-06-25
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `a79ab023-0d88-5693-8761-98aae5dfd8f5`
**Checkpoint:** `tinker://a79ab023-0d88-5693-8761-98aae5dfd8f5:train:0/sampler_weights/final`

## What changed from Experiment 1

The reward. Each task now asks for `forge solve(...)` and the grader calls it on several
**hidden inputs not shown in the prompt**, rewarding the fraction correct (`tasks.py` /
`grading.py`). A constant emit can no longer win, because it can't match multiple distinct
expected outputs. Everything else (proxy + app endpoints, GRPO, datum-per-turn) is the same.

## Config

`group_size=8`, `groups_per_batch=8`, 30 batches, `learning_rate=4e-5`, `max_turns=2`,
`max_tokens=1024`, temperature 1.0. 20 train tasks across 9 families, cycled (the trainer
wraps around the pool). On-policy and sequential — no sampling/training overlap.

## Result

Held-out (5 disjoint families, 4 samples each, n=20):

| | pass@1 | mean reward |
|---|---|---|
| base (before)   | 0.200 | 0.340 |
| after 30 steps  | 0.550 | 0.671 |

Per family, before → after pass@1: `factorial` 0.75→1.00, `palindrome` 0.00→1.00,
`gcd` 0.25→0.50, `nth_fib` 0.00→0.25, `reverse_text` 0.00→0.00. Training metric climbed
pass@1 ≈ 0.41 → 0.89 (`training_metrics.jsonl`).

## Verdict: the hack is gone; this is real (partial) learning

**20/20 trained completions define `forge solve(...)`; 0 are constant emits** (Experiment 1
was 98% constants). See `trained_final_programs.jsonl` and `sample_rollouts.md`. The model
learned RILL *syntax* and structure: e.g. on `palindrome` the base model wrote invalid
`c = chars(w)` (wrong assignment form) and failed, while the trained model wrote correct
`chars(w) -> c` with `w @ i` indexing and passed all hidden inputs.

It is also honestly imperfect: `reverse_text` stays at 0 (the trained attempts `push` onto
an int instead of building a list), `nth_fib` only reaches 0.25, and the run is noisy (one
`gcd` sample regressed to a buggy loop). The held-out 0.55 is far below Experiment 1's
*fake* 0.97 precisely because it now measures generalizing programs, not memorized literals.

## Takeaways → next

- More steps + a larger held-out set would tighten the numbers and the weak families.
- Consider overlapping sampling and training (bounded off-policy) for throughput.
- `reverse_text`/`nth_fib` suggest adding a couple of text-building and sequence tasks to
  training so those structures are in distribution.
