# Experiment 3 — async off-policy loop, more steps, bigger held-out

**Date:** 2026-06-25
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `94bfd2f0-b580-5a75-8d6a-499c017070f2`
**Checkpoint:** `tinker://94bfd2f0-b580-5a75-8d6a-499c017070f2:train:0/sampler_weights/final`

## What changed from Experiment 2

- **Async off-policy loop** (`max_steps_off_policy=2`): a producer keeps firing rollouts at
  the current proxy policy while the trainer steps in a thread, bounded by a lookahead
  semaphore; `importance_sampling` on the captured logprobs corrects for staleness.
- **More steps:** 50 batches (was 30), `groups_per_batch=16` (was 8).
- **Bigger held-out set:** 10 disjoint families (was 5) — added `lcm`, `list_max`,
  `is_sorted`, `count_char`, `power`.

## Result

Held-out (10 families, 4 samples each, n=40):

| | pass@1 | mean reward |
|---|---|---|
| base (before)   | 0.250 | 0.418 |
| after 50 steps  | **0.750** | **0.838** |

Per family, before → after pass@1: `count_char` 0.25→1.00, `factorial` 1.00→1.00,
`gcd` 0.00→0.75, `is_sorted` 0.00→0.75, `lcm` 0.00→0.50, `list_max` 0.25→0.50,
`nth_fib` 0.00→0.50, `palindrome` 0.25→0.50, `power` 0.75→1.00, `reverse_text` 0.00→1.00.
Training metric climbed pass@1 ≈ 0.39 → 0.98 (`training_metrics.jsonl`). **40/40 trained
completions are real `forge solve(...)` functions; 0 constants** (the Exp-1 hack stays
gone). The Exp-2 weak spot `reverse_text` went 0.00 → 1.00.

## The async/off-policy finding (honest)

The async machinery works (no deadlock; sampling of batch N+1 overlaps training of batch N
in a thread; weights republish every step), **but `lag` stayed 0 for all 50 batches** — the
sampler never ran more than ~1 batch ahead. A LoRA step on a 4B is about as fast as
sampling a batch here, so the producer and trainer stay in lockstep and the data each step
trains on is current. In other words, the *staleness* path was implemented and bounded but
not actually exercised: this run is **effectively on-policy**. `max_steps_off_policy` would
bite with a slower trainer (larger model / full fine-tune) or much longer generations.

Total wall for 50 batches: ~266s.

## Takeaways → next

- `lcm`, `list_max`, `palindrome`, `nth_fib` lag at ~0.50; more steps or curriculum on
  multi-step/text families would likely help.
- To genuinely study off-policy, throttle the trainer or scale the model so sampling is the
  bottleneck, and watch `off_policy/lag_batches` rise.
