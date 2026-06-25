# Experiment 1 — baseline GRPO, output-match reward

**Date:** 2026-06-25
**Model:** `Qwen/Qwen3.5-4B` (LoRA rank 32)
**Tinker session:** `12b4035b-a37e-5e37-890f-00ad82a795ed`
**Checkpoint:** `tinker://12b4035b-a37e-5e37-890f-00ad82a795ed:train:0/sampler_weights/final`

## State of the training loop

- Rollouts triggered through the production app's `/solve` endpoint; the sampling proxy
  served the policy and captured `(prompt, sampled, logprob)` tokens per rollout.
- GRPO: one datum per turn, advantage = reward − group mean, `importance_sampling` loss.
- Config: `group_size=8`, `groups_per_batch=8`, 30 batches, `learning_rate=4e-5`,
  `max_turns=2`, `max_tokens=1024`, temperature 1.0.

## Reward design (the part that mattered)

- Tasks were **single fixed instances** with the inputs written into the prompt
  ("gcd of 109 and 446", "7th Fibonacci", word "kayak").
- Reward was **output-match**: shaped `parse → run → emit → exact-match` against the one
  expected string for that instance.
- Held-out = disjoint families (`gcd`, `nth_fib`, `palindrome`), still single fixed
  instances.

## Result

| | pass@1 | mean reward |
|---|---|---|
| base (before)    | 0.100 | 0.270 |
| after 30 steps   | 0.967 | 0.979 |

Training metric climbed pass@1 ≈ 0.41 → 0.98 (see `training_metrics.jsonl`).

## Verdict: reward hacking

The number is real but it is *correct output*, not *learned to code*. **117 of 120 trained
completions (98%) are hardcoded constant emits** (`emit 1`, `emit "yes"`) — the model
computes the answer in its reasoning and prints the literal (`trained_final_programs.jsonl`,
`sample_rollouts.md`). Because each task's inputs are fixed in the prompt and the reward
only checks output, emitting the constant is the cheapest way to score 1.0.

## Takeaway → Experiment 2

Grade a **function of input** against **hidden inputs not shown in the prompt**, so a
constant fails the held-out inputs and only a general RILL program scores.
