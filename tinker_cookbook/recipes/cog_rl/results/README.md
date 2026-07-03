# Logbook

Running record of training experiments for the Cog recipe: the state of the training
loop and reward design at the time, what we ran, and what we found. Newest insights drive
the next experiment.

| # | Date | Model | Reward design | Held-out pass@1 | Verdict |
|---|------|-------|---------------|-----------------|---------|
| [1](./experiment_1/) | 2026-06-25 | Qwen3.5-4B | output-match on fixed-input tasks | 0.10 → 0.97 | reward-hacked (98% constant emits) |
| [2](./experiment_2/) | 2026-06-25 | Qwen3.5-4B | `solve(...)` graded on **hidden inputs** | 0.20 → 0.55 | hack fixed; writes real Cog (0% constants) |
| [3](./experiment_3/) | 2026-06-25 | Qwen3.5-4B | Exp-2 reward + async loop, 50 steps, 10 eval families | 0.25 → 0.75 | bigger/cleaner; async ran but lag≈0 (effectively on-policy) |
| [4](./experiment_4/) | 2026-06-26 | Qwen3.5-4B | + 5 structure-matched train families, 60 steps | 0.25 → 0.78 | matched families lifted (list_max/palindrome 0.50→1.00); overall flat — n=4/family eval too noisy |
| [5](./experiment_5/) | 2026-06-26 | Qwen3.5-4B | + Lucas (two-state recurrence) family, **16 samples/family eval** (n=160) | 0.275 → **0.944** | stable eval; nth_fib 0.19→0.62 (Lucas transferred); Exp-4 "regressions" were noise; 0 constants |
| [6](./experiment_6/) | 2026-06-26 | Qwen3.5-4B | **handbook system prompt** + keyed-lookup train shapes; held-out keyed-lookup eval families | 0.600 → **0.933** | handbook alone doubled the base floor (0.275→0.600); rebuild idiom transferred to held-out `mode_count` (1.00); 0 constants |
| [7](./experiment_7/) | 2026-06-26 | Qwen3.5-4B | **scalable corpus source**: grade Cog against MBPP's Python ref I/O; 561 train + 99 held-out, zero hand-written Cog | 0.283 → **0.364** (reward 0.467 → 0.665) | shape coverage from corpus diversity, not enumerated families; honest harder distribution (some tasks Cog-infeasible) |
| [8](./experiment_8/) | 2026-06-27 | Qwen3.5-4B | **self-distillation** (rejection-sample verified Cog from the open model itself) SFT warm-start, then GRPO; no frontier model in training | 0.283 → **0.495** (reward 0.467 → 0.708) | distill-then-RL compounds (>GRPO-only 0.364, >SFT-only 0.394); expert iteration; closes ~45% of gap to gpt-5.5 (0.751) self-contained |
| [9](./experiment_9/) | 2026-06-30 | Qwen3.5-4B | **expert-iteration sweep** — 3 more harvest/SFT/GRPO rounds; clean repeat=5 (n=495) eval | exp8 0.503 → exp10 0.513 (within ±0.022 SE) | converged ~0.51; SFT warm-start was the win (exp7 0.374→exp8 0.503), more RL rounds insignificant; 4B is capacity-bound, 9B is the next lever |
| [10](./experiment_10/) | 2026-06-30 | **Qwen3.5-9B** | same recipe (SFT on the 408 self-distilled gold, then GRPO) on a bigger base; repeat=5 | base 0.347 → SFT 0.519 → **SFT+GRPO 0.562** | capacity was the lever: 9B-SFT alone matches 4B's trained ceiling; GRPO has juice again on 9B (+0.043, flat on 4B); ~60% of gap to gpt-5.5 (0.751) closed, self-contained |

| [11](./experiment_11/) | 2026-07-03 | **Qwen3.5-9B** | autonomous hill-climb loop: CoT-SFT(neg), OPD tuning(null), **OPD->GRPO@8e-5 chain (best)**, GRPO extension(null), +HumanEval data, test-time example+BoN (+0.14, product win) | best **0.618** (harness 0.768) | 9B trained ceiling ~0.62 = untrained 397B/Kimi-K2.6; gap to teacher is capacity; next lever is a bigger student |

## How this is organized

Each `experiment_N/` directory has its own `README.md` (the logbook entry: config, the
loop + reward design used, the result, and the takeaway) plus the raw artifacts from that
run (`training_metrics.jsonl`, `trained_final_programs.jsonl`, `sample_rollouts.md`).

The headline so far: a verifiable-reward RL loop will exploit whatever the reward actually
measures. Experiment 1 measured "did the emitted output match," on tasks whose inputs were
fixed in the prompt, so the policy learned to print the literal answer. Experiment 2
changes the reward to grade `solve(...)` against hidden inputs, which a constant can't
satisfy.
