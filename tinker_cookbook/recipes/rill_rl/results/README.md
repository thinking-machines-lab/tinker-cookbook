# Logbook

Running record of training experiments for the RILL recipe: the state of the training
loop and reward design at the time, what we ran, and what we found. Newest insights drive
the next experiment.

| # | Date | Model | Reward design | Held-out pass@1 | Verdict |
|---|------|-------|---------------|-----------------|---------|
| [1](./experiment_1/) | 2026-06-25 | Qwen3.5-4B | output-match on fixed-input tasks | 0.10 → 0.97 | reward-hacked (98% constant emits) |
| [2](./experiment_2/) | 2026-06-25 | Qwen3.5-4B | `solve(...)` graded on **hidden inputs** | 0.20 → 0.55 | hack fixed; writes real RILL (0% constants) |

## How this is organized

Each `experiment_N/` directory has its own `README.md` (the logbook entry: config, the
loop + reward design used, the result, and the takeaway) plus the raw artifacts from that
run (`training_metrics.jsonl`, `trained_final_programs.jsonl`, `sample_rollouts.md`).

The headline so far: a verifiable-reward RL loop will exploit whatever the reward actually
measures. Experiment 1 measured "did the emitted output match," on tasks whose inputs were
fixed in the prompt, so the policy learned to print the literal answer. Experiment 2
changes the reward to grade `solve(...)` against hidden inputs, which a constant can't
satisfy.
