# Experiment 01: Arithmetic baseline (all 4 loss functions)

**Commit:** 9fd0e81e
**Date:** 2026-04-03

## Config
- Model: meta-llama/Llama-3.2-1B
- Task: arithmetic (addition, e.g. "What is 42 + 57?")
- group_size=4, groups_per_batch=100, lr=1e-4, max_tokens=5
- 50 training steps per loss function
- Log paths: /tmp/tinker-examples/loss_cmp/{is,ppo,cispo,dro}/

## Results

| Metric | IS | PPO | CISPO | DRO |
|--------|---:|----:|------:|----:|
| Steps to reward >= 0.95 | 3 | 3 | 3 | 12 |
| Steps to reward >= 0.99 | 5 | 4 | 5 | 14 |
| Final reward | 1.000 | 1.000 | 1.000 | 1.000 |
| Tail stability (std, last 10) | 0.003 | 0.005 | 0.001 | 0.001 |
| Entropy at step 5 | 0.022 | 0.018 | 0.040 | 0.577 |

## Interpretation

- IS, PPO, CISPO are essentially identical on this task — all converge in 3-5 steps.
- DRO converges 3-4x slower due to its quadratic KL penalty constraining update size.
  Visible in entropy: DRO entropy at step 5 is 0.577 vs ~0.02 for others.
- CISPO and DRO have the most stable tail (std 0.001 vs 0.003-0.005).
- All reach 1.0 reward. Task is too easy to differentiate final performance.

## Limitations

Arithmetic is a toy task (short outputs, trivial reasoning). The properties that
differentiate these loss functions — CISPO's rare-token gradient preservation,
DRO's robustness to distribution shift — cannot manifest here because:
1. Outputs are 1-2 tokens (no chain-of-thought, no rare correction tokens)
2. Training is fully on-policy (no stale data, so DRO's robustness doesn't help)
3. The task saturates at 100% within a few steps for all methods
