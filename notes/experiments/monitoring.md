# Experiment Monitoring Log

## 2026-04-08 06:27 UTC — Experiments Launched

**FIPO** (PID 2175419):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=4096, lr=1e-6, 50 steps
- FIPO config: tau=32, clip=[0.8, 1.2], ppo_clip=0.2
- Log: /tmp/tinker-fipo-math-qwen3-8b

**GRPO Baseline** (PID 2175565):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=4096, lr=1e-6, 50 steps, loss_fn=importance_sampling
- Log: /tmp/tinker-grpo-baseline-qwen3-8b

Both processes started successfully. Waiting for first metrics.
Commit: d65aaf3 (research/fipo branch)

## 2026-04-08 06:40 UTC — Restarted with improved config (v2)

Killed v1 experiments (max_tokens=4096 too small for math reasoning).

**FIPO v2** (PID 2182775):
- Model: Qwen/Qwen3-8B, env=math, group_size=8, groups_per_batch=16
- max_tokens=16384 (using Qwen3-8B's 32k context window), lr=1e-6, 50 steps
- FIPO config: tau=32, clip=[0.8, 1.2], ppo_clip=0.2
- TINKER_SUBPROCESS_SAMPLING=1 enabled
- Log: /tmp/tinker-fipo-math-v2

**GRPO Baseline v2** (PID 2182853):
- Same config but loss_fn=importance_sampling (standard GRPO)
- Log: /tmp/tinker-grpo-baseline-v2

Monitoring cron set up (every 30 min).
