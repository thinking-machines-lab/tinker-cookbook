# Model Decision: Nemotron-3-Super-120B with 262K Context

**Date:** 2026-03-27

## Decision

Use `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144` for all training.

## Rationale

- Nano (30B/3B) only supports 65K context on Tinker — no `:peft:N` variants available
- Super (120B/12B) with `:peft:262144` gives 262K context — verified working
- 262K matches all paper settings: Code RL (118K), SWE Agentless (98K), SWE Agentic (256K)
- 4x more active params than Nano (12B vs 3B) but still MoE-efficient

## Verification

```
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144
  65,535 + 1 = 65,536:  OK
 131,072 + 1 = 131,073: OK
 200,000 + 1 = 200,001: OK
 262,143 + 1 = 262,144: OK
```

## Impact on Existing Work

- All Nano experiments (SFT checkpoint, RL tests) were for dev/debug only
- Need new SFT run on Super before launching full RL cascade
- LoRA LR scaling (10x paper) needs re-validation on Super

## Tinker Capacity Notes (2026-03-27)

- Running 7 concurrent Super 120B experiments saturates Tinker backend
- Max practical concurrent sessions: 2-3 for Super 120B
- Backend returns: "Tinker backend is running short on capacity, please wait"
- Sessions queue and eventually get served, but can cause stale warnings
- For full SFT run: should be the only job running to get max throughput
