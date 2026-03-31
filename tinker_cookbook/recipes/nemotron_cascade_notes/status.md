# Overall Status

## Model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144` -- MoE (12B active), LoRA rank 64, 262K context.

See `model_decision.md` for rationale.

## SFT Progress

- Data: All 8 subsets downloaded (24.5M examples, 553GB)
- Config: batch=2048, lr=3e-4 cosine, rank=64, 49K tokens
- Training: step 169/33K, NLL 0.687 -> 0.547 (20% improvement), ~5.5 min/step

## RL Environment Status (Super base model, no SFT checkpoint)

| Env | Reward | frac_mixed | max_tokens | Status |
|-----|--------|-----------|------------|--------|
| IF-RL | 0.77-1.0 | 1.0 | 49K | Working |
| MCQA | 1.0 | 0.0 (too easy) | 8K | Fixed (expanded extraction + overlong partial credit) |
| Structured Output | 0.67-0.89 | 1.0 | 49K | Working |
| Code RL | 0.75-1.0 | 1.0 | 118K | Working |
| Long-Context | 0.60-0.65 | 1.0 | 49K | Working |
| RLHF | 0.50 | 1.0 | 16K | Working (GenRM Kimi K2.5) |
| Workbench | 0.44 | 1.0 | 49K | Fixed (ground-truth seeded mocks + partial credit) |
| SWE Agentless | 0.12-0.46 | 1.0 | 98K | Working (LLM judge); sandbox errors in execution mode |
| SWE Agentic | TBD | -- | 262K | R2E-Gym Docker integration wired up, untested at scale |

## Key Findings

1. LoRA LR = 10x paper's full-FT LR (SFT: 3e-4, RL: 3e-5)
2. group_size=16 critical for mixed groups and GRPO signal
3. LLM judge envs need max_tokens >= 256 for thinking models
4. `<think>` tags must be stripped before answer/code/patch extraction
5. R2E-Gym Docker images solve SWE dependency issues

## Next Steps

1. Validate multi-domain RL (current run, groups_per_batch=16), then scale to 64
2. SFT continuing (~step 918 as of 2026-03-31, NLL 0.525)
3. Test SWE Agentic at scale with R2E-Gym Docker
4. Full cascade benchmark after each stage

## Wandb

(internal — see team Wandb project)

## SFT Checkpoint (2026-03-30)

- **Step 500** saved with 14-day TTL
- **State**: `tinker://dcac3236-4699-55be-8375-9e54e071c056:train:0/weights/sft_step500_permanent`
- **Sampler**: `tinker://dcac3236-4699-55be-8375-9e54e071c056:train:0/sampler_weights/sft_step500_permanent`
- **NLL**: 0.687 -> 0.542 (500 steps, Super 120B, LoRA rank 64)

## Eval Results (2026-03-30)

Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144 (200 samples per benchmark)

| Benchmark | Base | SFT step-500 | Delta |
|---|---|---|---|
| MATH-500 | 91.5% | 93.0% | +1.5 |
| GSM8K | 89.5% | 90.0% | +0.5 |
| MMLU-Redux | 89.5% | 89.0% | -0.5 |
| MMLU-Pro | 76.0% | 76.0% | 0.0 |
| GPQA-Diamond | 52.0% | 55.1% | +3.1 |
| IFEval (loose) | 77.2% | 74.6% | -2.6 |
| IFEval (strict) | 58.5% | 54.0% | -4.5 |

SFT improves math/science, degrades instruction following. IF-RL should recover IFEval.

## Eval Results (2026-03-31) — IF-RL Final

| Benchmark | Base | SFT-500 | IF-RL | Δ (Base→IF-RL) |
|---|---|---|---|---|
| GSM8K | 89.5% | 90.0% | 91.0% | +1.5 |
| MATH-500 | 91.5% | 93.0% | 94.5% | +3.0 |
| GPQA-Diamond | 52.0% | 55.1% | 59.6% | **+7.6** |
| IFEval (strict) | 58.5% | 54.0% | 61.0% | **+2.5** |
| IFEval (loose) | 77.2% | 74.6% | 79.0% | **+1.8** |
| MMLU-Redux | 89.5% | 89.0% | 90.5% | +1.0 |
| MMLU-Pro | 76.0% | 76.0% | 73.5% | -2.5 |

IF-RL recovered IFEval (54%→61% strict, above base 58.5%) and improved math/science across the board. Only MMLU-Pro slight regression.

## Super IF-RL from SFT Complete (2026-03-31)

30-step IF-RL on Super 120B, starting from SFT step-500 checkpoint.

- **Config**: group_size=16, groups_per_batch=8, lr=3e-5, max_tokens=49K, save_every=5
- **Rollout strategy**: RetryOnFailure(max_retries=3, per_rollout_timeout=1800)
- **Reward**: first5 avg=0.751, last5 avg=0.819, best=0.953 (step 13)
- **Time**: ~37.6 min/step avg
- **Issues**: Original run hung at step 25 (2 sampling groups stuck indefinitely).
  Killed and resumed from step 20 checkpoint with optimizer state + retry/timeout.

### Checkpoints (save with permanent TTL for future stages)

| Checkpoint | State Path | Sampler Path |
|---|---|---|
| Step 20 (session 1) | `tinker://a59335bc-efb7-5c3d-ac99-f9b5b4791367:train:0/weights/000020` | `tinker://a59335bc-efb7-5c3d-ac99-f9b5b4791367:train:0/sampler_weights/000020` |
| Step 25 (session 2) | `tinker://72b86fbb-afd9-5b12-b9a5-c1e553a3a18e:train:0/weights/000025` | `tinker://72b86fbb-afd9-5b12-b9a5-c1e553a3a18e:train:0/sampler_weights/000025` |
| Step 30 / Final | `tinker://72b86fbb-afd9-5b12-b9a5-c1e553a3a18e:train:0/weights/final` | `tinker://72b86fbb-afd9-5b12-b9a5-c1e553a3a18e:train:0/sampler_weights/final` |

**TTL**: 7 days. Session 1 (step 20) created 2026-03-30 ~03:40 UTC, expires **2026-04-06**.
Session 2 (steps 25, 30, final) created 2026-03-30 ~19:28 UTC, expires **2026-04-06**.
Save permanently before expiry:
```python
await training_client.save_state_permanently_async("ifrl_step30_permanent")
```

### Wandb

Project: `nemotron-cascade-2-replication`, run: `super-ifrl-from-sft500-resumed`

## Multi-Domain RL (Stage 2) — In Progress (2026-03-31)

Starting from IF-RL final checkpoint. Uses InterleavedRLDatasetBuilder.

### Rollout & Group Size Reference

| Stage | Paper | Our Run | Rollouts/Step |
|---|---|---|---|
| IF-RL | group=16, batch=128 | group=16, batch=8 | 128 |
| Multi-domain (validation) | group=16, batch=128 | group=16, batch=16 | 256 |
| Multi-domain (production) | group=16, batch=128 | group=16, **batch=64** | 1024 |

**Domain weights**: MCQA 55%, Workbench 30%, Structured Output 15% (paper Table 8)

At batch=64: ~35 MCQA, ~19 workbench, ~10 structured output groups per step.

### Current run (validation, groups_per_batch=16)

- **Config**: group_size=16, groups_per_batch=16, lr=3e-5, max_tokens=49K, 70 steps
- **From**: IF-RL final checkpoint (cold start — fresh optimizer)
- **Purpose**: Validate 3-domain interleaving works, all domains produce metrics
- **Issues found**: Previous run (batch=8) missed structured output due to jsonschema missing + Python import caching. Fixed by installing deps and restarting.

### Production run plan

Once validation confirms 3-domain blend works:
```bash
python -m tinker_cookbook.recipes.nemotron_cascade.rl.train \
  env=multi_domain groups_per_batch=64 group_size=16 \
  max_tokens=49152 max_steps=70 save_every=10 \
  load_checkpoint_path=<ifrl_final> \
  rollout_error_tolerance=True per_rollout_timeout=1800
```

## Nano RL Baselines Complete (2026-03-30)

30-step RL on Nemotron-3-Nano base model (group_size=16, groups_per_batch=8):

| Env | Avg Reward | First5 | Last5 | Notes |
|---|---|---|---|---|
| Structured Output | 0.796 | 0.825 | 0.842 | Stable, strong |
| IF-RL | 0.729 | 0.766 | 0.653 | Noisy, slight decline |
| Long-Context | 0.588 | 0.597 | 0.591 | Stable |
| MCQA | 0.415 | 0.334 | 0.432 | Improved over training |
| Workbench | 0.252 | 0.219 | 0.230 | Slow improvement |
| RLHF | 0.503 (13 steps) | Flat | — | GenRM hangs (parallelization helped but still flaky) |
