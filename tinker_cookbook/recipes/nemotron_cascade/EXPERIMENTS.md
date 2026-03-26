# Nemotron-Cascade-2 Replication Experiments

Replicating NVIDIA's Nemotron-Cascade-2 (arxiv:2603.19220) using Tinker API.

## Models
- `openai/gpt-oss-120b:peft:131072` (LoRA fine-tuning, 128K ctx, MoE, hidden=2880)
- `Qwen/Qwen3-8B-Base` (LoRA fine-tuning, 32K ctx, dense base model)

## Paper Pipeline
1. SFT on ~24.8M examples (multi-domain)
2. IF-RL (instruction following, ~180 steps)
3. Multi-domain RL (~70 steps)
4. MOPD (on-policy distillation)
5. RLHF
6. Long-context RL
7. Code RL
8. SWE RL

## Our Replication Plan
Focus on stages 1-3 (SFT -> IF-RL -> Multi-domain RL).

## Data Analysis Summary

| Subset | Size | Multi-turn | Avg Tokens | Has think | System Prompt |
|--------|------|------------|------------|-----------|---------------|
| Instruction Following | 10K | 30.8% | 947 | 63.2% | 67.7% empty |
| Math | 100K (of 5.2M) | 0% | 13,503 | 100% | 9.8% empty |
| Science | 50K (of 2.7M) | 0% | 3,841 | 100% | 45.4% empty |
| Safety | 3.5K | 0% | 959 | 100% | 100% empty |

Key observations:
- All math/science/safety assistant messages use `<think>` tags (reasoning)
- Math examples are very long (~13.5K tokens avg, up to ~62K tokens)
- IF data is ~30% multi-turn
- Most system prompts are empty or "You are a helpful and harmless assistant"
- Full analysis: ~/data/nemotron-cascade-2/data_analysis_report.txt

## Experiment Log

### SFT LR Sweep (commit: 5b17edc)
- Date: 2026-03-26
- Data: instruction_following subset, 500 examples
- LRs tested: 1e-4, 3e-4, 5e-4, 1e-3
- Batch size: 16, Max length: 8192, LoRA rank: 32
- Cosine LR schedule, AdamW (beta1=0.9, beta2=0.98)

#### gpt-oss-120b Results

| LR | Steps | First NLL | Min NLL | Final NLL | Notes |
|----|-------|-----------|---------|-----------|-------|
| 1e-4 | 53 | 1.008 | 0.409 | 0.483 | Too conservative |
| 3e-4 | 53 | 0.900 | 0.367 | 0.449 | Good |
| **5e-4** | **31** | **0.900** | **0.362** | **0.452** | **Best - matches paper 5e-5 * 10x LoRA** |
| 1e-3 | 31 | 0.900 | 0.426 | 0.488 | Too aggressive |

#### Qwen3-8B-Base Results

| LR | Steps | First NLL | Min NLL | Final NLL | Notes |
|----|-------|-----------|---------|-----------|-------|
| 1e-4 | 31 | 1.310 | 0.932 | 0.946 | Too conservative |
| 3e-4 | 31 | 1.310 | 0.854 | 0.868 | OK |
| 5e-4 | 31 | 1.310 | 0.824 | 0.839 | Good (matches get_lr ~4.7e-4) |
| 1e-3 | 31 | 1.310 | 0.800 | 0.815 | Good, still improving |

**Decision**: Use lr=5e-4 for gpt-oss-120b, lr=5e-4 for Qwen3-8B-Base.

### Medium-scale SFT (10K examples) (commit: d5f474b)
- Date: 2026-03-26
- Data: instruction_following subset, 10K examples
- batch_size=64, max_length=8192

| Model | Steps | First NLL | Min NLL | Final NLL |
|-------|-------|-----------|---------|-----------|
| gpt-oss-120b (lr=5e-4) | 156 | 1.057 | 0.368 | 0.448 |
| Qwen3-8B-Base (lr=5e-4) | 156 | 1.182 | 0.682 | 0.829 |

### Full SFT (commit: 929a90d)
- Date: 2026-03-26
- Data: math (100K) + science (50K) + IF (10K) + safety (3.5K) = 163.5K examples
- batch_size=64, max_length=16384, cosine LR, AdamW (beta1=0.9, beta2=0.98)
- Total steps: ~2555 per epoch

#### gpt-oss-120b (lr=5e-4) - COMPLETED
- 2547 steps, final NLL=0.589, min NLL=0.487
- NLL plateaued at ~0.58 by step 200, continued slowly improving to ~0.49
- Checkpoints at: steps 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400 + final
- Final checkpoint: tinker://a27528b8-b83f-59a0-9334-78129ec565d0:train:0/weights/final
- Log: /tmp/tinker-examples/nemotron_cascade_full_sft/openai-gpt-oss-120b-peft-131072_lr0.0005/

#### Qwen3-8B-Base (lr=5e-4) - COMPLETED
- 2547 steps, final NLL=0.693, min NLL=0.589
- Final checkpoint: tinker://38e7fb7e-d5be-54b9-a4a4-164e2e65de6f:train:0/weights/final
- Log: /tmp/tinker-examples/nemotron_cascade_full_sft/Qwen-Qwen3-8B-Base_lr0.0005/

### IF-RL Environment Test (commit: 929a90d)
- Date: 2026-03-26
- Model: gpt-oss-120b base (no SFT), 3 steps, group_size=4, batch=4
- Result: reward=0.92-1.0 (base model already strong at IF)
- With improved verifier (all 48 types): reward ~0.67 (more discriminative)
- Dynamic filtering needed for meaningful training

### IF-RL Training (commit: 4487acd, 3bc5b53)
- Date: 2026-03-26
- Config: group_size=16, batch=32, lr=3e-6, max_tokens=16384
- Dynamic filtering: enabled
- ~10-15 min per step with batch=32, group=16

#### Run 1: from SFT step 1000 - COMPLETED
- 50/50 steps
- Reward: first5_avg=0.662, last5_avg=0.719, max=0.763
- **Improvement: +0.058 (+8.8% relative)**
- Checkpoint: tinker://6c99cabe-10a4-5e30-a32e-7ffdc39d896a:train:0/weights/final
- Log: /tmp/tinker-examples/nemotron_cascade_ifrl_from_sft/

#### Run 2: from final SFT checkpoint - COMPLETED
- 50/50 steps
- Reward: first5_avg=0.658, last5_avg=0.724, max=0.784
- **Improvement: +0.066 (+10.0% relative)**
- Checkpoint: tinker://ec4ef250-02aa-5f91-b141-fe7c6f8f7d95:train:0/weights/final
- Log: /tmp/tinker-examples/nemotron_cascade_ifrl_final_sft/

Note: IF-RL shows clear reward improvement (+0.058 over 50 steps).
Paper used batch=128 (vs our 32) and ran 180 steps with dynamic filtering.
Scaling up batch size and running more steps would likely yield stronger gains.

### Benchmark Evaluation (commit: 3317b54)
- Date: 2026-03-26
- Model: gpt-oss-120b
- Benchmarks: GSM8K (200 samples), IFEval (200 samples)
- Temperature: 0.6, max_tokens: 4096
- Concurrent sampling (128 parallel requests)

| Metric | Base | SFT | IF-RL | Delta |
|--------|------|-----|-------|-------|
| **GSM8K** | 90.5% | **93.5%** | 92.0% | +3 then -1.5 |
| **IFEval loose** | **79.6%** | 72.6% | 72.8% | -7 then +0.2 |
| **IFEval strict** | **61.0%** | 49.0% | 53.0% | -12 then +4 |

Note: Initial eval showed inflated GSM8K gains due to answer extraction bug
(nested LaTeX braces in \boxed{}). Fixed in commit 3317b54.

Key observations:
- Base model is strong (90.5% GSM8K) as expected for gpt-oss-120b
- SFT improves math slightly (+3 pts) but regresses IFEval (-7 loose, -12 strict)
  - Likely due to heavy math bias in SFT data (100K math vs 10K IF)
- IF-RL recovers IFEval strict (+4 pts) without losing much math (-1.5 pts)
- The cascade approach works: IF-RL partially reverses SFT's IF regression
- To improve: balance SFT data mix (more IF/chat data) or use more IF-RL steps

## Commit History
- `8ddbd53` - Initial recipe structure (SFT + IF-RL)
- `5b17edc` - LR sweep script, fix type annotation
- `d5f474b` - LR sweep results
- `929a90d` - Full SFT training script
- `6b4d518` - Medium SFT results, IF-RL test, data analysis
- `1967a4b` - Improved IF-RL verifier (all 48 types)
- `285973c` - IF-RL launch script
- `4487acd` - IF-RL from SFT checkpoint
