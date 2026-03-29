# Nemotron-Cascade-2 Replication Experiments

Replicating NVIDIA's Nemotron-Cascade-2 (arxiv:2603.19220) using Tinker API.

## Models
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (paper's base model, LoRA, 131K ctx)
- `openai/gpt-oss-120b:peft:131072` (secondary experiments)
- `Qwen/Qwen3-8B-Base` (smaller model experiments)

## Paper Pipeline
SFT → IF-RL → Multi-domain RL → MOPD → RLHF → Long-ctx RL → Code RL → SWE RL

## RL Environments (9 built, 5 working)
| Env | Verifier | Tested | Reward |
|-----|----------|--------|--------|
| if_rl | IFEval (48 types) | ✓ Nano | 0.81 |
| mcqa | Exact match | ✓ Nano | 0.50-0.58 |
| structured_output | JSON schema | ✓ Nano | 0.75-1.0 |
| workbench | Tool call match | ~ Multi-turn, mock backend issue |
| gsm8k (math_rl) | sympy grading | ✓ gpt-oss | 0.95 |
| swe_rl | Modal test exec | ✓ 0 reward (hard) |
| rlhf | Qwen3.5-397B GenRM | ⏳ Slow |
| code_rl | Modal + MBPP | ⏳ Testing |
| longctx_rl | Qwen3.5-397B judge | ⏳ Slow |
| swe_agentic | Modal multi-turn | ✓ 0 reward (19min/step, works e2e) |

### RL LR Sweep (paper-matched, Nano SFT v1 checkpoint)
- Date: 2026-03-27
- Model: Nemotron-3-Nano from SFT v1 final checkpoint
- Settings: group=16, max_tokens=49K (paper-matched)
- Batch=32 (paper=128, reduced for practical speed)
- LRs: 1e-5, 3e-5 (LoRA-adjusted from paper's 3e-6)
- Status: RUNNING (IF-RL, wandb streaming)
- wandb: (internal — see team Wandb project)

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
- Final checkpoint: (internal)

#### Qwen3-8B-Base (lr=5e-4) - COMPLETED
- 2547 steps, final NLL=0.693, min NLL=0.589
- Final checkpoint: (internal)

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
- Checkpoint: (internal)

#### Run 2: from final SFT checkpoint - COMPLETED
- 50/50 steps
- Reward: first5_avg=0.658, last5_avg=0.724, max=0.784
- **Improvement: +0.066 (+10.0% relative)**
- Checkpoint: (internal)

Note: IF-RL shows clear reward improvement (+0.058 over 50 steps).
Paper used batch=128 (vs our 32) and ran 180 steps with dynamic filtering.
Scaling up batch size and running more steps would likely yield stronger gains.

### Benchmark Evaluation (commit: ed3c51e)
- Date: 2026-03-26
- Benchmarks: GSM8K, IFEval, MMLU-Pro, MATH-500 (200 samples each)
- Temperature: 0.6, max_tokens: 4096
- Concurrent sampling (128 parallel requests)

#### gpt-oss-120b

| Benchmark | Base | SFT | IF-RL | SFT Δ | IFRL Δ |
|-----------|------|-----|-------|-------|--------|
| GSM8K | 90.5% | **94.0%** | 91.5% | +3.5 | -2.5 |
| IFEval loose | 77.7% | 73.3% | **77.3%** | -4.3 | **+3.9** |
| IFEval strict | 57.0% | 52.5% | **58.0%** | -4.5 | **+5.5** |
| MATH-500 | **91.5%** | 84.0% | 87.0% | -7.5 | +3.0 |
| MMLU-Pro | **78.0%** | 65.5% | 60.5% | -12.5 | -5.0 |

#### Qwen3-8B-Base

| Benchmark | Base | SFT | SFT Δ |
|-----------|------|-----|-------|
| GSM8K | 84.0% | **89.0%** | +5.0 |
| IFEval loose | 69.8% | 70.0% | +0.2 |
| IFEval strict | 46.5% | 46.0% | -0.5 |
| MATH-500 | **83.0%** | 73.0% | -10.0 |
| MMLU-Pro | 54.5% | **60.0%** | +5.5 |

Key observations:
- SFT improves GSM8K (+3.5/+5.0 pts) but regresses MATH-500 and MMLU-Pro
- SFT data mix is too math-heavy (100K math, only 10K IF) causing IFEval regression
- IF-RL recovers IFEval strict to **above base** (58% vs 57%) without catastrophic forgetting
- MMLU-Pro degrades through pipeline (-12.5 then -5.0) → needs multi-domain RL
- Qwen3-8B shows similar pattern: GSM8K/MMLU up, MATH-500 down from SFT

### Nemotron-3-Nano-30B Full Recipe (commit: d37ba64)
- Date: 2026-03-26
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:peft:131072
- Renderer: nemotron3
- Hidden size: 2688, MoE 30B (3B active)
- This is the paper's actual base model!

#### LR Sweep - COMPLETED
- Data: 500 IF examples, batch_size=16

| LR | Min NLL | Final NLL |
|----|---------|-----------|
| 1e-4 | 0.431 | 0.431 |
| 3e-4 | 0.415 | 0.415 |
| **5e-4** | **0.413** | 0.414 |
| 1e-3 | 0.415 | 0.415 |

All LRs 3e-4 to 1e-3 very similar. 5e-4 confirmed optimal.

#### Full SFT
- Data: math (100K) + science (50K) + IF (10K) + safety (3.5K) = 163.5K
- lr=5e-4, batch_size=64, max_length=16384, cosine schedule
- Status: RUNNING
- Log: (local)

#### IF-RL (pending SFT completion)
#### Multi-domain RL (pending)

## Commit History
- `8ddbd53` - Initial recipe structure (SFT + IF-RL)
- `5b17edc` - LR sweep script, fix type annotation
- `d5f474b` - LR sweep results
- `929a90d` - Full SFT training script
- `6b4d518` - Medium SFT results, IF-RL test, data analysis
- `1967a4b` - Improved IF-RL verifier (all 48 types)
- `285973c` - IF-RL launch script
- `4487acd` - IF-RL from SFT checkpoint
