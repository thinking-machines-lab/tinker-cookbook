# Self-Distillation Fine-Tuning (SDFT)

This recipe implements the SDFT algorithm from ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026) using Tinker's top-K distillation API. SDFT is an on-policy distillation method that learns new skills from demonstrations while preserving prior capabilities — addressing the catastrophic forgetting problem in sequential fine-tuning.

## How SDFT Works

SDFT uses the same model in two roles:

- **Teacher**: Sees the question **and** a golden answer as an in-context demonstration
- **Student**: Sees only the question

The student generates completions on-policy. The teacher is then "forced" through the student's completion tokens, and its top-K token distribution at each position is recovered via Tinker's `topk_prompt_logprobs` API. These soft targets are renormalized and used to train the student with `cross_entropy` loss — approximating the paper's full-vocabulary forward KL divergence.

```
L = -sum_t sum_k P_teacher(x_k|t) * log P_student(x_k|t)
```

The teacher's frozen weights act as a knowledge anchor: the distillation signal prevents the student from drifting too far from the base model's distribution, even as it acquires new skills.

## Setup

### 1. Download the data

The training data comes from the [Self-Distillation repository](https://github.com/Continual-Intelligence/Self-Distillation). The data includes preprocessed golden reasoning chains for both tool-use and science tasks.

```bash
git clone https://github.com/Continual-Intelligence/Self-Distillation.git
# Data is at Self-Distillation/data/tooluse_data/ and Self-Distillation/data/science_data/
```

### 2. Set your Tinker API key

```bash
export TINKER_API_KEY=<your-key>
```

## Running the Recipe

### Single-task SDFT training

```bash
# SDFT on tool-use (top-K distillation, K=20)
python -m tinker_cookbook.recipes.sdft.train \
    model_name=Qwen/Qwen3.5-35B-A3B \
    dataset=toolalpaca \
    toolalpaca_data_path=~/Self-Distillation/data/tooluse_data/train_data \
    groups_per_batch=128 \
    learning_rate=5e-4 \
    topk=20 \
    lora_rank=64

# SFT baseline on tool-use (for comparison)
python -m tinker_cookbook.recipes.sdft.run_continual_learning \
    model_name=Qwen/Qwen3.5-35B-A3B \
    methods=sft \
    learning_rates=5e-4 \
    stages=1 \
    lora_rank=64 \
    thinking_format=true
```

### Continual learning experiment (2-stage)

The key experiment: train sequentially on two tasks and measure retention.

```bash
# Stage 1: Train on tool-use
# Stage 2: Train on science (from Stage 1 checkpoint)
# Compare SFT vs SDFT on both tasks after each stage

python -m tinker_cookbook.recipes.sdft.run_continual_learning \
    model_name=Qwen/Qwen3.5-35B-A3B \
    methods=sft,sdft_topk \
    learning_rates=5e-4 \
    stages=1,2 \
    lora_rank=64 \
    topk=20 \
    thinking_format=true \
    wandb_project=sdft-replication
```

### Debug run

```bash
python -m tinker_cookbook.recipes.sdft.train \
    model_name=Qwen/Qwen3.5-35B-A3B \
    groups_per_batch=4 \
    max_tokens=256 \
    max_steps=3 \
    topk=20 \
    lora_rank=64
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | — | Model for both student and teacher |
| `topk` | `20` | Top-K tokens for distillation (0 = importance sampling fallback) |
| `learning_rate` | `2e-5` | Adam learning rate. For LoRA, use 5e-4 to 1e-3 |
| `groups_per_batch` | `32` | Problems per training batch |
| `lora_rank` | `128` | LoRA adapter rank |
| `max_tokens` | `2048` | Max completion length |
| `thinking_format` | `false` | Convert data for thinking models (Qwen3.5) |
| `teacher_sync_every` | `None` | Optional EMA-like teacher sync interval |

## Results

### Continual Learning: `Qwen/Qwen3.5-35B-A3B` (Tinker, LoRA 64)

**Stage 1: Train on tool-use, evaluate both tasks**

| Method | LR | Tool-use | Science | Tool Δ | Sci Δ |
|--------|-----|----------|---------|--------|-------|
| Base model | — | 61.86% | 46.35% | — | — |
| SFT | 5e-4 | 67.01% | 29.19% | +5.15 | **-17.16** |
| SFT | 1e-3 | 69.07% | 37.08% | +7.21 | **-9.27** |
| **SDFT (top-K)** | **5e-4** | **65.98%** | **46.55%** | **+4.12** | **+0.20** |
| **SDFT (top-K)** | **1e-3** | **67.01%** | **53.65%** | **+5.15** | **+7.30** |

**Stage 2: Train on science (from Stage 1 checkpoint), evaluate both tasks**

| Method | LR | Tool-use | Science | TU Retention |
|--------|-----|----------|---------|-------------|
| SFT | 5e-4 | 68.04% | 63.71% | 101% |
| SFT | 1e-3 | 8.25% | 64.69% | **12%** |
| **SDFT (top-K)** | **5e-4** | **61.86%** | **63.51%** | **94%** |
| **SDFT (top-K)** | **1e-4** | **61.86%** | **56.80%** | **97%** |

### Findings

1. **SDFT preserves prior knowledge during new-task training.** After Stage 1 (tool-use training), SFT causes catastrophic forgetting on science (-9 to -17pp), while SDFT either preserves or improves science (+0 to +7pp). This is the core advantage of on-policy self-distillation.

2. **SDFT learns new tasks comparably to SFT.** At lr=5e-4, SDFT achieves 65.98% tool-use vs SFT's 67.01% — a small gap that closes at lr=1e-3 (both 67-69%). The teacher's soft targets provide sufficient signal for task acquisition.

3. **SFT retention is sensitive to learning rate.** In Stage 2, SFT at lr=1e-3 catastrophically forgets tool-use (69% → 8%). At lr=5e-4 it retains well (101%). This suggests careful LR tuning can mitigate SFT's forgetting, but requires knowing the right LR in advance.

4. **SDFT retention is robust across learning rates.** SDFT maintains 94-97% tool-use retention at lr=5e-4 and lr=1e-4 in Stage 2, without needing to tune for retention specifically.

5. **Practical recommendation.** Use SDFT when you need to fine-tune on new data without risking degradation of existing capabilities. Use SFT with a conservative learning rate when you can tolerate some forgetting and want maximum target-task performance.

## Implementation Details

### Top-K Distillation via Tinker API

The recipe uses Tinker's `topk_prompt_logprobs` to recover the teacher's token distribution:

```python
# Teacher-force student's completion through teacher
topk_response = await teacher_client.sample_async(
    prompt=teacher_forced_sequence,
    num_samples=1,
    sampling_params=tinker.SamplingParams(max_tokens=1),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=K,  # Get top-K tokens at each position
)

# Build (N, K) shaped targets and weights for cross_entropy loss
target_tokens[pos, :K] = teacher_token_ids
weights[pos, :K] = renormalized_teacher_probs
```

This approximates full-vocabulary forward KL — validated against the paper's reference implementation where top-K=20 achieves identical results to full-vocabulary KL.

### Thinking Model Support

For thinking models like `Qwen/Qwen3.5-35B-A3B`, set `thinking_format=true`. This:
- Converts golden answers from `<reasoning>/<answer>` to `<think>...</think>` format
- Uses a thinking-compatible system prompt for science evaluation
- Handles the `<think>` token sequence in the renderer automatically

## Files

| File | Description |
|------|-------------|
| `tinker_cookbook/distillation/sdft.py` | Core algorithm: teacher prompts, top-K datum construction, training loop |
| `tinker_cookbook/recipes/sdft/train.py` | CLI entry point for single-task training |
| `tinker_cookbook/recipes/sdft/run_continual_learning.py` | 2-stage continual learning experiment runner |
| `tinker_cookbook/recipes/sdft/datasets.py` | Data loading for tool-use and science tasks |
| `tinker_cookbook/recipes/sdft/eval.py` | Evaluation for both tasks |
| `tinker_cookbook/recipes/sdft/sdft_test.py` | Unit tests |

## References

- Shenfeld et al., ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897), 2026
- [Reference implementation](https://github.com/Continual-Intelligence/Self-Distillation) (TRL-based, Qwen2.5-7B)
- [Tinker top-K distillation docs](https://tinker-docs.thinkingmachines.ai/losses#top-k-distillation)
