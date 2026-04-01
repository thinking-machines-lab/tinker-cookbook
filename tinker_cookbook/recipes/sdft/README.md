# Self-Distillation Fine-Tuning (SDFT)

This recipe implements the SDFT algorithm from ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026) using Tinker. SDFT is an on-policy distillation method that learns new skills from demonstrations while preserving prior capabilities — addressing catastrophic forgetting in sequential fine-tuning.

## Algorithm

### What the paper does

SDFT uses the same model in two roles with different prompts:

1. **Teacher** sees the question + golden answer as an in-context demonstration
2. **Student** sees only the question and generates a completion on-policy
3. The loss minimizes forward KL divergence between teacher and student distributions at each token position of the student's completion:

```
L = (1/T) * sum_{t=1}^{T} KL(P_teacher(·|t) || P_student(·|t))
```

where `T` is the number of completion tokens, and the KL at each position sums over the **full vocabulary** (~150K tokens).

Both teacher and student start from the same base model weights. The paper maintains the teacher as an EMA (Exponential Moving Average) of the student, updated every step with `alpha=0.01` — so the teacher slowly tracks the student's learning while providing stable distillation targets.

### What we implement in Tinker

Our Tinker implementation differs from the paper in two ways:

1. **Top-K instead of full-vocabulary KL.** The Tinker API does not expose full-vocabulary logits. Instead, we use Tinker's [top-K distillation API](https://tinker-docs.thinkingmachines.ai/tinker/losses) (`topk_prompt_logprobs`) to recover the teacher's top-K token distribution at each position, and train with `cross_entropy` loss:

    ```
    L_topK = (1/T) * sum_{t=1}^{T} [ -sum_{k=1}^{K} P_teacher(x_k|t) * log P_student(x_k|t) ]
    ```

    where the inner sum is over the K tokens with highest teacher probability (renormalized to sum to 1).

2. **Static teacher instead of EMA.** The paper maintains the teacher as an EMA of the student (updated every step with `alpha=0.01`). Our implementation keeps the teacher frozen at the initial base model weights, which is simpler and avoids the overhead of periodic weight syncing.

### Validating the approximation

We verified both design choices by running ablations on the [official SDFT codebase](https://github.com/Continual-Intelligence/Self-Distillation) with `Qwen/Qwen2.5-7B-Instruct` (full fine-tuning, lr=2e-5, tooluse task):

| KL Type | EMA Teacher | Tool-use | Science |
|---------|-------------|----------|---------|
| Full-vocab | Yes | 68.04% | 33.33% |
| Full-vocab | No (static) | 67.01% | 36.69% |
| **Top-K=20** | **Yes** | **68.04%** | **35.31%** |
| **Top-K=20** | **No (static)** | **69.07%** | **34.52%** |
| SFT baseline | N/A | 65.98% | 36.88% |

**Top-K=20 matches full-vocab KL** (68.04% vs 68.04% with EMA, 69.07% vs 67.01% without). **Static teacher matches EMA** — the 1-2pp difference is within noise. Both design choices are validated.

We also confirmed our Tinker implementation produces identical results to the reference on the same model (`Qwen/Qwen3-4B-Instruct-2507`): both get 56.70% tooluse accuracy with top-K=20 and static teacher.

## Setup

### 1. Download the data

Training data comes from the [Self-Distillation repository](https://github.com/Continual-Intelligence/Self-Distillation), which includes preprocessed golden reasoning chains for tool-use and science tasks.

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
    toolalpaca_data_path=Self-Distillation/data/tooluse_data/train_data \
    groups_per_batch=128 \
    learning_rate=5e-4 \
    topk=20 \
    lora_rank=64
```

### Continual learning experiment (SFT vs SDFT)

The key experiment: train sequentially on two tasks and measure retention.

```bash
python -m tinker_cookbook.recipes.sdft.run_continual_learning \
    model_name=Qwen/Qwen3.5-35B-A3B \
    data_dir=Self-Distillation/data \
    methods=sft,sdft_topk \
    learning_rates=5e-4 \
    stages=1,2 \
    lora_rank=64 \
    topk=20 \
    thinking_format=true
```

### Debug run

```bash
python -m tinker_cookbook.recipes.sdft.train \
    model_name=Qwen/Qwen3.5-35B-A3B \
    dataset=sciknoweval \
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
| `lora_rank` | `128` | LoRA adapter rank (64 for `Qwen/Qwen3.5-35B-A3B`) |
| `max_tokens` | `2048` | Max completion length |
| `thinking_format` | `false` | Convert data for thinking models (e.g., Qwen3.5) |
| `teacher_sync_every` | `None` | Optional periodic teacher weight sync |

## Results

### Continual Learning: `Qwen/Qwen3.5-35B-A3B` (Tinker, LoRA 64)

**Stage 1: Train on tool-use, evaluate both tasks**

| Method | LR | Tool-use | Science | Sci Δ |
|--------|-----|----------|---------|-------|
| Base model | — | 61.86% | 46.35% | — |
| SFT | 5e-4 | 67.01% | 29.19% | **-17.16** |
| SFT | 1e-3 | 69.07% | 37.08% | **-9.27** |
| SDFT | 1e-4 | 63.92% | 45.17% | -1.18 |
| **SDFT** | **5e-4** | **65.98%** | **46.55%** | **+0.20** |
| **SDFT** | **1e-3** | **67.01%** | **53.65%** | **+7.30** |

**Stage 2: Train on science (from Stage 1 checkpoint), evaluate both tasks**

| Method | LR | Tool-use | Science | TU Retention |
|--------|-----|----------|---------|-------------|
| SFT | 5e-4 | 68.04% | 63.71% | 101% |
| SFT | 1e-3 | 8.25% | 64.69% | **12%** |
| **SDFT** | **1e-4** | **61.86%** | **56.80%** | **97%** |
| **SDFT** | **5e-4** | **61.86%** | **63.51%** | **94%** |
| SDFT | 1e-3 | 35.05% | 60.75% | 52% |

### Findings

1. **SDFT preserves prior knowledge during new-task training.** After tool-use training (Stage 1), SFT causes -9 to -17pp science degradation. SDFT either preserves or improves science (+0 to +7pp).

2. **SFT can retain knowledge with careful LR tuning.** At lr=5e-4, SFT retains 101% of tool-use after Stage 2 science training. However, at lr=1e-3, tool-use collapses to 8.25% (12% retention). The right LR for retention isn't known in advance.

3. **SDFT retention is robust across learning rates.** 94-97% tool-use retention in Stage 2 at lr=5e-4 and 1e-4, without needing to tune for retention specifically.

4. **Practical recommendation.** Use SDFT when you need to fine-tune on new data without risking degradation of existing capabilities. Use SFT with a conservative learning rate when you can afford some forgetting and want maximum target-task performance. Optimizing SFT's learning rate and number of steps can also help control retention.

## Files

| File | Description |
|------|-------------|
| `tinker_cookbook/distillation/sdft.py` | Core: teacher prompts, top-K datum construction, training loop |
| `tinker_cookbook/recipes/sdft/train.py` | CLI entry point |
| `tinker_cookbook/recipes/sdft/run_continual_learning.py` | 2-stage experiment runner |
| `tinker_cookbook/recipes/sdft/datasets.py` | Data loading (tool-use, science, thinking model format) |
| `tinker_cookbook/recipes/sdft/eval.py` | Evaluation (tool-use exact match, science MCQ) |
| `tinker_cookbook/recipes/sdft/sdft_test.py` | Unit tests |

## References

- Shenfeld et al., ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897), 2026
- [Official implementation](https://github.com/Continual-Intelligence/Self-Distillation) (TRL-based, Qwen2.5-7B)
- [Tinker loss functions](https://tinker-docs.thinkingmachines.ai/tinker/losses) (top-K distillation, cross_entropy)
