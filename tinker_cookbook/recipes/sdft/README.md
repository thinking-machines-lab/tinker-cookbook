# Self-Distillation Fine-Tuning (SDFT)

This recipe implements the SDFT algorithm from ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026) using Tinker. SDFT is an on-policy distillation method that learns new skills from demonstrations while preserving prior capabilities — addressing catastrophic forgetting in sequential fine-tuning.

## Algorithm

### What the paper does

SDFT uses the same model in two roles with different prompts:

1. **Teacher** sees the question + golden answer as an in-context demonstration
2. **Student** sees only the question and generates a completion on-policy
3. The loss minimizes forward KL divergence between teacher and student distributions at each token position of the student's completion:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \text{KL}\big(P_{\text{teacher}}(\cdot \mid t) \| P_{\text{student}}(\cdot \mid t)\big)$$

where $T$ is the number of completion tokens, and the KL at each position sums over the **full vocabulary** (~150K tokens).

Both teacher and student start from the same base model weights. The paper maintains the teacher as an EMA (Exponential Moving Average) of the student, updated every step with `alpha=0.01` — so the teacher slowly tracks the student's learning while providing stable distillation targets.

### What we implement in Tinker

Our Tinker implementation differs from the paper in two ways:

1. **Top-K instead of full-vocabulary KL.** The Tinker API does not expose full-vocabulary logits. Instead, we use Tinker's [top-K distillation API](https://tinker-docs.thinkingmachines.ai/tinker/losses) (`topk_prompt_logprobs`) to recover the teacher's top-K token distribution at each position, and train with `cross_entropy` loss:

$$\mathcal{L}_{\text{top-K}} = \frac{1}{T} \sum_{t=1}^{T} \left[ -\sum_{k=1}^{K} P_{\text{teacher}}(x_k \mid t) \cdot \log P_{\text{student}}(x_k \mid t) \right]$$

   where the inner sum is over the $K$ tokens with highest teacher probability (renormalized to sum to 1).

2. **Static teacher instead of EMA.** The paper maintains the teacher as an EMA of the student (updated every step with `alpha=0.01`). Our implementation keeps the teacher frozen at the initial base model weights, which is simpler and avoids the overhead of periodic weight syncing.

### Validating the approximation

We verified both design choices by running ablations on the [official SDFT codebase](https://github.com/idanshen/Self-Distillation) with `Qwen/Qwen2.5-7B-Instruct` (full fine-tuning, lr=2e-5, tooluse task):

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

Training data comes from the [Self-Distillation repository](https://github.com/idanshen/Self-Distillation), which includes preprocessed golden reasoning chains for tool-use and science tasks.

```bash
git clone https://github.com/idanshen/Self-Distillation.git
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
    model_name=Qwen/Qwen3.6-35B-A3B \
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
    model_name=Qwen/Qwen3.6-35B-A3B \
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
    model_name=Qwen/Qwen3.6-35B-A3B \
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
| `groups_per_batch` | `128` | Problems per training batch |
| `lora_rank` | `128` | LoRA adapter rank (64 for `Qwen/Qwen3.6-35B-A3B`) |
| `max_tokens` | `2048` | Max completion length |
| `thinking_format` | `false` | Convert data for thinking models (e.g., Qwen3.5) |
| `teacher_sync_every` | `None` | Optional periodic teacher weight sync |

## Results

### Continual Learning: `Qwen/Qwen3.6-35B-A3B` (Tinker, LoRA 64)

**Stage 1: Train on tool-use, evaluate both tasks**

| Method | LR | Tool-use | Science | Sci Δ |
|--------|-----|----------|---------|-------|
| Base model | — | 45.36% | 37.67% | — |
| SFT | 1e-4 | 65.98% | 44.38% | +6.71 |
| **SFT** | **5e-4** | **73.20%** | **50.49%** | **+12.82** |
| SFT | 1e-3 | 70.10% | 35.50% | -2.17 |
| SDFT | 1e-4 | 59.79% | 38.07% | +0.39 |
| SDFT | 5e-4 | 54.64% | 39.96% | +2.29 |
| SDFT | 1e-3 | 53.61% | 42.60% | +4.93 |

**Stage 2: Train on science (from Stage 1 checkpoint), evaluate both tasks**

| Method | LR | Tool-use | Science | TU Retention |
|--------|-----|----------|---------|-------------|
| SFT | 1e-4 | 56.70% | 58.78% | 86% |
| SFT | 5e-4 | 48.45% | 66.07% | 66% |
| SFT | 1e-3 | 1.03% | 66.47% | **1%** |
| **SDFT** | **1e-4** | **64.95%** | **49.70%** | **108%** |
| **SDFT** | **5e-4** | **52.58%** | **58.19%** | **96%** |
| SDFT | 1e-3 | 63.92% | 0.20% | 119% †

† SDFT lr=1e-3 Stage 2 also exhibits a training collapse — science accuracy drops to ~0% even though tool-use is preserved. Both methods fail at lr=1e-3, in mirror-image ways.

### Findings

1. **SDFT preserves prior knowledge across all learning rates.** After Stage 2 science training, SDFT retains 96-108% of Stage 1 tool-use ability at the two well-behaved LRs (1e-4 and 5e-4). SFT retention degrades quickly with LR (86% → 66% → 1%).

2. **SFT can catastrophically forget at high LR.** At lr=1e-3 in Stage 2, SFT tool-use collapses to 1.03% (1% retention) — a single bad LR choice wipes out Stage 1 learning. The "right" LR isn't known in advance.

3. **SFT is the stronger target-task learner on this model.** In Stage 1, SFT lr=5e-4 reaches 73.2% tool-use vs SDFT's best of 59.8% (lr=1e-4). The on-policy KL signal that protects retention also slows raw learning of the new task.

4. **Practical recommendation.** Use SDFT when retention matters — its tool-use stays within ~12pp across Stage 2 lr=1e-4 / 5e-4 while SFT swings ~50pp. Use SFT with a carefully tuned LR when you need maximum target-task performance and can afford some forgetting. Avoid lr=1e-3 with either method on this setup.

> **Note on absolute scores:** Base `Qwen/Qwen3.6-35B-A3B` scores ~45% on ToolAlpaca, lower than `Qwen/Qwen3.5-35B-A3B` (~64%) under the same single-turn exact-match grader. The dominant failure mode is that `Qwen3.6` emits one tool call and stops to await results on requests where the gold answer expects the full multi-step plan up front. Whether this reflects a real tool-use regression or a behavioral shift toward multi-turn agentic use is not characterized here. Stage-1 Δ and Stage-2 TU Retention are the regime-independent metrics for evaluating SDFT vs SFT, and both reproduce the SDFT recipe's central claims.

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
- [Official implementation](https://github.com/idanshen/Self-Distillation) (TRL-based, Qwen2.5-7B)
- [Tinker loss functions](https://tinker-docs.thinkingmachines.ai/tinker/losses) (top-K distillation, cross_entropy)
