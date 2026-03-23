# Self-Distillation Fine-Tuning (SDFT)

This recipe implements the SDFT algorithm from ["Self-Distillation Enables Continual Learning"](https://arxiv.org/abs/2601.19897) (Shenfeld et al., 2026). SDFT is an on-policy distillation method that learns new skills from demonstrations while preserving prior capabilities.

## How SDFT Works

SDFT uses the same model in two roles:

- **Teacher**: Sees the question **and** a golden answer as an in-context demonstration
- **Student**: Sees only the question

The student generates completions from the plain question. The teacher scores those completions with its logprobs (conditioned on the demonstration). Per-token advantages are set to `teacher_lp - student_lp`, pushing the student toward tokens the teacher assigns high probability when it has seen the answer. Training uses the `importance_sampling` loss.

The teacher model is static by default (frozen base weights). Optional periodic hard-sync (`teacher_sync_every`) snapshots the student weights into the teacher, approximating the paper's EMA update.

## Datasets

Two datasets from the paper are supported:

- **SciKnowEval** (`sciknoweval`): Science multiple-choice questions from [hicai-zju/SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval). Filtered to L3-level MCQ tasks. Domains: chemistry, physics, biology, material.
- **ToolAlpaca** (`toolalpaca`): Tool-use tasks from [tangqiaoyu/ToolAlpaca](https://huggingface.co/datasets/tangqiaoyu/ToolAlpaca), or from local Arrow data matching the paper's format.

## Running This Recipe

### SciKnowEval (paper's science benchmark)

```bash
python -m tinker_cookbook.recipes.sdft.train \
    model_name=Qwen/Qwen3-8B \
    dataset=sciknoweval \
    sciknoweval_domain=chemistry \
    groups_per_batch=32 \
    learning_rate=2e-5 \
    max_tokens=1024 \
    lora_rank=128
```

### ToolAlpaca (paper's tool-use benchmark)

```bash
python -m tinker_cookbook.recipes.sdft.train \
    model_name=Qwen/Qwen3-8B \
    dataset=toolalpaca \
    groups_per_batch=32 \
    learning_rate=2e-5 \
    max_tokens=1024 \
    lora_rank=128
```

### Debug run (small batch)

```bash
python -m tinker_cookbook.recipes.sdft.train \
    groups_per_batch=4 \
    max_tokens=256 \
    max_steps=5
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `Qwen/Qwen3-8B` | Model for both student and teacher |
| `dataset` | `sciknoweval` | Dataset: `sciknoweval` or `toolalpaca` |
| `groups_per_batch` | `32` | Problems per training batch |
| `group_size` | `1` | Completions per problem (paper uses 1) |
| `learning_rate` | `2e-5` | Adam learning rate |
| `max_tokens` | `1024` | Max completion length |
| `lora_rank` | `128` | LoRA adapter rank |
| `teacher_sync_every` | `None` | Hard-sync teacher weights every N steps (None = static teacher) |
| `sciknoweval_domain` | `chemistry` | SciKnowEval domain filter |

## Architecture

The implementation consists of:

- `tinker_cookbook/distillation/sdft.py` — Core algorithm: teacher prompt construction, advantage computation, and training loop
- `tinker_cookbook/recipes/sdft/datasets.py` — Dataset loaders for SciKnowEval and ToolAlpaca
- `tinker_cookbook/recipes/sdft/train.py` — CLI entry point

The training loop reuses existing building blocks: `PromptOnlyEnv` for student generation, `do_group_rollout` for rollouts, `assemble_training_data` for datum construction, and `train_step` with `importance_sampling` loss for training.

## References

[1] Shenfeld, I., Damani, M., Hubotter, J., & Agrawal, P. (2026). Self-Distillation Enables Continual Learning. arXiv:2601.19897.
