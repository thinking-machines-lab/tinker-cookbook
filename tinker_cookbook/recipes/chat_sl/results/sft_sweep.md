# SFT Sweep Results

Empirical hyperparameter sweep results for supervised fine-tuning (SFT).
Use these as a reference when choosing learning rate and LoRA rank for your model.

**Setup:**
- Dataset: tulu3
- Batch size: 128
- Training steps: 780
- Adapter: LoRA

---

## DeepSeek V3.1 Base

**Configuration:**
- Model: `deepseek-ai/DeepSeek-V3.1-Base`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 2e-04, 4e-04, 1e-03]
- LoRA ranks: [1, 2, 4]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=deepseek-ai/DeepSeek-V3.1-Base \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 2e-04, 4e-04, 1e-03]' \
    'lora_ranks=[1, 2, 4]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 1e-04 | 1 | 0.4853 | 0.5137 | 557 |
| 1e-03 | 1 | 0.4955 | 0.5252 | 557 |
| 4e-05 | 2 | 0.4886 | 0.5160 | 571 |
| 2e-04 | 2 | 0.4835 | 0.5127 | 566 |
| 4e-04 | 2 | 0.4842 | 0.5132 | 550 |
| 1e-03 | 2 | 0.4938 | 0.5246 | 566 |
| 4e-05 | 4 | 0.4881 | 0.5159 | 561 |
| 1e-04 | 4 | 0.4848 | 0.5137 | 560 |
| 4e-04 | 4 | 0.4826 | 0.5128 | 566 |
| 1e-03 | 4 | 0.4904 | 0.5221 | 546 |

**Best config:** rank=4, lr=4e-04, test_nll=0.4826

**Avg wall time per run:** 560 min

![NLL curves for DeepSeek V3.1 Base](plots/deepseek-ai-DeepSeek-V3.1-Base_nll_curves.png)

---

## Nemotron Nano 30B (3B active)

**Configuration:**
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 2e-04, 4e-04, 2e-03, 4e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 2e-04, 4e-04, 2e-03, 4e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5656 | 0.6359 | 161 |
| 1e-04 | 1 | 0.5573 | 0.6288 | 159 |
| 2e-04 | 1 | 0.5546 | 0.6273 | 162 |
| 4e-04 | 1 | 0.5540 | 0.6275 | 161 |
| 4e-05 | 4 | 0.5649 | 0.6357 | 157 |
| 1e-04 | 4 | 0.5555 | 0.6274 | 162 |
| 2e-04 | 4 | 0.5515 | 0.6235 | 154 |
| 4e-04 | 4 | 0.5482 | 0.6228 | 161 |
| 4e-05 | 16 | 0.5647 | 0.6345 | 161 |
| 1e-04 | 16 | 0.5551 | 0.6273 | 152 |
| 4e-04 | 16 | 0.5455 | 0.6190 | 158 |
| 2e-03 | 16 | 0.5532 | 0.6329 | 160 |
| 4e-05 | 64 | 0.5649 | 0.6353 | 161 |
| 2e-04 | 64 | 0.5501 | 0.6219 | 162 |
| 4e-04 | 64 | 0.5449 | 0.6181 | 158 |
| 2e-03 | 64 | 0.5469 | 0.6240 | 67 |

**Best config:** rank=64, lr=4e-04, test_nll=0.5449

**Avg wall time per run:** 153 min

![NLL curves for Nemotron Nano 30B (3B active)](plots/nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_nll_curves.png)

> **Note:** 5 run(s) diverged (test_nll > 2.0) at lr={2e-03, 4e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

## Nemotron Super 120B (12B active)

**Configuration:**
- Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 2e-04, 4e-04, 1e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 2e-04, 4e-04, 1e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-04 | 1 | 0.4835 | 0.5401 | 155 |
| 4e-05 | 4 | 0.4899 | 0.5395 | 151 |
| 2e-04 | 4 | 0.4808 | 0.5354 | 153 |
| 4e-05 | 16 | 0.4900 | 0.5401 | 156 |
| 1e-04 | 16 | 0.4832 | 0.5353 | 155 |
| 2e-04 | 16 | 0.4802 | 0.5345 | 152 |
| 4e-04 | 16 | 0.4783 | 0.5334 | 152 |
| 2e-04 | 64 | 0.4799 | 0.5341 | 155 |
| 4e-04 | 64 | 0.4776 | 0.5330 | 150 |
| 1e-03 | 64 | 0.4767 | 0.5348 | 152 |

**Best config:** rank=64, lr=1e-03, test_nll=0.4767

**Avg wall time per run:** 153 min

![NLL curves for Nemotron Super 120B (12B active)](plots/nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16_nll_curves.png)

---
