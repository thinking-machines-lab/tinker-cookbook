# SFT Sweep Results

Empirical hyperparameter sweep results for supervised fine-tuning (SFT).
Use these as a reference when choosing learning rate and LoRA rank for your model.

**Setup:**
- Dataset: tulu3
- Batch size: 128
- Training steps: 780
- Adapter: LoRA

> **Note:** Wall times are approximate and depend on server load at the time of the run. They may fluctuate significantly between runs.

## Key Findings

**For the June 2026 reruns, `3e-04` was the best learning rate for every replacement or newly added model swept.** The best LoRA rank varied by model: ranks 2-4 for the largest MoE models, rank 16 for `Qwen/Qwen3.6-35B-A3B`, and rank 64 for the smaller/base Qwen models.

**Aggressive learning rates still diverge.** The excluded rows are all `test_nll > 2.0`, mostly at `3e-03`; `Qwen/Qwen3.6-35B-A3B` also diverged for rank 1 at `1e-03`.

**Historical supported-model results are retained separately.** Deprecated and replaced model tables were removed; the remaining historical sections are for models still present in the supported model list and not rerun in this pass.

---

## Table of Contents

- [Current Model Reruns (June 2026)](#current-model-reruns-june-2026)
- [Historical Supported Model Results](#historical-supported-model-results)
- [Qwen/Qwen3-8B](#qwen/qwen3-8b)
- [Qwen/Qwen3.5-397B-A17B](#qwen/qwen3.5-397b-a17b)
- [Qwen/Qwen3.5-4B](#qwen/qwen3.5-4b)
- [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](#nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16)
- [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](#nvidia/nvidia-nemotron-3-super-120b-a12b-bf16)
- [openai/gpt-oss-120b](#openai/gpt-oss-120b)
- [openai/gpt-oss-20b](#openai/gpt-oss-20b)

---

## Current Model Reruns (June 2026)

These reruns use the same SFT sweep setup as the earlier results: `tulu3`, batch size 128, 780 training steps, LoRA adapters, and `test/nll` on the held-out split. Long-context `:peft:` variants were excluded.

**Summary:**

| Model | Best LR | Best LoRA Rank | Test NLL | Runs | Diverged |
|-------|--------:|---------------:|---------:|-----:|---------:|
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 3e-04 | 4 | 0.5936 | 15 | 3 |
| `moonshotai/Kimi-K2.6` | 3e-04 | 2 | 0.5578 | 15 | 3 |
| `deepseek-ai/DeepSeek-V3.1` | 3e-04 | 4 | 0.6020 | 15 | 3 |
| `Qwen/Qwen3.6-35B-A3B` | 3e-04 | 16 | 0.6710 | 20 | 5 |
| `Qwen/Qwen3.6-27B` | 3e-04 | 64 | 0.6541 | 20 | 4 |
| `Qwen/Qwen3.5-35B-A3B-Base` | 3e-04 | 64 | 0.6388 | 20 | 4 |
| `Qwen/Qwen3.5-9B-Base` | 3e-04 | 64 | 0.6758 | 20 | 4 |
| `Qwen/Qwen3.5-9B` | 3e-04 | 64 | 0.7166 | 20 | 4 |

**Detailed results:**

### nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.6090 | 0.6195 |
| 1e-04 | 1 | 0.5986 | 0.6106 |
| 3e-04 | 1 | 0.5994 | 0.6145 |
| 1e-03 | 1 | 0.6340 | 0.6464 |
| 4e-05 | 2 | 0.6107 | 0.6220 |
| 1e-04 | 2 | 0.5989 | 0.6121 |
| 3e-04 | 2 | 0.5947 | 0.6093 |
| 1e-03 | 2 | 0.6282 | 0.6403 |
| 4e-05 | 4 | 0.6115 | 0.6215 |
| 1e-04 | 4 | 0.5990 | 0.6122 |
| 3e-04 | 4 | 0.5936 | 0.6065 |
| 1e-03 | 4 | 0.6186 | 0.6316 |

**Best config:** rank=4, lr=3e-04, test_nll=0.5936

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

### moonshotai/Kimi-K2.6

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.5653 | 0.5692 |
| 1e-04 | 1 | 0.5609 | 0.5622 |
| 3e-04 | 1 | 0.5595 | 0.5680 |
| 1e-03 | 1 | 0.5880 | 0.5946 |
| 4e-05 | 2 | 0.5653 | 0.5695 |
| 1e-04 | 2 | 0.5594 | 0.5623 |
| 3e-04 | 2 | 0.5578 | 0.5616 |
| 1e-03 | 2 | 0.5834 | 0.5883 |
| 4e-05 | 4 | 0.5651 | 0.5680 |
| 1e-04 | 4 | 0.5597 | 0.5630 |
| 3e-04 | 4 | 0.5579 | 0.5615 |
| 1e-03 | 4 | 0.5749 | 0.5910 |

**Best config:** rank=2, lr=3e-04, test_nll=0.5578

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

### deepseek-ai/DeepSeek-V3.1

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.6156 | 0.6184 |
| 1e-04 | 1 | 0.6076 | 0.6141 |
| 3e-04 | 1 | 0.6087 | 0.6057 |
| 1e-03 | 1 | 0.6540 | 0.6507 |
| 4e-05 | 2 | 0.6154 | 0.6181 |
| 1e-04 | 2 | 0.6065 | 0.6100 |
| 3e-04 | 2 | 0.6045 | 0.6094 |
| 1e-03 | 2 | 0.6434 | 0.6633 |
| 4e-05 | 4 | 0.6161 | 0.6187 |
| 1e-04 | 4 | 0.6079 | 0.6118 |
| 3e-04 | 4 | 0.6020 | 0.6060 |
| 1e-03 | 4 | 0.6600 | 0.6679 |

**Best config:** rank=4, lr=3e-04, test_nll=0.6020

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

### Qwen/Qwen3.6-35B-A3B

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.6921 | 0.6757 |
| 1e-04 | 1 | 0.6849 | 0.6743 |
| 3e-04 | 1 | 0.6931 | 0.6747 |
| 4e-05 | 4 | 0.6917 | 0.6759 |
| 1e-04 | 4 | 0.6793 | 0.6663 |
| 3e-04 | 4 | 0.6790 | 0.6642 |
| 1e-03 | 4 | 0.7179 | 0.7022 |
| 4e-05 | 16 | 0.6912 | 0.6774 |
| 1e-04 | 16 | 0.6795 | 0.6687 |
| 3e-04 | 16 | 0.6710 | 0.6621 |
| 1e-03 | 16 | 0.6917 | 0.6853 |
| 4e-05 | 64 | 0.6911 | 0.6774 |
| 1e-04 | 64 | 0.6783 | 0.6678 |
| 3e-04 | 64 | 0.6714 | 0.6610 |
| 1e-03 | 64 | 0.6829 | 0.6805 |

**Best config:** rank=16, lr=3e-04, test_nll=0.6710

> **Note:** 5 run(s) diverged (test_nll > 2.0) at lr={1e-03, 3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

### Qwen/Qwen3.6-27B

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.6747 | 0.6698 |
| 1e-04 | 1 | 0.6675 | 0.6622 |
| 3e-04 | 1 | 0.6635 | 0.6598 |
| 1e-03 | 1 | 0.7159 | 0.7063 |
| 4e-05 | 4 | 0.6746 | 0.6712 |
| 1e-04 | 4 | 0.6642 | 0.6596 |
| 3e-04 | 4 | 0.6585 | 0.6578 |
| 1e-03 | 4 | 0.6994 | 0.7032 |
| 4e-05 | 16 | 0.6741 | 0.6698 |
| 1e-04 | 16 | 0.6628 | 0.6589 |
| 3e-04 | 16 | 0.6557 | 0.6537 |
| 1e-03 | 16 | 0.6813 | 0.6836 |
| 4e-05 | 64 | 0.6741 | 0.6712 |
| 1e-04 | 64 | 0.6635 | 0.6625 |
| 3e-04 | 64 | 0.6541 | 0.6517 |
| 1e-03 | 64 | 0.6738 | 0.6702 |

**Best config:** rank=64, lr=3e-04, test_nll=0.6541

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

### Qwen/Qwen3.5-35B-A3B-Base

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 1 | 0.6565 | 0.6460 |
| 1e-04 | 1 | 0.6518 | 0.6438 |
| 3e-04 | 1 | 0.6562 | 0.6469 |
| 1e-03 | 1 | 0.6839 | 0.6737 |
| 4e-05 | 4 | 0.6560 | 0.6478 |
| 1e-04 | 4 | 0.6479 | 0.6383 |
| 3e-04 | 4 | 0.6439 | 0.6392 |
| 1e-03 | 4 | 0.6712 | 0.6599 |
| 4e-05 | 16 | 0.6561 | 0.6480 |
| 1e-04 | 16 | 0.6461 | 0.6380 |
| 3e-04 | 16 | 0.6398 | 0.6346 |
| 1e-03 | 16 | 0.6530 | 0.6554 |
| 4e-05 | 64 | 0.6565 | 0.6471 |
| 1e-04 | 64 | 0.6475 | 0.6385 |
| 3e-04 | 64 | 0.6388 | 0.6316 |
| 1e-03 | 64 | 0.6469 | 0.6479 |

**Best config:** rank=64, lr=3e-04, test_nll=0.6388

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

### Qwen/Qwen3.5-9B-Base

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 4 | 0.7012 | 0.6902 |
| 1e-04 | 4 | 0.6904 | 0.6790 |
| 3e-04 | 4 | 0.6804 | 0.6780 |
| 1e-03 | 4 | 0.7145 | 0.7109 |
| 4e-05 | 16 | 0.7006 | 0.6899 |
| 1e-04 | 16 | 0.6892 | 0.6786 |
| 3e-04 | 16 | 0.6774 | 0.6740 |
| 1e-03 | 16 | 0.6966 | 0.6934 |
| 4e-05 | 64 | 0.7010 | 0.6901 |
| 1e-04 | 64 | 0.6894 | 0.6796 |
| 3e-04 | 64 | 0.6758 | 0.6720 |
| 1e-03 | 64 | 0.6892 | 0.6872 |
| 4e-05 | 128 | 0.7011 | 0.6897 |
| 1e-04 | 128 | 0.6896 | 0.6783 |
| 3e-04 | 128 | 0.6758 | 0.6728 |
| 1e-03 | 128 | 0.6864 | 0.6856 |

**Best config:** rank=64, lr=3e-04, test_nll=0.6758

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

### Qwen/Qwen3.5-9B

| LR | LoRA Rank | Test NLL | Train NLL |
|---:|----------:|---------:|----------:|
| 4e-05 | 4 | 0.7444 | 0.7313 |
| 1e-04 | 4 | 0.7307 | 0.7182 |
| 3e-04 | 4 | 0.7231 | 0.7161 |
| 1e-03 | 4 | 0.7658 | 0.7589 |
| 4e-05 | 16 | 0.7436 | 0.7338 |
| 1e-04 | 16 | 0.7297 | 0.7176 |
| 3e-04 | 16 | 0.7178 | 0.7134 |
| 1e-03 | 16 | 0.7476 | 0.7352 |
| 4e-05 | 64 | 0.7449 | 0.7345 |
| 1e-04 | 64 | 0.7287 | 0.7183 |
| 3e-04 | 64 | 0.7166 | 0.7139 |
| 1e-03 | 64 | 0.7333 | 0.7211 |
| 4e-05 | 128 | 0.7447 | 0.7331 |
| 1e-04 | 128 | 0.7295 | 0.7173 |
| 3e-04 | 128 | 0.7172 | 0.7083 |
| 1e-03 | 128 | 0.7345 | 0.7261 |

**Best config:** rank=64, lr=3e-04, test_nll=0.7166

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## Historical Supported Model Results

These earlier sweep results are retained for supported non-long-context models that were not rerun in the June 2026 replacement sweep.

### Qwen/Qwen3-8B

**Configuration:**
- Model: `Qwen/Qwen3-8B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [4, 16, 64, 128]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-8B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[4, 16, 64, 128]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 4 | 0.5606 | 0.6214 | 76 |
| 4e-05 | 4 | 0.5606 | 0.6214 | 57 |
| 1e-04 | 4 | 0.5525 | 0.6148 | 62 |
| 3e-04 | 4 | 0.5458 | 0.6088 | 57 |
| 1e-03 | 4 | 0.5466 | 0.6131 | 76 |
| 4e-05 | 16 | 0.5605 | 0.6210 | 92 |
| 4e-05 | 16 | 0.5604 | 0.6213 | 55 |
| 1e-04 | 16 | 0.5520 | 0.6141 | 71 |
| 3e-04 | 16 | 0.5438 | 0.6065 | 52 |
| 1e-03 | 16 | 0.5405 | 0.6082 | 53 |
| 4e-05 | 64 | 0.5604 | 0.6209 | 93 |
| 4e-05 | 64 | 0.5604 | 0.6211 | 62 |
| 1e-04 | 64 | 0.5518 | 0.6138 | 83 |
| 3e-04 | 64 | 0.5433 | 0.6065 | 64 |
| 1e-03 | 64 | 0.5382 | 0.6040 | 71 |
| 4e-05 | 128 | 0.5603 | 0.6209 | 69 |
| 1e-04 | 128 | 0.5518 | 0.6136 | 88 |
| 3e-04 | 128 | 0.5433 | 0.6061 | 56 |
| 1e-03 | 128 | 0.5374 | 0.6027 | 71 |

**Best config:** rank=128, lr=1e-03, test_nll=0.5374

**Avg wall time per run:** 69 min

![NLL curves for Qwen/Qwen3-8B](plots/Qwen-Qwen3-8B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

### Qwen/Qwen3.5-397B-A17B

**Configuration:**
- Model: `Qwen/Qwen3.5-397B-A17B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 2, 4]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-397B-A17B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 2, 4]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.4467 | 0.4788 | 369 |
| 4e-05 | 1 | 0.4466 | 0.4793 | 362 |
| 1e-04 | 1 | 0.4444 | 0.4772 | 378 |
| 3e-04 | 1 | 0.4447 | 0.4789 | 356 |
| 1e-03 | 1 | 0.4538 | 0.4874 | 363 |
| 4e-05 | 2 | 0.4465 | 0.4791 | 303 |
| 4e-05 | 2 | 0.4464 | 0.4789 | 361 |
| 1e-04 | 2 | 0.4440 | 0.4768 | 374 |
| 3e-04 | 2 | 0.4431 | 0.4761 | 359 |
| 1e-03 | 2 | 0.4516 | 0.4873 | 802 |
| 4e-05 | 4 | 0.4464 | 0.4792 | 367 |
| 4e-05 | 4 | 0.4464 | 0.4793 | 312 |
| 1e-04 | 4 | 0.4438 | 0.4765 | 358 |
| 3e-04 | 4 | 0.4419 | 0.4745 | 361 |
| 1e-03 | 4 | 0.4589 | 0.4969 | 362 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4419

**Avg wall time per run:** 386 min

![NLL curves for Qwen/Qwen3.5-397B-A17B](plots/Qwen-Qwen3.5-397B-A17B_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

### Qwen/Qwen3.5-4B

**Configuration:**
- Model: `Qwen/Qwen3.5-4B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [4, 16, 64, 128]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-4B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[4, 16, 64, 128]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 4 | 0.5553 | 0.6004 | 57 |
| 4e-05 | 4 | 0.5552 | 0.6003 | 86 |
| 1e-04 | 4 | 0.5498 | 0.5961 | 74 |
| 3e-04 | 4 | 0.5467 | 0.5945 | 62 |
| 1e-03 | 4 | 0.5551 | 0.6071 | 65 |
| 4e-05 | 16 | 0.5547 | 0.5998 | 58 |
| 4e-05 | 16 | 0.5547 | 0.5997 | 85 |
| 1e-04 | 16 | 0.5483 | 0.5938 | 71 |
| 3e-04 | 16 | 0.5424 | 0.5893 | 58 |
| 1e-03 | 16 | 0.5498 | 0.6041 | 64 |
| 4e-05 | 64 | 0.5545 | 0.5992 | 57 |
| 4e-05 | 64 | 0.5548 | 0.5997 | 85 |
| 1e-04 | 64 | 0.5478 | 0.5932 | 61 |
| 3e-04 | 64 | 0.5406 | 0.5879 | 58 |
| 1e-03 | 64 | 0.5438 | 0.5955 | 63 |
| 4e-05 | 128 | 0.5546 | 0.5994 | 73 |
| 1e-04 | 128 | 0.5477 | 0.5932 | 62 |
| 3e-04 | 128 | 0.5402 | 0.5869 | 56 |
| 1e-03 | 128 | 0.5420 | 0.5948 | 59 |

**Best config:** rank=128, lr=3e-04, test_nll=0.5402

**Avg wall time per run:** 66 min

![NLL curves for Qwen/Qwen3.5-4B](plots/Qwen-Qwen3.5-4B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

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

![NLL curves for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](plots/nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16_nll_curves.png)

> **Note:** 5 run(s) diverged (test_nll > 2.0) at lr={2e-03, 4e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

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

![NLL curves for nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](plots/nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16_nll_curves.png)

---

### openai/gpt-oss-120b

**Configuration:**
- Model: `openai/gpt-oss-120b`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=openai/gpt-oss-120b \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5142 | 0.5497 | 150 |
| 1e-04 | 1 | 0.5092 | 0.5446 | 122 |
| 1e-03 | 1 | 0.5111 | 0.5483 | 121 |
| 4e-05 | 4 | 0.5131 | 0.5474 | 120 |
| 1e-04 | 4 | 0.5076 | 0.5439 | 149 |
| 3e-04 | 4 | 0.5048 | 0.5418 | 119 |
| 1e-03 | 4 | 0.5086 | 0.5452 | 148 |
| 3e-03 | 4 | 1.7223 | 1.8536 | 133 |
| 4e-05 | 16 | 0.5129 | 0.5469 | 57 |
| 3e-04 | 16 | 0.5032 | 0.5400 | 113 |
| 1e-03 | 16 | 0.5053 | 0.5434 | 49 |
| 3e-03 | 16 | 0.5477 | 0.5878 | 77 |

**Best config:** rank=16, lr=3e-04, test_nll=0.5032

**Avg wall time per run:** 113 min

![NLL curves for openai/gpt-oss-120b](plots/openai-gpt-oss-120b_nll_curves.png)

> **Note:** 1 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1} and are excluded from the table above.

---

### openai/gpt-oss-20b

**Configuration:**
- Model: `openai/gpt-oss-20b`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04]
- LoRA ranks: [4, 16]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=openai/gpt-oss-20b \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04]' \
    'lora_ranks=[4, 16]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 4 | 0.5474 | 0.5830 | 67 |
| 4e-05 | 4 | 0.5480 | 0.5836 | 49 |
| 1e-04 | 4 | 0.5431 | 0.5790 | 53 |
| 4e-05 | 16 | 0.5473 | 0.5828 | 54 |
| 4e-05 | 16 | 0.5473 | 0.5830 | 34 |
| 1e-04 | 16 | 0.5420 | 0.5779 | 34 |

**Best config:** rank=16, lr=1e-04, test_nll=0.5420

**Avg wall time per run:** 49 min

![NLL curves for openai/gpt-oss-20b](plots/openai-gpt-oss-20b_nll_curves.png)

---
