# SFT Sweep Results

Empirical hyperparameter sweep results for supervised fine-tuning (SFT).
Use these as a reference when choosing learning rate and LoRA rank for your model.

**Setup:**
- Dataset: tulu3
- Batch size: 128
- Training steps: 780
- Adapter: LoRA

> **Note:** Wall times are approximate and depend on server load at the time of the run. They may fluctuate significantly between runs.

## Table of Contents

- [Qwen/Qwen3-235B-A22B-Instruct-2507](#qwen/qwen3-235b-a22b-instruct-2507)
- [Qwen/Qwen3-30B-A3B](#qwen/qwen3-30b-a3b)
- [Qwen/Qwen3-30B-A3B-Base](#qwen/qwen3-30b-a3b-base)
- [Qwen/Qwen3-30B-A3B-Instruct-2507](#qwen/qwen3-30b-a3b-instruct-2507)
- [Qwen/Qwen3-32B](#qwen/qwen3-32b)
- [Qwen/Qwen3-4B-Instruct-2507](#qwen/qwen3-4b-instruct-2507)
- [Qwen/Qwen3-8B](#qwen/qwen3-8b)
- [Qwen/Qwen3-8B-Base](#qwen/qwen3-8b-base)
- [Qwen/Qwen3-VL-235B-A22B-Instruct](#qwen/qwen3-vl-235b-a22b-instruct)
- [Qwen/Qwen3-VL-30B-A3B-Instruct](#qwen/qwen3-vl-30b-a3b-instruct)
- [Qwen/Qwen3.5-27B](#qwen/qwen3.5-27b)
- [Qwen/Qwen3.5-35B-A3B](#qwen/qwen3.5-35b-a3b)
- [Qwen/Qwen3.5-397B-A17B](#qwen/qwen3.5-397b-a17b)
- [Qwen/Qwen3.5-4B](#qwen/qwen3.5-4b)
- [meta-llama/Llama-3.1-70B](#meta-llama/llama-3.1-70b)
- [meta-llama/Llama-3.1-8B](#meta-llama/llama-3.1-8b)
- [meta-llama/Llama-3.1-8B-Instruct](#meta-llama/llama-3.1-8b-instruct)
- [meta-llama/Llama-3.2-1B](#meta-llama/llama-3.2-1b)
- [meta-llama/Llama-3.2-3B](#meta-llama/llama-3.2-3b)
- [meta-llama/Llama-3.3-70B-Instruct](#meta-llama/llama-3.3-70b-instruct)
- [moonshotai/Kimi-K2-Thinking](#moonshotai/kimi-k2-thinking)
- [moonshotai/Kimi-K2.5](#moonshotai/kimi-k2.5)
- [openai/gpt-oss-120b](#openai/gpt-oss-120b)
- [openai/gpt-oss-20b](#openai/gpt-oss-20b)

---

## Qwen/Qwen3-235B-A22B-Instruct-2507

**Configuration:**
- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507`
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
    base.model_name=Qwen/Qwen3-235B-A22B-Instruct-2507 \
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
| 4e-05 | 1 | 0.4665 | 0.5141 | 259 |
| 1e-04 | 1 | 0.4624 | 0.5112 | 175 |
| 3e-04 | 1 | 0.4615 | 0.5106 | 185 |
| 1e-03 | 1 | 0.4702 | 0.5227 | 195 |
| 4e-05 | 2 | 0.4664 | 0.5138 | 185 |
| 1e-04 | 2 | 0.4618 | 0.5101 | 179 |
| 3e-04 | 2 | 0.4597 | 0.5092 | 166 |
| 1e-03 | 2 | 0.4677 | 0.5182 | 198 |
| 4e-05 | 4 | 0.4663 | 0.5136 | 184 |
| 1e-04 | 4 | 0.4616 | 0.5103 | 189 |
| 3e-04 | 4 | 0.4589 | 0.5082 | 182 |
| 1e-03 | 4 | 0.4648 | 0.5159 | 205 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4589

**Avg wall time per run:** 192 min

![NLL curves for Qwen/Qwen3-235B-A22B-Instruct-2507](plots/Qwen-Qwen3-235B-A22B-Instruct-2507_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## Qwen/Qwen3-30B-A3B

**Configuration:**
- Model: `Qwen/Qwen3-30B-A3B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-30B-A3B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5313 | 0.5840 | 104 |
| 1e-04 | 1 | 0.5247 | 0.5793 | 109 |
| 3e-04 | 1 | 0.5230 | 0.5790 | 87 |
| 1e-03 | 1 | 0.5264 | 0.5832 | 271 |
| 4e-05 | 4 | 0.5311 | 0.5843 | 110 |
| 1e-04 | 4 | 0.5235 | 0.5774 | 109 |
| 3e-04 | 4 | 0.5184 | 0.5734 | 106 |
| 1e-03 | 4 | 0.5206 | 0.5778 | 267 |
| 3e-03 | 4 | 0.5550 | 0.6222 | 216 |
| 4e-05 | 16 | 0.5311 | 0.5842 | 104 |
| 1e-04 | 16 | 0.5230 | 0.5763 | 84 |
| 3e-04 | 16 | 0.5171 | 0.5721 | 107 |
| 1e-03 | 16 | 0.5150 | 0.5733 | 265 |
| 3e-03 | 16 | 0.5630 | 0.6391 | 107 |
| 4e-05 | 64 | 0.5310 | 0.5839 | 107 |
| 1e-04 | 64 | 0.5231 | 0.5767 | 86 |
| 3e-04 | 64 | 0.5168 | 0.5715 | 112 |
| 1e-03 | 64 | 0.5126 | 0.5706 | 214 |
| 3e-03 | 64 | 0.5428 | 0.6113 | 103 |

**Best config:** rank=64, lr=1e-03, test_nll=0.5126

**Avg wall time per run:** 140 min

![NLL curves for Qwen/Qwen3-30B-A3B](plots/Qwen-Qwen3-30B-A3B_nll_curves.png)

> **Note:** 1 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1} and are excluded from the table above.

---

## Qwen/Qwen3-30B-A3B-Base

**Configuration:**
- Model: `Qwen/Qwen3-30B-A3B-Base`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-30B-A3B-Base \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5082 | 0.5530 | 64 |
| 1e-04 | 1 | 0.5037 | 0.5492 | 62 |
| 3e-04 | 1 | 0.5028 | 0.5490 | 59 |
| 1e-03 | 1 | 0.5061 | 0.5551 | 62 |
| 4e-05 | 4 | 0.5076 | 0.5523 | 62 |
| 1e-04 | 4 | 0.5025 | 0.5496 | 62 |
| 3e-04 | 4 | 0.4992 | 0.5473 | 63 |
| 1e-03 | 4 | 0.5027 | 0.5514 | 70 |
| 4e-05 | 16 | 0.5075 | 0.5524 | 28 |
| 1e-04 | 16 | 0.5022 | 0.5485 | 28 |
| 3e-04 | 16 | 0.4984 | 0.5460 | 29 |
| 1e-03 | 16 | 0.4986 | 0.5489 | 61 |
| 3e-03 | 16 | 0.5387 | 0.6007 | 370 |
| 4e-05 | 64 | 0.5074 | 0.5523 | 45 |
| 1e-04 | 64 | 0.5023 | 0.5490 | 29 |
| 3e-04 | 64 | 0.4983 | 0.5461 | 29 |
| 1e-03 | 64 | 0.4965 | 0.5474 | 41 |
| 3e-03 | 64 | 0.5333 | 0.5965 | 296 |

**Best config:** rank=64, lr=1e-03, test_nll=0.4965

**Avg wall time per run:** 81 min

![NLL curves for Qwen/Qwen3-30B-A3B-Base](plots/Qwen-Qwen3-30B-A3B-Base_nll_curves.png)

> **Note:** 2 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4} and are excluded from the table above.

---

## Qwen/Qwen3-30B-A3B-Instruct-2507

**Configuration:**
- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5153 | 0.5703 | 106 |
| 1e-04 | 1 | 0.5103 | 0.5667 | 88 |
| 3e-04 | 1 | 0.5088 | 0.5647 | 72 |
| 1e-03 | 1 | 0.5127 | 0.5698 | 80 |
| 4e-05 | 4 | 0.5154 | 0.5700 | 68 |
| 1e-04 | 4 | 0.5087 | 0.5646 | 63 |
| 3e-04 | 4 | 0.5047 | 0.5617 | 62 |
| 1e-03 | 4 | 0.5068 | 0.5658 | 78 |
| 4e-05 | 16 | 0.5155 | 0.5695 | 66 |
| 1e-04 | 16 | 0.5084 | 0.5641 | 69 |
| 3e-04 | 16 | 0.5032 | 0.5599 | 43 |
| 1e-03 | 16 | 0.5029 | 0.5622 | 70 |
| 4e-05 | 64 | 0.5154 | 0.5696 | 89 |
| 1e-04 | 64 | 0.5083 | 0.5635 | 68 |
| 3e-04 | 64 | 0.5029 | 0.5591 | 49 |
| 1e-03 | 64 | 0.5004 | 0.5587 | 46 |

**Best config:** rank=64, lr=1e-03, test_nll=0.5004

**Avg wall time per run:** 70 min

![NLL curves for Qwen/Qwen3-30B-A3B-Instruct-2507](plots/Qwen-Qwen3-30B-A3B-Instruct-2507_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

## Qwen/Qwen3-32B

**Configuration:**
- Model: `Qwen/Qwen3-32B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-32B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5130 | 0.5668 | 100 |
| 1e-04 | 1 | 0.5073 | 0.5622 | 90 |
| 3e-04 | 1 | 0.5043 | 0.5595 | 455 |
| 4e-05 | 4 | 0.5124 | 0.5659 | 54 |
| 1e-04 | 4 | 0.5061 | 0.5612 | 86 |
| 3e-04 | 4 | 0.5011 | 0.5567 | 470 |
| 4e-05 | 16 | 0.5124 | 0.5660 | 105 |
| 1e-04 | 16 | 0.5058 | 0.5608 | 71 |
| 4e-05 | 64 | 0.5124 | 0.5655 | 80 |
| 1e-04 | 64 | 0.5055 | 0.5604 | 436 |

**Best config:** rank=4, lr=3e-04, test_nll=0.5011

**Avg wall time per run:** 195 min

![NLL curves for Qwen/Qwen3-32B](plots/Qwen-Qwen3-32B_nll_curves.png)

> **Note:** 1 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4} and are excluded from the table above.

---

## Qwen/Qwen3-4B-Instruct-2507

**Configuration:**
- Model: `Qwen/Qwen3-4B-Instruct-2507`
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
    base.model_name=Qwen/Qwen3-4B-Instruct-2507 \
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
| 4e-05 | 4 | 0.5817 | 0.6545 | 24 |
| 1e-04 | 4 | 0.5737 | 0.6485 | 31 |
| 3e-04 | 4 | 0.5676 | 0.6438 | 25 |
| 1e-03 | 4 | 0.5702 | 0.6497 | 31 |
| 4e-05 | 16 | 0.5815 | 0.6543 | 37 |
| 1e-04 | 16 | 0.5730 | 0.6478 | 28 |
| 3e-04 | 16 | 0.5651 | 0.6408 | 26 |
| 1e-03 | 16 | 0.5631 | 0.6426 | 31 |
| 4e-05 | 64 | 0.5815 | 0.6541 | 37 |
| 1e-04 | 64 | 0.5725 | 0.6466 | 28 |
| 3e-04 | 64 | 0.5641 | 0.6397 | 35 |
| 1e-03 | 64 | 0.5596 | 0.6391 | 26 |
| 4e-05 | 128 | 0.5815 | 0.6541 | 31 |
| 1e-04 | 128 | 0.5725 | 0.6469 | 29 |
| 3e-04 | 128 | 0.5639 | 0.6398 | 35 |
| 1e-03 | 128 | 0.5588 | 0.6379 | 29 |

**Best config:** rank=128, lr=1e-03, test_nll=0.5588

**Avg wall time per run:** 30 min

![NLL curves for Qwen/Qwen3-4B-Instruct-2507](plots/Qwen-Qwen3-4B-Instruct-2507_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## Qwen/Qwen3-8B

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
| 4e-05 | 4 | 0.5606 | 0.6214 | 33 |
| 1e-04 | 4 | 0.5525 | 0.6148 | 38 |
| 3e-04 | 4 | 0.5458 | 0.6088 | 33 |
| 1e-03 | 4 | 0.5466 | 0.6131 | 49 |
| 4e-05 | 16 | 0.5604 | 0.6213 | 32 |
| 1e-04 | 16 | 0.5520 | 0.6141 | 44 |
| 3e-04 | 16 | 0.5438 | 0.6065 | 29 |
| 1e-03 | 16 | 0.5405 | 0.6082 | 31 |
| 4e-05 | 64 | 0.5604 | 0.6211 | 37 |
| 1e-04 | 64 | 0.5518 | 0.6138 | 54 |
| 3e-04 | 64 | 0.5433 | 0.6065 | 39 |
| 1e-03 | 64 | 0.5382 | 0.6040 | 43 |
| 4e-05 | 128 | 0.5603 | 0.6209 | 42 |
| 1e-04 | 128 | 0.5518 | 0.6136 | 57 |
| 3e-04 | 128 | 0.5433 | 0.6061 | 32 |
| 1e-03 | 128 | 0.5374 | 0.6027 | 44 |

**Best config:** rank=128, lr=1e-03, test_nll=0.5374

**Avg wall time per run:** 40 min

![NLL curves for Qwen/Qwen3-8B](plots/Qwen-Qwen3-8B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## Qwen/Qwen3-8B-Base

**Configuration:**
- Model: `Qwen/Qwen3-8B-Base`
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
    base.model_name=Qwen/Qwen3-8B-Base \
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
| 4e-05 | 4 | 0.5321 | 0.5864 | 60 |
| 1e-04 | 4 | 0.5271 | 0.5824 | 58 |
| 3e-04 | 4 | 0.5234 | 0.5794 | 53 |
| 1e-03 | 4 | 0.5261 | 0.5852 | 52 |
| 4e-05 | 16 | 0.5320 | 0.5861 | 55 |
| 1e-04 | 16 | 0.5269 | 0.5823 | 59 |
| 3e-04 | 16 | 0.5222 | 0.5787 | 50 |
| 1e-03 | 16 | 0.5219 | 0.5820 | 54 |
| 4e-05 | 64 | 0.5319 | 0.5863 | 56 |
| 1e-04 | 64 | 0.5268 | 0.5820 | 54 |
| 3e-04 | 64 | 0.5216 | 0.5781 | 52 |
| 1e-03 | 64 | 0.5195 | 0.5794 | 62 |
| 4e-05 | 128 | 0.5321 | 0.5862 | 57 |
| 1e-04 | 128 | 0.5267 | 0.5820 | 54 |
| 3e-04 | 128 | 0.5217 | 0.5780 | 56 |
| 1e-03 | 128 | 0.5189 | 0.5784 | 63 |

**Best config:** rank=128, lr=1e-03, test_nll=0.5189

**Avg wall time per run:** 56 min

![NLL curves for Qwen/Qwen3-8B-Base](plots/Qwen-Qwen3-8B-Base_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## Qwen/Qwen3-VL-235B-A22B-Instruct

**Configuration:**
- Model: `Qwen/Qwen3-VL-235B-A22B-Instruct`
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
    base.model_name=Qwen/Qwen3-VL-235B-A22B-Instruct \
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
| 4e-05 | 1 | 0.4687 | 0.5199 | 221 |
| 1e-04 | 1 | 0.4645 | 0.5165 | 247 |
| 3e-04 | 1 | 0.4639 | 0.5166 | 216 |
| 1e-03 | 1 | 0.4733 | 0.5248 | 223 |
| 4e-05 | 2 | 0.4687 | 0.5199 | 223 |
| 1e-04 | 2 | 0.4637 | 0.5157 | 233 |
| 3e-04 | 2 | 0.4623 | 0.5147 | 225 |
| 1e-03 | 2 | 0.4703 | 0.5247 | 220 |
| 4e-05 | 4 | 0.4684 | 0.5197 | 230 |
| 1e-04 | 4 | 0.4639 | 0.5162 | 241 |
| 3e-04 | 4 | 0.4610 | 0.5137 | 215 |
| 1e-03 | 4 | 0.4668 | 0.5222 | 212 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4610

**Avg wall time per run:** 225 min

![NLL curves for Qwen/Qwen3-VL-235B-A22B-Instruct](plots/Qwen-Qwen3-VL-235B-A22B-Instruct_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## Qwen/Qwen3-VL-30B-A3B-Instruct

**Configuration:**
- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5094 | 0.5571 | 99 |
| 1e-04 | 1 | 0.5047 | 0.5547 | 98 |
| 3e-04 | 1 | 0.5044 | 0.5537 | 89 |
| 1e-03 | 1 | 0.5085 | 0.5592 | 114 |
| 4e-05 | 4 | 0.5083 | 0.5562 | 97 |
| 1e-04 | 4 | 0.5031 | 0.5520 | 97 |
| 3e-04 | 4 | 0.4997 | 0.5497 | 100 |
| 1e-03 | 4 | 0.5046 | 0.5567 | 115 |
| 3e-03 | 4 | 0.5726 | 0.6289 | 123 |
| 4e-05 | 16 | 0.5083 | 0.5562 | 98 |
| 1e-04 | 16 | 0.5022 | 0.5512 | 82 |
| 3e-04 | 16 | 0.4980 | 0.5473 | 97 |
| 1e-03 | 16 | 0.4994 | 0.5531 | 119 |
| 4e-05 | 64 | 0.5083 | 0.5560 | 96 |
| 1e-04 | 64 | 0.5022 | 0.5513 | 88 |
| 3e-04 | 64 | 0.4975 | 0.5474 | 114 |
| 1e-03 | 64 | 0.4971 | 0.5496 | 121 |

**Best config:** rank=64, lr=1e-03, test_nll=0.4971

**Avg wall time per run:** 103 min

![NLL curves for Qwen/Qwen3-VL-30B-A3B-Instruct](plots/Qwen-Qwen3-VL-30B-A3B-Instruct_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 16, 64} and are excluded from the table above.

---

## Qwen/Qwen3.5-27B

**Configuration:**
- Model: `Qwen/Qwen3.5-27B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-27B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.4774 | 0.5191 | 78 |
| 1e-04 | 1 | 0.4745 | 0.5164 | 93 |
| 3e-04 | 1 | 0.4733 | 0.5165 | 57 |
| 1e-03 | 1 | 0.4829 | 0.5258 | 89 |
| 4e-05 | 4 | 0.4769 | 0.5187 | 56 |
| 1e-04 | 4 | 0.4731 | 0.5154 | 93 |
| 3e-04 | 4 | 0.4701 | 0.5133 | 82 |
| 1e-03 | 4 | 0.4793 | 0.5260 | 94 |
| 4e-05 | 16 | 0.4770 | 0.5189 | 80 |
| 1e-04 | 16 | 0.4728 | 0.5150 | 69 |
| 3e-04 | 16 | 0.4688 | 0.5119 | 70 |
| 1e-03 | 16 | 0.4738 | 0.5199 | 84 |
| 4e-05 | 64 | 0.4770 | 0.5185 | 84 |
| 1e-04 | 64 | 0.4727 | 0.5151 | 81 |
| 3e-04 | 64 | 0.4682 | 0.5118 | 71 |
| 1e-03 | 64 | 0.4704 | 0.5171 | 77 |

**Best config:** rank=64, lr=3e-04, test_nll=0.4682

**Avg wall time per run:** 79 min

![NLL curves for Qwen/Qwen3.5-27B](plots/Qwen-Qwen3.5-27B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

## Qwen/Qwen3.5-35B-A3B

**Configuration:**
- Model: `Qwen/Qwen3.5-35B-A3B`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 4, 16, 64]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=Qwen/Qwen3.5-35B-A3B \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 4, 16, 64]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.4926 | 0.5308 | 94 |
| 1e-04 | 1 | 0.4900 | 0.5288 | 88 |
| 3e-04 | 1 | 0.4906 | 0.5309 | 90 |
| 1e-03 | 1 | 0.4951 | 0.5365 | 94 |
| 4e-05 | 4 | 0.4917 | 0.5298 | 93 |
| 1e-04 | 4 | 0.4879 | 0.5265 | 87 |
| 3e-04 | 4 | 0.4856 | 0.5251 | 96 |
| 1e-03 | 4 | 0.4928 | 0.5330 | 79 |
| 4e-05 | 16 | 0.4917 | 0.5295 | 32 |
| 1e-04 | 16 | 0.4873 | 0.5257 | 29 |
| 3e-04 | 16 | 0.4835 | 0.5226 | 28 |
| 1e-03 | 16 | 0.4875 | 0.5303 | 27 |
| 4e-05 | 64 | 0.4916 | 0.5296 | 27 |
| 1e-04 | 64 | 0.4872 | 0.5252 | 30 |
| 3e-04 | 64 | 0.4827 | 0.5226 | 50 |
| 1e-03 | 64 | 0.4841 | 0.5264 | 67 |

**Best config:** rank=64, lr=3e-04, test_nll=0.4827

**Avg wall time per run:** 63 min

![NLL curves for Qwen/Qwen3.5-35B-A3B](plots/Qwen-Qwen3.5-35B-A3B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 4, 16, 64} and are excluded from the table above.

---

## Qwen/Qwen3.5-397B-A17B

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
| 4e-05 | 1 | 0.4466 | 0.4793 | 276 |
| 1e-04 | 1 | 0.4444 | 0.4772 | 294 |
| 3e-04 | 1 | 0.4447 | 0.4789 | 275 |
| 1e-03 | 1 | 0.4538 | 0.4874 | 278 |
| 4e-05 | 2 | 0.4464 | 0.4789 | 274 |
| 1e-04 | 2 | 0.4440 | 0.4768 | 292 |
| 3e-04 | 2 | 0.4431 | 0.4761 | 276 |
| 1e-03 | 2 | 0.4516 | 0.4873 | 272 |
| 4e-05 | 4 | 0.4464 | 0.4793 | 236 |
| 1e-04 | 4 | 0.4438 | 0.4765 | 276 |
| 3e-04 | 4 | 0.4419 | 0.4745 | 278 |
| 1e-03 | 4 | 0.4589 | 0.4969 | 277 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4419

**Avg wall time per run:** 275 min

![NLL curves for Qwen/Qwen3.5-397B-A17B](plots/Qwen-Qwen3.5-397B-A17B_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## Qwen/Qwen3.5-4B

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
| 4e-05 | 4 | 0.5552 | 0.6003 | 58 |
| 1e-04 | 4 | 0.5498 | 0.5961 | 47 |
| 3e-04 | 4 | 0.5467 | 0.5945 | 38 |
| 1e-03 | 4 | 0.5551 | 0.6071 | 39 |
| 4e-05 | 16 | 0.5547 | 0.5997 | 57 |
| 1e-04 | 16 | 0.5483 | 0.5938 | 45 |
| 3e-04 | 16 | 0.5424 | 0.5893 | 36 |
| 1e-03 | 16 | 0.5498 | 0.6041 | 39 |
| 4e-05 | 64 | 0.5548 | 0.5997 | 56 |
| 1e-04 | 64 | 0.5478 | 0.5932 | 37 |
| 3e-04 | 64 | 0.5406 | 0.5879 | 34 |
| 1e-03 | 64 | 0.5438 | 0.5955 | 38 |
| 4e-05 | 128 | 0.5546 | 0.5994 | 46 |
| 1e-04 | 128 | 0.5477 | 0.5932 | 38 |
| 3e-04 | 128 | 0.5402 | 0.5869 | 33 |
| 1e-03 | 128 | 0.5420 | 0.5948 | 35 |

**Best config:** rank=128, lr=3e-04, test_nll=0.5402

**Avg wall time per run:** 42 min

![NLL curves for Qwen/Qwen3.5-4B](plots/Qwen-Qwen3.5-4B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## meta-llama/Llama-3.1-70B

**Configuration:**
- Model: `meta-llama/Llama-3.1-70B`
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
    base.model_name=meta-llama/Llama-3.1-70B \
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
| 4e-05 | 1 | 0.5131 | 0.5557 | 217 |
| 1e-04 | 1 | 0.5064 | 0.5500 | 242 |
| 3e-04 | 1 | 0.5045 | 0.5486 | 223 |
| 4e-05 | 2 | 0.5125 | 0.5549 | 232 |
| 1e-04 | 2 | 0.5053 | 0.5496 | 243 |
| 3e-04 | 2 | 0.5015 | 0.5464 | 222 |
| 4e-05 | 4 | 0.5120 | 0.5548 | 223 |
| 1e-04 | 4 | 0.5047 | 0.5487 | 241 |
| 3e-04 | 4 | 0.4997 | 0.5454 | 221 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4997

**Avg wall time per run:** 229 min

![NLL curves for meta-llama/Llama-3.1-70B](plots/meta-llama-Llama-3.1-70B_nll_curves.png)

> **Note:** 6 run(s) diverged (test_nll > 2.0) at lr={1e-03, 3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## meta-llama/Llama-3.1-8B

**Configuration:**
- Model: `meta-llama/Llama-3.1-8B`
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
    base.model_name=meta-llama/Llama-3.1-8B \
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
| 4e-05 | 4 | 0.6120 | 0.6617 | 57 |
| 1e-04 | 4 | 0.6005 | 0.6512 | 51 |
| 3e-04 | 4 | 0.5925 | 0.6440 | 55 |
| 1e-03 | 4 | 0.6021 | 0.6611 | 52 |
| 4e-05 | 16 | 0.6112 | 0.6612 | 54 |
| 1e-04 | 16 | 0.5983 | 0.6494 | 55 |
| 3e-04 | 16 | 0.5863 | 0.6388 | 54 |
| 1e-03 | 16 | 0.5931 | 0.6525 | 66 |
| 4e-05 | 64 | 0.6110 | 0.6610 | 51 |
| 1e-04 | 64 | 0.5977 | 0.6485 | 53 |
| 3e-04 | 64 | 0.5839 | 0.6373 | 53 |
| 1e-03 | 64 | 0.5839 | 0.6424 | 65 |
| 4e-05 | 128 | 0.6109 | 0.6609 | 54 |
| 1e-04 | 128 | 0.5976 | 0.6484 | 58 |
| 3e-04 | 128 | 0.5834 | 0.6371 | 52 |
| 1e-03 | 128 | 0.5847 | 0.6413 | 56 |

**Best config:** rank=128, lr=3e-04, test_nll=0.5834

**Avg wall time per run:** 55 min

![NLL curves for meta-llama/Llama-3.1-8B](plots/meta-llama-Llama-3.1-8B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## meta-llama/Llama-3.1-8B-Instruct

**Configuration:**
- Model: `meta-llama/Llama-3.1-8B-Instruct`
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
    base.model_name=meta-llama/Llama-3.1-8B-Instruct \
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
| 4e-05 | 4 | 0.5941 | 0.6517 | 32 |
| 1e-04 | 4 | 0.5851 | 0.6435 | 45 |
| 3e-04 | 4 | 0.5792 | 0.6409 | 43 |
| 1e-03 | 4 | 0.5867 | 0.6527 | 28 |
| 4e-05 | 16 | 0.5933 | 0.6508 | 28 |
| 1e-04 | 16 | 0.5837 | 0.6417 | 28 |
| 3e-04 | 16 | 0.5750 | 0.6349 | 27 |
| 1e-03 | 16 | 0.5809 | 0.6483 | 39 |
| 4e-05 | 64 | 0.5932 | 0.6509 | 27 |
| 1e-04 | 64 | 0.5830 | 0.6408 | 27 |
| 3e-04 | 64 | 0.5730 | 0.6327 | 41 |
| 1e-03 | 64 | 0.5756 | 0.6423 | 27 |
| 4e-05 | 128 | 0.5932 | 0.6509 | 44 |
| 1e-04 | 128 | 0.5828 | 0.6409 | 41 |
| 3e-04 | 128 | 0.5727 | 0.6328 | 42 |
| 1e-03 | 128 | 0.5743 | 0.6390 | 38 |

**Best config:** rank=128, lr=3e-04, test_nll=0.5727

**Avg wall time per run:** 35 min

![NLL curves for meta-llama/Llama-3.1-8B-Instruct](plots/meta-llama-Llama-3.1-8B-Instruct_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## meta-llama/Llama-3.2-1B

**Configuration:**
- Model: `meta-llama/Llama-3.2-1B`
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
    base.model_name=meta-llama/Llama-3.2-1B \
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
| 4e-05 | 4 | 0.8562 | 0.9285 | 24 |
| 1e-04 | 4 | 0.8355 | 0.9102 | 24 |
| 3e-04 | 4 | 0.8192 | 0.8974 | 25 |
| 1e-03 | 4 | 0.8162 | 0.8958 | 23 |
| 4e-05 | 16 | 0.8540 | 0.9259 | 24 |
| 1e-04 | 16 | 0.8297 | 0.9051 | 24 |
| 3e-04 | 16 | 0.8038 | 0.8841 | 25 |
| 1e-03 | 16 | 0.7948 | 0.8800 | 23 |
| 4e-05 | 64 | 0.8533 | 0.9251 | 24 |
| 1e-04 | 64 | 0.8275 | 0.9029 | 23 |
| 3e-04 | 64 | 0.7978 | 0.8768 | 23 |
| 1e-03 | 64 | 0.7802 | 0.8656 | 24 |
| 4e-05 | 128 | 0.8532 | 0.9251 | 25 |
| 1e-04 | 128 | 0.8271 | 0.9027 | 25 |
| 3e-04 | 128 | 0.7966 | 0.8759 | 24 |
| 1e-03 | 128 | 0.7763 | 0.8600 | 26 |

**Best config:** rank=128, lr=1e-03, test_nll=0.7763

**Avg wall time per run:** 24 min

![NLL curves for meta-llama/Llama-3.2-1B](plots/meta-llama-Llama-3.2-1B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## meta-llama/Llama-3.2-3B

**Configuration:**
- Model: `meta-llama/Llama-3.2-3B`
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
    base.model_name=meta-llama/Llama-3.2-3B \
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
| 4e-05 | 4 | 0.7132 | 0.7738 | 39 |
| 1e-04 | 4 | 0.6966 | 0.7577 | 38 |
| 3e-04 | 4 | 0.6819 | 0.7458 | 38 |
| 1e-03 | 4 | 0.6808 | 0.7502 | 37 |
| 4e-05 | 16 | 0.7120 | 0.7722 | 36 |
| 1e-04 | 16 | 0.6936 | 0.7550 | 39 |
| 3e-04 | 16 | 0.6737 | 0.7380 | 35 |
| 1e-03 | 16 | 0.6661 | 0.7371 | 35 |
| 4e-05 | 64 | 0.7119 | 0.7719 | 37 |
| 1e-04 | 64 | 0.6927 | 0.7542 | 36 |
| 3e-04 | 64 | 0.6710 | 0.7355 | 36 |
| 1e-03 | 64 | 0.6580 | 0.7269 | 38 |
| 4e-05 | 128 | 0.7119 | 0.7716 | 39 |
| 1e-04 | 128 | 0.6925 | 0.7544 | 38 |
| 3e-04 | 128 | 0.6706 | 0.7353 | 36 |
| 1e-03 | 128 | 0.6559 | 0.7245 | 35 |

**Best config:** rank=128, lr=1e-03, test_nll=0.6559

**Avg wall time per run:** 37 min

![NLL curves for meta-llama/Llama-3.2-3B](plots/meta-llama-Llama-3.2-3B_nll_curves.png)

> **Note:** 4 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={4, 16, 64, 128} and are excluded from the table above.

---

## meta-llama/Llama-3.3-70B-Instruct

**Configuration:**
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 1e-04, 3e-04]
- LoRA ranks: [1, 2, 4]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=meta-llama/Llama-3.3-70B-Instruct \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 1e-04, 3e-04]' \
    'lora_ranks=[1, 2, 4]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.5160 | 0.5632 | 415 |
| 1e-04 | 1 | 0.5095 | 0.5585 | 313 |
| 3e-04 | 1 | 0.5068 | 0.5562 | 381 |
| 4e-05 | 2 | 0.5158 | 0.5632 | 388 |
| 1e-04 | 2 | 0.5078 | 0.5564 | 343 |
| 3e-04 | 2 | 0.5042 | 0.5548 | 365 |
| 4e-05 | 4 | 0.5153 | 0.5630 | 377 |
| 1e-04 | 4 | 0.5076 | 0.5560 | 355 |
| 3e-04 | 4 | 0.5021 | 0.5526 | 367 |

**Best config:** rank=4, lr=3e-04, test_nll=0.5021

**Avg wall time per run:** 367 min

![NLL curves for meta-llama/Llama-3.3-70B-Instruct](plots/meta-llama-Llama-3.3-70B-Instruct_nll_curves.png)

---

## moonshotai/Kimi-K2-Thinking

**Configuration:**
- Model: `moonshotai/Kimi-K2-Thinking`
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
    base.model_name=moonshotai/Kimi-K2-Thinking \
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
| 4e-05 | 1 | 0.4760 | 0.5060 | 317 |
| 1e-04 | 1 | 0.4729 | 0.5041 | 413 |
| 1e-03 | 1 | 0.4778 | 0.5111 | 434 |
| 4e-05 | 2 | 0.4760 | 0.5055 | 394 |
| 3e-04 | 2 | 0.4710 | 0.5029 | 421 |
| 1e-03 | 2 | 0.4761 | 0.5081 | 432 |
| 4e-05 | 4 | 0.4760 | 0.5051 | 397 |
| 3e-04 | 4 | 0.4707 | 0.5020 | 420 |
| 1e-03 | 4 | 0.4741 | 0.5067 | 405 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4707

**Avg wall time per run:** 404 min

![NLL curves for moonshotai/Kimi-K2-Thinking](plots/moonshotai-Kimi-K2-Thinking_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## moonshotai/Kimi-K2.5

**Configuration:**
- Model: `moonshotai/Kimi-K2.5`
- Dataset: tulu3 (train split for training, test split for evaluation)
- Batch size: 128
- Learning rates: [4e-05, 3e-04, 1e-03, 3e-03]
- LoRA ranks: [1, 2, 4]
- Metric: `test/nll` — negative log-likelihood on held-out test split

<details>
<summary>Reproduce</summary>

```bash
uv run python -m tinker_cookbook.recipes.chat_sl.sweep \
    recipe=sft \
    base.model_name=moonshotai/Kimi-K2.5 \
    base.dataset=tulu3 \
    base.batch_size=128 \
    metric=test/nll \
    'learning_rates=[4e-05, 3e-04, 1e-03, 3e-03]' \
    'lora_ranks=[1, 2, 4]'
```

</details>

**Results:**

| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |
|---:|----------:|---------:|----------:|----------------:|
| 4e-05 | 1 | 0.4669 | 0.4919 | 461 |
| 1e-03 | 1 | 0.4766 | 0.5050 | 542 |
| 4e-05 | 2 | 0.4669 | 0.4918 | 376 |
| 1e-03 | 2 | 0.4736 | 0.5025 | 510 |
| 4e-05 | 4 | 0.4666 | 0.4915 | 459 |
| 3e-04 | 4 | 0.4634 | 0.4907 | 522 |
| 1e-03 | 4 | 0.4696 | 0.4980 | 554 |

**Best config:** rank=4, lr=3e-04, test_nll=0.4634

**Avg wall time per run:** 489 min

![NLL curves for moonshotai/Kimi-K2.5](plots/moonshotai-Kimi-K2.5_nll_curves.png)

> **Note:** 3 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1, 2, 4} and are excluded from the table above.

---

## openai/gpt-oss-120b

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
| 4e-05 | 1 | 0.5142 | 0.5497 | 110 |
| 1e-04 | 1 | 0.5092 | 0.5446 | 87 |
| 1e-03 | 1 | 0.5111 | 0.5483 | 86 |
| 4e-05 | 4 | 0.5131 | 0.5474 | 85 |
| 1e-04 | 4 | 0.5076 | 0.5439 | 109 |
| 3e-04 | 4 | 0.5048 | 0.5418 | 83 |
| 1e-03 | 4 | 0.5086 | 0.5452 | 111 |
| 3e-03 | 4 | 1.7223 | 1.8536 | 101 |
| 3e-04 | 16 | 0.5032 | 0.5400 | 79 |
| 1e-03 | 16 | 0.5053 | 0.5434 | 29 |
| 3e-03 | 16 | 0.5477 | 0.5878 | 51 |

**Best config:** rank=16, lr=3e-04, test_nll=0.5032

**Avg wall time per run:** 84 min

![NLL curves for openai/gpt-oss-120b](plots/openai-gpt-oss-120b_nll_curves.png)

> **Note:** 1 run(s) diverged (test_nll > 2.0) at lr={3e-03} with rank={1} and are excluded from the table above.

---

## openai/gpt-oss-20b

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
| 4e-05 | 4 | 0.5480 | 0.5836 | 28 |
| 1e-04 | 4 | 0.5431 | 0.5790 | 32 |
| 4e-05 | 16 | 0.5473 | 0.5830 | 17 |
| 1e-04 | 16 | 0.5420 | 0.5779 | 18 |

**Best config:** rank=16, lr=1e-04, test_nll=0.5420

**Avg wall time per run:** 24 min

![NLL curves for openai/gpt-oss-20b](plots/openai-gpt-oss-20b_nll_curves.png)

---
