# RL for Instruction Following (IFBench)

[IFBench](https://github.com/allenai/IFBench) is a challenging benchmark for precise instruction following from AllenAI. It tests whether models can satisfy specific constraints embedded in prompts (e.g., word count limits, formatting rules, keyword inclusion).

This recipe trains models using RL on the [IFBench test dataset](https://huggingface.co/datasets/allenai/IFBench_test).

## Reward Types

The environment supports four reward strategies:

| Type             | Description                                        |
| ---------------- | -------------------------------------------------- |
| `FULL_STRICT`    | 1.0 if ALL instructions pass strict eval, else 0.0 |
| `FULL_LOOSE`     | 1.0 if ALL instructions pass loose eval, else 0.0  |
| `PARTIAL_STRICT` | Fraction of instructions passing strict eval       |
| `PARTIAL_LOOSE`  | Fraction of instructions passing loose eval        |

## Example Command

python -m tinker_cookbook.recipes.if_rl.train \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    batch_size=32 group_size=16 \
    learning_rate=1e-5 \
    reward_type=FULL_STRICT \
    lora_rank=32

- [IFBench Paper (NeurIPS 2025 D&B)](https://arxiv.org/pdf/2507.02833)
- [IFBench GitHub](https://github.com/allenai/IFBench)
- Pyatkin et al., "Generalizing Verifiable Instruction Following", 2025
