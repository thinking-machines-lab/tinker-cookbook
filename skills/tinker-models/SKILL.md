---
name: tinker-models
description: Guide for choosing models in Tinker — available model families, model types (base, instruction, reasoning, hybrid, vision), architecture (dense vs MoE), and how to match renderers to models. Use when the user asks which model to use, what models are available, or how to pick a model for their task.
---

# Model Selection

Help the user choose the right model for their task.

## Reference

- `docs/model-lineup.mdx` — Full model listing with types, sizes, and architecture
- `tinker_cookbook/model_info.py` — Model metadata and renderer mapping

## Available models

### Qwen family
| Model | Type | Arch | Size |
|-------|------|------|------|
| `Qwen/Qwen3.5-397B-A17B` | Hybrid + Vision | MoE | Large |
| `Qwen/Qwen3.5-35B-A3B` | Hybrid + Vision | MoE | Medium |
| `Qwen/Qwen3.5-27B` | Hybrid + Vision | Dense | Medium |
| `Qwen/Qwen3.5-4B` | Hybrid + Vision | Dense | Compact |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | Instruction | MoE | Large |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Instruction | MoE | Medium |
| `Qwen/Qwen3-30B-A3B` | Hybrid | MoE | Medium |
| `Qwen/Qwen3-30B-A3B-Base` | Base | MoE | Medium |
| `Qwen/Qwen3-32B` | Hybrid | Dense | Medium |
| `Qwen/Qwen3-8B` | Hybrid | Dense | Small |
| `Qwen/Qwen3-8B-Base` | Base | Dense | Small |
| `Qwen/Qwen3-4B-Instruct-2507` | Instruction | Dense | Compact |
| `Qwen/Qwen3-VL-235B-A22B-Instruct` | Vision | MoE | Large |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | Vision | MoE | Medium |

### Llama family
| Model | Type | Arch | Size |
|-------|------|------|------|
| `meta-llama/Llama-3.3-70B-Instruct` | Instruction | Dense | Large |
| `meta-llama/Llama-3.1-70B` | Base | Dense | Large |
| `meta-llama/Llama-3.1-8B` | Base | Dense | Small |
| `meta-llama/Llama-3.1-8B-Instruct` | Instruction | Dense | Small |
| `meta-llama/Llama-3.2-3B` | Base | Dense | Compact |
| `meta-llama/Llama-3.2-1B` | Base | Dense | Compact |

### Nemotron family
| Model | Type | Arch | Size |
|-------|------|------|------|
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Hybrid | MoE | Medium |

### Other families
| Model | Type | Arch | Size |
|-------|------|------|------|
| `openai/gpt-oss-120b` | Reasoning | MoE | Medium |
| `openai/gpt-oss-20b` | Reasoning | MoE | Small |
| `deepseek-ai/DeepSeek-V3.1` | Hybrid | MoE | Large |
| `deepseek-ai/DeepSeek-V3.1-Base` | Base | MoE | Large |
| `moonshotai/Kimi-K2-Thinking` | Reasoning | MoE | Large |
| `moonshotai/Kimi-K2.5` | Reasoning + Vision | MoE | Large |

## How to choose

### By task type

- **Instruction tuning / chat SFT**: Start with an Instruction model (e.g., `Llama-3.1-8B-Instruct`, `Qwen3-30B-A3B-Instruct-2507`)
- **RL with verifiable rewards (GRPO)**: Use Instruction or Hybrid models — they already follow instructions
- **Reasoning / chain-of-thought**: Use Reasoning or Hybrid models (`Kimi-K2-Thinking`, `Qwen3-8B`)
- **Full post-training pipeline**: Start with a Base model (e.g., `Qwen3-8B-Base`, `Llama-3.1-8B`)
- **Vision tasks**: Use Vision or Hybrid+Vision models (`Qwen3.5-35B-A3B`, `Qwen3-VL-*`)
- **Distillation (student)**: Use a Base model as student
- **Quick prototyping**: Use compact models (`Llama-3.2-1B`, `Qwen3.5-4B`)

### By cost

**Prefer MoE models** — they're much more cost-effective than dense models because cost scales with active parameters, not total parameters. For example, `Qwen3-30B-A3B` (MoE, 3B active) is cheaper than `Qwen3-32B` (Dense, 32B active) despite similar quality.

### Model types explained

- **Base**: Pre-trained on raw text. For research or full post-training pipelines.
- **Instruction**: Fine-tuned for instruction following. Fast inference, no chain-of-thought.
- **Reasoning**: Always uses chain-of-thought before visible output.
- **Hybrid**: Can operate in both thinking and non-thinking modes.
- **Vision**: Processes images alongside text. See `/tinker-renderers` skill for vision input handling.

### Size categories
- **Compact**: 1B–4B parameters
- **Small**: 8B parameters
- **Medium**: 27B–32B parameters
- **Large**: 70B+ parameters

## Renderer matching

Every model needs a matching renderer. **Always use the automatic lookup**:

```python
from tinker_cookbook import model_info

renderer_name = model_info.get_recommended_renderer_name(model_name)
```

Never hardcode renderer names — the mapping is maintained in `model_info.py`.

## Learning rate by model

Use `hyperparam_utils.get_lr(model_name)` for model-specific LR recommendations. See the `/tinker-hyperparams` skill for details.

## Common pitfalls
- MoE models are cheaper than dense — prefer them unless you have a specific reason
- Base models need full post-training (SFT + alignment) to be useful for chat
- Instruction models are best for tasks where you want to start from a capable baseline
- Vision models require `ImageChunk` in messages — see `/tinker-renderers` skill
- Llama models require `HF_TOKEN` for tokenizer download (gated on HuggingFace)
