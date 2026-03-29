# Model Lineup

Full listing of available models with types, architecture, and sizes.

## Qwen family

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

## Llama family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `meta-llama/Llama-3.3-70B-Instruct` | Instruction | Dense | Large |
| `meta-llama/Llama-3.1-70B` | Base | Dense | Large |
| `meta-llama/Llama-3.1-8B` | Base | Dense | Small |
| `meta-llama/Llama-3.1-8B-Instruct` | Instruction | Dense | Small |
| `meta-llama/Llama-3.2-3B` | Base | Dense | Compact |
| `meta-llama/Llama-3.2-1B` | Base | Dense | Compact |

## Nemotron family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Hybrid | MoE | Medium |

## Other families

| Model | Type | Arch | Size |
|-------|------|------|------|
| `openai/gpt-oss-120b` | Reasoning | MoE | Medium |
| `openai/gpt-oss-20b` | Reasoning | MoE | Small |
| `deepseek-ai/DeepSeek-V3.1` | Hybrid | MoE | Large |
| `deepseek-ai/DeepSeek-V3.1-Base` | Base | MoE | Large |
| `moonshotai/Kimi-K2-Thinking` | Reasoning | MoE | Large |
| `moonshotai/Kimi-K2.5` | Reasoning + Vision | MoE | Large |

## Model types explained

- **Base**: Pre-trained on raw text. For research or full post-training pipelines.
- **Instruction**: Fine-tuned for instruction following. Fast inference, no chain-of-thought.
- **Reasoning**: Always uses chain-of-thought before visible output.
- **Hybrid**: Can operate in both thinking and non-thinking modes.
- **Vision**: Processes images alongside text.

## Size categories

- **Compact**: 1B-4B parameters
- **Small**: 8B parameters
- **Medium**: 27B-32B parameters
- **Large**: 70B+ parameters

## Renderer matching

Every model needs a matching renderer. Always use automatic lookup:
```python
from tinker_cookbook import model_info
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

The mapping is maintained in `tinker_cookbook/model_info.py`. Never hardcode renderer names.

## Reference

- `tinker_cookbook/model_info.py` — Model metadata and renderer mapping
