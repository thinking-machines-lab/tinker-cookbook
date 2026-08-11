# Model Lineup

Full listing of available models with types, architecture, and sizes.

## Thinking Machines family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `thinkingmachines/Inkling` | Hybrid + Audio + Vision | MoE | Large |
| `thinkingmachines/Inkling-Small` | Hybrid + Audio + Vision | MoE | Medium |

These models render through `tml-renderers` (the `inkling` extra) and require an explicit
thinking-effort value at sampling and training time. Use the `inkling` skill for the details.

## Qwen family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `Qwen/Qwen3.6-35B-A3B` | Hybrid + Vision | MoE | Medium |
| `Qwen/Qwen3.6-27B` | Hybrid + Vision | Dense | Medium |
| `Qwen/Qwen3.5-397B-A17B` | Hybrid + Vision | MoE | Large |
| `Qwen/Qwen3.5-35B-A3B-Base` | Base | MoE | Medium |
| `Qwen/Qwen3.5-9B` | Hybrid + Vision | Dense | Small |
| `Qwen/Qwen3.5-9B-Base` | Base | Dense | Small |
| `Qwen/Qwen3.5-4B` | Hybrid + Vision | Dense | Compact |
| `Qwen/Qwen3-8B` | Hybrid | Dense | Small |

Use the `_disable_thinking` renderer variant when you want direct instruction-following behavior from a hybrid Qwen model.

## Nemotron family

| Model | Type | Arch | Size |
|-------|------|------|------|
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | Hybrid | MoE | Large |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | Hybrid | MoE | Medium |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | Hybrid | MoE | Medium |

## Other families

| Model | Type | Arch | Size |
|-------|------|------|------|
| `openai/gpt-oss-120b` | Reasoning | MoE | Medium |
| `openai/gpt-oss-20b` | Reasoning | MoE | Small |
| `deepseek-ai/DeepSeek-V3.1` | Hybrid | MoE | Large |
| `moonshotai/Kimi-K2.6` | Hybrid + Vision | MoE | Large |

## Model types explained

- **Base**: Pre-trained on raw text. For research or full post-training pipelines.
- **Reasoning**: Always uses chain-of-thought before visible output.
- **Hybrid**: Can operate in both thinking and non-thinking modes.
- **Vision**: Processes images alongside text.
- **Audio**: Processes audio alongside text.

## Size categories

Sizes are relative tiers (Compact < Small < Medium < Large) matching the
[Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
table. For MoE models, compute cost tracks active parameters rather than
total, so a large-total-parameter MoE model can sit in a smaller tier.

## Renderer matching

Every model needs a matching renderer. Always use automatic lookup:
```python
from tinker_cookbook import model_info
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

The mapping is maintained in `tinker_cookbook/model_info.py`. Never hardcode renderer names.

## Reference

- `tinker_cookbook/model_info.py` — Model metadata and renderer mapping

## Retired models

Models retired from the Tinker service can no longer be used for training or
sampling, even though `model_info.py` retains their metadata (existing
checkpoints can still be exported). See the Tinker docs
([Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/) and
[Model Deprecations](https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/))
for the retired list and recommended replacements.
