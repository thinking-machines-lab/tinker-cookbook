---
name: tinker-setup
description: Guide for installing Tinker, setting up the environment, getting an API key, and verifying everything works. Use when the user is getting started, setting up their environment, or troubleshooting installation issues.
---

# Setup & Installation

Get Tinker and tinker-cookbook running from scratch.

## Reference

- `docs/install.mdx` — Official installation guide
- `CONTRIBUTING.md` — Development setup
- `README.md` — Project overview

## Step 1: Sign up and get an API key

1. Sign up at [https://auth.thinkingmachines.ai/sign-up](https://auth.thinkingmachines.ai/sign-up)
2. Create an API key from the [console](https://tinker-console.thinkingmachines.ai)
3. Export it:
```bash
export TINKER_API_KEY=<your-key>
```

Add to your shell profile (`.bashrc`, `.zshrc`) for persistence.

## Step 2: Install Tinker SDK

```bash
pip install tinker
```

This gives you:
- **Python SDK** — `TrainingClient`, `SamplingClient`, low-level training/sampling APIs
- **Tinker CLI** — `tinker` or `python -m tinker` for management tasks

## Step 3: Install tinker-cookbook

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook
pip install -e .
```

Or with dev dependencies (for contributing):
```bash
uv sync --extra dev
pre-commit install
```

## Step 4: Verify installation

```python
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.2-1B", rank=32,
)
info = training_client.get_info()
print(info)  # Should print model info
```

## Step 5: Run a minimal example

```bash
# Supervised learning
python -m tinker_cookbook.recipes.sl_basic

# Reinforcement learning
python -m tinker_cookbook.recipes.rl_basic
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `TINKER_API_KEY` | Required — authenticates with Tinker service |
| `HF_TOKEN` | Optional — access gated HuggingFace models (Llama, etc.) |
| `HF_TRUST_REMOTE_CODE` | Optional — allow custom tokenizer code |
| `WANDB_API_KEY` | Optional — log to Weights & Biases |

## Common issues

- **`TINKER_API_KEY not set`**: Export the key in your shell or `.env` file
- **Tokenizer download fails**: Set `HF_TOKEN` for gated models (e.g., Llama)
- **Import errors**: Ensure `pip install -e .` was run from the repo root
- **`uv` not found**: Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
