<h1 align="center">Tinker Cookbook</h1>
<div align="center">
  <img src="assets/tinker-cover.png" width="60%" />
</div>

<div align="center">

[![pytest](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/pytest.yaml/badge.svg)](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/pytest.yaml)
[![pyright](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/pyright.yaml/badge.svg)](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/pyright.yaml)
[![smoke-test-recipes](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/smoke-test-recipes.yaml/badge.svg)](https://github.com/thinking-machines-lab/tinker-cookbook/actions/workflows/smoke-test-recipes.yaml)
[![PyPI](https://img.shields.io/pypi/v/tinker-cookbook)](https://pypi.org/project/tinker-cookbook/)

</div>

We provide two libraries for the broader community to customize their language models: `tinker` and `tinker-cookbook`.

- `tinker` is a training SDK for researchers and developers to fine-tune language models. You send API requests to us and we handle the complexities of distributed training.
- `tinker-cookbook` includes realistic examples of fine-tuning language models. It builds on the Tinker API and provides common abstractions to fine-tune language models.

## Installation

1. Sign up for Tinker [here](https://auth.thinkingmachines.ai/sign-up).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. Install `tinker-cookbook` (includes the `tinker` SDK as a dependency):
   ```bash
   # Latest stable release from PyPI
   uv pip install tinker-cookbook

   # Or install the nightly build
   uv pip install 'tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
   ```

## Tinker

Refer to the [docs](https://tinker-docs.thinkingmachines.ai/training-sampling) to start from basics.
Here we introduce a few Tinker primitives - the basic components to fine-tune LLMs:

```python
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
  base_model="meta-llama/Llama-3.2-1B", rank=32,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)

sampling_client = training_client.save_weights_and_get_sampling_client()
sampling_client.sample(...)
```

See [tinker_cookbook/recipes/sl_loop.py](tinker_cookbook/recipes/sl_loop.py) and [tinker_cookbook/recipes/rl_loop.py](tinker_cookbook/recipes/rl_loop.py) for minimal examples of using these primitives to fine-tune LLMs.

### Tutorials

New to Tinker? The [`tutorials/`](tutorials/) directory has 6 progressive [marimo](https://marimo.io/) notebooks that guide you from your first API call to building custom RL training pipelines:

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 01 | [Hello Tinker](tutorials/01_hello_tinker.py) | Architecture overview, client hierarchy, sampling |
| 02 | [First SFT](tutorials/02_first_sft.py) | Renderers, datum construction, training loop, Kimi K2.5 scaling demo |
| 03 | [Efficient Sampling](tutorials/03_async_patterns.py) | Concurrent futures, `num_samples`, batch evaluation throughput |
| 04 | [First RL](tutorials/04_first_rl.py) | GRPO on GSM8K: rewards, advantages, degenerate groups |
| 05 | [Cookbook RL Abstractions](tutorials/05_custom_task.py) | `Env`, `EnvGroupBuilder`, `RLDataset`, `ProblemEnv` |
| 06 | [Custom RL Environment](tutorials/06_custom_env.py) | Build your own `ProblemEnv` and `RLDataset` |

Run any tutorial with `marimo edit tutorials/01_hello_tinker.py`. Rendered versions are available on the [Tinker docs site](https://tinker-docs.thinkingmachines.ai/tutorials).

To download the weights of any model:
```python
rest_client = service_client.create_rest_client()
future = rest_client.get_checkpoint_archive_url_from_tinker_path(sampling_client.model_path)
with open(f"model-checkpoint.tar.gz", "wb") as f:
    f.write(future.result())
```

### Tinker Cookbook

Besides these primitives, we also offer **Tinker Cookbook** (a.k.a. this repo), a library of a wide range of abstractions to help you customize training environments.
[`tinker_cookbook/recipes/sl_basic.py`](tinker_cookbook/recipes/sl_basic.py) and [`tinker_cookbook/recipes/rl_basic.py`](tinker_cookbook/recipes/rl_basic.py) contain minimal examples to configure supervised learning and reinforcement learning.

We also include a wide range of more sophisticated examples in the [`tinker_cookbook/recipes/`](tinker_cookbook/recipes/) folder:
1. **[Chat supervised learning](tinker_cookbook/recipes/chat_sl/)**: supervised fine-tuning on conversational datasets like Tulu3.
2. **[Math reasoning](tinker_cookbook/recipes/math_rl/)**: improve LLM reasoning capability by rewarding it for answering math questions correctly.
3. **[Preference learning](tinker_cookbook/recipes/preference/)**: showcase a three-stage RLHF pipeline: 1) supervised fine-tuning, 2) learning a reward model, 3) RL against the reward model.
4. **[Tool use](tinker_cookbook/recipes/search_tool/)**: train LLMs to better use retrieval tools to answer questions more accurately.
5. **[Prompt distillation](tinker_cookbook/recipes/prompt_distillation/)**: internalize long and complex instructions into LLMs.
6. **[Multi-Agent](tinker_cookbook/recipes/multiplayer_rl/)**: optimize LLMs to play against another LLM or themselves.

These examples are located in each subfolder, and their `README.md` files will walk you through the key implementation details, the commands to run them, and the expected performance.

### Documentation

For the full Tinker documentation, visit [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai).

### Import our utilities

Tinker cookbook includes several utilities. Here's a quick overview:
- [`renderers`](tinker_cookbook/renderers/) converts tokens from/to structured chat message objects
- [`hyperparam_utils`](tinker_cookbook/hyperparam_utils.py) helps calculate hyperparameters suitable for LoRAs
- [`evaluation`](tinker_cookbook/eval/evaluators.py) provides abstractions for evaluating Tinker models and [`inspect_evaluation`](tinker_cookbook/eval/inspect_evaluators.py) shows how to integrate with InspectAI to make evaluating on standard benchmarks easy.

## Claude Code Skills

Tinker Cookbook ships with [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills) that teach Claude how to use the Tinker API. Install them so Claude can help you write training code in any project:

```
/plugin marketplace add thinking-machines-lab/tinker-cookbook
```

Then install the **tinker** plugin from the Discover tab (`/plugin` → Discover). Once installed, the following skills are available:

| Command | What it does |
|---|---|
| `/tinker:core` | Getting started — installation, models, SDK basics, hyperparameters |
| `/tinker:sft` | Supervised fine-tuning, datasets, renderers, distillation |
| `/tinker:rl` | Reinforcement learning — GRPO, custom environments, multi-turn |
| `/tinker:preferences` | DPO and RLHF pipelines |
| `/tinker:ops` | Checkpoints, weight export, logging, evaluation |
| `/tinker:debug` | Diagnose slow training, hangs, output mismatches, errors |
| `/tinker:dev` | Contributing to this repo — tests, CI, recipes |

Skills also trigger automatically based on context — ask Claude to "set up SFT training" and it will load the right skill without a slash command. Skills update automatically when the repo is updated.

## Development Setup

```bash
uv sync --extra dev
pre-commit install
```

This installs dev dependencies and registers pre-commit hooks that run `ruff` formatting and linting on every commit. CI enforces these checks on all pull requests.

## Contributing

This project is built in the spirit of open science and collaborative development. We believe that the best tools emerge through community involvement and shared learning.

We welcome PR contributions after our private beta is over. If you have any feedback, please email us at tinker@thinkingmachines.ai.

## Citation
If you use Tinker for your research, please cite it as:
```
Thinking Machines Lab, 2025. Tinker. https://thinkingmachines.ai/tinker/.
```

Or use this BibTeX citation:
```
@misc{tml2025tinker,
  author = {Thinking Machines Lab},
  title = {Tinker},
  year = {2025},
  url = {https://thinkingmachines.ai/tinker/},
}
```
