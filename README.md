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

Here we introduce a few Tinker primitives — the basic components to fine-tune LLMs (see the [quickstart guide](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/) for more details):

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

New to Tinker? The [`tutorials/`](tutorials/) directory contains 20+ progressive [marimo](https://marimo.io/) notebooks that walk through core concepts — rendering, loss functions, completers, weight management — and advanced topics such as custom RL environments, DPO, RLHF, and weight export. Run any tutorial with `marimo edit tutorials/101_hello_tinker.py`. See the [tutorials README](tutorials/README.md) for the full list, or browse rendered versions on the [Tinker docs site](https://tinker-docs.thinkingmachines.ai/tutorials).

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

We also include more complete examples in the [`tinker_cookbook/recipes/`](tinker_cookbook/recipes/) folder:
- **[Chat SFT](tinker_cookbook/recipes/chat_sl/)**: supervised fine-tuning on conversational datasets (e.g., Tulu3).
- **[Math RL](tinker_cookbook/recipes/math_rl/)**: reinforcement learning for mathematical reasoning with verifiable rewards.
- **[Code RL](tinker_cookbook/recipes/code_rl/)**: RL on competitive programming with sandboxed code execution (DeepCoder replication).
- **[Preference learning](tinker_cookbook/recipes/preference/)**: DPO and a three-stage RLHF pipeline (SFT, reward model, RL).
- **[Distillation](tinker_cookbook/recipes/distillation/)**: on-policy and off-policy knowledge distillation with single- and multi-teacher configurations.
- **[Tool use](tinker_cookbook/recipes/search_tool/)**: RL for retrieval-augmented generation (Search-R1 replication).
- **[Multi-agent](tinker_cookbook/recipes/multiplayer_rl/)**: multi-agent RL with self-play and cross-play.

The [recipes README](tinker_cookbook/recipes/README.md) covers all available recipes, including Harbor RL, rubric-based grading, VLM classification, and SDFT. Each recipe includes a `README.md` with implementation details, launch commands, and expected results.

### Evaluation (experimental)

Tinker Cookbook includes a [benchmark framework](tinker_cookbook/eval/) for evaluating trained models:

```python
from tinker_cookbook.eval.benchmarks import run_benchmarks, BenchmarkConfig

results = await run_benchmarks(
    ["gsm8k", "mmlu_pro", "ifeval"],
    sampling_client, renderer,
    BenchmarkConfig(save_dir="evals/step500"),
)
```

The framework currently supports 12 benchmarks (GSM8K, MATH-500, MMLU-Pro, MMLU-Redux, GPQA, IFEval, MBPP, C-Eval, SuperGPQA, IFBench, AIME 2025, AIME 2026) with verified scores against published results, plus experimental benchmarks such as LiveCodeBench, Terminal Bench, and SWE-bench. Benchmarks can also serve as inline training evaluators via `BenchmarkEvaluator`.

**Note:** Benchmark scores are sensitive to evaluation configuration — system prompts, `max_tokens`, temperature, and timeout settings can shift results significantly. We document our exact settings alongside all reported scores. This framework is under active development; feedback and contributions are welcome. See the [eval README](tinker_cookbook/eval/README.md) for verified scores, configuration details, and instructions for adding new benchmarks.

### Documentation

For the full Tinker documentation, visit [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai).

### Utilities

Tinker Cookbook also provides reusable building blocks:
- [`renderers`](tinker_cookbook/renderers/) — bidirectional conversion between token sequences and structured chat messages
- [`hyperparam_utils`](tinker_cookbook/hyperparam_utils.py) — learning rate and hyperparameter scaling for LoRA training
- [`eval`](tinker_cookbook/eval/) — benchmark framework and inline training evaluators (see [Evaluation](#evaluation-experimental) above)

## Claude Code Skills

Tinker Cookbook ships with [Claude Code skills](https://docs.anthropic.com/en/docs/claude-code/skills) that teach Claude how to use the Tinker API. Install them so Claude can help you write training code in any project:

```
/plugin marketplace add thinking-machines-lab/tinker-cookbook
```

Then install the **tinker** plugin from the Discover tab (`/plugin` → Discover). Once installed, two skills are available:

| Command | What it does |
|---|---|
| `/tinker:research` | Plan and run post-training experiments — SFT, RL, DPO, distillation, evaluation, hyperparameters, model selection, and more |
| `/tinker:debug` | Diagnose slow training, hangs, output mismatches, renderer issues, and errors |

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
