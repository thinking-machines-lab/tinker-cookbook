# RL training with Tinker and verifiers v1

This recipe runs native [`verifiers.v1`](https://github.com/primeintellect-ai/verifiers)
tasksets and harnesses against a Tinker sampling client, then trains on the tokenized v1
traces. Multi-turn conversations, tool calls, group rewards, retries, runtime-backed
harnesses, and branching traces use the verifiers lifecycle directly.

Install the optional dependencies and the package that provides your taskset:

```bash
uv sync --extra verifiers
uv pip install your-taskset-package
```

The recipe accepts one TOML file in the native `vf.EnvConfig` shape. Taskset-specific
dataset, split, and seed fields belong under `taskset`; harness and runtime fields belong
under `harness`.

```toml
# env.toml
[taskset]
id = "your-taskset"

[harness]
id = "default"
```

```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
  env_config_path=env.toml \
  model_name=Qwen/Qwen3.5-4B \
  num_tasks=256 \
  group_size=8
```

Evaluate a base model or saved Tinker sampler checkpoint with the same environment config:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate \
  env_config_path=env.toml \
  model_path=tinker://your/checkpoint \
  num_tasks=32 \
  rollouts_per_task=4
```

## Inference and rendering

`TinkerClient` implements the same chat-completions dialect as `vf.TrainClient`. It uses the
external `renderers` package for model chat templates, tool schemas, response parsing, token
attribution, and incremental multi-turn rendering; Tinker's `sample` API supplies generation.
This keeps the trace tokens aligned with the policy inputs used for training.

`renderer_model_name` defaults to `model_name`. Set it when the Tinker model identifier and
the tokenizer/chat-template identifier differ. The current integration accepts text-only
rendered prompts; multimodal tasksets fail explicitly.
