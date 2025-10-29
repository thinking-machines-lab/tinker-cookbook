# Cookbook Recipes

Tinker allows you to flexibly customize your training environment.
We will first introduce a few simple training scripts to help you get started, and then cover a broad range of different use cases.

## Getting Started

Tinker Cookbook comes with useful abstractions so you can flexibly customize your experiments. Here are some minimal launch scripts:
- [`rl_basic.py`](./rl_basic.py): a template script to configure reinforcement learning.
- [`sl_basic.py`](./sl_basic.py): a template script to configure supervised learning.

To explain what goes under-the-hood, we also provide minimal, self-contained scripts that directly use the TinkerAPI to train LLMs.
- [`rl_loop.py`](./rl_loop.py): a minimal reinforcement learning training loop.
- [`sl_loop.py`](./sl_loop.py): a minimal supervised learning training loop.

## More Post-Training Examples

Building on Tinker and Tinker Cookbook, we can easily customize a wide range of training environments for LLMs.
We provide the following examples:
1. **[Chat supervised learning](./chat_sl/)**: supervised fine-tuning on conversational datasets like Tulu3.
2. **[Math reasoning](./math_rl/)**: improve LLM reasoning capability by rewarding it for answering math questions correctly.
3. **[Preference learning](./preference/)**: showcase a three-stage RLHF pipeline: 1) supervised fine-tuning, 2) learning a reward model, 3) RL against the reward model.
4. **[Tool use](./tool_use/)**: train LLMs to better use retrieval tools to answer questions more accurately.
5. **[Prompt distillation](./prompt_distillation/)**: internalize long and complex instructions into LLMs.
6. **[Multi-Agent](./multiplayer_rl/)**: optimize LLMs to play against another LLM or themselves.

These examples are located in each subfolder, and their `README.md` file will walk you through the key implementation details, the commands to run them, and the expected performance.

### Logging and Recovering From Training Interruptions

Our examples support the following CLI arguments to log the results.

1. `wandb_project`: When provided, logs will be sent to your Weights & Biases project. Without this argument, training scripts save logs locally only.
2. `log_path`: Controls where training artifacts are saved.
  - Default behavior: If not specified, each run generates a unique name and saves to `/tmp/tinker-examples`
  - Output files:
    - `{log_path}/metrics.jsonl` saves training metrics.
    - `{log_path}/checkpoints.jsonl` records all the checkpoints saved during training. You can share these checkpoints for model release, offline evaluation, etc.
  - Resuming: When using an existing `log_path`, you can either overwrite the previous run or resume training. This is particularly useful for recovering from runtime interruptions.
