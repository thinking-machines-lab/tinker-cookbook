<h1 align="center">Tinker Cookbook</h1>
<div align="center">
  <img src="assets/tinker-cover.png" width="60%" />
</div>

We provide two libraries for the broader community to customize their language models: `tinker` and `tinker-cookbook`.

- `tinker` is a training SDK for researchers and developers to fine-tune language models. You send API requests to us and we handle the complexities of distributed training.
- `tinker-cookbook` includes realistic examples of fine-tuning language models. It builds on the Tinker API and provides common abstractions to fine-tune language models.

## Installation

1. Sign up for Tinker through the [waitlist](https://thinkingmachines.ai/tinker).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. Install tinker python client via `pip install tinker`
4. We recommend installing `tinker-cookbook` in a virtual env either with `conda` or `uv`. For running most examples, you can install via `pip install -e .`.

## Tinker

Refer to the [docs](https://tinker-docs.thinkingmachines.ai/training-sampling) to start from basics.
Here we introduce a few Tinker primitives - the basic components to fine-tune LLMs:

```python
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
  base_model="meta-llama/Llama-3.2-1B", rank=32,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)

sampling_client = training_client.save_weights_and_get_sampling_client(name="my_model")
sampling_client.sample(...)
```

See [tinker_cookbook/recipes/sl_loop.py](tinker_cookbook/recipes/sl_loop.py) and [tinker_cookbook/recipes/rl_loop.py](tinker_cookbook/recipes/rl_loop.py) for minimal examples of using these primitives to fine-tune LLMs.

To download the weights of any model:
```python
rest_client = service_client.create_rest_client()
future = rest_client.download_checkpoint_archive_from_tinker_path(sampling_client.model_path)
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
4. **[Tool use](tinker_cookbook/recipes/tool_use/)**: train LLMs to better use retrieval tools to answer questions more accurately.
5. **[Prompt distillation](tinker_cookbook/recipes/prompt_distillation/)**: internalize long and complex instructions into LLMs.
6. **[Multi-Agent](tinker_cookbook/recipes/multiplayer_rl/)**: optimize LLMs to play against another LLM or themselves.

These examples are located in each subfolder, and their `README.md` files will walk you through the key implementation details, the commands to run them, and the expected performance.

### Import our utilities

Tinker cookbook includes several utilities. Here's a quick overview:
- [`renderers`](tinker_cookbook/renderers.py) converts tokens from/to structured chat message objects
- [`hyperparam_utils`](tinker_cookbook/hyperparam_utils.py) helps calculate hyperparameters suitable for LoRAs
- [`evaluation`](tinker_cookbook/eval/evaluators.py) provides abstractions for evaluating Tinker models and [`inspect_evaluation`](tinker_cookbook/eval/inspect_evaluators.py) shows how to integrate with InspectAI to make evaluating on standard benchmarks easy.

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
