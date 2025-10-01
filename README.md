# Tinker is a training API for researchers and developers

![Tinker Cover](tinker-cover.png | width=100)

To help the broader community customize their language models, we release two libraries: Tinker and Tinker Cookbook.
- Tinker includes primitives to fine-tune language models. It sends API requests to us, while we handle the complexity of distributed training.
- Tinker Cookbook includes realistic examples to fine-tune language models. It builds on the Tinker API and provides commonly used abstractions to fine-tune language models.

## Installation

1. Obtain a Tinker API token and export it as `TINKER_API_KEY`. // TODO(tianyi): add onboarding flow link
2. Install tinker python client via `pip install git+https://github.com/thinking-machines-lab/tinker.git` // TODO(tianyi): update to pypi
3. As a starting point, we recommend cloning this repo locally and installing it via `pip install -e .`.

## Tinker

Refer to the [docs](https://tinker-docs.thinkingmachines.ai/training-sampling) to start from basics.
We introduce a few Tinker primitives, the basic components to fine-tune LLMs.

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

`tinker_cookbook/recipes/sl_loop.py` and `tinker_cookbook/recipes/rl_loop.py` include minimal examples of using these primitives to fine-tune LLMs.

### Tinker Cookbook

Besides these primitives, we also offer **Tinker Cookbook** (a.k.a. this repo), a library of a wide range of abstractions to help you customize training environments.
`tinker_cookbook/recipes/sl_basic.py` and `tinker_cookbook/recipes/rl_basic.py` contain minimal examples to configure supervised learning and reinforcement learning.

We also include a wide range of more sophisticated examples in the `tinker_cookbook/recipes/` folder:
1. **Math reasoning**: improve LLM reasoning capability by rewarding it for answering math questions correctly.
2. **Preference learning**: showcase a three-stage RLHF pipeline: 1) supervised fine-tuning, 2) learning a reward model, 3) RL against the reward model.
3. **Tool use**: train LLMs to better use retrieval tools to answer questions more accurately.
4. **Prompt distillation**: internalize long and complex instructions into LLMs.
5. **Multi-Agent**: optimize LLMs to play against another LLM or themselves.

These examples are located in each subfolder, and their `README.md` files will walk you through the key implementation details, the commands to run them, and the expected performance.

### Import our utilities

Tinker cookbook includes several utilities. Here's a quick overview:
- [renderers](tinker_cookbook/renderers.py) converts tokens from/to structured chat message objects
- [hyperparam_utils](tinker_cookbook/hyperparam_utils.py) helps calculate hyperparameters suitable for LoRAs
- [evaluation](tinker_cookbook/evaluators.py) provides abstractions for evaluating Tinker models and [inspect_evaluation](tinker_cookbook/inspect_evaluators.py) shows how to integrate with InspectAI to make evaluating on standard benchmarks easy.

## Contributing

This project is built in the spirit of open science and collaborative development. We believe that the best tools emerge through community involvement and shared learning.

We welcome PR contributions after our private beta is over. If you have any feedback, please email us at tinker@thinkingmachines.ai.
