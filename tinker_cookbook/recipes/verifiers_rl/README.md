# RL Training with Tinker + Environments Hub (Verifiers) 

[Verifiers](https://github.com/primeintellect-ai/verifiers) is a library for creating RL environments for LLMs, and is used by hundreds of community implementations featured on Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments). This recipe allows all text-based environments from the Environments Hub to be used with Tinker for RL training.

To use this recipe, you need to have the Environment module installed in your project. You can install Environments from the Environments Hub using the `prime` CLI:
```bash
uv tool install prime # for global install with uv; `(uv) pip install prime` for local install
prime env install user/env-id # ex. prime env install primeintellect/math-python
```

You can then run the recipe with the following command, where `vf_env_id` is the ID (just `env-id`) of the environment, and `vf_env_args` is an optional JSON string of arguments to pass when loading the environment.
```bash
python -m tinker_cookbook.recipes.verifiers_rl.train --vf_env_id env-id --vf_env_args "{}"
```
