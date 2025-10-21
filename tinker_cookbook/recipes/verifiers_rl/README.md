# RL Training with Tinker + Environments Hub (Verifiers)

[Verifiers](https://github.com/primeintellect-ai/verifiers) is a library for creating RL environments for LLMs, including many community implementations featured on Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments). This recipe allows all text-based environments from the Environments Hub to be used with Tinker for RL training.

To use this recipe, you need to have your chosen environment module (a self-contained Python package) installed in your project. You can install environments from the Environments Hub using the `prime` CLI:

```bash
uv tool install prime # or pipx install prime
prime env install user/env-id # ex. prime env install primeintellect/reverse-text
```

Examples:
- [primeintellect/reverse-text](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text)
- [primeintellect/alphabet-sort](https://app.primeintellect.ai/dashboard/environments/primeintellect/alphabet-sort)
- [primeintellect/math-python](https://app.primeintellect.ai/dashboard/environments/primeintellect/math-python)
- [will/wordle](https://app.primeintellect.ai/dashboard/environments/will/wordle)

You can then run the recipe with the following command, where `vf_env_id` is the ID (just `env-id`) of the environment, and `vf_env_args` is an optional JSON string of arguments to pass when loading the environment.

```bash
python -m tinker_cookbook.recipes.verifiers_rl.train vf_env_id=env-id vf_env_args='{}' ...
```

The reverse-text example as configured should climb from ~0.2 to ~0.35 in 32 steps.

You can also evaluate offline:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate vf_env_id=env-id vf_env_args='{}' ...
```

This recipe also includes a standalone `AsyncOpenAI`-compatible client implemented with Tinker, which can be adapted for other applications.

**Potential footgun:**
- Some Environments Hub implementations involve users writing their own `<think>` parsers (e.g. for use with reasoning RL starting on Instruct models). Despite being Instruct models, the Qwen3 models/tokenizers all use the same tokenizer chat template, which will strip any observed `<think>` sections automatically (which may be inadvertently penalized by reward functions). Users should either modify the renderer, tokenizer chat template, or environment module if observing issues with thinking sections from Qwen3 models.
