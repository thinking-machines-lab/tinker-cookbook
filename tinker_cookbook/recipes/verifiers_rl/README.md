# RL Training with Tinker + Environments Hub (Verifiers)

[Verifiers](https://github.com/primeintellect-ai/verifiers) is a library for creating RL environments for LLMs, including many community implementations featured on Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments). This recipe runs Hub environments against Tinker for RL training, driving the verifiers v1 stack (`verifiers.v1`) natively.

To use this recipe, you need to have your chosen environment (a self-contained Python package exporting a verifiers taskset) installed in your project. You can install environments from the Environments Hub using the `prime` CLI:

```bash
uv tool install prime # or pipx install prime
prime env install user/env-id # ex. prime env install primeintellect/reverse-text
```

You can then run the recipe with the following command, where `vf_env_id` is the ID (just `env-id`) of the environment, and `vf_env_args` is an optional JSON object of `vf.EnvConfig` overrides merged over the taskset id — for example, per-task config under `{"taskset": {"task": {...}}}`, or agent-seat pins under `{"agent": {...}}`:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.train \
    vf_env_id=reverse-text \
    vf_env_args='{"agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}}' \
    chat_template_kwargs='{"enable_thinking": false}' ...
```

Two `vf_env_args` pins worth knowing:

- `{"agent": {"harness": {"id": "null"}}}` runs a plain chat agent on a taskset that doesn't bundle its own harness (the fallback harness is `bash`, a coding agent).
- `{"agent": {"runtime": {"type": "subprocess"}}}` runs harness programs as local subprocesses. The verifiers default runtime provisions Prime sandboxes, which costs money and is capped per account — pin `subprocess` (or `docker`) for local training unless you want that isolation.

`chat_template_kwargs` (JSON) is passed through verifiers' train client to the chat template of the [`renderers`](https://github.com/PrimeIntellect-ai/renderers) package that tokenizes each turn — e.g. `{"enable_thinking": false}` to switch Qwen thinking off so short `max_tokens` budgets aren't consumed by reasoning.

You can also evaluate offline:

```bash
python -m tinker_cookbook.recipes.verifiers_rl.evaluate vf_env_id=env-id vf_env_args='{}' ...
```

This recipe requires `verifiers>=0.3.1`, installed by `pip install 'tinker_cookbook[verifiers]'`.

## How it plugs in

verifiers' train client renders each turn to token IDs with Prime Intellect's `renderers` package (a verifiers dependency, distinct from this cookbook's `tinker_cookbook.renderers`) and POSTs them to a vLLM-style `/inference/v1/generate` endpoint, expecting sampled token IDs and per-token logprobs back. That endpoint is the one seam where an inference engine plugs into the v1 rollout stack, so this recipe serves it locally over a `tinker.SamplingClient` (`tinker_generate.TinkerGenerateServer`) — harness programs, interception, renderer bridging, scoring, and trace/token bookkeeping are all verifiers running natively. Each rollout comes back as a `vf.Episode` whose agent trace carries the exact sampled token IDs and logprobs, which `verifiers_env.convert_episodes_to_trajectory_group` turns into tinker trajectories.

The recipe trains single-agent environments (one agent trace per episode, one trainable branch per trace); multi-agent or branching envs, and traces whose agent is not the policy, are rejected with an explicit error. An episode or trace that verifiers marks failed stays in its group as an empty trajectory with a `rollout_failed` metric, so reward centering sees the failure; a sampled token without a logprob is an error, never a guess.

**Potential footgun:**

- The tokenizer/renderer used for sampling is resolved from the base model name by verifiers' `renderers` dependency (pinned via `renderer_model_name`), not by this cookbook's own renderers. For reasoning-mode subtleties (e.g. Qwen thinking sections being stripped or re-rendered by the chat template), check the resolved renderer's behavior against your environment's parsers before ascribing reward drops to the policy.
