---
name: inkling
description: Sample, evaluate, and post-train Inkling and Inkling-Small, Thinking Machines Lab's models built for Tinker. Use this skill whenever the user mentions Inkling, `thinkingmachines/Inkling`, tml-renderers, `tml_v0` / `TmlV0Renderer`, or thinking/reasoning effort — and whenever they are choosing a model, building training data, running evals, setting up SFT or RL, handling parse errors, or working with audio or image inputs for an Inkling model. Inkling has requirements that differ from other Tinker models (mandatory effort conditioning, its own renderer and tokenizer, a learning rate you calibrate yourself), so load this skill before writing any Inkling code.
---

# Inkling

[Inkling](https://thinkingmachines.ai/inkling/) is Thinking Machines Lab's open-weight model family built for Tinker: general-purpose models that code, reason, call tools, and accept image and audio input.

Inkling is a Mixture-of-Experts transformer with 975B total parameters and 41B active. Inkling-Small is an efficient sibling at 276B total and 12B active, reaching comparable performance to Inkling at a quarter of its size. Both offer native reasoning over audio and images, variable thinking effort, a context window of up to 1M tokens, and well-rounded performance across a range of benchmarks.

Inkling-Small's efficiency makes it a reasonable default for most tasks, and a natural fit for workloads where cost and latency matter, such as coding, using LLMs to grade, or generating synthetic data for other models. The two share a renderer, tokenizer, and effort interface, so moving between them is a one-line change to the model name, and benchmarking both on your own task is cheap.

Both models are offered as post-trained versions on Tinker, not base models: they arrive instruction-tuned and effort-conditioned, so treat them as starting points for further post-training — SFT, RL, distillation — rather than for continued pretraining.

Working with Inkling differs from other Tinker models in three ways, each covered below: every render needs an explicit thinking-effort value, rendering and tokenization go through `tml-renderers` rather than a Hugging Face chat template, and the learning rate is yours to calibrate.

## Setup

Inkling renders through the standalone [`tml-renderers`](https://pypi.org/project/tml-renderers/) package, included in the default installation together with the required `torch>=2.10`.

Pass `thinkingmachines/Inkling` or `thinkingmachines/Inkling-Small` anywhere the cookbook takes a model name. The tokenizer and renderer (`tml_v0`) are selected automatically for any `thinkingmachines/Inkling*` model, including the [`:peft:` long-context variants](https://tinker-docs.thinkingmachines.ai/tinker/models/):

```python
from tinker_cookbook import model_info
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

model_name = "thinkingmachines/Inkling"
renderer = get_renderer(model_info.get_recommended_renderer_name(model_name), get_tokenizer(model_name))
```

Never hardcode a renderer name, and never call `tokenizer.encode()` to build a prompt — the Inkling tokenizer adapter has no chat template, so raw encoding produces out-of-distribution tokens. The [Using Inkling guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/) is the canonical reference for setup and supported inputs.

## Thinking effort: set it everywhere

Inkling's flagship control is [continuous thinking effort](https://thinkingmachines.ai/news/introducing-inkling/#controllable-thinking-effort) — a finite scalar in `[0.0, 1.0)`. The renderer turns it into a `Thinking effort level` system message inserted before the first non-system message. **The model is post-trained with that message present**, so leaving it to chance means out-of-distribution behavior and unreliable eval numbers.

Rules:

- Set `effort` explicitly at sampling time *and* when building training data. The renderer defaults to `0.9` (high) if you omit it, which is a reasonable default but should be a deliberate choice.
- Do not write the effort system message yourself, and do not include a second one in the conversation. The renderer owns it.
- Effort and `max_tokens` are independent. Higher effort tends to produce longer traces, so raise the generation budget with it or responses truncate mid-thought.

### Preset values

These are the same scalars the [OpenAI-compatible API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) uses for its named `reasoning_effort` presets. Any finite value in `[0.0, 1.0)` works.

| Name | `none` | `minimal` | `low` | `medium` | `high` (default) | `xhigh` |
|---|---|---|---|---|---|---|
| Effort | `0.0` | `0.1` | `0.2` | `0.7` | `0.9` | `0.99` |

For evaluation: use `0.99` for maximum reasoning performance, `0.0` for benchmarks that do not need chain-of-thought, and sweep the middle for accuracy-versus-latency trade-offs — the [thinking-effort guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/) has effort-scaling results on selected benchmarks. Note that `effort=0.0` conditions the model toward no reasoning; it is not a hard constraint.

Think of effort as a general rollout-effort knob, not just a thinking-length knob. Even on tasks that need little reasoning, higher effort can produce more tool calls and more turns, which may help agentic environments.

## Sampling

```python
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

renderer = TmlV0Renderer(get_tokenizer("thinkingmachines/Inkling"))
messages = [Message(role="user", content="Solve this problem step by step.")]
prompt = renderer.build_generation_prompt(messages, effort=0.9)
```

Sweep effort on your own task before committing to a value, with [`sample_reasoning.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/scripts/inkling/sample_reasoning.py):

```bash
python -m tinker_cookbook.scripts.inkling.sample_reasoning efforts='[0.0,0.2,0.7,0.9,0.99]'
```

Sample and run evals at `temperature=1.0`, which is what every script in `tinker_cookbook/scripts/inkling/` defaults to. Lowering it is not a substitute for lowering effort.

`TmlV0Renderer` does not support assistant prefill — pass complete messages and let the model open its own turn. Always pass `renderer.get_stop_sequences()` in `SamplingParams`.

## Training data

Render supervised data at the same effort you will use at test time, so training tokens match sampling tokens exactly:

```python
model_input, weights = renderer.build_supervised_example(messages_with_response, effort=0.9)
```

We suggest picking **one** effort for the whole dataset and using it at test time too, leaning toward the higher end — `0.9` or `0.99` — unless your task genuinely does not need reasoning; mixed-effort training recipes are something we plan to share later. Generic supervised dataset builders currently render at the default `0.9`; call `build_supervised_example` (or `build_supervised_examples`, which returns multiple examples when a conversation expands to several) directly to render at another effort.

For custom training code built directly on the Tinker SDK and `tml-renderers`, use `render_for_completion_with_effort` rather than `render_for_completion` — the latter omits the effort message.

## Post-training

We suggest calibrating the learning rate to your own task and dataset rather than starting from a general-purpose default. `hyperparam_utils.get_lr("thinkingmachines/Inkling")` deliberately raises `NotImplementedError` instead of returning a number, so pick a starting value and sweep from there.

- **Sweep learning rate and effort together over the first few RL steps.** We suggest doing this before committing to any long run, and trying the higher end of your learning-rate range as part of the sweep.
- **Watch entropy.** The cookbook logs `optim/entropy` each step. Healthy entropy means the model is still producing variance, which is what makes rewards differ within a group and keeps the model responsive to training.
- **Track all-fail and all-success groups.** Groups with uniform reward contribute no gradient — see the `remove_constant_reward_groups` option. If most groups are degenerate, we suggest adjusting task difficulty, group size, or effort.
- **Consider an SFT warm-start if RL is slow to move.** When your domain or output format sits far from the model's defaults, a short SFT phase on in-format data can reach a regime where RL hill-climbs more readily.

RL defaults are already tuned for Inkling in the cookbook: `thinkingmachines/Inkling*` models default to the `agentic` rollout preset and the `MinViableGroup` rollout strategy (tool-backed rollouts shouldn't discard a whole group over one infrastructure failure). Override with an explicit `rollout_config` or `rollout_error_tolerance` if your task is single-turn.

## Evaluation

Evaluate through the cookbook sampling path, or through Tinker's [OpenAI-compatible](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) or [Anthropic-compatible](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/) endpoints if your harness expects one of those interfaces. However you sample, set effort explicitly and record it alongside the score — an eval number without its effort value is not reproducible.

The two endpoints expose effort differently. The OpenAI endpoint takes `reasoning_effort` as either a named preset or a raw float, matching the table above. The Anthropic endpoint takes named levels only, through a Tinker-specific `output_config.effort` in `extra_body` — Anthropic's own `thinking.budget_tokens` is accepted for wire compatibility but ignored — and it does not accept audio input.

**Handle parse errors.** They occur during training, eval, and normal use, usually when a response is truncated mid-structure. Handle them however fits your pipeline; we suggest retrying recoverable errors with a corrective message that includes the parser error detail.

- Through the cookbook renderer: `parse_response()` never raises. It returns `(message, ParseTermination.MALFORMED)` on unparseable output and falls back to decoded raw text. Check the termination value rather than assuming success.
- Through `tml-renderers` directly: catch `tml_renderers.v0.ParseError` and feed its string into the corrective message.

If malformed terminations are frequent, check `max_tokens` first — truncation at an unlucky cut point is the most common cause.

## Audio and images

Both models accept [audio](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/audio/) and [image](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/images/) input. Encode media locally through `tml-renderers`; the renderer accepts raw bytes, a local path, or a base64 `data:` URI, and deliberately refuses remote URLs. Audio must be WAV, MP3, or FLAC, and non-WAV formats must supply `num_frames` and `sample_rate` together. Clean, adequately loud audio noticeably improves results.

```bash
python -m tinker_cookbook.scripts.inkling.sample_audio
python -m tinker_cookbook.scripts.inkling.sample_vision
```

Audio fine-tuning recipes (speech recognition, emotion plus transcription, medical ASR domain adaptation) live in `tinker_cookbook/recipes/audio/` and need `pip install 'tinker_cookbook[audio]'`.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Odd or off-distribution responses | Effort message missing | Use `TmlV0Renderer`, or `render_for_completion_with_effort` in custom code |
| Eval scores that don't reproduce | Effort not recorded or not set | Pin and log the effort value for every eval run |
| Train/test mismatch after SFT | Data rendered at a different effort than sampling | Render training data with the effort you deploy at |
| Responses truncated mid-thought | `max_tokens` not scaled with effort | Raise the budget; high effort can need 16k+ tokens |
| `ModuleNotFoundError: tml_renderers` | Incomplete installation | Reinstall `tinker-cookbook` |
| `TmlV0Renderer requires PyTorch 2.10 or newer` | Old torch | Reinstall `tinker-cookbook`, or run `pip install "torch>=2.10"` |
| `get_lr` raises `NotImplementedError` | No general-purpose default LR is published for Inkling | Calibrate for your task: pick a starting value and sweep |
| `NotImplementedError` on prefill | TMLv0 sampling rejects partial assistant messages | Pass complete messages instead |
| Frequent `MALFORMED` terminations | Truncated or unparseable output | Raise `max_tokens`; retry with a corrective message carrying the parse error |

## Reference

Model and documentation:

- [Inkling](https://thinkingmachines.ai/inkling/) — model overview and capabilities
- [Introducing Inkling](https://thinkingmachines.ai/news/introducing-inkling/#controllable-thinking-effort) — announcement, including controllable thinking effort
- [Using Inkling](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/) — canonical setup and usage guide
- [Thinking effort](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/) — presets, sweeps, and effort-scaling results
- [Audio](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/audio/) and [Images](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/images/) — multimodal input requirements
- [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/) — context lengths and pricing per model
- [OpenAI-compatible endpoint](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/) — `reasoning_effort` as a named preset or a raw float
- [Anthropic-compatible endpoint](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/) — point Claude Code or the Anthropic SDKs at a Tinker model; in beta, and no audio input

Packages and code:

- [`tml-renderers`](https://pypi.org/project/tml-renderers/) — the rendering library behind `tml_v0`
- [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook) — this repository
- [`sample_reasoning.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/scripts/inkling/sample_reasoning.py) — the effort sweep script
- `tinker_cookbook/renderers/tml_v0.py` — the cookbook renderer adapter
- `tinker_cookbook/scripts/inkling/` — runnable sampling scripts for effort, audio, and images
- `tinker_cookbook/recipes/audio/` — audio SFT and RL recipes

For general post-training methodology use the `research` skill; for training or deployment triage use the `debug` skill.
