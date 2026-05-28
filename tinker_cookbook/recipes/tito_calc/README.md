# tito_calc — TITO vs the cookbook's renderers, audited

This recipe is a small, runnable audit of one question: for a multi-turn
tool-calling rollout, do `tinker_cookbook.renderers` produce the **same
tokens** as the model's official HuggingFace chat template?

Empirically across every supported family we could test, the answer is **no.**

It's a deliberately tiny recipe — a four-message calculator rollout (system /
user / assistant-tool-call / tool / assistant-answer) — but the divergences it
surfaces are systematic, model-family-independent, and have a direct bearing
on training quality. The recipe is the [TITO blog post]'s argument applied to
the cookbook itself.

[TITO blog post]: https://qgallouedec-tito.hf.space/


## The finding in one table

On the canonical 5-message tool-calling conversation, with `train_on_what=ALL_ASSISTANT_MESSAGES`:

| family            | cookbook renderer        | HF tokens | rend tokens | byte-equal? | stock prefix-preserving |
|-------------------|--------------------------|-----------|-------------|-------------|-------------------------|
| Llama-3.1         | `llama3`                 | 89        | 69          | **no**      | yes |
| Qwen3 (strip)     | `qwen3`                  | 80        | 75          | **no**      | no — needs §6 patch |
| Qwen3 instruct    | `qwen3_instruct`         | 80        | 75          | **no**      | no |
| Qwen3 no-think    | `qwen3_disable_thinking` | 80        | 79          | **no**      | no |
| DeepSeek-V3       | `deepseekv3`             | 50        | 56          | **no**      | yes |
| GPT-OSS           | `gpt_oss_no_sysprompt`   | 131       | 70          | **no**      | yes |
| SmolLM3           | *no cookbook renderer*   | 100       | —           | n/a         | yes |
| Laguna XS.2       | *no cookbook renderer*   | 93        | —           | n/a         | yes |

What's diverging in each row:

- **Llama-3.1** — cookbook omits HF's `Cutting Knowledge Date: …\nToday Date: …` preamble injected into the system message.
- **Qwen3 (all three modes)** — cookbook omits HF's empty `<think>\n\n</think>` block (auto-emitted on the last assistant turn) and the trailing `\n` after the final `<|im_end|>`.
- **DeepSeek-V3** — cookbook *adds* small content the stock HF template doesn't (+6 tokens).
- **GPT-OSS** — the largest gap (131 vs 70). Cookbook strips most of the Harmony preamble, channel headers, and tool-namespace declarations that HF's template emits.
- **SmolLM3, Laguna XS.2** — no cookbook renderer at all; TITO is the only path.

Reproduce with:

Reproduce with:

```bash
python -m tinker_cookbook.recipes.tito_calc.compare --n 5
```

## Why this matters: which rendering is in-distribution?

The model that is about to be RL-trained was originally trained on a specific
input format. At inference time, that same model will see the HuggingFace
chat template's output — because that's what `tokenizer.apply_chat_template`
produces and that's what every inference engine (vLLM, SGLang, OpenAI-compatible
servers, Tinker's deployed endpoint) uses.

So the HuggingFace chat template is the **canonical, declared, in-distribution
format** for the model. Training on a different rendering — even one that's
"the same conversation, minor decorations changed" — trains on bytes the model
never saw during pre/post-training and won't see at deployment. That's the
out-of-distribution case in textbook form, and it's the train/deploy gap §8.2
of the blog covers.

This is also exactly what `tinker_cookbook/AGENTS.md` itself warns against:

> Never call `tokenizer.encode(prompt)` directly on a chat-tuned model. Raw
> encoding skips the chat template, producing OOD prompt tokens. The sampler
> and trainer then take subtly different code paths on those OOD inputs, and
> per-token sampler/trainer logprob KL can inflate by 5×+ (max ratios in the
> tens), silently breaking PPO/CISPO/GRPO importance ratios.

The cookbook's renderers don't raw-encode — they're proper renderings — but
each one is *a different rendering* than the chat template the model was
trained on. Same failure mechanism, smaller magnitude. The renderer fork is
what the cookbook trains against; HF's template is what the model expects and
what production renders.

## How TITO removes the gap

The whole TITO recipe for any of these models is one call:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("poolside/Laguna-XS.2", trust_remote_code=True)
out = tok.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)
token_ids = out["input_ids"]
loss_mask = out["assistant_masks"]
```

When the chat template carries `{% generation %}` markers (Laguna XS.2 ships
them, TRL's training templates inject them for the major families), this is
the entire path: tokens and assistant-only loss mask in one call, with no
per-family Python. The mask is exactly the tokens we want gradient on; the
ids are byte-identical to what production will render. No train/deploy gap.

For multi-turn rollouts where the next-turn prompt must extend the prior
sampled tokens byte-for-byte, the **one structural condition** §6 of the
blog identifies — *prefix preservation for tool messages* — has to hold on
the template. It does for every open-weights family the blog tested except
Qwen3 stock; TRL ships a one-line-patched training template for the
exceptions.

For models like **Laguna XS.2** the picture is even cleaner. Laguna's stock
template:

- already passes `is_chat_template_prefix_preserving`,
- already has `{% generation %}` markers (no mask hand-rolling),
- and ships an explicit `render_assistant_messages_raw` Jinja flag that
  implements the assert-equality invariant described in
  §4.1.3 of the Laguna XS.2 technical report (re-render after every step,
  assert the result equals the decoded rollout prefix).

There is no `LagunaXS2Renderer` in tinker-cookbook. Under TITO it doesn't
need one — `apply_chat_template` is the renderer.

## What's in this directory

```
tito_calc/
├── env.py                — canonical 5-message calc-tool conversation
├── rollout_renderer.py   — cookbook path: build (tokens, weights) via Renderer
├── rollout_tito.py       — TITO path: build (tokens, weights) via apply_chat_template
├── compare.py            — runs both across N families, prints the audit table
├── README.md             — you are here
```

## How to run

```bash
# install the cookbook + trl (the audit uses TRL's is_chat_template_prefix_preserving)
uv pip install -e .
uv pip install trl

# run the audit; downloads tokenizers as needed
python -m tinker_cookbook.recipes.tito_calc.compare --n 5
```

Models the harness probes: a representative tool-capable instruct from each
of the families the cookbook ships a renderer for, plus Laguna XS.2 as the
unsupported case.

## Mapping to the blog

| Recipe artifact | Blog section |
|---|---|
| `is_chat_template_prefix_preserving(tok)` baseline test | §6 *Prefix preservation* |
| Audit table showing cookbook-renderer ≠ HF template | §7 *Do you need a renderer for this?* — "a renderer is, in effect, a programmable fork of the chat template" |
| The cookbook trains on different bytes than HF renders | §8.2 *Honest edges — train/deploy template divergence* |
| Laguna's `render_assistant_messages_raw` invariant | §8.2 — "a cleaner construction worth knowing… pure TITO, no renderer required" |
| `apply_chat_template(..., return_assistant_tokens_mask=True)` as the whole TITO surface | §9 *The right primitive — a property test on the chat template, not a Python re-implementation* |

## Honest caveats

- **The cookbook's renderer divergences may be intentional.** `Llama3Renderer`
  explicitly documents "Omits the HF preamble" — that's a deliberate
  simplification, not a bug. The audit doesn't claim the cookbook renderers
  are *wrong*; it claims they're *forks of the chat template, not faithful
  reimplementations*, and that the fork has a measurable train/deploy gap.
- **One conversation isn't a benchmark.** The divergences shown are
  reproduced across all four `Qwen3` renderer modes and across families;
  they're structural, not accidents of the probe. But this recipe doesn't
  attempt to quantify downstream impact on RL metrics — just to show the
  gap exists and where it comes from.
- **Some divergences only fire in specific conditions** (e.g., the Qwen3
  trailing `\n` only appears at the end of the assembled training sample;
  the Llama 3 preamble only appears when no system message overrides it).
  The probe conversation is constructed to surface the structural cases.
- **GPT-OSS' Harmony template requires a tool-call link** (`tool_call_id`
  / `name`) that the simple probe doesn't provide; the audit lists it as
  template-raises rather than testing further.
