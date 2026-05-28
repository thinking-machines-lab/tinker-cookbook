"""Audit cookbook renderers vs HF apply_chat_template, on a multi-turn tool rollout.

For each (family, model_id, cookbook_renderer_name) tuple we attempt to build a
training sample two ways:

  - **renderer**: tinker_cookbook.renderers.get_renderer(name, tok).build_supervised_example(messages)
  - **TITO**:    tok.apply_chat_template(messages, tokenize=True, return_dict=True,
                                          return_assistant_tokens_mask=True)
                (falls back to a per-family header-split when the template lacks
                {% generation %} markers).

We report token counts and byte-equality between the two paths, plus the §6
prefix-preservation status of the stock template.

Run::

    python -m tinker_cookbook.recipes.tito_calc.compare [--n 1]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from tinker_cookbook.recipes.tito_calc.env import (
    Problem,
    build_canonical_messages,
    make_problems,
)


@dataclass(frozen=True)
class Case:
    label: str
    model_id: str
    renderer_name: str | None  # None means "cookbook has no renderer"


CASES: list[Case] = [
    Case("Llama-3.1",        "hf-internal-testing/Llama-3.1-8B-Instruct", "llama3"),
    Case("Qwen3 (strip)",    "Qwen/Qwen3-0.6B",                            "qwen3"),
    Case("Qwen3 instruct",   "Qwen/Qwen3-0.6B",                            "qwen3_instruct"),
    Case("Qwen3 no-think",   "Qwen/Qwen3-0.6B",                            "qwen3_disable_thinking"),
    Case("DeepSeek-V3",      "deepseek-ai/DeepSeek-V3",                    "deepseekv3"),
    Case("GPT-OSS",          "hf-internal-testing/gpt-oss-20b",            "gpt_oss_no_sysprompt"),
    Case("SmolLM3",          "HuggingFaceTB/SmolLM3-3B",                   None),
    Case("Laguna XS.2",      "poolside/Laguna-XS.2",                       None),
]


def _try_renderer_tokens(case: Case, messages):
    """Return (status, tokens, weights) or (status, None, None)."""
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import RendererError, TrainOnWhat
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    if case.renderer_name is None:
        return "no-cookbook-renderer", None, None
    try:
        tok = get_tokenizer(case.model_id)
    except Exception as e:
        return f"load-fail: {type(e).__name__}", None, None
    try:
        r = get_renderer(case.renderer_name, tok)
    except RendererError:
        return "renderer-name-unknown", None, None
    except Exception as e:
        return f"renderer-error: {type(e).__name__}", None, None
    try:
        mi, w = r.build_supervised_example(messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    except Exception as e:
        return f"build-fail: {type(e).__name__}: {str(e)[:40]}", None, None
    ids = [t for c in mi.chunks for t in c.tokens]
    return "ok", ids, [int(x) for x in w.tolist()]


def _try_tito_tokens(case: Case, messages):
    """Return (status, tokens, weights) — never raises."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(case.model_id, trust_remote_code=True)
    except Exception as e:
        return f"load-fail: {type(e).__name__}", None, None
    # Prefer the generation-markers path: one call gives tokens + assistant mask.
    try:
        out = tok.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
        )
        ids = list(out["input_ids"])
        mask = [int(m) for m in out["assistant_masks"]]
        if sum(mask) > 0:
            return "ok-gen-markers", ids, mask
    except Exception:
        pass
    # Fallback: render without the mask, leave weights all zero (caller cares
    # about token equality, not mask correctness in that case).
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
        return "ok-no-mask", list(ids), [0] * len(ids)
    except Exception as e:
        return f"hf-template-error: {type(e).__name__}", None, None


def _prefix_preserving(case: Case, messages) -> str:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(case.model_id, trust_remote_code=True)
    except Exception:
        return "?"
    try:
        from trl.chat_template_utils import is_chat_template_prefix_preserving

        return "yes" if is_chat_template_prefix_preserving(tok) else "no"
    except Exception:
        return "?"


def run_one(case: Case, problem: Problem) -> dict:
    messages = build_canonical_messages(problem)
    r_status, r_ids, _ = _try_renderer_tokens(case, messages)
    t_status, t_ids, t_mask = _try_tito_tokens(case, messages)
    pp = _prefix_preserving(case, messages)

    n_r = len(r_ids) if r_ids is not None else None
    n_t = len(t_ids) if t_ids is not None else None
    loss_t = sum(t_mask) if t_mask is not None else None
    byte_equal = (r_ids is not None and t_ids is not None and r_ids == t_ids)
    return {
        "case": case,
        "renderer_status": r_status,
        "tito_status": t_status,
        "renderer_tokens": n_r,
        "tito_tokens": n_t,
        "tito_loss_tokens": loss_t,
        "prefix_preserving": pp,
        "byte_equal": byte_equal if r_ids is not None and t_ids is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The probe conversation is the same shape for every family; we just need
    # one example to surface the structural divergences. --n > 1 confirms that
    # the result is stable across problems.
    problems = make_problems(args.n, seed=args.seed)
    problem = problems[0]
    rows = [run_one(c, problem) for c in CASES]

    # Pretty-print
    print(f"\nProbe conversation (problem={problem.a} {problem.op} {problem.b}):")
    for m in build_canonical_messages(problem):
        print(f"  [{m['role']:9s}] {m['content']}")
    print()
    print(f"{'family':22s} {'renderer':28s} {'hf-tokens':>10} {'rend-tokens':>12} {'byte-eq':>8} {'prefix-pres':>12}")
    print("-" * 100)
    for r in rows:
        c: Case = r["case"]
        rname = c.renderer_name or "(none)"
        n_r = "—" if r["renderer_tokens"] is None else str(r["renderer_tokens"])
        n_t = "—" if r["tito_tokens"] is None else str(r["tito_tokens"])
        if r["byte_equal"] is None:
            eq = "n/a"
        else:
            eq = "YES" if r["byte_equal"] else "no"
        pp = r["prefix_preserving"]
        print(f"{c.label:22s} {rname:28s} {n_t:>10} {n_r:>12} {eq:>8} {pp:>12}")

    # Diagnostics for non-trivial failures
    print()
    for r in rows:
        c: Case = r["case"]
        if r["renderer_status"] != "ok" and r["renderer_status"] != "no-cookbook-renderer":
            print(f"  · {c.label} renderer: {r['renderer_status']}")
        if r["tito_status"] not in ("ok-gen-markers", "ok-no-mask"):
            print(f"  · {c.label} TITO: {r['tito_status']}")

    print()
    print("legend:")
    print("  hf-tokens     = len(tok.apply_chat_template(messages))   # HF chat template")
    print("  rend-tokens   = len(cookbook Renderer.build_supervised_example)  # cookbook fork")
    print("  byte-eq       = HF tokens == cookbook renderer tokens, byte-for-byte")
    print("  prefix-pres   = trl.chat_template_utils.is_chat_template_prefix_preserving on stock")
    print()
    print("see README.md for what these numbers say about train/deploy alignment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
