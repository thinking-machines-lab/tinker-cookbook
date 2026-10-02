"""Check that SGLang serves each Tinker adapter faithfully.

    python -m tinker_cookbook.inference.sglang.compare \
        --base-model Qwen/Qwen3.8-27B --url http://localhost:30000 \
        --adapter lora1=tinker://<run-id>/sampler_weights/final \
        --adapter lora2=tinker://<run-id>/sampler_weights/final

For every prompt and adapter this samples greedily from SGLang, then scores the
sampled tokens four ways -- SGLang with the adapter, SGLang with no adapter, and
Tinker with each adapter and with none -- and reports three verdicts:

``applied`` (per row)
    The served output tracks Tinker's adapter more closely than it tracks the
    un-adapted base model. This fails when SGLang silently drops part of the
    adapter, which is what happens to a module whose weight keys resolve to no
    parameter -- SGLang reserves the slot and never fills it.

``routed`` (per row)
    Of all the adapters, the served output tracks the one that was requested.
    This fails when a request is answered with another adapter's weights. Needs
    two or more adapters; reported as ``n/a`` with one.

``close`` (per adapter, pooled over prompts)
    The mean per-token distance from Tinker stays within a threshold derived
    from the noise floor: what SGLang and Tinker disagree by on the very same
    tokens with no adapter anywhere. Measured in the same run because the
    floor is a property of the model and serving config -- 0.06 nats for bf16
    Qwen3.8, ~15x that for FP8 GLM-5.3 -- so no constant could stand in for
    it. ``--tolerance`` substitutes an absolute bound; ``--skip-closeness``
    drops the verdict. The floor needs a base id Tinker can sample (GLM-5.3
    needs ``--tinker-base`` with its ``:peft:`` variant); without one the
    verdict reports itself unavailable, never silently passed.

Distances are the mean absolute per-token logprob difference over the sampled
tokens, in nats. Comparing scores on one fixed token sequence -- rather than
comparing two independently sampled strings -- keeps sampling divergence out of
the measurement.

Note that ``routed`` can only discriminate between adapters that actually behave
differently. Two adapters trained to near-identical behavior will produce
near-identical logprobs, and the verdict becomes meaningless rather than wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

import aiohttp
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.inference.sglang.common import parse_adapter_spec
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain gradient descent in one sentence.",
    "Write a haiku about GPUs.",
    "What is 17 times 24?",
]
MAX_TOKENS = 128
REQUEST_TIMEOUT = 600

# Closeness threshold = max(CLOSENESS_SLACK x floor mean, CLOSENESS_MIN).
# Measured floors span 0.06 (Qwen3.8 bf16) to 0.90 nats (GLM-5.3 FP8), and
# every correctly served adapter pooled below its own floor, so twice the
# floor passes all with >=2x margin. CLOSENESS_MIN guards a near-zero floor.
CLOSENESS_SLACK = 2.0
CLOSENESS_MIN = 0.05


def _abs_diffs(a: list[float | None], b: list[float | None]) -> list[float]:
    """Per-token absolute differences, over positions where both sides scored.

    Positions are compared by index, so every producer must keep ``None`` in
    place rather than dropping it -- SGLang reports ``None`` for the first token
    of a rescoring window, and Tinker for any token it did not score. Filtering
    those out would shift one array against the other and silently compare each
    position with its neighbour.
    """
    return [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]


def _summarize(label: str, vals: list[float]) -> str:
    """One line of distribution stats, for choosing a closeness threshold."""
    if not vals:
        return f"  {label:<22} (no comparable positions)"
    v = sorted(vals)
    n = len(v)

    def pick(q: float) -> float:
        return v[min(n - 1, int(n * q))]

    return (
        f"  {label:<22} n={n:<5} mean={sum(v) / n:8.4f}  median={pick(0.5):8.4f}"
        f"  p90={pick(0.9):8.4f}  p95={pick(0.95):8.4f}  max={v[-1]:8.4f}"
    )


def _mean_abs_diff(a: list[float | None], b: list[float | None]) -> float:
    """Mean over ``_abs_diffs``; ``inf`` when no position is comparable."""
    diffs = _abs_diffs(a, b)
    if not diffs:
        return float("inf")
    return sum(diffs) / len(diffs)


def _logprob_values(entries: list[list[object]]) -> list[float | None]:
    """Pull the logprob out of SGLang's ``[logprob, token_id, text]`` triples.

    Keeps ``None`` entries so the result stays index-aligned with the tokens.
    """
    return [None if e[0] is None else float(cast(float, e[0])) for e in entries]


def _token_ids(entries: list[list[object]]) -> list[int]:
    return [int(cast(int, e[1])) for e in entries]


def _meta_entries(result: dict[str, object], key: str) -> list[list[object]]:
    """The ``[logprob, token_id, text]`` triples under ``meta_info[key]``."""
    meta = cast(dict[str, object], result["meta_info"])
    return cast(list[list[object]], meta[key])


async def _generate(
    session: aiohttp.ClientSession, url: str, body: dict[str, object]
) -> list[dict[str, object]]:
    async with session.post(f"{url}/generate", json=body) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    return payload if isinstance(payload, list) else [payload]


async def _tinker_score(
    client: tinker.SamplingClient, prompt: tinker.ModelInput, out_ids: list[int]
) -> list[float | None]:
    """Score the full prompt+output sequence with a Tinker sampling client.

    ``logprobs[i]`` scores the token at absolute position i (``None`` at 0 and
    wherever Tinker did not score), verified position-aligned with SGLang's
    ``logprob_start_len: 0`` scoring — so callers slice ``[prompt.length:]``
    for the sampled tokens. Beware: an off-by-one here hides on
    near-deterministic continuations, where every logprob is ~0.
    """
    full = prompt.append(tinker.EncodedTextChunk(tokens=out_ids))
    logprobs = await client.compute_logprobs_async(full)
    return list(logprobs[: prompt.length + len(out_ids)])


def _generation_prompt(base_model: str, tokenizer: Tokenizer, prompt: str) -> tinker.ModelInput:
    """Tokenize one user turn, ready for generation.

    Prefers the cookbook renderer. Models the cookbook has no renderer for —
    ``model_info`` raises ``ConfigurationError`` for them — fall back to the
    tokenizer's own chat template, which is the other prompt path CLAUDE.md
    sanctions. Either way both sides of the comparison get the same token ids,
    so a logprob difference still reflects weights rather than formatting.

    A fallback prompt is only as canonical as the checkpoint's HF chat template.
    If Tinker trained the model against a different one, the prompt is somewhat
    off-distribution — equally for Tinker and SGLang, so the distance between
    them stays meaningful.
    """
    messages: list[renderers.Message] = [{"role": "user", "content": prompt}]
    try:
        renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(base_model), tokenizer
        )
    except ConfigurationError:
        logger.warning(
            "No cookbook renderer for %s; using the tokenizer's chat template instead.",
            base_model,
        )
        # This branch exists for tokenizers outside the cookbook's typed
        # surface, so the call is deliberately untyped.
        encoded = cast(Any, tokenizer).apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        ids = encoded if isinstance(encoded, list) else encoded["input_ids"]
        while ids and isinstance(ids[0], list):
            ids = ids[0]
        return tinker.ModelInput.from_ints([int(t) for t in ids])
    return renderer.build_generation_prompt(messages)


async def run(
    *,
    base_model: str,
    url: str,
    adapters: list[tuple[str, str]],
    prompts: list[str],
    tolerance: float | None = None,
    tinker_base: str | None = None,
    skip_closeness: bool = False,
) -> int:
    names = [name for name, _ in adapters]
    tokenizer = get_tokenizer(base_model)
    prompt_inputs = [_generation_prompt(base_model, tokenizer, p) for p in prompts]
    prompt_ids = [mi.to_ints() for mi in prompt_inputs]

    service = tinker.ServiceClient()
    # tinker_base pins which variant of the base model Tinker samples the adapter
    # through. It must name the same base as the checkpoint: ``:peft:`` context
    # variants are accepted, but a ``:sampling-nvfp4`` id is rejected for a
    # checkpoint trained on the plain model, so it cannot be used to match
    # Tinker's precision to a quantized SGLang deployment.
    clients = {
        name: service.create_sampling_client(model_path=path, base_model=tinker_base)
        for name, path in adapters
    }

    # Tinker scoring the un-adapted base model: paired with SGLang's own
    # base-model scoring of the same tokens, this is the engine-vs-engine
    # noise floor the closeness threshold is derived from.
    floor_client: tinker.SamplingClient | None = None
    floor_base = tinker_base or base_model
    try:
        floor_client = service.create_sampling_client(base_model=floor_base)
    except Exception as exc:  # any failure just disables the floor
        logger.warning(
            "No noise floor: Tinker will not sample base model %r (%s). Pass "
            "--tinker-base with an id Tinker accepts to enable it.",
            floor_base,
            exc,
        )

    # One row per (prompt, adapter): all adapters share a single mixed batch, which
    # is also how a real multi-adapter server sees traffic.
    rows = [(p, a) for p in range(len(prompts)) for a in names]
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        sampled = await _generate(
            session,
            url,
            {
                "input_ids": [prompt_ids[p] for p, _ in rows],
                "lora_path": [a for _, a in rows],
                "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_TOKENS},
                "return_logprob": True,
            },
        )
        out_ids: list[list[int]] = []
        served: list[list[float | None]] = []
        for result in sampled:
            entries = _meta_entries(result, "output_token_logprobs")
            out_ids.append(_token_ids(entries))
            served.append(_logprob_values(entries))

        # Re-score the same tokens with no adapter, from position 0. The
        # output slice is the control behind the "applied" verdict; the prompt
        # slice widens the noise floor, which matters for overfit adapters
        # that stop after a handful of tokens.
        scored = await _generate(
            session,
            url,
            {
                "input_ids": [prompt_ids[p] + out_ids[i] for i, (p, _) in enumerate(rows)],
                "lora_path": [None] * len(rows),
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": [0] * len(rows),
            },
        )
        base_full = [_logprob_values(_meta_entries(r, "input_token_logprobs")) for r in scored]
        base_lps = [full[len(prompt_ids[p]) :] for full, (p, _) in zip(base_full, rows)]

    # Reference logprobs from Tinker: every sampled sequence scored by every adapter.
    reference_full = await asyncio.gather(
        *[
            _tinker_score(clients[ref_name], prompt_inputs[p], out_ids[i])
            for i, (p, _requested) in enumerate(rows)
            for ref_name in names
        ]
    )
    reference = [
        full[len(prompt_ids[rows[k // len(names)][0]]) :] for k, full in enumerate(reference_full)
    ]
    floor_ref: list[list[float | None]] | None = None
    if floor_client is not None:
        floor_ref = list(
            await asyncio.gather(
                *[
                    _tinker_score(floor_client, prompt_inputs[p], out_ids[i])
                    for i, (p, _requested) in enumerate(rows)
                ]
            )
        )

    failures = 0
    row_failures = False
    pooled_adapter: dict[str, list[float]] = {name: [] for name in names}
    pooled_floor: list[float] = []
    print(f"{'prompt':<8}{'adapter':<14}{'d(tinker)':>11}{'d(base)':>10}{'d(floor)':>10}  result")
    for i, (p, requested) in enumerate(rows):
        per_adapter = {
            name: _mean_abs_diff(served[i], reference[i * len(names) + j])
            for j, name in enumerate(names)
        }
        d_own = per_adapter[requested]
        d_base = _mean_abs_diff(base_lps[i], reference[i * len(names) + names.index(requested)])
        closest = min(per_adapter, key=lambda n: per_adapter[n])

        pooled_adapter[requested] += _abs_diffs(
            served[i], reference[i * len(names) + names.index(requested)]
        )
        floor_diffs: list[float] = []
        if floor_ref is not None:
            n_prompt = len(prompt_ids[p])
            floor_diffs = _abs_diffs(base_full[i], floor_ref[i])
            # Pool output-token diffs from every row, prompt-token diffs only
            # from each prompt's first row -- the other rows re-score the same
            # prompt positions and would count the same measurement twice.
            pooled_floor += _abs_diffs(base_full[i][n_prompt:], floor_ref[i][n_prompt:])
            if i % len(names) == 0:
                pooled_floor += _abs_diffs(base_full[i][:n_prompt], floor_ref[i][:n_prompt])
        d_floor = sum(floor_diffs) / len(floor_diffs) if floor_diffs else float("nan")

        if d_own >= d_base:
            result = (
                "FAIL: lora not applied -- served logprobs are no closer to the fine-tune "
                "than to the base model"
            )
        elif len(names) > 1 and closest != requested:
            result = (
                f"FAIL: wrong adapter -- served logprobs match {closest!r} "
                f"({per_adapter[closest]:.4f}) better than {requested!r} ({d_own:.4f})"
            )
        else:
            result = "ok"

        if result != "ok":
            failures += 1
            row_failures = True
        floor_cell = "       n/a" if math.isnan(d_floor) else f"{d_floor:>10.4f}"
        print(f"{p:<8}{requested:<14}{d_own:>11.4f}{d_base:>10.4f}{floor_cell}  {result}")

    # The closeness verdict pools per-token diffs: per-row means are a poor
    # basis when an overfit adapter stops after two tokens and one argmax flip
    # moves the mean by half a nat.
    print("\n=== per-token |logprob difference|, pooled over prompts (nats) ===")
    print(_summarize("floor (base vs base)", pooled_floor))
    for name in names:
        print(_summarize(f"{name} (adapter)", pooled_adapter[name]))

    checks = len(rows)
    print("\n=== close: pooled per-adapter mean vs threshold ===")
    if skip_closeness:
        print("  skipped (--skip-closeness)")
    elif tolerance is not None or pooled_floor:
        if tolerance is not None:
            threshold = tolerance
            print(f"  threshold = {threshold:.4f} (--tolerance)")
        else:
            floor_mean = sum(pooled_floor) / len(pooled_floor)
            threshold = max(CLOSENESS_SLACK * floor_mean, CLOSENESS_MIN)
            print(
                f"  threshold = max({CLOSENESS_SLACK} x floor mean {floor_mean:.4f}, "
                f"{CLOSENESS_MIN}) = {threshold:.4f}"
            )
        for name in names:
            vals = pooled_adapter[name]
            mean = sum(vals) / len(vals) if vals else float("inf")
            checks += 1
            if mean <= threshold:
                print(f"  {name:<14}{mean:>8.4f}  ok")
            else:
                failures += 1
                print(
                    f"  {name:<14}{mean:>8.4f}  FAIL: exceeds {threshold:.4f} -- the "
                    "adapter is applied but not faithfully; suspect scaling "
                    "(lora_alpha), rank truncation, or a partially loaded module"
                )
    else:
        print(
            "  UNAVAILABLE -- Tinker cannot sample the base model, so there is no "
            "noise floor to derive a threshold from. Pass --tinker-base with a "
            "sampleable id (e.g. the :peft: variant), or --tolerance for an "
            "absolute bound. Only applied/routed were checked."
        )

    print(f"\n{checks - failures}/{checks} checks passed")
    if row_failures:
        print(
            "Compare the adapter's adapter_config.json against the target_modules the "
            "server logs at startup. A module whose weight keys SGLang cannot resolve is "
            "dropped in silence."
        )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-model", required=True, help="HF model name or local directory")
    parser.add_argument(
        "--url", required=True, help="SGLang server URL, e.g. http://localhost:30000"
    )
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        metavar="NAME=tinker://...",
        help="Adapter to check; NAME must match the one given to --lora-paths",
    )
    parser.add_argument("--prompts", help="JSON file holding a list of prompt strings")
    parser.add_argument(
        "--tinker-base",
        help=(
            "Base-model variant Tinker should sample the adapters through, e.g. "
            "a :sampling-nvfp4 id when SGLang is serving a quantized checkpoint. "
            "Must name the same base as the adapters; defaults to the checkpoint's own."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help=(
            "Absolute closeness bound in nats, replacing the threshold derived from the "
            "measured noise floor. The applied and routed checks run regardless."
        ),
    )
    parser.add_argument(
        "--skip-closeness",
        action="store_true",
        help="Only check applied and routed; skip the closeness verdict entirely.",
    )
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else DEFAULT_PROMPTS
    adapters = [parse_adapter_spec(spec) for spec in args.adapter]
    sys.exit(
        asyncio.run(
            run(
                base_model=args.base_model,
                url=args.url.rstrip("/"),
                adapters=adapters,
                prompts=prompts,
                tolerance=args.tolerance,
                tinker_base=args.tinker_base,
                skip_closeness=args.skip_closeness,
            )
        )
    )


if __name__ == "__main__":
    main()
