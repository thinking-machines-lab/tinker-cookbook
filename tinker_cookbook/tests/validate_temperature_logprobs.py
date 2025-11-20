"""
Validate temperature scaling in sampling by comparing pairwise logprob differences.

Two complementary checks ensure correctness across temperatures and sequence positions:
1. Temperature scaling: Verifies (log p_τ(i) - log p_τ(j)) ≈ (1/τ) * (log p_1(i) - log p_1(j))
2. Sequence-level consistency: Validates multi-token sampling returns accurate logprobs at each step.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import chz
import numpy as np
import tinker

from tinker_cookbook.tokenizer_utils import get_tokenizer


def _default_temperatures() -> list[float]:
    return [0.5, 0.7, 1.0, 1.2, 1.5, 1.8]


@chz.chz
class Config:
    base_model: str
    prompt: str = (
        "Explain temperature scaling in language model sampling, include a brief "
        "example, and discuss calibration vs diversity trade-offs."
    )
    temperatures: list[float] = chz.field(default_factory=_default_temperatures)
    baseline_temperature: float = 1.0
    num_trials: int = 20
    check_sequence_consistency: bool = True
    consistency_check_length: int = 20
    consistency_check_temp: float = 0.5
    seed: int | None = 42
    base_url: str | None = None


async def _sample_next_token(
    sampling_client: tinker.SamplingClient,
    model_input: tinker.ModelInput,
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> tuple[list[int], list[float]]:
    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        ),
    )
    seq = resp.sequences[0]
    if seq.logprobs is None:
        raise RuntimeError("Sampling response did not include logprobs")
    return seq.tokens, seq.logprobs


async def _collect_sampled_token_logprobs(
    sampling_client: tinker.SamplingClient,
    model_input: tinker.ModelInput,
    *,
    temperature: float,
    num_trials: int,
    max_tokens: int,
    seed: int | None,
) -> dict[int, float]:
    """Collect token_id -> logprob at a given temperature over several trials."""
    out: dict[int, float] = {}
    base = 0 if seed is None else seed
    for i in range(num_trials):
        s = base + i if seed is not None else None
        tokens, lps = await _sample_next_token(
            sampling_client,
            model_input,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=s,
        )
        if not tokens:
            continue
        t = tokens[0]
        out.setdefault(t, lps[0])
    return out


async def _compute_logp1_for_tokens(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    tokens: Sequence[int],
) -> dict[int, float]:
    """Compute baseline log p_1(token|prompt) for each token via compute_logprobs_async."""
    res: dict[int, float] = {}
    for tok in tokens:
        seq = tinker.ModelInput.from_ints(prompt_tokens + [tok])
        lps = await sampling_client.compute_logprobs_async(seq)
        lp = lps[len(prompt_tokens)]
        if lp is None:
            raise RuntimeError(
                "compute_logprobs_async did not return a logprob for the sampled token"
            )
        res[tok] = lp
    return res


def _pairwise_ratio_metrics(
    base_logp: dict[int, float],
    temp_logp: dict[int, float],
    temperature: float,
) -> dict[str, float]:
    """Compare pairwise logprob differences: (log p_τ(i) - log p_τ(j)) vs (1/τ) * (log p_1(i) - log p_1(j))."""
    common = sorted(set(base_logp) & set(temp_logp))
    if len(common) < 2:
        return {
            "tokens": float(len(common)),
            "pairs": 0.0,
            "mean_abs_err": float("nan"),
            "max_abs_err": float("nan"),
        }
    base_diffs: list[float] = []
    temp_diffs: list[float] = []
    inv_tau = 1.0 / max(temperature, 1e-9)
    for a in range(len(common)):
        for b in range(a + 1, len(common)):
            i, j = common[a], common[b]
            base_diffs.append(inv_tau * (base_logp[i] - base_logp[j]))
            temp_diffs.append(temp_logp[i] - temp_logp[j])
    x = np.array(base_diffs, dtype=float)
    y = np.array(temp_diffs, dtype=float)
    abs_err = np.abs(y - x)
    mean_abs_err = float(np.mean(abs_err))
    max_abs_err = float(np.max(abs_err))
    return {
        "tokens": float(len(common)),
        "pairs": float(len(base_diffs)),
        "mean_abs_err": mean_abs_err,
        "max_abs_err": max_abs_err,
    }


# ============================================================================
# Sequence-level consistency validation
# ============================================================================


async def _sample_sequence_oneshot(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None,
) -> tuple[list[int], list[float]]:
    """Sample a sequence in one call with max_tokens > 1."""
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        ),
    )
    seq = resp.sequences[0]
    if seq.logprobs is None:
        raise RuntimeError("Sampling response did not include logprobs")
    return seq.tokens, seq.logprobs


async def _resample_tokens_individually(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    *,
    temperature: float,
    length: int,
    seed: int | None,
) -> tuple[list[int], list[float]]:
    """Sample tokens one at a time, feeding each back into the prefix.

    This mimics what max_tokens > 1 should do internally: sample token i,
    append to context, then sample token i+1.
    """
    tokens: list[int] = []
    logprobs: list[float] = []
    current_prefix = prompt_tokens.copy()

    for i in range(length):
        model_input = tinker.ModelInput.from_ints(current_prefix)
        # Increment seed for each position to get different random states
        pos_seed = (seed + i) if seed is not None else None

        resp = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=1,
                temperature=temperature,
                seed=pos_seed,
            ),
        )
        seq = resp.sequences[0]
        if not seq.tokens or seq.logprobs is None:
            break

        tok = seq.tokens[0]
        logprob = seq.logprobs[0]
        tokens.append(tok)
        logprobs.append(logprob)
        current_prefix.append(tok)

    return tokens, logprobs


def _compare_logprobs(
    sampled_logprobs: list[float],
    computed_logprobs: list[float],
) -> dict[str, float]:
    """Compare sampled vs recomputed logprobs."""
    min_len = min(len(sampled_logprobs), len(computed_logprobs))
    if min_len == 0:
        return {
            "length": 0.0,
            "mean_diff": float("nan"),
            "max_diff": float("nan"),
        }

    diffs = [abs(sampled_logprobs[i] - computed_logprobs[i]) for i in range(min_len)]

    return {
        "length": float(min_len),
        "mean_diff": float(np.mean(diffs)),
        "max_diff": float(np.max(diffs)),
    }


async def validate_sequence_consistency(
    sampling_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    *,
    temperature: float,
    length: int,
    seed: int | None,
    tokenizer,
) -> None:
    """Validate that sample_async(max_tokens > 1) returns accurate per-step logprobs.

    Generates a sequence then resamples each position individually to find matching tokens
    and compare their logprobs, validating correctness at each step.
    """
    print("\n" + "=" * 75)
    print("SEQUENCE-LEVEL CONSISTENCY CHECK (multi-token logprob validation)")
    print("=" * 75)
    print(
        f"Generate with max_tokens={length} at temp={temperature}, then resample each position individually to verify logprob consistency."
    )
    print(f"{'Temp':>8}  {'Length':>8}  {'Matches':>8}  {'Mean Diff':>12}  {'Max Diff':>12}")
    print("-" * 75)

    tau = temperature
    gen_tokens, gen_logprobs = await _sample_sequence_oneshot(
        sampling_client, prompt_tokens, temperature=tau, max_tokens=length, seed=seed
    )

    matching_diffs: list[float] = []
    num_attempts_per_position = 5

    for i in range(len(gen_tokens)):
        prefix = prompt_tokens + gen_tokens[:i]
        model_input = tinker.ModelInput.from_ints(prefix)

        for attempt in range(num_attempts_per_position):
            resp = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=1,
                    temperature=tau,
                    seed=(seed + 1000 * (i + 1) + attempt) if seed is not None else None,
                ),
            )
            seq = resp.sequences[0]
            if not seq.tokens or seq.logprobs is None:
                continue

            if seq.tokens[0] == gen_tokens[i]:
                matching_diffs.append(abs(gen_logprobs[i] - seq.logprobs[0]))
                break

    if len(matching_diffs) == 0:
        print(f"{tau:>8.3f}  {len(gen_tokens):>8}  {0:>8}  {'N/A':>12}  {'N/A':>12}  {'N/A':>8}")
        return

    mean_diff = float(np.mean(matching_diffs))
    max_diff = float(np.max(matching_diffs))
    print(
        f"{tau:>8.3f}  {len(gen_tokens):>8}  {len(matching_diffs):>8}  {mean_diff:>12.6f}  {max_diff:>12.6f}"
    )
    print()


async def main(cfg: Config) -> None:
    tokenizer = get_tokenizer(cfg.base_model)
    prompt_tokens = tokenizer.encode(cfg.prompt)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)

    service = tinker.ServiceClient(base_url=cfg.base_url)
    sampler = service.create_sampling_client(base_model=cfg.base_model)

    print("\n" + "=" * 75)
    print("TEMPERATURE SCALING VALIDATION")
    print("=" * 75)

    base_seen = await _collect_sampled_token_logprobs(
        sampler,
        model_input,
        temperature=cfg.baseline_temperature,
        num_trials=cfg.num_trials,
        max_tokens=1,
        seed=cfg.seed,
    )
    base_logp = await _compute_logp1_for_tokens(sampler, prompt_tokens, list(base_seen))

    print(f"Model: {cfg.base_model}, {cfg.num_trials} trials per temperature")
    print(f"{'Temp':>8}  {'Unique Tokens':>15}  {'Pairs':>8}  {'Mean Diff':>12}  {'Max Diff':>12}")
    print("-" * 75)

    for tau in cfg.temperatures:
        temp_seen = await _collect_sampled_token_logprobs(
            sampler,
            model_input,
            temperature=tau,
            num_trials=cfg.num_trials,
            max_tokens=1,
            seed=cfg.seed,
        )
        missing = [t for t in temp_seen if t not in base_logp]
        if missing:
            base_logp.update(await _compute_logp1_for_tokens(sampler, prompt_tokens, missing))
        metrics = _pairwise_ratio_metrics(base_logp, temp_seen, tau)

        mean_diff = metrics["mean_abs_err"]
        max_diff = metrics["max_abs_err"]
        print(
            f"{tau:>8.3f}  {int(metrics['tokens']):>15}  {int(metrics['pairs']):>8}  {mean_diff:>12.6f}  {max_diff:>12.6f}"
        )

    if cfg.check_sequence_consistency:
        await validate_sequence_consistency(
            sampler,
            prompt_tokens,
            temperature=cfg.consistency_check_temp,
            length=cfg.consistency_check_length,
            seed=cfg.seed,
            tokenizer=tokenizer,
        )

    print()


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
