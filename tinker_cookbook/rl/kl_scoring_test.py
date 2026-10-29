"""The KL scoring calls keep the whole ``SampleResponse``: same logprobs, plus reported usage."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
import tinker
import torch
from tinker.types import SampledSequence, SampleResponse

from tinker_cookbook.rl import metrics as rl_metrics
from tinker_cookbook.utils.cost_estimation import TokenUsage


class _ScoringClient:
    """Fake sampling client: fixed prompt logprobs per position, configurable response metadata."""

    def __init__(
        self, logprob_at: float, *, cache_hits: int = 0, returned_tokens: list[int] | None = None
    ) -> None:
        self.logprob_at = logprob_at
        self.cache_hits = cache_hits
        self.returned_tokens = [7] if returned_tokens is None else returned_tokens
        self.requests: list[dict[str, Any]] = []

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        **kwargs: Any,
    ) -> SampleResponse:
        self.requests.append(
            {
                "length": prompt.length,
                "num_samples": num_samples,
                "max_tokens": sampling_params.max_tokens,
                "include_prompt_logprobs": include_prompt_logprobs,
            }
        )
        # The first prompt position has no logprob, as from compute_logprobs.
        logprobs: list[float | None] = [None] + [self.logprob_at] * (prompt.length - 1)
        return SampleResponse(
            sequences=[
                SampledSequence(
                    stop_reason="length",
                    sequence_id="seq",
                    _tokens_list=list(self.returned_tokens),
                    _logprobs_list=[-1.0] * len(self.returned_tokens),
                )
            ],
            prompt_cache_hit_tokens=self.cache_hits,
            _prompt_logprobs_list=logprobs,
        )

    async def compute_logprobs_async(self, prompt: tinker.ModelInput) -> list[float | None]:
        """What the SDK's wrapper would have returned for the same request."""
        return [None] + [self.logprob_at] * (prompt.length - 1)


def _datum(n_tokens: int, sampled_logprob: float) -> tinker.Datum:
    tokens = list(range(n_tokens + 1))
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.full((n_tokens,), sampled_logprob)),
            "mask": tinker.TensorData.from_torch(torch.ones(n_tokens)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(n_tokens)),
        },
    )


@pytest.mark.parametrize("cache_hits", [0, 3])
def test_score_sequences_reports_cache_hits_and_returned_tokens(cache_hits: int):
    client = _ScoringClient(-0.5, cache_hits=cache_hits, returned_tokens=[7, 8])
    inputs = [tinker.ModelInput.from_ints([1, 2, 3, 4]), tinker.ModelInput.from_ints([5, 6, 7])]

    logprobs, usage = asyncio.run(rl_metrics.score_sequences(cast(Any, client), inputs))

    assert logprobs == [[None, -0.5, -0.5, -0.5], [None, -0.5, -0.5]]
    assert usage == TokenUsage(
        sampled=4, prefill_cached=2 * cache_hits, prefill_uncached=7 - 2 * cache_hits
    )
    # Each sequence is one request, shaped like the one compute_logprobs makes.
    assert client.requests == [
        {"length": 4, "num_samples": 1, "max_tokens": 1, "include_prompt_logprobs": True},
        {"length": 3, "num_samples": 1, "max_tokens": 1, "include_prompt_logprobs": True},
    ]


def test_compute_post_kl_matches_compute_logprobs_and_reports_usage():
    data_D = [_datum(3, sampled_logprob=-1.0), _datum(2, sampled_logprob=-1.0)]
    client = _ScoringClient(-1.5, cache_hits=2)

    out, usage = asyncio.run(rl_metrics.compute_post_kl(data_D, cast(Any, client)))

    # Same arithmetic as before over compute_logprobs' output: diff = -1.0 - (-1.5)
    expected = []
    for datum in data_D:
        new = asyncio.run(client.compute_logprobs_async(datum.model_input.append_int(0)))
        expected.append(datum.loss_fn_inputs["logprobs"].to_torch() - torch.tensor(new[1:]))
    diffs = torch.cat(expected)
    assert out["kl_pre_post_v1"] == pytest.approx(diffs.mean().item())
    assert out["kl_pre_post_v2"] == pytest.approx(0.5 * (diffs**2).mean().item())
    # Full sequences are 4 and 3 tokens; each response reports 2 cache hits.
    assert usage == TokenUsage(sampled=2, prefill_cached=4, prefill_uncached=3)


def test_incorporate_kl_penalty_matches_compute_logprobs_and_reports_usage():
    data_D = [_datum(3, sampled_logprob=-1.0), _datum(2, sampled_logprob=-2.0)]
    client = _ScoringClient(-1.5, cache_hits=1)

    out, usage = asyncio.run(
        rl_metrics.incorporate_kl_penalty(
            data_D, cast(Any, client), kl_penalty_coef=0.1, kl_discount_factor=0.0
        )
    )

    # diffs: 3 tokens of (-1.0 + 1.5) and 2 tokens of (-2.0 + 1.5), masks all ones
    avg = (3 * 0.5 + 2 * -0.5) / 5
    assert out["kl_policy_base"] == pytest.approx(avg)
    assert data_D[0].loss_fn_inputs["advantages"].to_torch().tolist() == pytest.approx(
        [0.1 * (avg - 0.5)] * 3
    )
    assert data_D[1].loss_fn_inputs["advantages"].to_torch().tolist() == pytest.approx(
        [0.1 * (avg + 0.5)] * 2
    )
    assert out == {"kl_policy_base": pytest.approx(avg)}
    assert usage == TokenUsage(sampled=2, prefill_cached=2, prefill_uncached=5)
