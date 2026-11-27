"""``compute_prob_change_by_advantage`` splits an update's effect on action tokens by advantage sign."""

from __future__ import annotations

import math

import pytest
import tinker
import torch

from tinker_cookbook.rl.metrics import compute_prob_change_by_advantage

POS = "prob_change_pre_post_pos_adv"
NEG = "prob_change_pre_post_neg_adv"
GAP = "prob_change_pre_post_adv_gap"


def _datum(
    sampling_logprobs: list[float], advantage: float, mask: list[float] | None = None
) -> tinker.Datum:
    """A datum whose action tokens (mask 1) all carry ``advantage``, as the RL loop builds them."""
    n_tokens = len(sampling_logprobs)
    mask_T = torch.tensor([1.0] * n_tokens if mask is None else mask)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(n_tokens))),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.arange(1, n_tokens + 1)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(sampling_logprobs)),
            "mask": tinker.TensorData.from_torch(mask_T),
            "advantages": tinker.TensorData.from_torch(advantage * mask_T),
        },
    )


def _post_logprobs(datum: tinker.Datum, ratio: float) -> torch.Tensor:
    """Post-update logprobs under which every token's probability is ``ratio`` times its sampling one."""
    return datum.loss_fn_inputs["logprobs"].to_torch() + math.log(ratio)


# Two positive-advantage datums of different lengths, a negative-advantage one whose first
# token is an observation, and a zero-advantage one that moved the most.
DATA_D = [
    _datum([-0.5, -0.25], advantage=0.5),
    _datum([-1.0, -0.5, -0.25, -0.125], advantage=0.5),
    _datum([-3.0, -0.5, -0.5], advantage=-1.0, mask=[0.0, 1.0, 1.0]),
    _datum([-0.5], advantage=0.0),
]
RATIOS = [1.02, 1.05, 0.97, 2.0]


def _post_logprobs_D() -> list[torch.Tensor]:
    post_logprobs_D = [_post_logprobs(datum, ratio) for datum, ratio in zip(DATA_D, RATIOS)]
    # An observation token is not part of the change, however far it moved.
    post_logprobs_D[2][0] += 5.0
    return post_logprobs_D


def test_splits_the_change_by_advantage_sign_over_action_tokens():
    metrics = compute_prob_change_by_advantage(DATA_D, _post_logprobs_D())

    # Each side is a mean over its action tokens: two at +2% and four at +5%.
    assert metrics[POS] == pytest.approx((2 * 0.02 + 4 * 0.05) / 6, abs=1e-6)
    assert metrics[NEG] == pytest.approx(-0.03, abs=1e-6)
    assert metrics[GAP] == pytest.approx(0.04 + 0.03, abs=1e-6)
    assert set(metrics) == {POS, NEG, GAP}


def test_a_change_unrelated_to_what_was_sampled_reads_zero():
    # Four tokens with sampling probabilities 1/2, 1/4, 1/8, 1/8 appear in those proportions,
    # so a mean over the datum's tokens is an expectation under the sampling distribution.
    sampling_probs = [0.5] * 4 + [0.25] * 2 + [0.125] * 2
    post_probs = [0.4] * 4 + [0.3] * 2 + [0.2, 0.1]
    sampling_logprobs = [math.log(p) for p in sampling_probs]
    post_logprobs = torch.tensor([math.log(p) for p in post_probs])
    data_D = [
        _datum(sampling_logprobs, advantage=1.0),
        _datum(sampling_logprobs, advantage=-1.0),
    ]

    metrics = compute_prob_change_by_advantage(data_D, [post_logprobs, post_logprobs])

    # The mean logprob change of the same tokens is negative (minus a KL divergence).
    assert (post_logprobs - torch.tensor(sampling_logprobs)).mean().item() < -0.03
    assert metrics[POS] == pytest.approx(0.0, abs=1e-6)
    assert metrics[NEG] == pytest.approx(0.0, abs=1e-6)
    assert metrics[GAP] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("advantages", [[0.5, 1.0], [-1.0], [0.0], [0.5, 0.0]])
def test_reports_nothing_without_action_tokens_of_both_signs(advantages: list[float]):
    data_D = [_datum([-0.5, -0.25], advantage=advantage) for advantage in advantages]
    post_logprobs_D = [_post_logprobs(datum, 1.1) for datum in data_D]

    assert compute_prob_change_by_advantage(data_D, post_logprobs_D) == {}


def test_a_datum_without_action_tokens_changes_nothing():
    unsampled = _datum([-0.5, -0.5], advantage=0.5, mask=[0.0, 0.0])

    assert compute_prob_change_by_advantage(
        [*DATA_D, unsampled], [*_post_logprobs_D(), _post_logprobs(unsampled, 9.0)]
    ) == compute_prob_change_by_advantage(DATA_D, _post_logprobs_D())
