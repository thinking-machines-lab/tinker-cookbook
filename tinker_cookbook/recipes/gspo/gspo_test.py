"""Unit tests for the GSPO recipe loss (no API key needed)."""

import math

import pytest
import torch

from tinker_cookbook.recipes.gspo.loss import make_gspo_loss


def test_sequence_ratio_uses_completion_tokens_only():
    loss_fn = make_gspo_loss(
        old_logprobs_D=[torch.tensor([0.0, 50.0, -2.0, -3.0])],
        ob_lens_D=[2],
        advantages_D=[1.0],
        clip_low=0.0,
        clip_high=10.0,
    )

    loss, metrics = loss_fn(
        [],
        [torch.tensor([100.0, -100.0, -1.0, -2.0])],
    )

    expected_ratio = math.exp(1.0)
    assert loss.item() == pytest.approx(-expected_ratio)
    assert metrics["gspo_loss"] == pytest.approx(-expected_ratio)
    assert metrics["mean_log_ratio"] == pytest.approx(1.0)
    assert metrics["clip_frac"] == pytest.approx(0.0)


def test_sequence_ratio_clips_positive_advantage():
    loss_fn = make_gspo_loss(
        old_logprobs_D=[torch.zeros(2)],
        ob_lens_D=[0],
        advantages_D=[1.0],
        clip_low=0.9,
        clip_high=1.1,
    )

    loss, metrics = loss_fn(
        [],
        [torch.full((2,), math.log(2.0))],
    )

    assert loss.item() == pytest.approx(-1.1)
    assert metrics["clip_frac"] == pytest.approx(1.0)
    assert metrics["mean_log_ratio"] == pytest.approx(math.log(2.0))


def test_loss_averages_over_datums():
    loss_fn = make_gspo_loss(
        old_logprobs_D=[torch.zeros(1), torch.zeros(1)],
        ob_lens_D=[0, 0],
        advantages_D=[1.0, -1.0],
        clip_low=0.0,
        clip_high=10.0,
    )

    loss, metrics = loss_fn(
        [],
        [
            torch.tensor([math.log(2.0)]),
            torch.tensor([math.log(0.5)]),
        ],
    )

    expected_loss = (-2.0 + 0.5) / 2.0
    expected_mean_log_ratio = (math.log(2.0) + math.log(0.5)) / 2.0
    assert loss.item() == pytest.approx(expected_loss)
    assert metrics["gspo_loss"] == pytest.approx(expected_loss)
    assert metrics["mean_log_ratio"] == pytest.approx(expected_mean_log_ratio)


def test_empty_batch_returns_zero_metrics():
    loss_fn = make_gspo_loss(
        old_logprobs_D=[],
        ob_lens_D=[],
        advantages_D=[],
    )

    loss, metrics = loss_fn([], [])

    assert loss.item() == pytest.approx(0.0)
    assert metrics == {
        "gspo_loss": 0.0,
        "clip_frac": 0.0,
        "mean_log_ratio": 0.0,
    }
