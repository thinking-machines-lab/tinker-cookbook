"""Tests for supervised/common.py weight normalization in datum_from_model_input_weights."""

import math

import tinker
import torch

from tinker_cookbook.supervised.common import datum_from_model_input_weights


def _make_model_input(tokens: list[int]) -> tinker.ModelInput:
    return tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])


def _extract_weights(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["weights"].data


def _extract_targets(datum: tinker.Datum) -> list[int]:
    return datum.loss_fn_inputs["target_tokens"].data


def test_normalize_weights_true_sums_to_one():
    """With normalize_weights=True, output weights should sum to 1.0."""
    # 5 tokens -> 4 targets after right-shift. weights[1:5] = [0, 1, 1, 1]
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=True)
    out_weights = _extract_weights(datum)
    assert len(out_weights) == 4
    assert math.isclose(sum(out_weights), 1.0, rel_tol=1e-6)


def test_normalize_weights_false_preserves_original():
    """With normalize_weights=False, output weights should be unchanged from the slice."""
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=False)
    out_weights = _extract_weights(datum)
    # After right-shift slicing: weights[1:5] = [0.0, 1.0, 1.0, 1.0]
    assert out_weights == [0.0, 1.0, 1.0, 1.0]


def test_normalize_weights_default_is_true():
    """The default for normalize_weights should be True (normalizes)."""
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights)
    out_weights = _extract_weights(datum)
    assert math.isclose(sum(out_weights), 1.0, rel_tol=1e-6)


def test_all_zero_weights_no_division_by_zero():
    """All-zero weights should not cause division by zero; they should remain zero."""
    model_input = _make_model_input([10, 20, 30])
    weights = torch.tensor([0.0, 0.0, 0.0])
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=True)
    out_weights = _extract_weights(datum)
    assert all(w == 0.0 for w in out_weights)


def test_single_nonzero_weight_normalizes_to_one():
    """A single non-zero weight should normalize to 1.0."""
    model_input = _make_model_input([10, 20, 30])
    weights = torch.tensor([0.0, 0.0, 5.0])
    datum = datum_from_model_input_weights(model_input, weights, normalize_weights=True)
    out_weights = _extract_weights(datum)
    # weights[1:3] = [0.0, 5.0], normalized -> [0.0, 1.0]
    assert math.isclose(out_weights[-1], 1.0, rel_tol=1e-6)
    assert out_weights[0] == 0.0


def test_target_tokens_are_left_shifted():
    """Target tokens should be the input tokens shifted left by one position."""
    model_input = _make_model_input([10, 20, 30, 40])
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights)
    targets = _extract_targets(datum)
    assert targets == [20, 30, 40]


def test_max_length_truncation_with_normalization():
    """Truncation + normalization should produce weights summing to 1.0 over the truncated region."""
    model_input = _make_model_input([10, 20, 30, 40, 50, 60])
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(
        model_input, weights, max_length=4, normalize_weights=True
    )
    out_weights = _extract_weights(datum)
    # Truncated to 4 tokens -> 3 targets
    assert len(out_weights) == 3
    assert math.isclose(sum(out_weights), 1.0, rel_tol=1e-6)


def test_max_length_truncation_without_normalization():
    """Truncation without normalization should preserve raw weight values."""
    model_input = _make_model_input([10, 20, 30, 40, 50, 60])
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(
        model_input, weights, max_length=4, normalize_weights=False
    )
    out_weights = _extract_weights(datum)
    assert len(out_weights) == 3
    # weights[1:4] = [1.0, 1.0, 1.0]
    assert out_weights == [1.0, 1.0, 1.0]
