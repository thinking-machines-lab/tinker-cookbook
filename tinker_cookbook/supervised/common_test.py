"""Tests for supervised/common.py weight reduction in datum_from_model_input_weights."""

import math
from typing import Literal, cast

import pytest
import tinker
import torch

from tinker_cookbook.supervised.common import (
    _counted_byte_count,
    compute_bpb,
    datum_from_model_input_weights,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


def _make_model_input(tokens: list[int]) -> tinker.ModelInput:
    return tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=tokens)])


def _extract_weights(datum: tinker.Datum) -> list[int] | list[float]:
    return datum.loss_fn_inputs["weights"].data


def _extract_targets(datum: tinker.Datum) -> list[int] | list[float]:
    return datum.loss_fn_inputs["target_tokens"].data


def test_reduction_mean_sums_to_one():
    """With reduction='mean', output weights should sum to 1.0."""
    # 5 tokens -> 4 targets after right-shift. weights[1:5] = [0, 1, 1, 1]
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, reduction="mean")
    out_weights = _extract_weights(datum)
    assert len(out_weights) == 4
    assert math.isclose(sum(out_weights), 1.0, rel_tol=1e-6)


def test_reduction_none_preserves_original():
    """With reduction='none', output weights should be unchanged from the slice."""
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, reduction="none")
    out_weights = _extract_weights(datum)
    # After right-shift slicing: weights[1:5] = [0.0, 1.0, 1.0, 1.0]
    assert out_weights == [0.0, 1.0, 1.0, 1.0]


def test_reduction_default_is_none():
    """The default for reduction should be 'none' (no reduction)."""
    model_input = _make_model_input([10, 20, 30, 40, 50])
    weights = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights)
    out_weights = _extract_weights(datum)
    # Without reduction, raw weights are preserved: [0.0, 1.0, 1.0, 1.0]
    assert out_weights == [0.0, 1.0, 1.0, 1.0]


def test_all_zero_weights_no_division_by_zero():
    """All-zero weights with 'mean' should not cause division by zero."""
    model_input = _make_model_input([10, 20, 30])
    weights = torch.tensor([0.0, 0.0, 0.0])
    datum = datum_from_model_input_weights(model_input, weights, reduction="mean")
    out_weights = _extract_weights(datum)
    assert all(w == 0.0 for w in out_weights)


def test_single_nonzero_weight_normalizes_to_one():
    """A single non-zero weight with 'mean' should normalize to 1.0."""
    model_input = _make_model_input([10, 20, 30])
    weights = torch.tensor([0.0, 0.0, 5.0])
    datum = datum_from_model_input_weights(model_input, weights, reduction="mean")
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


def test_trailing_empty_text_chunk_does_not_break_right_shift():
    """A trailing empty text chunk should not keep the last real token in the input."""
    model_input = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(tokens=[10]),
            tinker.types.EncodedTextChunk(tokens=[20, 30]),
            tinker.types.EncodedTextChunk(tokens=[]),
        ]
    )
    weights = torch.tensor([1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights)

    assert datum.model_input.length == 2
    assert _extract_targets(datum) == [20, 30]


def test_max_length_truncation_with_mean_reduction():
    """Truncation + 'mean' reduction should produce weights summing to 1.0."""
    model_input = _make_model_input([10, 20, 30, 40, 50, 60])
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, max_length=4, reduction="mean")
    out_weights = _extract_weights(datum)
    # Truncated to 4 tokens -> 3 targets
    assert len(out_weights) == 3
    assert math.isclose(sum(out_weights), 1.0, rel_tol=1e-6)


def test_max_length_truncation_without_reduction():
    """Truncation with 'none' should preserve raw weight values."""
    model_input = _make_model_input([10, 20, 30, 40, 50, 60])
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    datum = datum_from_model_input_weights(model_input, weights, max_length=4, reduction="none")
    out_weights = _extract_weights(datum)
    assert len(out_weights) == 3
    # weights[1:4] = [1.0, 1.0, 1.0]
    assert out_weights == [1.0, 1.0, 1.0]


def test_invalid_reduction_raises():
    """An unrecognized reduction string should raise ValueError."""
    model_input = _make_model_input([10, 20, 30])
    weights = torch.tensor([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Unknown reduction mode"):
        datum_from_model_input_weights(model_input, weights, reduction="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bits-per-byte (compute_bpb)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stand-in mapping each token id to a fixed string.

    Exposes ``all_special_ids`` (used by ``compute_bpb`` to exclude special
    tokens from both numerator and denominator) and a ``decode`` that renders
    the given ids to their strings.
    """

    def __init__(self, vocab: dict[int, str], special_ids: set[int] | None = None):
        self.vocab = vocab
        self.all_special_ids = list(special_ids or [])

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(
            self.vocab[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids)
        )


class _MethodOnlySpecialTokenizer:
    """Tokenizer that exposes special tokens by method, not ``all_special_ids``."""

    def __init__(self, vocab: dict[int, str], special_ids: set[int]):
        self.vocab = vocab
        self._special_ids = special_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.vocab[i] for i in ids)

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self._special_ids


def _fake_tokenizer(vocab: dict[int, str], special_ids: set[int] | None = None) -> Tokenizer:
    """Build a `_FakeTokenizer` typed as `Tokenizer` for the BPB helpers."""
    return cast(Tokenizer, _FakeTokenizer(vocab, special_ids))


def _td(data: list[float] | list[int], dtype: Literal["float32", "int64"]) -> tinker.TensorData:
    return tinker.TensorData(data=list(data), dtype=dtype, shape=[len(data)])


def test_compute_bpb_matches_hand_computation():
    """bpb = -sum(logprobs * weights) / (ln(2) * target_bytes)."""
    tok = _fake_tokenizer({1: "he", 2: "llo"})  # decodes to "hello" -> 5 bytes
    logprobs = _td([-0.5, -1.5], "float32")
    weights = _td([1.0, 1.0], "float32")
    targets = _td([1, 2], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    expected = 2.0 / (math.log(2) * 5)  # total nats = 0.5 + 1.5; bytes = 5
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_ignores_zero_weight_tokens():
    """Weight-0 (prompt) tokens affect neither the numerator nor the byte count."""
    tok = _fake_tokenizer({1: "AB", 2: "x", 3: "yz"})
    logprobs = _td([-9.0, -1.0, -1.0], "float32")  # big loss on the prompt token
    weights = _td([0.0, 1.0, 1.0], "float32")
    targets = _td([1, 2, 3], "int64")  # weighted run [2, 3] -> "xyz" -> 3 bytes
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    expected = 2.0 / (math.log(2) * 3)
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_invariant_to_weight_magnitude():
    """Weights are a mask, not a multiplier: per-example normalized weights
    (reduction="mean") give the same BPB as 0/1 weights."""
    tok = _fake_tokenizer({1: "ab", 2: "cd"})  # "abcd" -> 4 bytes
    logprobs = _td([-1.0, -3.0], "float32")  # total nats = 1 + 3 = 4 in both cases
    targets = _td([1, 2], "int64")
    expected = 4.0 / (math.log(2) * 4)
    bpb_binary = compute_bpb([logprobs], [_td([1.0, 1.0], "float32")], [targets], tok)
    bpb_mean = compute_bpb([logprobs], [_td([0.5, 0.5], "float32")], [targets], tok)
    assert math.isclose(bpb_binary, expected, rel_tol=1e-6)
    assert math.isclose(bpb_mean, expected, rel_tol=1e-6)


def test_compute_bpb_counts_utf8_bytes_not_chars():
    """Multi-byte characters count as their UTF-8 byte length, not 1 char each."""
    tok = _fake_tokenizer({1: "café", 2: "中"})  # "café" = 5 bytes, "中" = 3 bytes
    logprobs = _td([-1.0, -1.0], "float32")
    weights = _td([1.0, 1.0], "float32")
    targets = _td([1, 2], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    expected = 2.0 / (math.log(2) * 8)
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_excludes_special_tokens_from_both_sides():
    """A trained end-of-turn special token is excluded from BOTH the numerator
    and the byte denominator, so it does not bias BPB."""
    tok = _fake_tokenizer({1: "hi", 99: "<|end|>"}, special_ids={99})
    logprobs = _td([-1.0, -5.0], "float32")  # large loss on the special token
    weights = _td([1.0, 1.0], "float32")
    targets = _td([1, 99], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    # Only token 1 ("hi") counts: 1 nat over 2 bytes. The special token's -5.0
    # nats are dropped along with its (zero) bytes.
    expected = 1.0 / (math.log(2) * len("hi"))
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_excludes_method_only_special_tokens():
    """Some tokenizers expose special/control IDs via an ``is_special_token`` method."""
    tok = cast(Tokenizer, _MethodOnlySpecialTokenizer({1: "hi", 99: "<|end|>"}, {99}))
    logprobs = _td([-1.0, -5.0], "float32")
    weights = _td([1.0, 1.0], "float32")
    targets = _td([1, 99], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    expected = 1.0 / (math.log(2) * len("hi"))
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_excludes_special_tokens_under_mean_reduction():
    """Both fixes together: special tokens excluded from both sides, and weights
    treated as a mask (mean-reduction normalized weights)."""
    tok = _fake_tokenizer({1: "ab", 2: "cd", 99: "<|end|>"}, special_ids={99})
    logprobs = _td([-1.0, -3.0, -9.0], "float32")  # special token has huge loss
    weights = _td([1 / 3, 1 / 3, 1 / 3], "float32")  # normalized over 3 trained tokens
    targets = _td([1, 2, 99], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    # Counted = tokens 1, 2 ("abcd"): nats = 1 + 3 = 4, bytes = 4.
    expected = 4.0 / (math.log(2) * 4)
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_multi_turn_excludes_specials_and_splits_runs():
    """Special tokens between trained content runs (multi-turn) are excluded from
    both sides, and each content run is byte-counted independently."""
    tok = _fake_tokenizer({1: "ab", 2: "cd", 3: "ef", 99: "<|end|>"}, special_ids={99})
    # layout: run1=[1,2]  <|end|>  run2=[3]  <|end|>   (all trained)
    logprobs = _td([-1.0, -2.0, -8.0, -3.0, -8.0], "float32")  # big loss on both specials
    weights = _td([1.0, 1.0, 1.0, 1.0, 1.0], "float32")
    targets = _td([1, 2, 99, 3, 99], "int64")
    bpb = compute_bpb([logprobs], [weights], [targets], tok)
    # counted = tokens 1,2,3: nats = 1 + 2 + 3 = 6; bytes = "ab" + "cd" + "ef" = 6.
    expected = 6.0 / (math.log(2) * 6)
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_aggregates_across_batch():
    """Numerator and denominator sum over all datums in the batch."""
    tok = _fake_tokenizer({1: "ab", 2: "cde"})
    d1 = ([_td([-1.0], "float32")], [_td([1.0], "float32")], [_td([1], "int64")])  # "ab" = 2
    d2 = ([_td([-3.0], "float32")], [_td([1.0], "float32")], [_td([2], "int64")])  # "cde" = 3
    bpb = compute_bpb(d1[0] + d2[0], d1[1] + d2[1], d1[2] + d2[2], tok)
    expected = 4.0 / (math.log(2) * 5)  # nats = 1 + 3; bytes = 2 + 3
    assert math.isclose(bpb, expected, rel_tol=1e-6)


def test_compute_bpb_zero_bytes_returns_nan():
    """No weighted tokens -> zero bytes -> nan (no division by zero)."""
    tok = _fake_tokenizer({1: "a"})
    logprobs = _td([-1.0], "float32")
    weights = _td([0.0], "float32")
    targets = _td([1], "int64")
    assert math.isnan(compute_bpb([logprobs], [weights], [targets], tok))


def test_counted_byte_count_splits_runs():
    """Counted tokens separated by an uncounted gap are decoded independently."""
    tok = _fake_tokenizer({1: "ab", 2: "GAP", 3: "cde"})
    n = _counted_byte_count([1, 2, 3], [True, False, True], tok)
    assert n == len("ab") + len("cde")  # 2 + 3; the middle token is excluded
