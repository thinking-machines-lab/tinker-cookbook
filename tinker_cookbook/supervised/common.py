import logging
import math
from collections.abc import Sequence
from typing import Literal

import tinker
import torch

from tinker_cookbook.exceptions import DataValidationError
from tinker_cookbook.tokenizer_utils import Tokenizer

_logged_reduction_modes: set[str] = set()

logger = logging.getLogger(__name__)


def compute_mean_nll(
    logprobs_list: list[tinker.TensorData], weights_list: list[tinker.TensorData]
) -> float:
    """Compute weighted mean negative log-likelihood across a batch.

    For each (logprobs, weights) pair the dot product gives the weighted
    log-probability; the result is ``-sum(logprobs * weights) / sum(weights)``
    over the entire batch.

    Args:
        logprobs_list (list[tinker.TensorData]): Per-token log-probabilities
            returned by a forward pass, one entry per datum in the batch.
        weights_list (list[tinker.TensorData]): Per-token loss weights aligned
            with ``logprobs_list``.

    Returns:
        float: The mean NLL, or ``nan`` if total weight is zero.
    """
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    if total_weights == 0:
        logger.warning("No valid weights found for NLL computation")
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)


def _weighted_target_byte_count(
    target_tokens: Sequence[float], weights: Sequence[float], tokenizer: Tokenizer
) -> int:
    """UTF-8 byte count of the text formed by the loss-weighted target tokens.

    Each contiguous run of nonzero-weight tokens is decoded separately so that
    weight-0 spans between trained regions (e.g. user turns in a multi-turn
    conversation) do not get merged across the gap. Special tokens are skipped
    so the count reflects natural-language bytes, which keeps the denominator
    comparable across tokenizers with different control-token strings.

    Args:
        target_tokens (Sequence[float]): Target token IDs
            (``loss_fn_inputs["target_tokens"].data``, stored as floats and cast
            back to ``int`` here).
        weights (Sequence[float]): Per-token loss weights aligned with
            ``target_tokens``.
        tokenizer (Tokenizer): Tokenizer used to decode tokens back to text.

    Returns:
        int: Total number of UTF-8 bytes across all weighted spans.
    """
    total_bytes = 0
    run: list[int] = []
    for token, weight in zip(target_tokens, weights, strict=True):
        if weight > 0:
            run.append(int(token))
        elif run:
            total_bytes += _decoded_byte_length(run, tokenizer)
            run = []
    if run:
        total_bytes += _decoded_byte_length(run, tokenizer)
    return total_bytes


def _decoded_byte_length(token_ids: list[int], tokenizer: Tokenizer) -> int:
    """UTF-8 byte length of ``token_ids`` decoded to natural-language text."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    if isinstance(text, list):
        # Some tokenizers can return a list of piece strings; join them.
        text = "".join(text)
    return len(text.encode("utf-8"))


def compute_bpb(
    logprobs_list: list[tinker.TensorData],
    weights_list: list[tinker.TensorData],
    target_tokens_list: list[tinker.TensorData],
    tokenizer: Tokenizer,
) -> float:
    """Compute bits-per-byte (BPB) across a batch: a tokenizer-independent NLL.

    Per-token mean NLL (:func:`compute_mean_nll`) is not comparable across
    models that use different tokenizers: a coarser tokenizer packs more text
    into each token (higher nats/token) while a finer one spreads it over more
    tokens (lower nats/token), even at equal modeling quality. Bits-per-byte
    removes this dependence by dividing the *total* log-loss (converted to bits)
    by the number of UTF-8 bytes of the target text::

        bpb = -sum(logprobs * weights) / (ln(2) * total_target_bytes)

    The numerator is the same weighted sum used by :func:`compute_mean_nll`
    (only without the per-token averaging); the denominator counts the bytes of
    the decoded loss-weighted tokens, which is fixed by the raw text rather than
    by the tokenization. Both cover the same weighted span, so the ratio is a
    proper compression rate that can be compared across tokenizers.

    Args:
        logprobs_list (list[tinker.TensorData]): Per-token log-probabilities,
            one entry per datum in the batch.
        weights_list (list[tinker.TensorData]): Per-token loss weights aligned
            with ``logprobs_list``.
        target_tokens_list (list[tinker.TensorData]): Per-token target IDs
            (``loss_fn_inputs["target_tokens"]``) aligned with ``weights_list``,
            used to recover the target text for the byte count.
        tokenizer (Tokenizer): Tokenizer used to decode target tokens to bytes.

    Returns:
        float: Bits per byte, or ``nan`` if the weighted target has zero bytes.
    """
    total_weighted_logprobs = 0.0
    total_bytes = 0

    for logprobs, weights, target_tokens in zip(
        logprobs_list, weights_list, target_tokens_list, strict=True
    ):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += float(logprobs_torch.dot(weights_torch))
        total_bytes += _weighted_target_byte_count(
            list(target_tokens.data), list(weights.data), tokenizer
        )

    if total_bytes == 0:
        logger.warning("No target bytes found for BPB computation")
        return float("nan")

    return float(-total_weighted_logprobs / (math.log(2) * total_bytes))


def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """Create right-shifted inputs and left-shifted target tokens from a chunk sequence.

    Given the full sequence of model-input chunks, produce:

    * **inputs** -- all chunks with the last token removed (retains images).
    * **targets** -- all token IDs with the first token removed (image positions
      are represented as ``0``).

    This implements the standard next-token-prediction shift used for causal
    language-model training.

    Args:
        chunks (list[tinker.ModelInputChunk]): Full sequence of text and/or
            image chunks.  The last chunk must be an ``EncodedTextChunk``.

    Returns:
        tuple[tinker.ModelInput, list[int]]: A ``(model_input, target_tokens)``
        pair where ``model_input`` has length ``N-1`` and ``target_tokens`` is
        a list of ``N-1`` integer token IDs.

    Raises:
        DataValidationError: If the last chunk is not a text chunk or the
            total length is less than 2.
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise DataValidationError(
            "The last chunk must be a text chunk. This is because images are 0-loss anyways, so we should remove them beforehand."
        )

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise DataValidationError("need at least 2 tokens for input/target split")

    # Build input chunks: all but last, then append truncated last chunk
    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    # Build target tokens: collect all tokens, then slice off first
    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: torch.Tensor,
    max_length: int | None = None,
    reduction: Literal["none", "mean"] = "none",
) -> tinker.Datum:
    """Create a training Datum from a ModelInput and per-token weights tensor.

    Performs ``max_length`` truncation and next-token slicing to produce the
    input / target pair consumed by the cross-entropy loss function.  Text
    chunks can be partially truncated; image chunks are discarded whole if
    they would exceed ``max_length``.

    Args:
        model_input (tinker.ModelInput): The model input containing a sequence
            of text and/or image chunks.
        weights (torch.Tensor): 1-D float tensor of per-token loss weights,
            aligned with ``model_input.length``.
        max_length (int | None): Optional maximum sequence length.  If
            provided, the input is truncated from the right to fit.
        reduction (Literal["none", "mean"]): How to reduce per-token loss
            weights after slicing.  ``"none"`` preserves raw weights
            (token-sum loss).  ``"mean"`` normalizes weights to sum to 1.0
            per example (token-mean loss), making gradient magnitudes
            consistent across variable-length sequences.
            Defaults to ``"none"``.

    Returns:
        tinker.Datum: A datum whose ``model_input`` holds the right-shifted
        input tokens and whose ``loss_fn_inputs`` contains ``"weights"`` and
        ``"target_tokens"`` ``TensorData`` entries.

    Example::

        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        datum = datum_from_model_input_weights(model_input, weights, max_length=2048)
        datum = datum_from_model_input_weights(
            model_input, weights, max_length=2048, reduction="mean",
        )
    """

    model_input_chunks = list(model_input.chunks)

    # Truncate to max_length by popping from end
    if max_length is not None:
        total_length = sum(chunk.length for chunk in model_input_chunks)

        while total_length > max_length and model_input_chunks:
            last = model_input_chunks[-1]
            if isinstance(last, tinker.types.EncodedTextChunk):
                overflow = total_length - max_length
                if overflow < last.length:
                    # Partial truncation of text chunk
                    model_input_chunks[-1] = tinker.types.EncodedTextChunk(
                        tokens=list(last.tokens[:-overflow])
                    )
                    total_length = max_length
                else:
                    # Remove entire text chunk
                    model_input_chunks.pop()
                    total_length -= last.length
            else:
                # Image chunk - must remove entirely
                model_input_chunks.pop()
                total_length -= last.length

    # Empty text chunks can appear when a renderer emits a header for an empty
    # assistant message. They have no targets/weights, but a trailing empty
    # chunk would prevent the right-shift below from dropping the last real token.
    model_input_chunks = [
        chunk
        for chunk in model_input_chunks
        if not (isinstance(chunk, tinker.types.EncodedTextChunk) and chunk.length == 0)
    ]

    # Remove trailing images (no text to predict after them)
    while model_input_chunks and isinstance(
        model_input_chunks[-1], (tinker.types.ImageChunk, tinker.types.ImageAssetPointerChunk)
    ):
        model_input_chunks.pop()

    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        model_input_chunks
    )
    weights = weights[1 : len(target_tokens) + 1]

    # Apply weight reduction
    if reduction == "mean":
        total = float(weights.sum())
        if total > 0:
            weights = weights / total
        if "mean" not in _logged_reduction_modes:
            logger.info("Weight reduction: 'mean' (token-mean loss)")
            _logged_reduction_modes.add("mean")
    elif reduction != "none":
        raise ValueError(f"Unknown reduction mode: {reduction!r}")

    return tinker.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )
