import logging

import tinker
import torch

from tinker_cookbook.exceptions import DataValidationError

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

    Returns:
        tinker.Datum: A datum whose ``model_input`` holds the right-shifted
        input tokens and whose ``loss_fn_inputs`` contains ``"weights"`` and
        ``"target_tokens"`` ``TensorData`` entries.

    Example::

        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        datum = datum_from_model_input_weights(model_input, weights, max_length=2048)
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

    # Remove trailing images (no text to predict after them)
    while model_input_chunks and isinstance(
        model_input_chunks[-1], (tinker.types.ImageChunk, tinker.types.ImageAssetPointerChunk)
    ):
        model_input_chunks.pop()

    input_model_input, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        model_input_chunks
    )
    weights = weights[1 : len(target_tokens) + 1]

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
