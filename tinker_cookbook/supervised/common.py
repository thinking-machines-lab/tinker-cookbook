import logging

import tinker
import torch

logger = logging.getLogger(__name__)


def compute_mean_nll(
    logprobs_list: list[tinker.TensorData], weights_list: list[tinker.TensorData]
) -> float:
    """Compute weighted mean negative log likelihood."""
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
    """
    Given a full sequence of model input chunks, create
     "inputs" (with last token removed); these are also list[ModelInputChunk] because text+images
     "targets" (with first token removed); these are list[int] text tokens
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise ValueError(
            "The last chunk must be a text chunk. This is because images are 0-loss anyways, so we should remove them beforehand."
        )

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens for input/target split")

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
    """
    Create a Datum from a ModelInput and weights tensor.

    Performs max_length truncation and next-token slicing to create input and target.
    Text chunks can be truncated, but image chunks must be wholly discarded to stay
    within max_length.

    Args:
        model_input: The model input containing a sequence of text and/or image chunks
        weights: The weights tensor aligned with the model_input length
        max_length: Optional maximum sequence length. If provided, truncates to this length.
                   Image chunks are discarded entirely if they would exceed max_length.

    Returns:
        A Datum with model_input (input tokens) and loss_fn_inputs (target tokens and weights)
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
