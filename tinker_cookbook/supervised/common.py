import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
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


def _counted_byte_count(token_ids: list[int], counted: list[bool], tokenizer: Tokenizer) -> int:
    """UTF-8 byte count of the ``counted`` tokens, decoded per contiguous run.

    ``counted[i]`` marks whether ``token_ids[i]`` contributes to BPB (i.e. it is
    trained and not a special token). Contiguous runs of counted tokens are
    decoded separately so that gaps (untrained or special tokens) don't merge
    unrelated spans across the hole.

    Args:
        token_ids (list[int]): Target token IDs.
        counted (list[bool]): Per-token inclusion mask, aligned with
            ``token_ids``.
        tokenizer (Tokenizer): Tokenizer used to decode tokens back to text.

    Returns:
        int: Total number of UTF-8 bytes across all counted spans.
    """
    total_bytes = 0
    run: list[int] = []
    for token_id, is_counted in zip(token_ids, counted, strict=True):
        if is_counted:
            run.append(token_id)
        elif run:
            total_bytes += _decoded_byte_length(run, tokenizer)
            run = []
    if run:
        total_bytes += _decoded_byte_length(run, tokenizer)
    return total_bytes


def _decoded_byte_length(token_ids: list[int], tokenizer: Tokenizer) -> int:
    """UTF-8 byte length of ``token_ids`` decoded to text.

    Decodes with ``skip_special_tokens=False``; the caller is responsible for
    excluding special tokens beforehand so the numerator and denominator cover
    the same span.
    """
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if isinstance(text, list):
        # Some tokenizers can return a list of piece strings; join them.
        text = "".join(text)
    return len(text.encode("utf-8"))


def compute_bpb(
    logprobs_list: list[tinker.TensorData],
    weights_list: list[tinker.TensorData],
    target_tokens_list: list[tinker.TensorData],
    tokenizer: Tokenizer,
    content_bytes_list: Sequence[int | None] | None = None,
) -> float:
    """Compute bits-per-byte (BPB) across a batch: a tokenizer-independent NLL.

    Per-token mean NLL (:func:`compute_mean_nll`) is not comparable across
    models that use different tokenizers: a coarser tokenizer packs more text
    into each token (higher nats/token) while a finer one spreads it over more
    tokens (lower nats/token), even at equal modeling quality. Bits-per-byte
    removes this dependence by dividing the *total* log-loss (converted to
    bits) by a byte count of the target text.

    A datum is scored in one of two modes:

    **Content-byte mode** (used when its ``content_bytes_list`` entry is not
    ``None``, as reported by the renderer via
    ``Renderer.build_supervised_example_with_metadata``)::

        bpb = -sum(logprob_i for i where weight_i > 0) / (ln(2) * content_bytes)

    The numerator is the model's *entire* trained code length, including
    chat-template scaffolding (think tags, tool-call framing, end-of-turn
    markers). The denominator counts only the UTF-8 bytes of the semantic
    message content, which is identical across renderers and tokenizers (see
    ``message_content_byte_count``). Scaffolding therefore never pads the
    denominator: a template whose markup costs real bits pays for them in the
    numerator (a cost that vanishes as the template is learned), and a verbose
    template gains no artificial BPB advantage from its extra markup bytes.
    This is the preferred mode for cross-model comparison.

    **Token-byte fallback** (entry is ``None``: renderer without content-byte
    support, or the trained region was truncated by ``max_length``)::

        bpb = -sum(logprob_i for i in counted) / (ln(2) * bytes(counted tokens))

    where a token is *counted* iff it is trained (``weight > 0``) and not a
    special token (per ``tokenizer.all_special_ids``). The same mask drives
    both sides, so they cover the identical span. Caveat: non-special
    scaffolding that renderers inject as plain text (e.g. ``<think></think>``)
    is counted as target bytes here, which slightly deflates BPB by an amount
    that varies with the chat template.

    In both modes weights are used only as a ``> 0`` mask, not as multipliers,
    so the total nats stay correct even when weights are normalized per example
    (``reduction="mean"``, the SFT default, where a datum's weights sum to 1.0
    instead of being 0/1).

    Args:
        logprobs_list (list[tinker.TensorData]): Per-token log-probabilities,
            one entry per datum in the batch.
        weights_list (list[tinker.TensorData]): Per-token loss weights aligned
            with ``logprobs_list``.
        target_tokens_list (list[tinker.TensorData]): Per-token target IDs
            (``loss_fn_inputs["target_tokens"]``) aligned with ``weights_list``,
            used to recover the target text for the byte count in fallback mode.
        tokenizer (Tokenizer): Tokenizer used to decode target tokens to bytes
            in fallback mode.
        content_bytes_list (Sequence[int | None] | None): Per-datum semantic
            content byte counts (``DatumWithContentBytes.trained_content_bytes``),
            aligned with ``logprobs_list``. ``None`` (or a ``None`` entry)
            selects the token-byte fallback for the batch (or that datum).

    Returns:
        float: Bits per byte, or ``nan`` if the counted target has zero bytes.
    """
    if content_bytes_list is not None and len(content_bytes_list) != len(logprobs_list):
        raise ValueError(
            f"content_bytes_list has length {len(content_bytes_list)}, "
            f"expected {len(logprobs_list)}"
        )
    special_ids = frozenset(getattr(tokenizer, "all_special_ids", None) or [])
    total_nll_nats = 0.0
    total_bytes = 0

    for i, (logprobs, weights, target_tokens) in enumerate(
        zip(logprobs_list, weights_list, target_tokens_list, strict=True)
    ):
        content_bytes = content_bytes_list[i] if content_bytes_list is not None else None
        if content_bytes is not None:
            # Content-byte mode: full trained NLL over renderer-reported
            # content bytes.
            trained_mask = weights.to_torch() > 0
            total_nll_nats += float(-logprobs.to_torch()[trained_mask].sum())
            total_bytes += content_bytes
            continue
        token_ids = [int(t) for t in target_tokens.data]
        # Token-byte fallback: a token contributes iff it is trained
        # (weight > 0) and not special. This single mask is used for both the
        # numerator and the byte denominator, guaranteeing they cover the
        # identical span.
        counted = [
            weight > 0 and token_id not in special_ids
            for weight, token_id in zip(weights.data, token_ids, strict=True)
        ]
        mask = torch.tensor(counted, dtype=torch.bool)
        total_nll_nats += float(-logprobs.to_torch()[mask].sum())
        total_bytes += _counted_byte_count(token_ids, counted, tokenizer)

    if total_bytes == 0:
        logger.warning("No target bytes found for BPB computation")
        return float("nan")

    return float(total_nll_nats / (math.log(2) * total_bytes))


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


@dataclass(frozen=True)
class DatumWithContentBytes(tinker.Datum):
    """A ``tinker.Datum`` carrying client-side metadata for metric computation.

    ``trained_content_bytes`` is a plain Python attribute, deliberately *not*
    a ``loss_fn_inputs`` entry: the Tinker SDK serializes datums from
    ``model_input`` and ``loss_fn_inputs`` only, so this field never reaches
    the service (which restricts ``loss_fn_inputs`` keys), and the datum can
    be passed to ``forward_backward`` / ``forward_backward_custom`` unchanged.
    Read it with ``getattr(datum, "trained_content_bytes", None)`` so plain
    datums keep working.

    Attributes:
        trained_content_bytes: UTF-8 bytes of the semantic content of the
            loss-weighted messages (see
            ``tinker_cookbook.renderers.message_content_byte_count``). Used as
            the bits-per-byte denominator by :func:`compute_bpb`.
    """

    trained_content_bytes: int | None = None


def datum_from_model_input_weights(
    model_input: tinker.ModelInput,
    weights: torch.Tensor,
    max_length: int | None = None,
    reduction: Literal["none", "mean"] = "none",
    trained_content_bytes: int | None = None,
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
        trained_content_bytes (int | None): Semantic content byte count of the
            loss-weighted messages, as reported by
            ``Renderer.build_supervised_example_with_metadata``. When provided
            and no loss-weighted token is lost to truncation, the returned
            datum is a :class:`DatumWithContentBytes` carrying the count for
            the bits-per-byte metric. If truncation removes any trained token,
            the count no longer matches the surviving tokens and is dropped
            (the metric then falls back to token-based byte counting).

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
    # Trained-token count over the full (untruncated) sequence, post-shift.
    n_trained_full = int((weights[1:] > 0).sum())
    weights = weights[1 : len(target_tokens) + 1]

    if trained_content_bytes is not None and int((weights > 0).sum()) != n_trained_full:
        # Truncation removed loss-weighted tokens, so the content byte count no
        # longer describes the surviving trained span. Drop it; BPB falls back
        # to token-based byte counting for this datum.
        trained_content_bytes = None

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

    loss_fn_inputs = {
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
    }
    if trained_content_bytes is not None:
        return DatumWithContentBytes(
            model_input=input_model_input,
            loss_fn_inputs=loss_fn_inputs,
            trained_content_bytes=trained_content_bytes,
        )
    return tinker.Datum(model_input=input_model_input, loss_fn_inputs=loss_fn_inputs)
