"""
Canonical bin-packing algorithm for supervised fine-tuning.

This module is the single source of truth for packing multiple tokenized
examples into fixed-length sequences. The core algorithm (``greedy_pack``)
operates on plain Python lists and has no dependencies on torch or tinker,
making it picklable and easy to test. Convenience wrappers build tinker
``Datum`` objects from the packed output.
"""

from __future__ import annotations

from collections.abc import Iterable

import tinker
import torch

from tinker_cookbook.supervised.common import datum_from_model_input_weights


def greedy_pack(
    items: Iterable[tuple[list[int], list[float]]],
    max_length: int,
) -> list[tuple[list[int], list[float]]]:
    """Greedy first-fit bin-packing on token sequences.

    Iterates over ``(token_ids, weights)`` pairs and concatenates them into
    bins of at most ``max_length`` tokens. When an item would overflow the
    current bin, the bin is flushed and a new one is started. Items longer
    than ``max_length`` are truncated and emitted alone.

    Args:
        items: An iterable of ``(token_ids, weights)`` pairs. Each pair
            must have equal length.
        max_length: Maximum number of tokens per packed sequence.

    Returns:
        A list of ``(tokens, weights)`` pairs, each with length
        ``<= max_length``.
    """
    packed: list[tuple[list[int], list[float]]] = []
    current_tokens: list[int] = []
    current_weights: list[float] = []

    for tokens, weights in items:
        example_len = len(tokens)
        if example_len == 0:
            continue

        # Oversized example: flush buffer, emit alone (truncated).
        if example_len > max_length:
            if current_tokens:
                packed.append((current_tokens, current_weights))
                current_tokens = []
                current_weights = []
            packed.append((tokens[:max_length], weights[:max_length]))
            continue

        # Would adding this example overflow the current buffer?
        if len(current_tokens) + example_len > max_length:
            packed.append((current_tokens, current_weights))
            current_tokens = []
            current_weights = []

        current_tokens.extend(tokens)
        current_weights.extend(weights)

    if current_tokens:
        packed.append((current_tokens, current_weights))

    return packed


def make_datum(tokens: list[int], weights: list[float], max_length: int) -> tinker.Datum:
    """Convert a packed ``(tokens, weights)`` pair into a training Datum.

    Wraps the tokens in a ``ModelInput``, the weights in a ``torch.Tensor``,
    and delegates to ``datum_from_model_input_weights`` for the standard
    next-token input/target split.

    Args:
        tokens: Token IDs for the packed sequence.
        weights: Per-token loss weights, same length as *tokens*.
        max_length: Maximum sequence length (used for padding/truncation
            inside ``datum_from_model_input_weights``).

    Returns:
        A ``tinker.Datum`` ready for training.
    """
    model_input = tinker.ModelInput.from_ints(tokens[:max_length])
    weight_tensor = torch.tensor(weights[:max_length], dtype=torch.float32)
    return datum_from_model_input_weights(model_input, weight_tensor, max_length)


def pack_to_datums(
    items: Iterable[tuple[list[int], list[float]]],
    max_length: int,
) -> list[tinker.Datum]:
    """Pack and convert to Datums in one step.

    Convenience wrapper that calls ``greedy_pack`` followed by ``make_datum``
    for each resulting bin.

    Args:
        items: An iterable of ``(token_ids, weights)`` pairs.
        max_length: Maximum number of tokens per packed sequence.

    Returns:
        A list of packed ``tinker.Datum`` objects.
    """
    return [
        make_datum(tokens, weights, max_length)
        for tokens, weights in greedy_pack(items, max_length)
    ]
