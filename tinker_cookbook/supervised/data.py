"""
Supervised learning dataset implementations from HuggingFace datasets.
"""

import json
import logging
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import blobfile
import chz
import datasets
import numpy as np
import tinker

from tinker_cookbook.exceptions import DataFormatError, DataValidationError
from tinker_cookbook.renderers import Message, Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    reduction: Literal["none", "mean"] = "mean",
) -> tinker.Datum:
    """Convert a chat conversation into a training Datum.

    This is the primary entry point for turning a list of chat messages into a
    ``tinker.Datum`` suitable for supervised training.  It delegates to the
    renderer for tokenisation and weight assignment, then wraps the result with
    ``datum_from_model_input_weights``.

    Args:
        conversation (list[Message]): Chat messages (each a dict with
            ``"role"`` and ``"content"`` keys).
        renderer (Renderer): Renderer that tokenises the conversation.
        max_length (int | None): Optional maximum sequence length.  The
            resulting datum is truncated to this many tokens.
        train_on_what (TrainOnWhat): Which tokens receive non-zero loss
            weight.  Default ``TrainOnWhat.ALL_ASSISTANT_MESSAGES``.
        reduction (Literal["none", "mean"]): How to reduce per-token loss
            weights.  ``"mean"`` (default for SFT) normalizes weights to
            sum to 1.0 per example, producing consistent gradient
            magnitudes across variable-length sequences.  ``"none"``
            preserves raw weights (token-sum loss).

    Returns:
        tinker.Datum: A training datum with model input, target tokens,
        and per-token loss weights.

    Example::

        from tinker_cookbook.renderers import get_renderer, TrainOnWhat
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer("Qwen/Qwen3-8B")
        renderer = get_renderer("qwen3", tokenizer)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        datum = conversation_to_datum(messages, renderer, max_length=2048)
    """
    model_input, weights = renderer.build_supervised_example(
        conversation, train_on_what=train_on_what
    )
    return datum_from_model_input_weights(model_input, weights, max_length, reduction=reduction)


def _one_of(a: Any, b: Any) -> bool:
    return (a is not None and b is None) or (a is None and b is not None)


class SupervisedDatasetFromHFDataset(SupervisedDataset):
    """A supervised dataset backed by a HuggingFace dataset.

    Args:
        hf_dataset (datasets.Dataset): The HuggingFace dataset to draw rows from.
        batch_size (int): Number of rows per batch.
        map_fn (Callable[[dict], tinker.Datum] | None): Function mapping a single
            row to a Datum. Mutually exclusive with ``flatmap_fn``.
        flatmap_fn (Callable[[dict], list[tinker.Datum]] | None): Function mapping
            a single row to multiple Datums. Mutually exclusive with ``map_fn``.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        batch_size: int,
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset
        self.shuffle_dataset = (
            hf_dataset  # Keep a reference to the original dataset to avoid statefulness
        )
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return a batch of Datum objects at the given index.

        Args:
            index (int): Zero-based batch index.

        Returns:
            list[tinker.Datum]: Training datums for this batch.
        """
        rows = self.shuffle_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows.to_list()]
        else:
            assert self.flatmap_fn is not None
            return [datum for row in rows.to_list() for datum in self.flatmap_fn(row)]

    def set_epoch(self, seed: int = 0):
        """Shuffle the dataset for a new epoch.

        Args:
            seed (int): Random seed for shuffling. Default ``0``.
        """
        self.shuffle_dataset = self.hf_dataset.shuffle(seed=seed)

    def __len__(self) -> int:
        """Return the number of complete batches in the dataset."""
        return len(self.hf_dataset) // self.batch_size


class StreamingSupervisedDatasetFromHFDataset(SupervisedDataset):
    """A supervised dataset that streams from HuggingFace, reducing memory usage.

    Only supports forward iteration; seeking backward raises an error.

    Args:
        hf_dataset (datasets.IterableDataset): The streaming HuggingFace dataset.
        batch_size (int): Number of rows per batch.
        length (int): Total number of rows in the dataset (streaming datasets
            do not expose a length).
        map_fn (Callable[[dict], tinker.Datum] | None): Function mapping a single
            row to a Datum. Mutually exclusive with ``flatmap_fn``.
        flatmap_fn (Callable[[dict], list[tinker.Datum]] | None): Function mapping
            a single row to multiple Datums. Mutually exclusive with ``map_fn``.
        buffer_size (int): Shuffle buffer size. Default: 10000.
    """

    def __init__(
        self,
        hf_dataset: datasets.IterableDataset,
        batch_size: int,
        length: int,
        map_fn: Callable[[dict], tinker.Datum] | None = None,
        flatmap_fn: Callable[[dict], list[tinker.Datum]] | None = None,
        buffer_size: int = 10_000,
    ):
        assert _one_of(map_fn, flatmap_fn), "Only one of map_fn or flatmap_fn can be provided"
        self.hf_dataset = hf_dataset.shuffle(seed=0, buffer_size=buffer_size).batch(
            batch_size=batch_size, drop_last_batch=True
        )
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1
        self.batch_size = batch_size
        self.map_fn = map_fn
        self.flatmap_fn = flatmap_fn
        # We pass the length to the dataset, since streaming HF datasets don't have a length attribute
        self.length = length

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Return a batch of Datum objects at the given index.

        Only forward iteration is supported. Requesting a batch at or before
        the most recently returned index raises ``DataValidationError``.

        Args:
            index (int): Zero-based batch index (must be strictly greater than
                the previous call's index).

        Returns:
            list[tinker.Datum]: Training datums for this batch.

        Raises:
            DataValidationError: If ``index`` would require backward seeking.
        """
        # Error on backward seeks
        if index < self.index + 1:
            raise DataValidationError(
                f"StreamingSupervisedDatasetFromHFDataset only supports forward iteration. "
                f"Cannot seek backward from batch {self.index} to {index}."
            )

        # Skip forward if needed by consuming intermediate batches
        batches_to_skip = index - self.index - 1
        for _ in range(batches_to_skip):
            next(self.dataset_iterator)
            self.index += 1

        # Get the actual batch
        self.index = index
        batch = next(self.dataset_iterator)
        rows = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        if self.map_fn is not None:
            return [self.map_fn(row) for row in rows]
        else:
            assert self.flatmap_fn is not None
            return [datum for row in rows for datum in self.flatmap_fn(row)]

    def set_epoch(self, seed: int = 0):
        """Reset the stream for a new epoch.

        Args:
            seed (int): Epoch seed forwarded to the underlying iterable
                dataset. Default ``0``.
        """
        self.hf_dataset.set_epoch(seed)
        self.dataset_iterator = iter(self.hf_dataset)
        self.index = -1

    def __len__(self) -> int:
        """Return the number of complete batches in the dataset."""
        return self.length // self.batch_size


@chz.chz
class FromConversationFileBuilder(ChatDatasetBuilder):
    """Build a supervised dataset from a JSONL file of chat conversations.

    Each line of the file must be a JSON object with a ``"messages"`` key whose
    value is a list of chat messages (dicts with ``"role"`` and ``"content"``).

    Attributes:
        file_path (str): Path (local or ``blobfile``-compatible) to the JSONL
            file.
        test_size (int): Number of examples to hold out for evaluation.
            Default ``0`` (no held-out set).
        shuffle_seed (int): Seed used to shuffle before splitting. Default ``0``.

    Example::

        builder = FromConversationFileBuilder(
            file_path="data/conversations.jsonl",
            test_size=50,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="Qwen/Qwen3-8B",
                renderer_name="qwen3",
                max_length=2048,
                batch_size=8,
            ),
        )
        train_ds, test_ds = builder()
    """

    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Load the JSONL file and return (train_dataset, test_dataset).

        Returns:
            tuple[SupervisedDataset, SupervisedDataset | None]: Training
                dataset and an optional held-out evaluation dataset.

        Raises:
            DataFormatError: If any line in the file lacks a ``"messages"`` key.
        """
        # Load conversations from JSONL file
        conversations = []
        with blobfile.BlobFile(self.file_path, "r", streaming=False) as f:
            for line in f:
                data = json.loads(line.strip())
                if "messages" not in data:
                    raise DataFormatError(
                        f"Each line in the JSONL file must contain a 'messages' field. Got: {data.keys()}"
                    )
                conversations.append(data)

        # Create HuggingFace dataset from the loaded data
        dataset = datasets.Dataset.from_list(conversations)

        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Split into train and test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.take(self.test_size)
            train_ds = dataset.skip(self.test_size)
        else:
            # If test_size is 0 or dataset is too small, use all data for training
            train_ds = dataset
            test_ds = None

        # Use train_on_what from common_config if provided, otherwise use default
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Define mapping function
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        # Create supervised dataset
        supervised_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        # Create evaluator if we have test data
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), map_fn=map_fn
            )
        else:
            test_dataset = None

        return supervised_dataset, test_dataset


@chz.chz
class HFDatasetSource:
    """A single HuggingFace dataset to include in an interleaved mix.

    Attributes:
        path: HuggingFace dataset path (e.g. ``"allenai/tulu-3-sft-mixture"``).
        name: HuggingFace dataset config name (passed as ``name`` to ``load_dataset``).
        split: Dataset split to load.
        weight: Relative mixing weight. When set, weights are normalized across sources
            to determine sampling probabilities. When left as ``None`` (default) for all
            sources, each original row has equal sampling probability. The stopping
            strategy determines whether source rows can repeat.
        message_field: Column name containing conversation messages.
    """

    path: str
    name: str | None = None
    split: str = "train"
    weight: float | None = None
    message_field: str = "messages"


def _split_interleaved_sources(
    source_datasets: list[datasets.Dataset],
    weights: list[float],
    test_size: int,
    seed: int,
    uniform_rows: bool,
) -> tuple[list[datasets.Dataset], list[float], datasets.Dataset]:
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Source weights must be finite and non-negative for a holdout split")
    if not math.isfinite(sum(weights)) or sum(weights) <= 0:
        raise ValueError("Total weight across all sources must be finite and positive")
    for i, (ds, weight) in enumerate(zip(source_datasets, weights, strict=True)):
        if weight > 0 and len(ds) == 0:
            raise ValueError(f"Cannot split empty positively weighted source at index {i}")

    capacities = np.array(
        [
            len(ds) - 1 if weight > 0 else 0
            for ds, weight in zip(source_datasets, weights, strict=True)
        ],
        dtype=np.int64,
    )
    max_test_size = int(capacities.sum())
    if test_size > max_test_size:
        raise ValueError(
            f"test_size ({test_size}) exceeds the maximum unique holdout size ({max_test_size}) "
            "while retaining one training row per positively weighted source"
        )

    rng = np.random.default_rng(seed)
    remaining_rows = np.array([len(ds) for ds in source_datasets], dtype=np.int64)
    counts = [0] * len(source_datasets)
    for _ in range(test_size):
        probabilities = np.where(capacities > 0, remaining_rows if uniform_rows else weights, 0.0)
        probabilities /= probabilities.sum()
        source_index = int(rng.choice(len(source_datasets), p=probabilities))
        counts[source_index] += 1
        capacities[source_index] -= 1
        remaining_rows[source_index] -= 1

    train_sources: list[datasets.Dataset] = []
    train_weights: list[float] = []
    heldout_sources: list[datasets.Dataset] = []
    for ds, weight, count in zip(source_datasets, weights, counts, strict=True):
        if weight == 0:
            continue
        shuffled = ds.shuffle(seed=int(rng.integers(0, 2**32)))
        train_ds = shuffled.skip(count)
        train_sources.append(train_ds)
        train_weights.append(float(len(train_ds)) if uniform_rows else weight)
        if count > 0:
            heldout_sources.append(shuffled.take(count))

    heldout = datasets.concatenate_datasets(heldout_sources).shuffle(seed=seed)
    return train_sources, train_weights, heldout


@chz.chz
class InterleavedChatDatasetBuilder(ChatDatasetBuilder):
    """Builds an SFT dataset by interleaving multiple HuggingFace datasets.

    Uses ``datasets.interleave_datasets`` to mix rows from multiple sources according
    to the configured weights. When no weights are specified, sources are weighted by
    size so that original rows have equal sampling probabilities. Interleaving can
    repeat source rows, depending on the stopping strategy. The resulting dataset
    is a standard Arrow-backed ``datasets.Dataset`` with O(1)
    random access and deterministic per-epoch shuffling.

    A positive ``test_size`` holds out unique original rows before interleaving,
    keeping at least one training row per positively weighted source. Explicit
    weights guide source selection until a source's holdout capacity is exhausted;
    without explicit weights, eligible rows are sampled uniformly. Requests that
    cannot preserve the training sources raise ``ValueError``. Zero-weight sources
    do not contribute to either partition in holdout mode. Identical content in
    distinct original rows is not deduplicated.
    """

    sources: list[HFDatasetSource]
    test_size: int = 0
    shuffle_seed: int = 0
    stopping_strategy: Literal[
        "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
    ] = "all_exhausted"

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        if not self.sources:
            raise ValueError("At least one dataset source must be provided")

        hf_datasets: list[datasets.Dataset] = []

        for source in self.sources:
            ds = datasets.load_dataset(source.path, name=source.name, split=source.split)
            if not isinstance(ds, datasets.Dataset):
                raise TypeError(
                    f"Expected a Dataset but got {type(ds).__name__}. "
                    f"Check that split='{source.split}' is valid for '{source.path}'."
                )
            if source.message_field != "messages":
                ds = ds.rename_column(source.message_field, "messages")
            ds = ds.select_columns(["messages"])
            hf_datasets.append(ds)
            logger.info(f"Loaded '{source.path}' ({len(ds)} rows, weight={source.weight})")

        # If all weights are None, weight by dataset size (uniform row sampling).
        # If any weight is set, all must be set.
        if all(s.weight is None for s in self.sources):
            weights: list[float] = [float(len(ds)) for ds in hf_datasets]
        elif any(s.weight is None for s in self.sources):
            raise ValueError(
                "Either all sources must have explicit weights or none of them. "
                "Got a mix of weighted and unweighted sources."
            )
        else:
            weights = [cast(float, s.weight) for s in self.sources]

        test_ds = None
        if self.test_size > 0:
            hf_datasets, weights, test_ds = _split_interleaved_sources(
                hf_datasets,
                weights,
                self.test_size,
                self.shuffle_seed,
                uniform_rows=all(source.weight is None for source in self.sources),
            )

        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Total weight across all sources must be positive")
        probabilities = [w / total_weight for w in weights]

        interleaved = datasets.interleave_datasets(
            hf_datasets,
            probabilities=probabilities,
            seed=self.shuffle_seed,
            stopping_strategy=self.stopping_strategy,
        )
        if not isinstance(interleaved, datasets.Dataset):
            raise TypeError(
                f"Expected Dataset from interleave_datasets, got {type(interleaved).__name__}"
            )
        logger.info(f"Interleaved dataset: {len(interleaved)} rows")

        train_ds = interleaved

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        test_dataset = (
            SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), map_fn=map_fn)
            if test_ds is not None
            else None
        )

        return train_dataset, test_dataset
