"""
Dataset builders for Nemotron-Cascade-2 SFT data.

Loads from nvidia/Nemotron-Cascade-2-SFT-Data on HuggingFace.
The dataset has subsets: math, science, chat, instruction_following, safety,
conversational_agent, swe, terminal_agent.

Conversations use standard OpenAI message format:
  [{"role": "system"|"user"|"assistant", "content": "..."}]
"""

import logging
from collections.abc import Callable
from typing import Literal, cast

import chz
import datasets
import tinker
import torch

from tinker_cookbook.renderers import Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import (
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)

DATASET_NAME = "nvidia/Nemotron-Cascade-2-SFT-Data"

# Subset sizes (approximate) for reference
SUBSET_SIZES = {
    "math": 5_226_364,
    "science": 2_717_163,
    "chat": 13_972_873,
    "instruction_following": 820_263,
    "safety": 3_570,
    "conversational_agent": 822_213,
    "swe": 439_610,
    "terminal_agent": 822_213,
}

SFTSubset = Literal[
    "math",
    "science",
    "chat",
    "instruction_following",
    "safety",
    "conversational_agent",
    "swe",
    "terminal_agent",
]


def pack_rendered_examples(
    rendered: list[tuple[tinker.ModelInput, torch.Tensor]],
    max_packed_length: int,
) -> list[tinker.Datum]:
    """Pack multiple pre-shift rendered examples into Datums using greedy bin-packing.

    Each rendered example is a (ModelInput, weights) pair as returned by
    ``renderer.build_supervised_example``. This function concatenates multiple
    examples' tokens and weights into packed sequences up to *max_packed_length*
    tokens, then applies the standard right-shift via ``datum_from_model_input_weights``.

    Args:
        rendered: List of (ModelInput, weights_tensor) pairs, one per example.
        max_packed_length: Maximum number of tokens per packed sequence.

    Returns:
        List of packed Datums. Each Datum may contain multiple concatenated examples.
    """
    packed_datums: list[tinker.Datum] = []
    current_tokens: list[int] = []
    current_weights: list[float] = []

    for model_input, weights in rendered:
        tokens = list(model_input.to_ints())
        w = weights.tolist()
        example_len = len(tokens)

        if example_len == 0:
            continue

        # If this single example exceeds the limit, pack it alone (truncated).
        if example_len > max_packed_length:
            if current_tokens:
                # Flush the current buffer first.
                _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)
                current_tokens = []
                current_weights = []
            _flush_packed(tokens, w, max_packed_length, packed_datums)
            continue

        # Would adding this example exceed the limit?
        if len(current_tokens) + example_len > max_packed_length:
            # Flush current buffer.
            _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)
            current_tokens = []
            current_weights = []

        current_tokens.extend(tokens)
        current_weights.extend(w)

    # Flush remaining.
    if current_tokens:
        _flush_packed(current_tokens, current_weights, max_packed_length, packed_datums)

    return packed_datums


def _flush_packed(
    tokens: list[int],
    weights: list[float],
    max_packed_length: int,
    out: list[tinker.Datum],
) -> None:
    """Create a Datum from concatenated tokens/weights and append to *out*."""
    combined_input = tinker.ModelInput.from_ints(tokens[:max_packed_length])
    combined_weights = torch.tensor(weights[:max_packed_length], dtype=torch.float32)
    out.append(datum_from_model_input_weights(combined_input, combined_weights, max_packed_length))


class PackedSupervisedDataset(SupervisedDataset):
    """A supervised dataset that packs multiple short examples into longer sequences.

    The paper (Nemotron-Cascade-2) packs many short SFT examples into 256K-token
    sequences.  With Tinker's current 49K per-sequence limit the default
    ``max_packed_length`` is 49152.

    Packing works at the *pre-shift* level:
      1. Render each conversation to ``(ModelInput, weights)`` via the renderer.
      2. Greedily bin-pack rendered examples until the next would exceed
         ``max_packed_length``.
      3. Apply the standard right-shift (``datum_from_model_input_weights``) to
         produce final Datums with ``model_input``, ``weights``, and
         ``target_tokens``.

    Because weights naturally encode which tokens are trainable (prompt=0,
    completion=1), concatenation preserves correct loss masking across example
    boundaries.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        batch_size: int,
        renderer: Renderer,
        train_on_what: TrainOnWhat,
        max_packed_length: int = 49152,
    ):
        self.hf_dataset = hf_dataset
        self.batch_size = batch_size
        self.renderer = renderer
        self.train_on_what = train_on_what
        self.max_packed_length = max_packed_length

        # Pre-render and pack all examples.
        self._packed_datums = self._render_and_pack(hf_dataset)
        logger.info(
            "Packed %d raw examples into %d sequences (max_packed_length=%d, "
            "avg %.1f examples/sequence)",
            len(hf_dataset),
            len(self._packed_datums),
            max_packed_length,
            len(hf_dataset) / max(len(self._packed_datums), 1),
        )

        # Shuffle indices (updated each epoch).
        self._indices = list(range(len(self._packed_datums)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_and_pack(self, hf_dataset: datasets.Dataset) -> list[tinker.Datum]:
        """Render every conversation and pack into sequences."""
        rendered: list[tuple[tinker.ModelInput, torch.Tensor]] = []
        for row in hf_dataset:
            messages = row["messages"]  # type: ignore[index]
            model_input, weights = self.renderer.build_supervised_example(
                messages, train_on_what=self.train_on_what
            )
            rendered.append((model_input, weights))

        return pack_rendered_examples(rendered, self.max_packed_length)

    # ------------------------------------------------------------------
    # SupervisedDataset interface
    # ------------------------------------------------------------------

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = start + self.batch_size
        return [self._packed_datums[self._indices[i]] for i in range(start, end)]

    def set_epoch(self, seed: int = 0) -> None:
        import random

        rng = random.Random(seed)
        self._indices = list(range(len(self._packed_datums)))
        rng.shuffle(self._indices)

    def __len__(self) -> int:
        return len(self._packed_datums) // self.batch_size


@chz.chz
class NemotronCascadeSFTBuilder(ChatDatasetBuilder):
    """Loads one or more subsets of Nemotron-Cascade-2-SFT-Data from HuggingFace."""

    subsets: tuple[SFTSubset, ...] = ("math",)
    max_examples: int | None = None
    test_size: int = 1024
    seed: int = 0
    streaming: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        if self.streaming:
            return self._build_streaming(map_fn)
        return self._build_eager(map_fn)

    def _build_eager(
        self, map_fn: Callable[[dict], tinker.Datum]
    ) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        all_datasets = []
        for subset in self.subsets:
            logger.info(f"Loading SFT subset: {subset}")
            ds = datasets.load_dataset(DATASET_NAME, name=subset, split="train")
            ds = cast(datasets.Dataset, ds)
            all_datasets.append(ds)

        if len(all_datasets) == 1:
            dataset = all_datasets[0]
        else:
            dataset = datasets.concatenate_datasets(all_datasets)

        dataset = dataset.shuffle(seed=self.seed)

        if self.max_examples is not None:
            dataset = dataset.select(range(min(self.max_examples, len(dataset))))

        # Split train/test
        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        logger.info(f"SFT dataset: {len(train_ds)} train examples")

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset

    def _build_streaming(
        self, map_fn: Callable[[dict], tinker.Datum]
    ) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Use streaming for very large datasets to avoid downloading everything upfront."""
        all_streams = []
        total_size = 0
        for subset in self.subsets:
            logger.info(f"Loading SFT subset (streaming): {subset}")
            ds = datasets.load_dataset(DATASET_NAME, name=subset, split="train", streaming=True)
            ds = cast(datasets.IterableDataset, ds)
            all_streams.append(ds)
            total_size += SUBSET_SIZES.get(subset, 100_000)

        if len(all_streams) == 1:
            stream = all_streams[0]
        else:
            stream = datasets.interleave_datasets(all_streams)

        if self.max_examples is not None:
            total_size = min(total_size, self.max_examples)
            stream = stream.take(self.max_examples)

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            stream,
            batch_size=self.common_config.batch_size,
            length=total_size,
            map_fn=map_fn,
        )
        return train_dataset, None


@chz.chz
class NemotronCascadeSFTFromFileBuilder(ChatDatasetBuilder):
    """Loads SFT data from a local JSONL file (pre-downloaded and preprocessed)."""

    file_path: str
    test_size: int = 1024
    seed: int = 0
    packing: bool = False
    max_packed_length: int = 49152

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        import json

        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        # Load JSONL
        rows = []
        with open(self.file_path) as f:
            for line in f:
                rows.append(json.loads(line))

        dataset = datasets.Dataset.from_list(rows)
        dataset = dataset.shuffle(seed=self.seed)

        if self.test_size > 0 and len(dataset) > self.test_size:
            test_ds = dataset.select(range(self.test_size))
            train_ds = dataset.select(range(self.test_size, len(dataset)))
        else:
            train_ds = dataset
            test_ds = None

        logger.info(f"SFT dataset from file: {len(train_ds)} train examples")

        if self.packing:
            logger.info(
                "Packing enabled: packing examples into sequences of up to %d tokens",
                self.max_packed_length,
            )
            train_dataset: SupervisedDataset = PackedSupervisedDataset(
                train_ds,
                batch_size=self.common_config.batch_size,
                renderer=self.renderer,
                train_on_what=train_on_what,
                max_packed_length=self.max_packed_length,
            )
        else:
            train_dataset = SupervisedDatasetFromHFDataset(
                train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )

        # Test dataset is never packed (we want per-example eval metrics).
        test_dataset = (
            SupervisedDatasetFromHFDataset(
                test_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset
