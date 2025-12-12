import logging
import random

import chz
import datasets
import tinker
from tinker_cookbook.preference.types import (
    Comparison,
    ComparisonRenderer,
    ComparisonRendererFromChatRenderer,
    LabeledComparison,
)
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Base Classes
# ============================================================================


@chz.chz
class ComparisonDatasetBuilder:
    """
    Builds HF datasets and converts to LabeledComparisons.
    This class is independent of rendering/tokenization.
    """

    swap: bool = False  # do data augmentation by swapping the order of the completions

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Get raw HuggingFace datasets for train and test."""
        raise NotImplementedError

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert a HuggingFace dataset example to a LabeledComparison."""
        raise NotImplementedError

    def get_labeled_comparisons(
        self,
    ) -> tuple[list[LabeledComparison], list[LabeledComparison] | None]:
        """Get all labeled comparisons for train and test sets."""
        train_dataset, test_dataset = self.get_train_and_test_datasets()

        # Process train dataset
        train_comparisons = []
        for i in range(len(train_dataset)):
            example = train_dataset[i]
            labeled_comparison = self.example_to_labeled_comparison(example)
            if labeled_comparison is not None:
                train_comparisons.append(labeled_comparison)

        # Process test dataset if it exists
        test_comparisons = None
        if test_dataset is not None:
            test_comparisons = []
            for i in range(len(test_dataset)):
                example = test_dataset[i]
                labeled_comparison = self.example_to_labeled_comparison(example)
                if labeled_comparison is not None:
                    test_comparisons.append(labeled_comparison)

        return train_comparisons, test_comparisons


@chz.chz
class ChatDatasetBuilderFromComparisons(ChatDatasetBuilder):
    """
    Abstract base for chat dataset builders that use comparisons.
    Subclasses must implement get_comparison_builder() to provide the dataset-specific logic.
    """

    comparison_builder: ComparisonDatasetBuilder
    swap: bool = False  # do data augmentation by swapping the order of the completions

    @property
    def comparison_renderer(self) -> ComparisonRenderer:
        return ComparisonRendererFromChatRenderer(self.renderer)

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_dataset, test_dataset = self.comparison_builder.get_train_and_test_datasets()
        comparison_renderer = self.comparison_renderer
        rng = random.Random(0)

        def comparison_to_datum(labeled_comparison: LabeledComparison) -> tinker.Datum:
            model_input, weights = comparison_renderer.to_model_input_weights(labeled_comparison)
            return datum_from_model_input_weights(
                model_input, weights, self.common_config.max_length
            )

        def example_to_data(example: dict[str, str]) -> list[tinker.Datum]:
            labeled_comparison = self.comparison_builder.example_to_labeled_comparison(example)
            if labeled_comparison is None:
                return []
            if self.swap:
                return [
                    comparison_to_datum(labeled_comparison),
                    comparison_to_datum(labeled_comparison.swap()),
                ]
            else:
                if rng.random() < 0.5:
                    labeled_comparison = labeled_comparison.swap()
                return [comparison_to_datum(labeled_comparison)]

        if test_dataset is not None:
            test_supervised_dataset = SupervisedDatasetFromHFDataset(
                test_dataset,
                batch_size=len(test_dataset),
                flatmap_fn=example_to_data,
            )
        else:
            test_supervised_dataset = None

        return SupervisedDatasetFromHFDataset(
            train_dataset,
            batch_size=self.common_config.batch_size,
            flatmap_fn=example_to_data,
        ), test_supervised_dataset


@chz.chz
class ComparisonBuilderFromJsonl(ComparisonDatasetBuilder):
    """Load LabeledComparisons from JSONL files produced by combine_preference_datasets.py."""

    train_path: str
    test_path: str | None = None

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load datasets from JSONL files."""
        import json

        import blobfile

        # Load train dataset
        train_data = []
        with blobfile.BlobFile(self.train_path, "r", streaming=False) as f:
            for line in f:
                train_data.append(json.loads(line.strip()))

        train_dataset = datasets.Dataset.from_list(train_data)

        # Load test dataset if provided
        test_dataset = None
        if self.test_path:
            test_data = []
            with blobfile.BlobFile(self.test_path, "r", streaming=False) as f:
                for line in f:
                    test_data.append(json.loads(line.strip()))
            test_dataset = datasets.Dataset.from_list(test_data)

        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert a dictionary (from JSONL) back to a LabeledComparison."""
        # The JSONL contains the raw LabeledComparison as a dict
        # with 'comparison' and 'label' keys
        if "comparison" not in example or "label" not in example:
            return None

        comparison_dict = example["comparison"]

        # Reconstruct the Comparison object
        comparison = Comparison(
            prompt_conversation=comparison_dict["prompt_conversation"],
            completion_A=comparison_dict["completion_A"],
            completion_B=comparison_dict["completion_B"],
        )

        return LabeledComparison(comparison=comparison, label=example["label"])
