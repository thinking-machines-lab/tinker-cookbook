import logging

import chz
import tinker

from tinker_cookbook.preference.preference_datasets import (
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.types import (
    LabeledComparison,
)
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


@chz.chz
class DPODatasetBuilderFromComparisons(ChatDatasetBuilder):
    """DPO dataset builder that converts labeled comparisons into paired datums.

    Each ``LabeledComparison`` is expanded into two ``Datum`` objects
    (chosen first, rejected second) interleaved in the batch so that the
    DPO loss function can pair them by index.

    Attributes:
        comparison_builder (ComparisonDatasetBuilder): Builder that provides
            raw HuggingFace datasets and the
            ``example_to_labeled_comparison`` conversion logic.

    Example::

        builder = DPODatasetBuilderFromComparisons(
            comparison_builder=my_comparison_builder,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="meta-llama/Llama-3.1-8B-Instruct",
                renderer_name="llama3",
                max_length=2048,
                batch_size=8,
            ),
        )
        train_ds, test_ds = builder()
    """

    comparison_builder: ComparisonDatasetBuilder

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and optional test supervised datasets for DPO.

        Each labeled comparison is split into a chosen and rejected datum
        pair.  Batches contain interleaved chosen/rejected datums at even/odd
        indices respectively.

        Returns:
            tuple[SupervisedDataset, SupervisedDataset | None]: The training
                dataset and an optional test dataset.
        """
        train_dataset, test_dataset = self.comparison_builder.get_train_and_test_datasets()
        renderer = self.renderer

        def comparison_to_datum(labeled_comparison: LabeledComparison) -> list[tinker.Datum]:
            chosen_completion = (
                labeled_comparison.comparison.completion_A
                if labeled_comparison.label == "A"
                else labeled_comparison.comparison.completion_B
            )
            rejected_completion = (
                labeled_comparison.comparison.completion_B
                if labeled_comparison.label == "A"
                else labeled_comparison.comparison.completion_A
            )

            chosen_convo = [
                *labeled_comparison.comparison.prompt_conversation,
                *chosen_completion,
            ]
            rejected_convo = [
                *labeled_comparison.comparison.prompt_conversation,
                *rejected_completion,
            ]

            chosen_tokens, chosen_weights = renderer.build_supervised_example(chosen_convo)
            rejected_tokens, rejected_weights = renderer.build_supervised_example(rejected_convo)

            # DPO uses sequence log-probability (token-sum), so normalization
            # must be "none" to avoid implicit length bias.
            return [
                datum_from_model_input_weights(
                    chosen_tokens, chosen_weights, self.common_config.max_length,
                    normalization="none",
                ),
                datum_from_model_input_weights(
                    rejected_tokens, rejected_weights, self.common_config.max_length,
                    normalization="none",
                ),
            ]

        def example_to_data(example: dict[str, str]) -> list[tinker.Datum]:
            labeled_comparison = self.comparison_builder.example_to_labeled_comparison(example)
            if labeled_comparison is None:
                logger.warning(
                    "Skipping invalid row: example_to_labeled_comparison returned None. "
                    "This may cause an odd number of datums in a batch."
                )
                return []
            return comparison_to_datum(labeled_comparison)

        if test_dataset is not None:
            test_supervised_dataset = SupervisedDatasetFromHFDataset(
                test_dataset,
                batch_size=len(test_dataset),
                flatmap_fn=example_to_data,
            )
        else:
            test_supervised_dataset = None

        return SupervisedDatasetFromHFDataset(
            train_dataset, batch_size=self.common_config.batch_size, flatmap_fn=example_to_data
        ), test_supervised_dataset
