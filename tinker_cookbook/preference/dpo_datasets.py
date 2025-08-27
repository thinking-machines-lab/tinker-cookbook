from typing import cast

import chz
import datasets
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.preference.preference_datasets import (
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
)
from tinker_cookbook.supervised.chat_datasets import (
    SupervisedDatasetFromHFDataset,
)
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset


@chz.chz
class DPODatasetBuilderFromComparisons(ChatDatasetBuilder):
    """
    DPO dataset builder that uses a ComparisonDatasetBuilder.
    DPO needs both chosen and rejected examples for training.
    """

    comparison_builder: ComparisonDatasetBuilder

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_dataset, test_dataset = self.comparison_builder.get_train_and_test_datasets()
        renderer = self.renderer

        def comparison_to_datum(labeled_comparison: LabeledComparison) -> list[types.Datum]:
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

            return [
                datum_from_tokens_weights(
                    chosen_tokens, chosen_weights, self.common_config.max_length
                ),
                datum_from_tokens_weights(
                    rejected_tokens, rejected_weights, self.common_config.max_length
                ),
            ]

        def example_to_data(example: dict[str, str]) -> list[types.Datum]:
            labeled_comparison = self.comparison_builder.example_to_labeled_comparison(example)
            if labeled_comparison is None:
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


# Tulu38B comparison builder - this is the only one not in preference_datasets.py yet


@chz.chz
class Tulu38BComparisonBuilder(ComparisonDatasetBuilder):
    """Tulu 3.8B preference dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        instruction = example["prompt"]
        chosen_response = example["chosen"][1]["content"]
        rejected_response = example["rejected"][1]["content"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")
