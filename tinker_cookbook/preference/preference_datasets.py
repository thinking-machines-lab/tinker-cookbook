import random
import re
from typing import cast

import chz
import datasets
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.preference.types import (
    Comparison,
    ComparisonRenderer,
    ComparisonRendererFromChatRenderer,
    LabeledComparison,
)
from tinker_cookbook.supervised.chat_datasets import (
    SupervisedDatasetFromHFDataset,
)
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset


@chz.chz
class DPODatasetBuilder(ChatDatasetBuilder):
    """
    Abstract class for supervised learning datasets based on pairwise comparisons
    """

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        raise NotImplementedError

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        raise NotImplementedError

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_dataset, test_dataset = self.get_train_and_test_datasets()
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
            labeled_comparison = self.example_to_labeled_comparison(example)
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


@chz.chz
class PairwiseComparisonDatasetBuilder(ChatDatasetBuilder):
    """
    Abstract class for supervised learning datasets based on pairwise comparisons
    """

    swap: bool = False  # do data augmentation by swapping the order of the completions

    @property
    def comparison_renderer(self) -> ComparisonRenderer:
        return ComparisonRendererFromChatRenderer(self.renderer)

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        raise NotImplementedError

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        raise NotImplementedError

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_dataset, test_dataset = self.get_train_and_test_datasets()
        comparison_renderer = self.comparison_renderer
        rng = random.Random(0)
        # XXX This depends on what order we request the batch
        # and violates determinism assumptions

        def comparison_to_datum(labeled_comparison: LabeledComparison) -> types.Datum:
            tokens, weights = comparison_renderer.to_tokens_weights(labeled_comparison)
            return datum_from_tokens_weights(tokens, weights, self.common_config.max_length)

        def example_to_data(example: dict[str, str]) -> list[types.Datum]:
            labeled_comparison = self.example_to_labeled_comparison(example)
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


def _hhh_parse_conversation(text: str) -> list[renderers.Message]:
    """Parse conversation text into message list format."""
    messages = []

    # Split by Human: or Assistant: and capture the delimiter
    parts = re.split(r"(Human:|Assistant:)", text)

    # Skip the first part if it's empty (text starts with a delimiter)
    if not parts[0].strip():
        parts = parts[1:]

    # Process parts in pairs: (delimiter, content)
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            delimiter = parts[i].strip()
            content = parts[i + 1].strip()

            if delimiter == "Human:":
                messages.append({"role": "user", "content": content})
            elif delimiter == "Assistant:":
                messages.append({"role": "assistant", "content": content})

    return messages


def hhh_example_to_comparison(example: dict[str, str]) -> LabeledComparison | None:
    """Process a single preference pair into the new format."""
    chosen = _hhh_parse_conversation(example["chosen"])
    rejected = _hhh_parse_conversation(example["rejected"])
    if len(chosen) != len(rejected):
        # Ran into at least one malformatted example like this
        return None
    match_bool_list = [
        chosen_msg == rejected_msg
        for chosen_msg, rejected_msg in zip(chosen, rejected, strict=True)
    ]
    if match_bool_list != [True] * (len(match_bool_list) - 1) + [False]:
        # Ran into at least one malformatted example like this
        return None
    comparison = Comparison(
        prompt_conversation=chosen[:-1],
        completion_A=[chosen[-1]],
        completion_B=[rejected[-1]],
    )
    return LabeledComparison(comparison=comparison, label="A")


@chz.chz
class HHHBuilder(PairwiseComparisonDatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("Anthropic/hh-rlhf")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"].shuffle(seed=0)
        test_dataset = dataset["test"].shuffle(seed=0).take(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        return hhh_example_to_comparison(example)


@chz.chz
class HHHDPOBuilder(DPODatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("Anthropic/hh-rlhf")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"].shuffle(seed=0)
        test_dataset = dataset["test"].shuffle(seed=0).take(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        return hhh_example_to_comparison(example)


@chz.chz
class HelpSteer3Builder(PairwiseComparisonDatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("nvidia/HelpSteer3", "preference")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"].shuffle(seed=0)
        test_dataset = dataset["validation"].shuffle(seed=0).take(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        context = example["context"]
        response1 = example["response1"]
        response2 = example["response2"]
        overall_preference = example["overall_preference"]

        # Skip ties
        if overall_preference == 0:
            return None

        # Convert context to message format
        prompt_conversation = []
        for msg in context:
            if msg["role"] == "user":
                prompt_conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                prompt_conversation.append({"role": "assistant", "content": msg["content"]})

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": response1}],
            completion_B=[{"role": "assistant", "content": response2}],
        )
        return LabeledComparison(
            comparison=comparison, label="A" if overall_preference > 0 else "B"
        )


@chz.chz
class HelpSteer3DPOBuilder(DPODatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("nvidia/HelpSteer3", "preference")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"].shuffle(seed=0)
        test_dataset = dataset["validation"].shuffle(seed=0).take(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        context = example["context"]
        response1 = example["response1"]
        response2 = example["response2"]
        overall_preference = example["overall_preference"]

        # Skip ties
        if overall_preference == 0:
            return None

        # Convert context to message format
        prompt_conversation = []
        for msg in context:
            if msg["role"] == "user":
                prompt_conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                prompt_conversation.append({"role": "assistant", "content": msg["content"]})

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": response1}],
            completion_B=[{"role": "assistant", "content": response2}],
        )
        return LabeledComparison(
            comparison=comparison, label="A" if overall_preference > 0 else "B"
        )


@chz.chz
class UltraFeedbackBuilder(PairwiseComparisonDatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "argilla/ultrafeedback-binarized-preferences", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        instruction = example["instruction"]
        chosen_response = example["chosen_response"]
        rejected_response = example["rejected_response"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")


@chz.chz
class UltraFeedbackDPOBuilder(DPODatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "argilla/ultrafeedback-binarized-preferences", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        instruction = example["instruction"]
        chosen_response = example["chosen_response"]
        rejected_response = example["rejected_response"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")


@chz.chz
class Tulu38BBuilder(PairwiseComparisonDatasetBuilder):
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


@chz.chz
class Tulu38BDPOBuilder(DPODatasetBuilder):
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
