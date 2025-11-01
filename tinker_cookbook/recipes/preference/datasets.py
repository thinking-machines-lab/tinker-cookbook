import logging
import re
from typing import cast

import chz
import datasets
import pandas as pd
from tinker_cookbook import renderers
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


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


def _arena_parse_conversation(conversation: list) -> list[renderers.Message] | None:
    """Parse arena conversation to message format."""
    messages = []
    for msg in conversation:
        assert isinstance(msg, dict)
        role = msg["role"]
        content_list = msg["content"]

        # Extract text content only
        text_parts = []
        for item in content_list:
            if isinstance(item, dict) and item["type"] == "text":
                text = item.get("text", "")
                text_parts.append(text)
            else:
                logger.info(f"Skipping arena conversation with non-text content: {msg}")
                return None

        if text_parts:
            content = " ".join(text_parts)
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

    return messages


# ============================================================================
# Concrete Dataset Implementations
# ============================================================================


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


@chz.chz
class HHHComparisonBuilder(ComparisonDatasetBuilder):
    """HHH dataset comparison builder."""

    test_size: int = 1024

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("Anthropic/hh-rlhf")
        dataset = cast(datasets.DatasetDict, dataset)
        train_dataset = dataset["train"].shuffle(seed=0)
        test_dataset = dataset["test"].shuffle(seed=0).take(self.test_size)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        return hhh_example_to_comparison(example)


@chz.chz
class HelpSteer3ComparisonBuilder(ComparisonDatasetBuilder):
    """HelpSteer3 dataset comparison builder."""

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
        comparison = Comparison(
            prompt_conversation=context,
            completion_A=[{"role": "assistant", "content": response1}],
            completion_B=[{"role": "assistant", "content": response2}],
        )
        return LabeledComparison(
            comparison=comparison, label="A" if overall_preference < 0 else "B"
        )


@chz.chz
class UltraFeedbackComparisonBuilder(ComparisonDatasetBuilder):
    """UltraFeedback dataset comparison builder."""

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
class ArenaComparisonBuilder(ComparisonDatasetBuilder):
    """Arena dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
        dataset = cast(datasets.Dataset, dataset)

        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        winner = example["winner"]

        # Skip ties or invalid winners
        if winner not in ["model_a", "model_b"]:
            # print(f"Skipping arena example with invalid winner: {winner}")
            return None

        conversation_a = _arena_parse_conversation(example["conversation_a"])
        conversation_b = _arena_parse_conversation(example["conversation_b"])

        # Skip if conversations are empty or malformed
        if not conversation_a or not conversation_b:
            logger.info("Skipping arena example with empty conversations")
            return None

        # The conversations should have same prompt (all messages except last assistant response)
        # Check that both have at least a user message and assistant response
        if len(conversation_a) < 2 or len(conversation_b) < 2:
            logger.info("Skipping arena example with too few messages")
            return None

        # Verify last message is assistant in both
        if conversation_a[-1]["role"] != "assistant" or conversation_b[-1]["role"] != "assistant":
            logger.info("Skipping arena example with non-assistant last message")
            return None

        comparison = Comparison(
            prompt_conversation=conversation_a[0:1],
            completion_A=conversation_a[1:],
            completion_B=conversation_b[1:],
        )

        return LabeledComparison(comparison=comparison, label="A" if winner == "model_a" else "B")


@chz.chz
class HelpSteer2ComparisonBuilder(ComparisonDatasetBuilder):
    """HelpSteer2 dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset("nvidia/HelpSteer2", split="train")
        dataset = cast(datasets.Dataset, dataset)

        # Create pairs by grouping examples with same prompt
        prompt_to_responses: dict[str, list] = {}
        for i in range(len(dataset)):
            example = dataset[i]
            prompt = example["prompt"]
            if prompt not in prompt_to_responses:
                prompt_to_responses[prompt] = []
            prompt_to_responses[prompt].append(example)

        # Create comparison pairs from examples with same prompt
        comparisons = []
        for prompt, responses in prompt_to_responses.items():
            if len(responses) >= 2:
                # Sort by helpfulness score to create preferences
                responses.sort(key=lambda x: x["helpfulness"], reverse=True)
                # Take best vs worst if significant difference
                if responses[0]["helpfulness"] > responses[-1]["helpfulness"]:
                    comparisons.append(
                        {
                            "prompt": prompt,
                            "chosen_response": responses[0]["response"],
                            "rejected_response": responses[-1]["response"],
                            "helpfulness_diff": responses[0]["helpfulness"]
                            - responses[-1]["helpfulness"],
                        }
                    )

        # Convert to dataset
        df = pd.DataFrame(comparisons)
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.shuffle(seed=0)

        test_dataset = dataset.take(min(1024, len(dataset) // 10))
        train_dataset = dataset.skip(min(1024, len(dataset) // 10))
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        prompt = example["prompt"]
        chosen_response = example["chosen_response"]
        rejected_response = example["rejected_response"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": prompt}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")
