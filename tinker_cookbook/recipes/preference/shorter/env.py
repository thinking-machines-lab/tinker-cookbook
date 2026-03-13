import chz
from datasets import Dataset

from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
    PreferenceModel,
    PreferenceModelBuilder,
)
from tinker_cookbook.renderers import Message

CONVO_PREFIX: list[Message] = [{"role": "user", "content": "Who are you?"}]
DUMMY_COMPLETION: list[Message] = [
    {
        "role": "assistant",
        "content": "Hello thre! I am a large language model. How can I assist you? Feel free to ask me anything.",
    }
]
DUMMY_COMPARISON: Comparison = Comparison(
    prompt_conversation=CONVO_PREFIX,
    completion_A=DUMMY_COMPLETION,
    completion_B=DUMMY_COMPLETION,
)
DUMMY_DATASET: Dataset = Dataset.from_list([{"id": None}] * 1024)


class PreferenceModelShorter(PreferenceModel):
    """
    A dummy preference model that always prefers a shorter response
    """

    def _get_completion_length(self, completion: list[Message]) -> int:
        char_count = 0
        for message in completion:
            char_count += len(message["content"])
        return char_count

    async def __call__(self, comparison: Comparison) -> float:
        length_a = self._get_completion_length(comparison.completion_A)
        length_b = self._get_completion_length(comparison.completion_B)
        if length_a > length_b:
            return 1.0
        elif length_b > length_a:
            return -1.0
        else:
            return 0.0


@chz.chz
class ShorterComparisonBuilder(ComparisonDatasetBuilder):
    def get_train_and_test_datasets(self) -> tuple[Dataset, Dataset | None]:
        return DUMMY_DATASET, None

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        return LabeledComparison(comparison=DUMMY_COMPARISON, label="Tie")


@chz.chz
class ShorterPreferenceModelBuilder(PreferenceModelBuilder):
    def __call__(self) -> PreferenceModel:
        return PreferenceModelShorter()
