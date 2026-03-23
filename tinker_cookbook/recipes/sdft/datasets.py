"""
Dataset loaders for SDFT recipe.

Provides datasets that return (env_group_builders, questions, golden_answers) batches
for use with the SDFT training loop. Each dataset loader returns an SDFTDataset
and optionally a test RLDataset for evaluation.
"""

import logging
import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import chz
import tinker
from datasets import load_dataset, load_from_disk

from tinker_cookbook import renderers
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDatasetBuilder
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class SDFTDataset:
    """Provides (builders, questions, golden_answers) batches for SDFT training.

    This is a recipe-private class, not part of the public tinker_cookbook API.
    """

    def __init__(
        self,
        questions: list[str],
        golden_answers: list[str],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        dataset_name: str = "sdft",
    ):
        assert len(questions) == len(golden_answers), (
            f"questions ({len(questions)}) and golden_answers ({len(golden_answers)}) must match"
        )
        self.questions = questions
        self.golden_answers = golden_answers
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        """Return (builders, questions, golden_answers) for the batch at index."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.questions))
        assert batch_start < batch_end, f"Batch index {index} out of range"

        batch_questions = self.questions[batch_start:batch_end]
        batch_golden = self.golden_answers[batch_start:batch_end]

        builders = [
            ProblemGroupBuilder(
                env_thunk=partial(PromptOnlyEnv, question, self.renderer),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for question in batch_questions
        ]
        return builders, batch_questions, batch_golden

    def __len__(self) -> int:
        return math.ceil(len(self.questions) / self.batch_size)


# ---------------------------------------------------------------------------
# SciKnowEval
# ---------------------------------------------------------------------------


def _format_sciknoweval_choices(choices: dict) -> str:  # type: ignore[type-arg]
    """Format MCQ choices as 'A: text\\nB: text\\n...'."""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))


def load_sciknoweval(
    domain: str = "Chemistry",
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load SciKnowEval L3 MCQ dataset.

    Returns (train_questions, train_answers, test_questions, test_answers).
    """
    ds = load_dataset("hicai-zju/SciKnowEval", split="test")
    # Filter to domain, L3 level, and MCQ types
    # Level is nested inside the 'details' dict field
    filtered = [
        row
        for row in ds  # type: ignore[union-attr]
        if row.get("domain") == domain  # type: ignore[union-attr]
        and row.get("details", {}).get("level") == "L3"  # type: ignore[union-attr]
        and "mcq" in str(row.get("type", ""))  # type: ignore[union-attr]
    ]
    if not filtered:
        raise ValueError(
            f"No SciKnowEval examples found for domain={domain}, level=L3, MCQ type. "
            f"Available domains may differ. Got {len(ds)} total examples."  # type: ignore[arg-type]
        )

    # Shuffle and split
    import random

    rng = random.Random(seed)
    rng.shuffle(filtered)
    split_idx = int(len(filtered) * train_fraction)
    train_rows = filtered[:split_idx]
    test_rows = filtered[split_idx:]

    def extract(rows: list) -> tuple[list[str], list[str]]:  # type: ignore[type-arg]
        questions = []
        answers = []
        for row in rows:
            q = row["question"]
            choices_str = _format_sciknoweval_choices(row["choices"])
            questions.append(
                f"{q}\n\n{choices_str}\n"
                "Please reason step by step, then give your answer as <answer>X</answer> "
                "where X is the letter of the correct option."
            )
            answers.append(str(row["answerKey"]))
        return questions, answers

    train_q, train_a = extract(train_rows)
    test_q, test_a = extract(test_rows)
    logger.info(f"Loaded SciKnowEval ({domain}): {len(train_q)} train, {len(test_q)} test examples")
    return train_q, train_a, test_q, test_a


@chz.chz
class SciKnowEvalSDFTBuilder(RLDatasetBuilder):
    """Builds SciKnowEval SDFT datasets."""

    groups_per_batch: int
    group_size: int = 1
    model_name_for_tokenizer: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"
    domain: str = "Chemistry"
    train_fraction: float = 0.9

    async def __call__(self) -> tuple[SDFTDataset, SDFTDataset | None]:  # type: ignore[override]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_q, train_a, test_q, test_a = load_sciknoweval(
            domain=self.domain,
            train_fraction=self.train_fraction,
        )

        train_dataset = SDFTDataset(
            questions=train_q,
            golden_answers=train_a,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            dataset_name="sciknoweval",
        )

        test_dataset = (
            SDFTDataset(
                questions=test_q,
                golden_answers=test_a,
                batch_size=self.groups_per_batch,
                group_size=1,
                renderer=renderer,
                dataset_name="sciknoweval_test",
            )
            if test_q
            else None
        )

        return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# ToolAlpaca
# ---------------------------------------------------------------------------


def load_toolalpaca(
    data_path: str | None = None,
    seed: int = 42,
    train_fraction: float = 0.9,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load ToolAlpaca dataset for SDFT.

    If data_path is provided, loads from local Arrow format (SDFT paper's format).
    Otherwise loads from HuggingFace.

    Returns (train_questions, train_answers, test_questions, test_answers).
    """
    if data_path:
        ds = load_from_disk(data_path)
        prompts = [row["prompt"] for row in ds]  # type: ignore[union-attr]
        golden_responses = [
            "\n".join(row["golden_response"])  # type: ignore[index]
            if isinstance(row["golden_response"], list)  # type: ignore[index]
            else str(row["golden_response"])  # type: ignore[index]
            for row in ds  # type: ignore[union-attr]
        ]
    else:
        ds = load_dataset("tangqiaoyu/ToolAlpaca", split="train")
        prompts = [row["instruction"] for row in ds]  # type: ignore[union-attr]
        golden_responses = [str(row["output"]) for row in ds]  # type: ignore[union-attr]

    # Shuffle and split
    import random

    rng = random.Random(seed)
    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * train_fraction)

    train_q = [prompts[i] for i in indices[:split_idx]]
    train_a = [golden_responses[i] for i in indices[:split_idx]]
    test_q = [prompts[i] for i in indices[split_idx:]]
    test_a = [golden_responses[i] for i in indices[split_idx:]]

    logger.info(f"Loaded ToolAlpaca: {len(train_q)} train, {len(test_q)} test examples")
    return train_q, train_a, test_q, test_a


@chz.chz
class ToolAlpacaSDFTBuilder(RLDatasetBuilder):
    """Builds ToolAlpaca SDFT datasets."""

    groups_per_batch: int
    group_size: int = 1
    model_name_for_tokenizer: str = "Qwen/Qwen3-8B"
    renderer_name: str = "qwen3"
    data_path: str | None = None

    async def __call__(self) -> tuple[SDFTDataset, SDFTDataset | None]:  # type: ignore[override]
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_q, train_a, test_q, test_a = load_toolalpaca(data_path=self.data_path)

        train_dataset = SDFTDataset(
            questions=train_q,
            golden_answers=train_a,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            dataset_name="toolalpaca",
        )

        test_dataset = (
            SDFTDataset(
                questions=test_q,
                golden_answers=test_a,
                batch_size=self.groups_per_batch,
                group_size=1,
                renderer=renderer,
                dataset_name="toolalpaca_test",
            )
            if test_q
            else None
        )

        return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# Arrow data loaders (paper's exact preprocessed data)
# ---------------------------------------------------------------------------


def load_science_from_arrow(
    data_dir: str,
) -> tuple[list[str], list[str], list, list[str]]:
    """Load science data from the SDFT paper's Arrow format.

    The paper's science dataset has 'messages' (conversation with system prompt)
    and 'output_text' (full reasoning chain + answer in XML tags).

    Args:
        data_dir: Path to the dataset directory (e.g., ~/Repos/Self-Distillation/data/science_data).
            Must contain 'train_data/' and 'eval_data/' subdirectories.

    Returns:
        (questions, golden_answers, eval_prompts):
        - questions: user messages from training data
        - golden_answers: full reasoning chains (output_text) from training data
        - eval_prompts: list of message lists from eval data
    """
    train_ds = load_from_disk(f"{data_dir}/train_data")
    eval_ds = load_from_disk(f"{data_dir}/eval_data")

    # Convert <reasoning> tags to <think> tags for Qwen3 compatibility
    def _convert_tags(text: str) -> str:
        return text.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")

    questions = []
    golden_answers = []
    for row in train_ds:  # type: ignore[union-attr]
        msgs = row["messages"]  # type: ignore[index]
        # Extract user question (messages[1] is the user turn after system prompt)
        if len(msgs) >= 2:
            questions.append(msgs[1]["content"])
        else:
            questions.append(msgs[0]["content"])
        golden_answers.append(_convert_tags(str(row["output_text"])))  # type: ignore[index]

    # Eval data has 'prompt' (message list) and 'answer' (letter)
    # Convert <reasoning> to <think> in eval system prompts too
    eval_prompts = []
    for row in eval_ds:  # type: ignore[union-attr]
        converted_msgs = [
            {**m, "content": _convert_tags(m["content"])}
            for m in row["prompt"]  # type: ignore[index]
        ]
        eval_prompts.append(converted_msgs)
    eval_answers = [row["answer"] for row in eval_ds]  # type: ignore[union-attr]

    logger.info(
        f"Loaded science Arrow data: {len(questions)} train, {len(eval_prompts)} eval examples"
    )
    return questions, golden_answers, eval_prompts, eval_answers


def load_tooluse_from_arrow(
    data_dir: str,
) -> tuple[list[str], list[str], list[str], list[list[dict[str, str]]]]:
    """Load tooluse data from the SDFT paper's Arrow format.

    Args:
        data_dir: Path to the dataset directory (e.g., ~/Repos/Self-Distillation/data/tooluse_data).

    Returns:
        (questions, golden_answers, eval_prompts, eval_golden_answers)
    """
    train_ds = load_from_disk(f"{data_dir}/train_data")
    eval_ds = load_from_disk(f"{data_dir}/eval_data")

    questions = [row["prompt"] for row in train_ds]  # type: ignore[union-attr]
    golden_answers = [
        "\n".join(row["golden_response"])  # type: ignore[index]
        if isinstance(row["golden_response"], list)  # type: ignore[index]
        else str(row["golden_response"])  # type: ignore[index]
        for row in train_ds  # type: ignore[union-attr]
    ]

    eval_prompts = [row["prompt"] for row in eval_ds]  # type: ignore[union-attr]
    eval_golden_answers = [row["golden_answer"] for row in eval_ds]  # type: ignore[union-attr]

    logger.info(
        f"Loaded tooluse Arrow data: {len(questions)} train, {len(eval_prompts)} eval examples"
    )
    return questions, golden_answers, eval_prompts, eval_golden_answers


# ---------------------------------------------------------------------------
# SFT dataset builders (for benchmark comparison)
# ---------------------------------------------------------------------------


@chz.chz
class ScienceArrowSFTBuilder(ChatDatasetBuilder):
    """Builds SFT dataset from the paper's Arrow data (with reasoning chains)."""

    data_dir: str = "~/Repos/Self-Distillation/data/science_data"

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        expanded_dir = str(Path(self.data_dir).expanduser())
        questions, golden_answers, _, _ = load_science_from_arrow(expanded_dir)

        import datasets as hf_datasets

        # Convert <reasoning> tags to <think> tags for Qwen3 compatibility
        def _convert_tags(text: str) -> str:
            return text.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")

        system_prompt = (
            "Given a question and four options, please select the right answer. "
            "Respond in the following format:\n<think>\n...\n</think>\n"
            "<answer>\n...\n</answer>\n\n"
            "For the answer, only output the letter corresponding to the correct option "
            "(A, B, C, or D), and nothing else."
        )
        train_hf = hf_datasets.Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": _convert_tags(a)},
                    ]
                    for q, a in zip(questions, golden_answers)
                ]
            }
        )

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length
            )

        train_dataset = SupervisedDatasetFromHFDataset(
            train_hf, batch_size=self.common_config.batch_size, map_fn=map_fn
        )
        # No test dataset for SFT — eval is done separately via the eval phase
        return train_dataset, None
