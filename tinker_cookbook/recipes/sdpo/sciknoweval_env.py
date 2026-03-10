"""SciKnowEval MCQ environment for SDPO.

Loads the SciKnowEval dataset (hicai-zju/SciKnowEval) and creates RL
environments for multiple-choice science questions. This matches the
evaluation setup used in the SDPO paper (arXiv:2601.20802).

Domains: Biology, Chemistry, Material, Physics
"""

import math
import re
from functools import partial
from typing import Any, Literal, Sequence, cast

import chz
from datasets import Dataset, load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

SCIKNOWEVAL_SYSTEM_PROMPT = """\
Given a question and answer options, please select the right answer. \
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option, and nothing else."""

SciKnowEvalDomain = Literal["biology", "chemistry", "material", "physics"]


def _extract_answer(text: str) -> str | None:
    """Extract the answer letter from <answer>A</answer> XML format."""
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _format_choices(choices: dict) -> str:
    """Format MCQ choices as 'A: text\\nB: text\\n...'."""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))


class SciKnowEvalEnv(ProblemEnv):
    """Environment for SciKnowEval multiple-choice questions."""

    def __init__(
        self,
        question: str,
        choices_str: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.question = question
        self.choices_str = choices_str
        self.answer = answer  # Ground truth letter (A/B/C/D)

    def get_question(self) -> str:
        return f"{self.question}\n\n{self.choices_str}\nPlease reason step by step."

    def check_format(self, sample_str: str) -> bool:
        return _extract_answer(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        predicted = _extract_answer(sample_str)
        if predicted is None:
            return False
        return predicted == self.answer

    def get_reference_answer(self) -> str:
        return self.answer


def _load_sciknoweval(
    domain: SciKnowEvalDomain,
    seed: int = 42,
    test_fraction: float = 0.1,
) -> tuple[Dataset, Dataset]:
    """Load and split SciKnowEval for a given domain.

    Filters to L3 level, mcq-4-choices and mcq-2-choices types,
    then splits 90/10 train/test (matching the paper).
    """
    ds = load_dataset("hicai-zju/SciKnowEval", split="test")
    ds = cast(Dataset, ds)

    # Filter by domain, level, and type
    ds = ds.filter(
        lambda x: (
            x["domain"].lower() == domain
            and x["level"] == "L3"
            and x["type"] in ("mcq-4-choices", "mcq-2-choices")
        )
    )

    # Train/test split
    split = ds.train_test_split(test_size=test_fraction, seed=seed)
    return split["train"], split["test"]


class SciKnowEvalDataset(RLDataset):
    def __init__(
        self,
        ds: Dataset,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        system_prompt: str = SCIKNOWEVAL_SYSTEM_PROMPT,
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.system_prompt = system_prompt

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        builders = []
        for row in self.ds.select(range(batch_start, batch_end)):
            builder = self._make_env_group_builder(row)  # pyright: ignore[reportArgumentType]
            if builder is not None:
                builders.append(builder)
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(self, row: dict[str, Any]) -> ProblemGroupBuilder | None:
        question = row.get("question", "")
        choices = row.get("choices", {})
        answer_key = row.get("answerKey", "")

        if not question or not answer_key:
            return None

        choices_str = _format_choices(choices)
        convo_prefix: list[renderers.Message] = [{"role": "system", "content": self.system_prompt}]

        return ProblemGroupBuilder(
            env_thunk=partial(
                SciKnowEvalEnv,
                question,
                choices_str,
                answer_key,
                self.renderer,
                convo_prefix=convo_prefix,
            ),
            num_envs=self.group_size,
            dataset_name="sciknoweval",
        )


@chz.chz
class SciKnowEvalDatasetBuilder(RLDatasetBuilder):
    """Builds SciKnowEval train/test datasets for a given domain."""

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    domain: SciKnowEvalDomain = "chemistry"
    seed: int = 42

    async def __call__(self) -> tuple[SciKnowEvalDataset, SciKnowEvalDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_ds, test_ds = _load_sciknoweval(domain=self.domain, seed=self.seed)

        train_dataset = SciKnowEvalDataset(
            ds=train_ds,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
        )
        test_dataset = SciKnowEvalDataset(
            ds=test_ds,
            batch_size=self.batch_size,
            group_size=1,  # Single rollout for eval
            renderer=renderer,
        )
        return train_dataset, test_dataset
