import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        return " Write your answer in \\boxed{} format."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, x["problem"], answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[MathDataset, MathDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            MathDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(
            seed=seed
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return PolarisDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
        ), None


class DeepMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        self.ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


@chz.chz
class DeepMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[DeepMathDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return DeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderers.get_renderer(self.renderer_name, tokenizer=tokenizer),
            seed=self.seed,
        ), None


class Gsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    @classmethod
    def question_suffix(cls) -> str:
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class Gsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0

    async def __call__(self) -> tuple[Gsm8kDataset, Gsm8kDataset]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            Gsm8kDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split=split,
                seed=self.seed,
            )
            for split in ("train", "test")
        ]
        return (datasets[0], datasets[1])


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
}


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    """
    Unified function to get any math dataset builder.
    Args:
        dataset_name: One of "math", "polaris", "deepmath", or "gsm8k"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
    Returns:
        The appropriate dataset builder instance
    """
    if dataset_name not in DATASET_BUILDER_MAP:
        raise ValueError(
            f"Unknown math dataset: {dataset_name}. Available: {list(DATASET_BUILDER_MAP.keys())}"
        )

    builder_class = DATASET_BUILDER_MAP[dataset_name]

    return builder_class(
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=seed,
    )
