import json
from functools import partial
from typing import Any, Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, load_dataset

import tinker
from tinker_cookbook.recipes.code_rl.code_grading import (
    extract_code_from_model,
    sandbox_check_correctness,
    taco_to_lcb_format,
)
from tinker_cookbook.recipes.code_rl.lcb_utils import fetch_live_code_bench_system_prompt
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree


def _load_deepcoder_split(split: Literal["train", "test"]) -> Dataset:
    if split == "train":
        datasets = [
            cast(
                Dataset,
                load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train"),
            )
            for name in ("primeintellect", "taco", "lcbv5")
        ]
    elif split == "test":
        datasets = [
            cast(
                Dataset,
                load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test"),
            )
            for name in ("codeforces", "lcbv5")
        ]
    return cast(Dataset, concatenate_datasets(datasets))


def _ensure_dict(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize metadata: %s", metadata)
            return {}
    if isinstance(metadata, dict):
        return metadata
    return {}


def _normalize_tests(raw_tests: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tests = raw_tests
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            logger.warning("Failed to deserialize tests. Dropping sample.")
            return []
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = taco_to_lcb_format(tests)
    if isinstance(tests, dict):
        tests = [tests]

    normalized: list[dict[str, Any]] = []
    for test in tests or []:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or "stdin_stdout"
        test_metadata = _ensure_dict(test.get("metadata", {}))
        if testtype == "functional":
            func_name = test_metadata.get("func_name") or metadata.get("func_name")
            if func_name is not None:
                test_metadata["func_name"] = str(func_name)
        normalized.append(
            {
                "input": str(test.get("input", "")),
                "output": str(test.get("output", "")),
                "testtype": testtype,
                "metadata": test_metadata or {"func_name": None},
            }
        )
    return normalized


def _build_question(example: dict[str, Any]) -> str | None:
    # Prefer preprocessed question if available.
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)


class CodeEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        tests: list[dict[str, Any]],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        reward_timeout: int = 6,
    ):
        super().__init__(renderer, convo_prefix, format_coef=format_coef)
        self.problem = problem
        self.tests = tests
        self.reward_timeout = reward_timeout

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        return extract_code_from_model(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        """Not used - CodeEnv uses async check_sandbox_correctness instead."""
        return False

    async def check_sandbox_correctness(self, sample_str: str) -> bool:
        """Check if the code passes all test cases using sandbox execution."""
        code = extract_code_from_model(sample_str)
        if code is None:
            logtree.log_text("No code block detected in response.")
            return False

        try:
            success, details = await sandbox_check_correctness(
                self.tests, code, timeout=self.reward_timeout
            )
            status = "✓" if success else "✗"
            logtree.log_text(
                f"Sandbox result {status}: {'All tests passed' if success else 'Failed'}"
            )
            return success
        except Exception as exc:
            logger.warning("Sandbox check failed: %s", exc, exc_info=True)
            logtree.log_text(f"Sandbox check failed: {exc}")
            return False

    def get_reference_answer(self) -> str:
        return ""

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.ensure_text(message["content"])
        format_ok_bool = bool(parse_success) and self.check_format(content)
        correct_answer_bool = await self.check_sandbox_correctness(content)
        format_score = float(format_ok_bool)
        correct_score = float(correct_answer_bool)
        total_reward = self.format_coef * (format_score - 1.0) + correct_score

        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {content}")
        if reference := self.get_reference_answer():
            logtree.log_text(f"Reference Answer: {reference}")
        logtree.log_text(
            f"Format Valid: {'✓' if format_ok_bool else '✗'}, "
            f"Correct: {'✓' if correct_answer_bool else '✗'}, "
            f"Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "correct": correct_score,
            },
        )


class DeepcoderDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        format_coef: float = 0.1,
        reward_timeout: int = 6,
    ):
        self.ds = _load_deepcoder_split(split)
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.format_coef = format_coef
        self.reward_timeout = reward_timeout

    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.ds))
        if start >= end:
            raise IndexError("Incorrect batch index for DeepcoderDataset")
        builders: list[EnvGroupBuilder] = []
        for row in self.ds.select(range(start, end)):
            builder = self._make_env_group_builder(cast(dict[str, Any], row), self.group_size)
            if builder is not None:
                builders.append(builder)
        return builders

    def _make_env_group_builder(
        self, example: dict[str, Any], group_size: int
    ) -> ProblemGroupBuilder | None:
        metadata = _ensure_dict(example.get("metadata", {}))
        tests = _normalize_tests(example.get("tests") or example.get("ground_truth"), metadata)
        if not tests:
            logger.warning("Skipping sample without valid tests.")
            return None
        question = _build_question(example)
        if question is None:
            logger.warning("Skipping sample without question text.")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                CodeEnv,
                question,
                tests,
                self.renderer,
                convo_prefix=self.convo_prefix,
                format_coef=self.format_coef,
                reward_timeout=self.reward_timeout,
            ),
            num_envs=group_size,
            dataset_name="deepcoder",
        )


@chz.chz
class DeepcoderDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None = None
    seed: int = 0
    format_coef: float = 0.1
    reward_timeout: int = 6

    async def __call__(self) -> tuple[DeepcoderDataset, DeepcoderDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_ds = DeepcoderDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            split="train",
            seed=self.seed,
            format_coef=self.format_coef,
            reward_timeout=self.reward_timeout,
        )
        test_ds = DeepcoderDataset(
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            split="test",
            seed=self.seed,
            format_coef=self.format_coef,
            reward_timeout=self.reward_timeout,
        )
        return train_ds, test_ds
