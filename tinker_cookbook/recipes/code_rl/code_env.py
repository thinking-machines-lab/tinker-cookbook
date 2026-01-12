"""Code environment with tool-use infrastructure.

This module provides:
- Task loading: load_deepcoder_tasks() for DeepCoder dataset
- Environment construction: build_env(), EnvGroupBuilder, DeepcoderDataset, DeepcoderDatasetBuilder
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, load_dataset

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.code_rl.code_grading import taco_to_lcb_format
from tinker_cookbook.recipes.code_rl.lcb_utils import fetch_live_code_bench_system_prompt
from tinker_cookbook.recipes.code_rl.tools import CodeReward, CodeTask, CodeTool
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl import types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.sandbox import SandboxBackend
from tinker_cookbook.tool_use import AgentToolMessageEnv

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Loading Helpers
# ============================================================================


def _load_deepcoder_split(split: Literal["train", "test"]) -> Dataset:
    """Load a split from the DeepCoder dataset."""
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
    """Parse JSON metadata if needed."""
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
    """Normalize test cases to standard format."""
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
    """Build the question text with LCB system prompt."""
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)


# ============================================================================
# Task Loading
# ============================================================================


def load_deepcoder_tasks(
    split: Literal["train", "test"] = "train",
    seed: int = 0,
) -> list[CodeTask]:
    """Load tasks from the DeepCoder dataset.

    Args:
        split: Which split to load ("train" or "test")
        seed: Random seed for shuffling (train split only)

    Returns:
        List of CodeTask instances with normalized test cases
    """
    ds: Dataset = _load_deepcoder_split(split)
    if split == "train":
        ds = ds.shuffle(seed=seed)

    tasks: list[CodeTask] = []
    for item in ds:
        row = cast(dict[str, Any], item)

        # Extract and normalize metadata
        metadata = _ensure_dict(row.get("metadata", {}))

        # Normalize test cases
        raw_tests = row.get("tests") or row.get("ground_truth")
        tests = _normalize_tests(raw_tests, metadata)
        if not tests:
            continue

        # Build problem prompt
        problem = _build_question(row)
        if problem is None:
            continue

        # Extract starter code if present
        starter_code = row.get("starter_code")
        if isinstance(starter_code, str) and not starter_code.strip():
            starter_code = None

        tasks.append(
            CodeTask(
                problem=problem,
                tests=tests,
                starter_code=starter_code if isinstance(starter_code, str) else None,
            )
        )

    return tasks


# ============================================================================
# Environment Construction
# ============================================================================


def _initial_messages(
    task: CodeTask,
    renderer: Renderer,
    code_tool: CodeTool,
) -> list[Message]:
    """Build initial messages with tool schemas and task problem.

    Note: task.problem already contains the full LCB system prompt (via _build_question),
    including starter code if present. The renderer adds tool-specific formatting
    automatically via create_conversation_prefix_with_tools().
    """
    tool_schemas = [code_tool.run_python.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=tool_schemas)
    return prefix + [{"role": "user", "content": task.problem}]


def build_env(
    task: CodeTask,
    model_name: str,
    *,
    renderer_name: str | None = None,
    max_turns: int = 1,
    sandbox_backend: SandboxBackend | None = None,
    timeout: int = 6,
    format_coef: float = 0.1,
) -> EnvFromMessageEnv:
    """Build an RL environment for a single code task."""
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    chosen_renderer = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(chosen_renderer, tokenizer)

    code_tool = CodeTool(task, sandbox_backend=sandbox_backend, timeout=timeout)
    msg_env = AgentToolMessageEnv(
        tools=[code_tool.run_python],
        initial_messages=_initial_messages(task, renderer, code_tool),
        max_turns=max_turns,
        reward_fn=CodeReward(code_tool=code_tool, format_coef=format_coef),
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=-0.1,
    )


class EnvGroupBuilder(types.EnvGroupBuilder):
    """EnvGroupBuilder that creates code environments with shared sandbox backend."""

    def __init__(
        self,
        task: CodeTask,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        sandbox_backend: SandboxBackend | None,
        timeout: int = 6,
        format_coef: float = 0.1,
    ):
        self.task = task
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.sandbox_backend = sandbox_backend
        self.timeout = timeout
        self.format_coef = format_coef

    async def make_envs(self) -> Sequence[types.Env]:
        return [
            build_env(
                task=self.task,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
                format_coef=self.format_coef,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["deepcoder"]


class DeepcoderDataset(types.RLDataset):
    """Dataset that processes code EnvGroupBuilders once per epoch."""

    def __init__(
        self,
        env_group_builders: list[EnvGroupBuilder],
        batch_size: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        # Ceiling division to include partial batches
        return (len(self.env_group_builders) + self.batch_size - 1) // self.batch_size


@chz.chz
class DeepcoderDatasetBuilder(types.RLDatasetBuilder):
    """Build an RL dataset over DeepCoder tasks with tool-use infrastructure."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 1
    format_coef: float = 0.1
    timeout: int = 6
    sandbox_backend: SandboxBackend | None = None
    seed: int = 0

    async def __call__(self) -> tuple[types.RLDataset, types.RLDataset | None]:
        # Load train tasks
        train_tasks = load_deepcoder_tasks("train", seed=self.seed)
        train_builders = [
            EnvGroupBuilder(
                task=task,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
                format_coef=self.format_coef,
            )
            for task in train_tasks
        ]
        train_dataset = DeepcoderDataset(
            env_group_builders=train_builders,
            batch_size=self.batch_size,
        )

        # Load test tasks (group_size=1 for eval)
        test_tasks = load_deepcoder_tasks("test", seed=self.seed)
        test_builders = [
            EnvGroupBuilder(
                task=task,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=1,  # Single sample per task for evaluation
                sandbox_backend=self.sandbox_backend,
                timeout=self.timeout,
                format_coef=self.format_coef,
            )
            for task in test_tasks
        ]
        test_dataset = DeepcoderDataset(
            env_group_builders=test_builders,
            batch_size=self.batch_size,
        )

        return train_dataset, test_dataset
