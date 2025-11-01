"""
Dataset utilities for on-policy distillation.

This module contains dataset configuration classes and environment definitions
for distillation where the only supervision comes from the KL penalty against
a teacher model. The environment provides no correctness or format rewards.
"""

import math
from functools import partial
from typing import List, Literal, Sequence

import chz
import tinker
from datasets import load_dataset
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


@chz.chz
class TeacherConfig:
    """Configuration for a teacher model."""

    base_model: str
    load_checkpoint_path: str | None = None


@chz.chz
class DistillationDatasetConfig:
    """Configuration for a dataset used in distillation."""

    dataset_builder: RLDatasetBuilder
    teacher_config: TeacherConfig
    groups_per_batch: int


class CompositeDataset:
    """Wraps multiple datasets and samples from each according to their groups_per_batch."""

    def __init__(self, datasets: List[RLDataset], groups_per_batch_list: List[int]):
        self.datasets = datasets
        self.groups_per_batch_list = groups_per_batch_list
        # Use the shortest dataset length
        if len(datasets) > 0:
            self.length = min(len(dataset) for dataset in datasets)
        else:
            self.length = 0

    def __len__(self) -> int:
        return self.length

    def get_batch(self, i_batch: int) -> tuple[List[EnvGroupBuilder], List[int]]:
        """
        Get a batch by sampling from each dataset according to groups_per_batch.

        Returns:
            env_group_builders: List of all env group builders
            dataset_indices: List of dataset indices corresponding to each env group builder
        """
        all_env_group_builders = []
        all_dataset_indices = []

        for dataset_idx, (dataset, groups_per_batch) in enumerate(
            zip(self.datasets, self.groups_per_batch_list)
        ):
            env_group_builders = dataset.get_batch(i_batch)
            # Each dataset should return exactly groups_per_batch items
            assert len(env_group_builders) == groups_per_batch, (
                f"Dataset {dataset_idx} returned {len(env_group_builders)} items, "
                f"expected {groups_per_batch}"
            )
            all_env_group_builders.extend(env_group_builders)
            all_dataset_indices.extend([dataset_idx] * groups_per_batch)

        return all_env_group_builders, all_dataset_indices


class PromptOnlyEnv(ProblemEnv):
    """Environment that only provides prompts with no rewards."""

    def __init__(
        self,
        prompt: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        # Set format_coef to 0 since we don't care about format
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.prompt = prompt

    def get_question(self) -> str:
        return self.prompt

    def check_format(self, sample_str: str) -> bool:
        # Always return True - no format checking for distillation
        return True

    def check_answer(self, sample_str: str) -> bool:
        # Always return False - no answer checking for distillation
        return False

    def get_reference_answer(self) -> str:
        """No reference answer needed for distillation."""
        return ""

    async def step(self, action: Action) -> StepResult:
        """Return zero reward always."""
        message, parse_success = self.renderer.parse_response(action)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class PromptOnlyDataset(RLDataset):
    """Dataset that provides prompts without rewards."""

    def __init__(
        self,
        prompts: list[str],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        tokenizer,
        max_prompt_tokens: int | None = None,
        convo_prefix: list[renderers.Message] | None = None,
        dataset_name: str = "prompts",
    ):
        self.prompts = prompts
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.convo_prefix = convo_prefix
        self.dataset_name = dataset_name

    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max_prompt_tokens if specified."""
        if self.max_prompt_tokens is None:
            return prompt

        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.max_prompt_tokens:
            tokens = tokens[: self.max_prompt_tokens]
            return self.tokenizer.decode(tokens)
        return prompt

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.prompts))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    PromptOnlyEnv,
                    self._truncate_prompt(prompt),
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name=self.dataset_name,
            )
            for prompt in self.prompts[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.prompts) / self.batch_size)


def load_deepmath_prompts(split: Literal["train", "test"] = "train") -> list[str] | None:
    """Load DeepMath prompts from HuggingFace. Returns None if split doesn't exist."""
    try:
        ds = load_dataset("zwhe99/DeepMath-103K", split=split)
        # DeepMath has 'question' field containing the math problem
        prompts = [row["question"] for row in ds]  # type: ignore
        return prompts
    except Exception as e:
        logger.warning(f"Could not load {split} split for DeepMath: {e}")
        return None


def load_tulu3_prompts() -> list[str] | None:
    """
    Load Tulu3 prompts from HuggingFace.

    Extracts the first user message from each conversation.
    Returns None if dataset cannot be loaded.
    """
    try:
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        prompts = []

        for row in ds:  # type: ignore
            messages = row["messages"]  # type: ignore
            # Extract first user message
            first_user_msg = None
            for msg in messages:
                if msg["role"] == "user":  # type: ignore
                    first_user_msg = msg["content"]  # type: ignore
                    break

            if first_user_msg:
                prompts.append(first_user_msg)

        return prompts
    except Exception as e:
        logger.warning(f"Could not load Tulu3 dataset: {e}")
        return None


@chz.chz
class PromptOnlyDatasetBuilder(RLDatasetBuilder):
    """Builder for prompt-only datasets."""

    dataset_name: str  # e.g., "deepmath"
    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    convo_prefix: list[renderers.Message] | None = None
    max_prompt_tokens: int | None = 1024  # Maximum tokens per prompt (None = no truncation)

    async def __call__(self) -> tuple[PromptOnlyDataset, PromptOnlyDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Load prompts based on dataset name
        if self.dataset_name == "deepmath":
            train_prompts = load_deepmath_prompts("train")
            test_prompts = None
        elif self.dataset_name == "tulu3":
            train_prompts = load_tulu3_prompts()
            test_prompts = None  # Tulu3 only has train split
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if train_prompts is None:
            raise ValueError(f"Could not load train split for {self.dataset_name}")

        train_dataset = PromptOnlyDataset(
            prompts=train_prompts,
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            max_prompt_tokens=self.max_prompt_tokens,
            convo_prefix=self.convo_prefix,
            dataset_name=self.dataset_name,
        )

        test_dataset = (
            PromptOnlyDataset(
                prompts=test_prompts,
                batch_size=self.groups_per_batch,
                group_size=1,  # Use group_size=1 for test
                renderer=renderer,
                tokenizer=tokenizer,
                max_prompt_tokens=self.max_prompt_tokens,
                convo_prefix=self.convo_prefix,
                dataset_name=f"{self.dataset_name}_test",
            )
            if test_prompts is not None
            else None
        )

        return train_dataset, test_dataset
