"""Dataset builder for the lightweight local-search Search-R1 recipe.

Parallels ``search_env.py`` but uses ``LocalSearchTool`` (pure-numpy BM25, no
external services) instead of ``ChromaTool``. See ``local_tools.py`` for the
tool implementation and the README for a comparison of the two backends.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import chz

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.recipes.search_tool.local_tools import LocalSearchTool
from tinker_cookbook.recipes.search_tool.search_env import (
    SEARCH_TASK_INSTRUCTIONS,
    SearchR1Datum,
    download_search_r1_dataset,
)
from tinker_cookbook.recipes.search_tool.tools import TextAnswerReward
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env


def _initial_messages(
    datum: SearchR1Datum,
    renderer: Renderer,
    search_tool: LocalSearchTool,
) -> list[Message]:
    """Build initial messages with tool schemas and task question."""
    tool_schemas = [search_tool.search.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=SEARCH_TASK_INSTRUCTIONS,
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


class LocalSearchEnvGroupBuilder(EnvGroupBuilder):
    """EnvGroupBuilder that creates search environments with a shared LocalSearchTool."""

    def __init__(
        self,
        datum: SearchR1Datum,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        search_tool: LocalSearchTool,
        format_coef: float = 0.1,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int | None = None,
        context_overflow_reward: float = -0.1,
    ) -> None:
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.search_tool = search_tool
        self.format_coef = format_coef
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        initial_messages = _initial_messages(self.datum, renderer, self.search_tool)
        reward_fn = TextAnswerReward(
            gold_answers=self.datum["answer"], format_coef=self.format_coef
        )

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=[self.search_tool.search],
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.datum.get("data_source", "unknown")]


class LocalSearchRLDataset(RLDataset):
    """Dataset that processes local-search EnvGroupBuilders once per epoch."""

    def __init__(
        self,
        env_group_builders: list[LocalSearchEnvGroupBuilder],
        batch_size: int,
    ) -> None:
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = start + self.batch_size
        return self.env_group_builders[start:end]

    def __len__(self) -> int:
        return len(self.env_group_builders) // self.batch_size


@chz.chz
class LocalSearchR1DatasetBuilder(RLDatasetBuilder):
    """Build an RL dataset over SearchR1 tasks with a local BM25 search tool."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    n_results: int = 3
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1
    seed: int = 0
    # Cap the number of training questions — the default corpus is small so retrieval
    # quality drops off on rare-entity questions. Use a subset that's well-covered.
    max_train_examples: int | None = 2048

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        search_tool = LocalSearchTool.build(n_results=self.n_results)

        data = download_search_r1_dataset("train")
        rng = random.Random(self.seed)
        rng.shuffle(data)
        if self.max_train_examples is not None:
            data = data[: self.max_train_examples]

        env_builders = [
            LocalSearchEnvGroupBuilder(
                datum=datum,
                model_name=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                search_tool=search_tool,
                format_coef=self.format_coef,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                context_overflow_reward=self.context_overflow_reward,
            )
            for datum in data
        ]
        dataset = LocalSearchRLDataset(
            env_group_builders=env_builders,
            batch_size=self.batch_size,
        )
        return dataset, None
