"""Search tool RL training infrastructure.

Provides environment building and RL dataset classes for training
search models with tool-based interaction.

Usage:
```bash
python -m tinker_cookbook.recipes.search.training model_name=<model>
```
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
from tinker_cookbook import cli_utils, model_info, tokenizer_utils
from tinker_cookbook.recipes.search.tools import (
    ChromaTool,
    ChromaToolConfig,
    TextAnswerReward,
    build_tools,
)
from tinker_cookbook.recipes.search_tool.search_env import (
    SEARCH_TASK_INSTRUCTIONS,
    SearchR1Datum,
    download_search_r1_dataset,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl import train, types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.tool_use import AgentToolMessageEnv


def _initial_messages(
    datum: SearchR1Datum,
    renderer: Renderer,
    chroma_tool: ChromaTool,
) -> list[Message]:
    """Build initial messages matching SearchEnv.standard_fewshot_prefix + question."""
    # Get tool schema from our @tool-decorated method
    tool_schemas = [chroma_tool.search.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_schemas,
        system_prompt=SEARCH_TASK_INSTRUCTIONS,
    )
    return prefix + [{"role": "user", "content": datum["question"]}]


def build_env(
    datum: SearchR1Datum,
    model_name: str,
    *,
    renderer_name: str | None = None,
    max_turns: int = 8,
    chroma_tool: ChromaTool,
    format_coef: float = 0.1,
) -> EnvFromMessageEnv:
    """Build an RL environment for a single search task."""
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    chosen_renderer = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(chosen_renderer, tokenizer)

    tools = build_tools(chroma_tool)
    msg_env = AgentToolMessageEnv(
        tools=tools,
        initial_messages=_initial_messages(datum, renderer, chroma_tool),
        max_turns=max_turns,
        reward_fn=TextAnswerReward(gold_answers=datum["answer"], format_coef=format_coef),
    )
    return EnvFromMessageEnv(
        renderer=renderer,
        message_env=msg_env,
        failed_parse_reward=-0.1,
    )


class EnvGroupBuilder(types.EnvGroupBuilder):
    """EnvGroupBuilder that creates search environments with a shared ChromaTool."""

    def __init__(
        self,
        datum: SearchR1Datum,
        model_name: str,
        renderer_name: str | None,
        max_turns: int,
        group_size: int,
        chroma_tool: ChromaTool,
        format_coef: float = 0.1,
    ):
        self.datum = datum
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.group_size = group_size
        self.chroma_tool = chroma_tool
        self.format_coef = format_coef

    async def make_envs(self) -> Sequence[types.Env]:
        return [
            build_env(
                datum=self.datum,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                chroma_tool=self.chroma_tool,
                format_coef=self.format_coef,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return [self.datum.get("data_source", "unknown")]


class RLDataset(types.RLDataset):
    """Dataset that cycles through search EnvGroupBuilders."""

    def __init__(
        self,
        env_group_builders: list[EnvGroupBuilder],
        batch_size: int,
        num_batches: int,
    ):
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size
        self.num_batches = num_batches

    def get_batch(self, index: int) -> Sequence[types.EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.env_group_builders)
        builders: list[types.EnvGroupBuilder] = []
        for i in range(self.batch_size):
            builders.append(self.env_group_builders[(start + i) % len(self.env_group_builders)])
        return builders

    def __len__(self) -> int:
        return self.num_batches


@dataclass(frozen=True)
class RLDatasetBuilder(types.RLDatasetBuilder):
    """Build an RL dataset over SearchR1 tasks with ChromaTool."""

    model_name: str
    chroma_config: ChromaToolConfig
    split: Literal["train", "test"] = "train"
    max_tasks: int | None = 100
    batch_size: int = 4
    group_size: int = 4
    renderer_name: str | None = None
    max_turns: int = 8
    num_batches: int = 50
    format_coef: float = 0.1
    seed: int = 0

    async def __call__(self) -> tuple[types.RLDataset, types.RLDataset | None]:
        # Create shared ChromaTool
        chroma_tool = await ChromaTool.create(self.chroma_config)

        # Load and shuffle tasks
        data = download_search_r1_dataset(self.split)
        rng = random.Random(self.seed)
        rng.shuffle(data)
        if self.max_tasks is not None:
            data = data[: self.max_tasks]

        env_builders = [
            EnvGroupBuilder(
                datum=datum,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                max_turns=self.max_turns,
                group_size=self.group_size,
                chroma_tool=chroma_tool,
                format_coef=self.format_coef,
            )
            for datum in data
        ]
        dataset = RLDataset(
            env_group_builders=env_builders,
            batch_size=self.batch_size,
            num_batches=self.num_batches,
        )
        return dataset, None


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Create a train.Config blueprint for search tool RL."""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    chroma_config = ChromaToolConfig(
        chroma_host="localhost",
        chroma_port=8000,
        chroma_collection_name="wiki_embeddings",
    )

    dataset_builder = RLDatasetBuilder(
        model_name=model_name,
        chroma_config=chroma_config,
        renderer_name=renderer_name,
        batch_size=4,
        group_size=4,
        max_turns=8,
        max_tasks=100,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "/tmp/tinker-examples/rl_search_tool",
            "dataset_builder": dataset_builder,
            "learning_rate": 4e-5,
            "max_tokens": 512,
            "eval_every": 0,
            "save_every": 0,
            "num_substeps": 1,
        }
    )


def main(config: train.Config) -> None:
    """Check log dir semantics then run the RL trainer."""
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    import sys

    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
