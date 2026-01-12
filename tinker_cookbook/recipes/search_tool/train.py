"""Search tool RL training CLI launcher.

Usage:
```bash
python -m tinker_cookbook.recipes.search_tool.train model_name=<model>
```
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.search_tool.search_env import SearchR1DatasetBuilder
from tinker_cookbook.recipes.search_tool.tools import ChromaToolConfig
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Create a train.Config blueprint for search tool RL."""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    chroma_config = ChromaToolConfig(
        chroma_host="localhost",
        chroma_port=8000,
        chroma_collection_name="wiki_embeddings",
    )

    dataset_builder = SearchR1DatasetBuilder(
        model_name_for_tokenizer=model_name,
        chroma_tool_config=chroma_config,
        renderer_name=renderer_name,
        batch_size=4,
        group_size=4,
        max_turns=5,
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
