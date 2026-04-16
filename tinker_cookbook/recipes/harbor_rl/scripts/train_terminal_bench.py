"""
Load Terminal-Bench tasks from the Harbor cache and launch RL training.

uv run python tinker_cookbook/recipes/harbor_rl/scripts/train_terminal_bench.py

"""

# Kimi-K2 tokenizer imports bytes_to_unicode from a location removed in transformers>=5.
# Patch it back before anything triggers tokenizer loading.
import transformers.models.gpt2.tokenization_gpt2 as _gpt2_tok

if not hasattr(_gpt2_tok, "bytes_to_unicode"):
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    _gpt2_tok.bytes_to_unicode = bytes_to_unicode

import asyncio

import modal
modal.enable_output()

from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory, load_harbor_tasks
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

if __name__ == "__main__":
    cli_config = CLIConfig()
    tasks = load_harbor_tasks("terminal-bench-2.0")
    asyncio.run(cli_main(cli_config, tasks, sandbox_factory=default_sandbox_factory))
