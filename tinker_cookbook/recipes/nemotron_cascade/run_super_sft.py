"""Full SFT on Nemotron-3-Super-120B using locally downloaded data.

Usage:
    python -m tinker_cookbook.recipes.nemotron_cascade.run_super_sft

Data must be pre-downloaded to ~/data/nemotron-cascade-2/ via download_all.py.
"""

import asyncio
import shutil
from pathlib import Path

from tinker_cookbook.supervised.data import HFDatasetSource, InterleavedChatDatasetBuilder
from tinker_cookbook.supervised.train import Config, main
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

DATA_DIR = Path.home() / "data" / "nemotron-cascade-2"

SUBSETS = [
    ("math", "sft_math_full.jsonl"),
    ("science", "sft_science_full.jsonl"),
    ("chat", "sft_chat_full.jsonl"),
    ("instruction_following", "sft_instruction_following_full.jsonl"),
    ("safety", "sft_safety_full.jsonl"),
    ("conversational_agent", "sft_conversational_agent_full.jsonl"),
    ("swe", "sft_swe_full.jsonl"),
    ("terminal_agent", "sft_terminal_agent_full.jsonl"),
]


def make_config(log_path: str = "/tmp/super_sft_full") -> Config:
    # Verify all data files exist
    sources = []
    for name, filename in SUBSETS:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing {filepath}. Run download_all.py first."
            )
        sources.append(HFDatasetSource(path=str(filepath)))

    return Config(
        model_name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144",
        dataset_builder=InterleavedChatDatasetBuilder(
            sources=sources,
            test_size=512,
            shuffle_seed=42,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
                renderer_name="nemotron3_disable_thinking",
                max_length=49152,
                batch_size=2048,
            ),
        ),
        lora_rank=64,
        learning_rate=3e-4,
        lr_schedule="cosine",
        num_epochs=2,
        max_steps=33000,
        save_every=5000,
        rolling_save_every=500,
        eval_every=1000,
        log_path=log_path,
        adam_beta2=0.98,
        wandb_project="nemotron-cascade-2-replication",
        wandb_name="super-262k-sft-full-8domain",
    )


if __name__ == "__main__":
    config = make_config()
    asyncio.run(main(config))
