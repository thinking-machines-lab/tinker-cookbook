"""
Script to convert supervised datasets to jsonl.

The script is adapted from the tinker_cookbook.viz_sft_dataset script. It might be faster to directly download the HF dataset and convert it to jsonl than to tokenize and decode the dataset.

uv run python -m tinker_cookbook.convert_sft_dataset_to_jsonl dataset_path=Tulu3Builder output_file=~/tinker_datasets/tulu3.jsonl
"""

import json
import os

import chz
from tqdm import tqdm

from tinker_cookbook import model_info
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.misc_utils import lookup_func


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"  # just for tokenizer
    dataset_path: str = "Tulu3Builder"
    output_file: str = "~/tinker_datasets/tulu3.jsonl"
    renderer_name: str | None = None
    max_length: int | None = None


def run(cfg: Config):
    # check if the output file exists
    assert not os.path.exists(cfg.output_file), (
        f"Output file {cfg.output_file} already exists. Please provide a different output file."
    )

    # create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)

    assert cfg.model_name.startswith("meta-llama/"), "Only meta-llama models are supported."

    n_examples_total = 100
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name),
        max_length=cfg.max_length,
        batch_size=n_examples_total,
    )
    dataset_builder = lookup_func(
        cfg.dataset_path, default_module="tinker_cookbook.supervised.chat_datasets"
    )(common_config=common_config)
    assert isinstance(dataset_builder, SupervisedDatasetBuilder)
    tokenizer = get_tokenizer(cfg.model_name)
    train_dataset, _ = dataset_builder()

    # Calculate total number of items for progress bar
    total_items = len(train_dataset)

    with tqdm(total=total_items, desc="Converting dataset") as pbar:
        for batch_idx in range(len(train_dataset)):
            batch = train_dataset.get_batch(batch_idx)
            for datum in batch:
                int_tokens = list(datum.model_input.to_ints()) + [
                    datum.loss_fn_inputs["target_tokens"].tolist()[-1]
                ]

                # TODO: this might break for non-meta-llama models.
                decoded = tokenizer.decode(int_tokens)
                user_prompt = decoded.split("Assistant:")[0].split("User:")[1].strip()
                assistant_response = decoded.split("Assistant:")[1].split("User:")[0].strip()

                json_data = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_response},
                    ],
                }

                # write to jsonl
                with open(cfg.output_file, "a") as f:
                    f.write(json.dumps(json_data) + "\n")

            # Update progress bar
            pbar.update(1)


if __name__ == "__main__":
    chz.nested_entrypoint(run)
