import asyncio
from typing import cast

import datasets
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl import arithmetic_env
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


def test_supervised():
    batch_size = 64
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = "role_colon"
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    max_length = 8192

    def dataset_builder():
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        dataset = dataset["train"]
        dataset = dataset.shuffle(seed=0)
        train_ds = dataset.take(batch_size * 3)

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(row["messages"], renderer, max_length)

        return SupervisedDatasetFromHFDataset(train_ds, batch_size=batch_size, map_fn=map_fn), None

    cfg = supervised_train.Config(
        model_name=model_name,
        dataset_builder=dataset_builder,  # type: ignore
        log_path="/tmp/tinker-smoke-test/supervised",
        wandb_project="tinker-smoke-test",
        learning_rate=1e-4,
    )
    asyncio.run(supervised_train.main(cfg))


async def test_rl():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank = 32
    renderer_name = "role_colon"
    tokenizer = get_tokenizer(model_name)
    renderers.get_renderer(renderer_name, tokenizer)

    dataset_builder = arithmetic_env.ArithmeticDatasetBuilder(
        batch_size=64,
        model_name_for_tokenizer=model_name,
        renderer_name="role_colon",
        n_batches=100,
        include_fewshot=True,
        group_size=16,
    )
    cfg = rl_train.Config(
        model_name=model_name,
        lora_rank=lora_rank,
        dataset_builder=dataset_builder,
        log_path="/tmp/tinker-smoke-test/rl-arithmetic",
        wandb_project="tinker-smoke-test",
        learning_rate=1e-4,
        max_tokens=5,
    )
    await rl_train.main(cfg)


def test_rl_async():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora_rank = 32
    renderer_name = "role_colon"
    tokenizer = get_tokenizer(model_name)
    renderers.get_renderer(renderer_name, tokenizer)

    dataset_builder = arithmetic_env.ArithmeticDatasetBuilder(
        batch_size=64,
        model_name_for_tokenizer=model_name,
        renderer_name="role_colon",
        n_batches=3,
        include_fewshot=True,
        group_size=16,
    )
    cfg = rl_train.Config(
        model_name=model_name,
        lora_rank=lora_rank,
        dataset_builder=dataset_builder,
        log_path="/tmp/tinker-smoke-test/rl-arithmetic",
        wandb_project="tinker-smoke-test",
        learning_rate=1e-4,
        max_tokens=5,
        async_config=rl_train.AsyncConfig(
            max_steps_off_policy=2,
            groups_per_batch=64,
        ),
    )
    asyncio.run(rl_train.main(cfg))


def test_rl_sync_stream_minibatch():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora_rank = 32
    renderer_name = "role_colon"
    tokenizer = get_tokenizer(model_name)
    renderers.get_renderer(renderer_name, tokenizer)

    dataset_builder = arithmetic_env.ArithmeticDatasetBuilder(
        batch_size=64,
        model_name_for_tokenizer=model_name,
        renderer_name="role_colon",
        n_batches=3,
        include_fewshot=True,
        group_size=16,
    )
    cfg = rl_train.Config(
        model_name=model_name,
        lora_rank=lora_rank,
        dataset_builder=dataset_builder,
        log_path="/tmp/tinker-smoke-test/rl-arithmetic-stream-minibatch",
        wandb_project="tinker-smoke-test",
        learning_rate=1e-4,
        max_tokens=5,
        stream_minibatch_config=rl_train.StreamMinibatchConfig(
            groups_per_batch=64,
            num_minibatches=2,
        ),
    )
    asyncio.run(rl_train.main(cfg))


if __name__ == "__main__":
    # test_supervised()
    asyncio.run(test_rl())
