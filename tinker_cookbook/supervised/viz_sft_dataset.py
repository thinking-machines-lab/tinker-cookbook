"""
Script to visualize supervised datasets in the terminal.
"""

import chz
from tinker_cookbook import model_info
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.misc_utils import lookup_func
from tinker_cookbook.renderers import TrainOnWhat


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"  # just for tokenizer
    dataset_path: str = "Tulu3Builder"
    renderer_name: str | None = None
    max_length: int | None = None
    train_on_what: TrainOnWhat | None = None


def run(cfg: Config):
    n_examples_total = 100
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name),
        max_length=cfg.max_length,
        batch_size=n_examples_total,
        train_on_what=cfg.train_on_what,
    )
    dataset_builder = lookup_func(
        cfg.dataset_path, default_module="tinker_cookbook.recipes.chat_sl.chat_datasets"
    )(common_config=common_config)
    assert isinstance(dataset_builder, SupervisedDatasetBuilder)
    tokenizer = get_tokenizer(cfg.model_name)
    train_dataset, _ = dataset_builder()
    batch = train_dataset.get_batch(0)

    for datum in batch:
        int_tokens = list(datum.model_input.to_ints()) + [
            datum.loss_fn_inputs["target_tokens"].tolist()[-1]
        ]
        weights = [0.0] + datum.loss_fn_inputs["weights"].tolist()
        print(format_colorized(int_tokens, weights, tokenizer))
        input("press enter")


if __name__ == "__main__":
    chz.nested_entrypoint(run)
