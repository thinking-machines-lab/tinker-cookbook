from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sft import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-30B-A3B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=32768,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.FromConversationFileBuilder(
        common_config=common_config,
        file_path="/tmp/tinker-datasets/prompt_distillation_lang.jsonl",
    )
    return train.Config(
        log_path="/tmp/tinker-examples/prompt_distillation_lang",
        model_name=model_name,
        dataset_builder=dataset,
        learning_rate=1e-4,
        lr_schedule="linear",
        num_epochs=4,
        eval_every=5,
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    train.main(config)


if __name__ == "__main__":
    main()
