import asyncio

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.twenty_questions.env import TwentyQuestionsDatasetBuilder
from tinker_cookbook.rl import train


def build_config() -> train.Config:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    dataset_builder = TwentyQuestionsDatasetBuilder(
        batch_size=100,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=4,
    )

    return train.Config(
        model_name=model_name,
        log_path="/tmp/tinker-examples/twenty-questions-rl",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,
        max_tokens=20,
        eval_every=0,
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
