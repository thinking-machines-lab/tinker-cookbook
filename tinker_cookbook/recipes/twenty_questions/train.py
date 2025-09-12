import asyncio
from time import time

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.twenty_questions.env import TwentyQuestionsDatasetBuilder
from tinker_cookbook.rl import train


def build_config() -> train.Config:
    # model_name = "Qwen/Qwen3-30B-A3B"
    if 0:
        model_name = "Qwen/Qwen3-30B-A3B"
        renderer_name = "qwen3_nothink"
    else:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    dataset_builder = TwentyQuestionsDatasetBuilder(
        batch_size=400,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=4,
        num_epochs=100,
        answerer_base_model=model_name,
    )

    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/twenty-questions-rl/{int(time())}",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,
        max_tokens=20,
        eval_every=5,
        compute_post_kl=True,
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
