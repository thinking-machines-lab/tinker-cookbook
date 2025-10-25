import asyncio
from time import time

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.preference.shorter.env import (
    ShorterComparisonBuilder,
    ShorterPreferenceModelBuilder,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.preference_envs import PairwisePreferenceRLDatasetBuilder


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name = model_info.get_recommended_renderer_name(model_name)

    comparison_builder = ShorterComparisonBuilder()
    dataset_builder = PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        batch_size=32,
        policy_renderer_name=renderer_name,
        policy_model_name=model_name,
        group_size=16,
        preference_model_builder=ShorterPreferenceModelBuilder(),
    )

    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/shorter/{int(time())}",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,
        max_tokens=64,
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
