from tests.smoke.helpers import run_recipe


def test_off_policy_reasoning():
    run_recipe(
        "tinker_cookbook.recipes.distillation.off_policy_reasoning",
        [
            "batch_size=16",
            "max_prompts=128",
            "buffer_size=128",
            "behavior_if_log_dir_exists=delete",
        ],
    )
