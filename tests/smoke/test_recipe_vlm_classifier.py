from tests.smoke.conftest import run_recipe


def test_vlm_classifier():
    run_recipe(
        "tinker_cookbook.recipes.vlm_classifier.train",
        [
            "experiment_dir=/tmp/tinker-smoke-test/vlm_classifier",
            "behavior_if_log_dir_exists=delete",
        ],
    )
