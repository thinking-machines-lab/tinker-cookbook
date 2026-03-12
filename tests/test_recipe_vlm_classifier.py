from tests.helpers import run_recipe


def test_vlm_classifier():
    run_recipe(
        "tinker_cookbook.recipes.vlm_classifier.train",
        [
            "experiment_dir=/tmp/tinker-smoke-test/vlm_classifier",
            "model_name=Qwen/Qwen3-VL-30B-A3B-Instruct",
            "batch_size=16",
            "num_epochs=1",
            "n_eval=16",
            "eval_every=0",
            "behavior_if_log_dir_exists=delete",
        ],
    )
