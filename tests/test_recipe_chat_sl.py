import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.chat_sl.train"
LOG_PATH = "/tmp/tinker-smoke-test/chat_sl_resume"


@pytest.mark.integration
def test_chat_sl():
    """Train SFT from scratch for 2 steps, saving a checkpoint at step 1."""
    run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=delete",
            f"log_path={LOG_PATH}",
            "save_every=1",
        ],
    )


@pytest.mark.integration
def test_chat_sl_resume():
    """Resume SFT training from the checkpoint saved by test_chat_sl."""
    run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=resume",
            f"log_path={LOG_PATH}",
            "save_every=1",
        ],
        max_steps=4,
    )
