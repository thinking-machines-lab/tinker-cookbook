import pytest

from tests.helpers import run_recipe


@pytest.mark.integration
def test_dpo():
    run_recipe(
        "tinker_cookbook.recipes.preference.dpo.train",
        ["behavior_if_log_dir_exists=delete"],
    )
