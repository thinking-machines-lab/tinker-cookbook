from tests.helpers import run_recipe


def test_rlhf_pipeline():
    run_recipe(
        "tinker_cookbook.recipes.preference.rlhf.rlhf_pipeline",
        ["short_name=smoke-test"],
    )
