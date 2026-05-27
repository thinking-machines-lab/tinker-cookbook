from __future__ import annotations

from tinker_cookbook import __version__
from tinker_cookbook.utils.git_rev import recipe_user_metadata


def test_recipe_user_metadata_shape():
    md = recipe_user_metadata("recipe_unit_test")
    assert md == {"recipe_name": "recipe_unit_test", "git_rev": __version__}


def test_git_rev_is_nonempty():
    md = recipe_user_metadata("recipe_unit_test")
    assert md["git_rev"]
