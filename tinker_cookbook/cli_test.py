"""Tests for the CLI entry point."""

from tinker_cookbook.cli import _find_recipes, _list_recipes


def test_find_recipes_returns_dict():
    recipes = _find_recipes()
    assert isinstance(recipes, dict)
    # Should find at least the basic recipes
    assert "sl_basic" in recipes
    assert "rl_basic" in recipes


def test_find_recipes_includes_subpackages():
    recipes = _find_recipes()
    # Sub-directories with train.py should be discovered
    assert "math_rl" in recipes
    assert recipes["math_rl"] == "tinker_cookbook.recipes.math_rl.train"


def test_find_recipes_module_paths():
    recipes = _find_recipes()
    # Top-level files map to the recipes package directly
    assert recipes["sl_basic"] == "tinker_cookbook.recipes.sl_basic"
    # Sub-packages map to their train.py
    assert recipes["chat_sl"] == "tinker_cookbook.recipes.chat_sl.train"


def test_list_recipes_formatted():
    output = _list_recipes()
    assert "sl_basic" in output
    assert "math_rl" in output
    # Should include module paths in parentheses
    assert "(tinker_cookbook.recipes." in output
