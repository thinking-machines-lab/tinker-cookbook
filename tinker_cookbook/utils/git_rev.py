from __future__ import annotations

from tinker_cookbook import __version__


def recipe_user_metadata(recipe_name: str) -> dict[str, str]:
    """Build the standard ``user_metadata`` dict attached to a recipe's ``ServiceClient``.

    The ``git_rev`` field carries the cookbook's PEP 440 / setuptools-scm version
    string (e.g. ``"0.4.2.dev14+gf8dcf5d5a"``), which embeds the short git SHA in
    the local-version segment for dev installs and resolves to the release version
    for tagged builds. Baked in at build time, so it works correctly for wheel
    installs without shelling out to ``git``.
    """
    return {"recipe_name": recipe_name, "git_rev": __version__}
