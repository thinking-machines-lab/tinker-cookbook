"""Tests for skills/ directory structure and marketplace.json consistency."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
MARKETPLACE_JSON = REPO_ROOT / ".claude-plugin" / "marketplace.json"


def _all_skill_dirs() -> set[str]:
    """Return all skill directory names."""
    return {d.name for d in SKILLS_DIR.iterdir() if d.is_dir()}


def _marketplace_skills() -> set[str]:
    """Return all skill directory names referenced in marketplace.json."""
    data = json.loads(MARKETPLACE_JSON.read_text())
    names = set()
    for plugin in data["plugins"]:
        for path in plugin["skills"]:
            # paths are like "./skills/tinker-sft"
            names.add(path.rsplit("/", 1)[-1])
    return names


class TestSkillStructure:
    def test_all_skills_have_tinker_prefix(self):
        """Every skill directory must start with tinker-."""
        for skill in _all_skill_dirs():
            assert skill.startswith("tinker-"), (
                f"Skill directory '{skill}' must start with 'tinker-'. "
                f"Rename it to 'tinker-{skill}'."
            )

    def test_all_skills_have_skill_md(self):
        """Every skill directory must contain a SKILL.md file."""
        for skill in _all_skill_dirs():
            assert (SKILLS_DIR / skill / "SKILL.md").exists(), (
                f"Skill directory '{skill}' is missing SKILL.md"
            )

    def test_skill_name_matches_directory(self):
        """The name field in SKILL.md must match the directory name."""
        import re

        for skill in _all_skill_dirs():
            text = (SKILLS_DIR / skill / "SKILL.md").read_text()
            match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
            assert match, f"{skill}/SKILL.md is missing a name field"
            assert match.group(1).strip() == skill, (
                f"{skill}/SKILL.md has name '{match.group(1).strip()}' but directory is '{skill}'"
            )

    def test_every_skill_in_marketplace(self):
        """Every skill directory must be listed in marketplace.json.

        If this test fails, a new skill was added without registering it
        in .claude-plugin/marketplace.json. Add it to the appropriate
        plugin bundle (tinker-training or tinker-dev).
        """
        on_disk = _all_skill_dirs()
        in_marketplace = _marketplace_skills()

        for skill in on_disk:
            assert skill in in_marketplace, (
                f"Skill '{skill}' exists on disk but is not listed in "
                f".claude-plugin/marketplace.json. Add it to a plugin bundle."
            )

    def test_marketplace_skills_exist(self):
        """Every skill in marketplace.json must exist on disk.

        Catches typos or stale entries in the marketplace config.
        """
        on_disk = _all_skill_dirs()
        in_marketplace = _marketplace_skills()

        for skill in in_marketplace:
            assert skill in on_disk, (
                f"Skill '{skill}' is in marketplace.json but does not exist "
                f"in skills/. Remove it or fix the path."
            )

    def test_marketplace_valid_json(self):
        """marketplace.json must be valid and have required fields."""
        data = json.loads(MARKETPLACE_JSON.read_text())
        assert "name" in data
        assert "plugins" in data
        for plugin in data["plugins"]:
            assert "name" in plugin
            assert "description" in plugin
            assert "skills" in plugin
            assert len(plugin["skills"]) > 0
