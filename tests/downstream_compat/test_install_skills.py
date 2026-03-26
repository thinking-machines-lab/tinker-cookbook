"""Tests for skills/ directory structure and marketplace.json consistency."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
            names.add(path.rsplit("/", 1)[-1])
    return names


def _frontmatter(text: str) -> str:
    """Extract YAML frontmatter from between --- fences."""
    # First fence may be at the very start of the file (no leading newline)
    if not text.startswith("---"):
        return ""
    parts = text.split("\n---", 2)
    return parts[0][3:] if len(parts) >= 2 else ""


class TestSkillStructure:
    def test_all_skills_have_tinker_prefix(self):
        """Every skill directory must start with tinker-."""
        for skill in _all_skill_dirs():
            assert skill.startswith("tinker-"), (
                f"Skill directory '{skill}' must start with 'tinker-'"
            )

    def test_all_skills_have_skill_md(self):
        """Every skill directory must contain a SKILL.md file."""
        for skill in _all_skill_dirs():
            assert (SKILLS_DIR / skill / "SKILL.md").exists(), (
                f"Skill directory '{skill}' is missing SKILL.md"
            )

    def test_skill_name_matches_directory(self):
        """The name field in SKILL.md frontmatter must match the directory name."""
        for skill in _all_skill_dirs():
            text = (SKILLS_DIR / skill / "SKILL.md").read_text()
            fm = _frontmatter(text)
            match = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
            assert match, f"{skill}/SKILL.md is missing a name field in frontmatter"
            assert match.group(1).strip() == skill, (
                f"{skill}/SKILL.md has name '{match.group(1).strip()}' but directory is '{skill}'"
            )

    def test_skills_and_marketplace_match(self):
        """Skills on disk and in marketplace.json must be the same set.

        If a skill exists on disk but not in marketplace.json, add it to
        the appropriate plugin bundle (tinker-cookbook or tinker-dev).
        """
        on_disk = _all_skill_dirs()
        in_marketplace = _marketplace_skills()
        assert on_disk == in_marketplace, (
            f"Mismatch between skills/ and marketplace.json.\n"
            f"  On disk only: {on_disk - in_marketplace or 'none'}\n"
            f"  In marketplace only: {in_marketplace - on_disk or 'none'}"
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
