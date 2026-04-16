"""Tests for skills/ directory structure and plugin manifest consistency."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
PLUGIN_JSON = REPO_ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_JSON = REPO_ROOT / ".claude-plugin" / "marketplace.json"


def _all_skill_dirs() -> set[str]:
    """Return all skill directory names."""
    return {d.name for d in SKILLS_DIR.iterdir() if d.is_dir()}


def _frontmatter(text: str) -> str:
    """Extract YAML frontmatter from between --- fences."""
    if not text.startswith("---"):
        return ""
    parts = text.split("\n---", 2)
    return parts[0][3:] if len(parts) >= 2 else ""


class TestSkillStructure:
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
                f"{skill}/SKILL.md has name '{match.group(1).strip()}' but expected '{skill}'"
            )

    def test_skill_has_description(self):
        """Every SKILL.md must have a description field in frontmatter."""
        for skill in _all_skill_dirs():
            text = (SKILLS_DIR / skill / "SKILL.md").read_text()
            fm = _frontmatter(text)
            match = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
            assert match, f"{skill}/SKILL.md is missing a description field in frontmatter"
            assert len(match.group(1).strip()) > 10, f"{skill}/SKILL.md has too short a description"


class TestPluginManifest:
    def test_plugin_json_exists(self):
        """plugin.json must exist in .claude-plugin/."""
        assert PLUGIN_JSON.exists(), "Missing .claude-plugin/plugin.json"

    def test_plugin_json_valid(self):
        """plugin.json must be valid JSON with required fields."""
        data = json.loads(PLUGIN_JSON.read_text())
        assert "name" in data, "plugin.json missing 'name'"
        assert "skills" in data, "plugin.json missing 'skills'"
        assert data["skills"] == "skills/", (
            f"plugin.json 'skills' should be 'skills/', got '{data['skills']}'"
        )

    def test_plugin_name_is_tinker(self):
        """Plugin name must be 'tinker' for /tinker:* namespace."""
        data = json.loads(PLUGIN_JSON.read_text())
        assert data["name"] == "tinker", (
            f"plugin.json name should be 'tinker', got '{data['name']}'"
        )

    def test_marketplace_valid_json(self):
        """marketplace.json must be valid and have required fields."""
        data = json.loads(MARKETPLACE_JSON.read_text())
        assert "name" in data
        assert "plugins" in data
        for plugin in data["plugins"]:
            assert "name" in plugin
            assert "description" in plugin
