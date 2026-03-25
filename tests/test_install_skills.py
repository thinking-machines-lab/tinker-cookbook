"""Tests for .claude/install-skills.sh"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SCRIPT = REPO_ROOT / ".claude" / "install-skills.sh"
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"


def _parse_excluded_from_script() -> set[str]:
    """Parse the EXCLUDED=(...) array from install-skills.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(r"EXCLUDED=\(([^)]+)\)", text)
    assert match, "Could not find EXCLUDED=(...) in install-skills.sh"
    return set(match.group(1).split())


# Derived from the script itself — not a separate hardcoded list
EXCLUDED_SKILLS = _parse_excluded_from_script()

# Skills already prefixed with tinker-
ALREADY_PREFIXED = {
    d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and d.name.startswith("tinker-")
}


def _all_skill_dirs() -> list[str]:
    """Return all skill directory names."""
    return [d.name for d in SKILLS_DIR.iterdir() if d.is_dir()]


def _expected_global_name(name: str) -> str:
    if name.startswith("tinker-"):
        return name
    return f"tinker-{name}"


def _run_install(home: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALL_SCRIPT), *extra_args],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    return tmp_path


class TestInstallSkills:
    def test_installs_all_non_excluded_skills(self, fake_home: Path):
        _run_install(fake_home)
        installed = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}
        all_skills = _all_skill_dirs()

        for skill in all_skills:
            expected = _expected_global_name(skill)
            if skill in EXCLUDED_SKILLS:
                assert expected not in installed, f"{skill} should be excluded"
            else:
                assert expected in installed, f"{skill} should be installed as {expected}"

    def test_excludes_dev_skills(self, fake_home: Path):
        _run_install(fake_home)
        installed = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}

        for skill in EXCLUDED_SKILLS:
            assert f"tinker-{skill}" not in installed

    def test_prefix_not_doubled(self, fake_home: Path):
        _run_install(fake_home)
        installed = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}

        for skill in ALREADY_PREFIXED:
            assert skill in installed, f"{skill} should keep its name"
            assert f"tinker-{skill}" not in installed, f"{skill} should not be double-prefixed"

    def test_symlinks_point_to_skill_dirs(self, fake_home: Path):
        _run_install(fake_home)
        skills_dst = fake_home / ".claude" / "skills"

        for link in skills_dst.iterdir():
            assert link.is_symlink(), f"{link.name} should be a symlink"
            target = link.resolve()
            assert target.is_dir(), f"{link.name} should point to a directory"
            assert (target / "SKILL.md").exists(), f"{link.name} target should contain SKILL.md"

    def test_remove_cleans_up(self, fake_home: Path):
        _run_install(fake_home)
        skills_dst = fake_home / ".claude" / "skills"
        assert any(skills_dst.iterdir()), "Skills should be installed first"

        _run_install(fake_home, "--remove")
        remaining = list(skills_dst.iterdir())
        assert remaining == [], f"All symlinks should be removed, but found: {remaining}"

    def test_idempotent(self, fake_home: Path):
        _run_install(fake_home)
        first_run = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}

        _run_install(fake_home)
        second_run = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}

        assert first_run == second_run

    def test_skips_non_symlink_collision(self, fake_home: Path):
        skills_dst = fake_home / ".claude" / "skills"
        skills_dst.mkdir(parents=True)
        # Create a real directory that would collide
        (skills_dst / "tinker-sft").mkdir()
        (skills_dst / "tinker-sft" / "dummy.txt").write_text("user content")

        result = _run_install(fake_home)
        assert "SKIP tinker-sft" in result.stdout

        # Verify the real directory was not replaced
        assert (skills_dst / "tinker-sft" / "dummy.txt").exists()
        assert not (skills_dst / "tinker-sft").is_symlink()

    def test_creates_skills_dir_if_missing(self, fake_home: Path):
        assert not (fake_home / ".claude").exists()
        _run_install(fake_home)
        assert (fake_home / ".claude" / "skills").is_dir()

    def test_skill_count_matches(self, fake_home: Path):
        """Sanity check: installed count matches expected."""
        _run_install(fake_home)
        installed = list((fake_home / ".claude" / "skills").iterdir())
        expected_count = len(_all_skill_dirs()) - len(EXCLUDED_SKILLS)
        assert len(installed) == expected_count

    def test_every_skill_is_accounted_for(self, fake_home: Path):
        """Every skill directory must be either installed or explicitly excluded.

        If this test fails, a new skill was added without deciding whether it
        should be globally installable. Either add it to the EXCLUDED list in
        install-skills.sh or leave it as-is to include it.
        """
        _run_install(fake_home)
        installed = {p.name for p in (fake_home / ".claude" / "skills").iterdir()}

        for skill in _all_skill_dirs():
            expected = _expected_global_name(skill)
            in_excluded = skill in EXCLUDED_SKILLS
            in_installed = expected in installed
            assert in_excluded or in_installed, (
                f"Skill '{skill}' is neither excluded in install-skills.sh "
                f"nor installed as '{expected}'. Add it to the EXCLUDED list "
                f"in .claude/install-skills.sh if it should not be globally installed."
            )

    def test_excluded_skills_exist(self):
        """Every entry in the EXCLUDED list must correspond to an actual skill directory.

        Catches typos or stale entries in the exclusion list.
        """
        all_skills = set(_all_skill_dirs())
        for excluded in EXCLUDED_SKILLS:
            assert excluded in all_skills, (
                f"EXCLUDED entry '{excluded}' in install-skills.sh does not match "
                f"any skill directory in .claude/skills/. Remove it or fix the typo."
            )
