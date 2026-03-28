from __future__ import annotations

import importlib
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast


def code_state(modules: Sequence[str | ModuleType] = ("tinker_cookbook",)) -> str:
    """Capture the current code state for one or more Python modules as a diff string.

    For each module this function locates its filesystem path, discovers the
    enclosing Git repository, records the current commit hash (HEAD), and
    includes combined staged+unstaged changes (``git diff HEAD``) for the
    entire repository.  The output is suitable for storage alongside experiment
    metadata to reproduce the exact code state later.

    Args:
        modules (Sequence[str | ModuleType]): Module import names
            (e.g. ``"tinker_cookbook.rl"``) or already-imported module objects.

    Returns:
        str: Concatenated sections, one per repository, each beginning with
            a header of the form
            ``### repo: <repo_root> @ <commit_hash>`` followed by the
            repo-wide diff.  Modules not in a Git repository get a short
            note instead.

    Example::

        state = code_state(["tinker_cookbook", "tinker"])
        with open("code.diff", "w") as f:
            f.write(state)
    """

    def ensure_module(obj: str | ModuleType) -> ModuleType:
        if isinstance(obj, ModuleType):
            return obj
        assert isinstance(obj, str), (
            "Each item in modules must be a module object or import path string"
        )
        return importlib.import_module(obj)

    def find_module_dir(mod: ModuleType) -> Path:
        # Prefer package path if available, else use the file's directory
        mod_file = cast(str | None, getattr(mod, "__file__", None))
        mod_path_list = cast(Sequence[str] | None, getattr(mod, "__path__", None))
        assert (mod_file is not None) or (mod_path_list is not None), (
            f"Module {mod!r} lacks __file__/__path__"
        )
        if mod_path_list is not None:  # packages expose __path__ (iterable); pick the first entry
            first_path = next(iter(mod_path_list))
            return Path(first_path).resolve()
        assert mod_file is not None
        return Path(mod_file).resolve().parent

    def git_toplevel(start_dir: Path) -> Path | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(start_dir), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            )
            return Path(completed.stdout.strip()).resolve()
        except subprocess.CalledProcessError:
            return None

    def git_rev(head_dir: Path) -> str:
        completed = subprocess.run(
            ["git", "-C", str(head_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def git_diff_vs_head(head_dir: Path) -> str:
        """Return a repo-wide unified diff of working tree + index (staged and
        unstaged) relative to HEAD."""
        args = ["git", "-C", str(head_dir), "diff", "--no-color", "HEAD"]
        completed = subprocess.run(args, check=False, capture_output=True, text=True)
        return completed.stdout

    sections: list[str] = []

    # Group modules by their enclosing repo and track non-git modules
    repos_to_modules: dict[Path, list[str]] = {}
    nongit_modules: list[tuple[str, Path]] = []

    for obj in modules:
        mod = ensure_module(obj)
        mod_name = mod.__name__
        mod_dir = find_module_dir(mod)
        repo_root = git_toplevel(mod_dir)

        if repo_root is None:
            nongit_modules.append((mod_name, mod_dir))
            continue

        if repo_root not in repos_to_modules:
            repos_to_modules[repo_root] = []
        if mod_name not in repos_to_modules[repo_root]:
            repos_to_modules[repo_root].append(mod_name)

    # Emit one section per repo with a single repo-wide diff
    for repo_root in sorted(repos_to_modules.keys(), key=lambda p: str(p)):
        try:
            head = git_rev(repo_root)
        except subprocess.CalledProcessError:
            head = "UNKNOWN"

        diff_repo = git_diff_vs_head(repo_root)
        mod_names = ", ".join(sorted(repos_to_modules[repo_root]))
        header = f"### repo: {repo_root} @ {head}\nmodules: {mod_names}\n"
        if diff_repo:
            body = "-- repo-wide (vs HEAD, staged+unstaged) --\n" + diff_repo.rstrip() + "\n"
        else:
            body = "(no local changes)\n"
        sections.append(header + body)

    # Notes for modules not in a git repo
    for mod_name, mod_dir in nongit_modules:
        sections.append(
            f"### module: {mod_name} (not in a git repository)\nmodule_path: {mod_dir}\n"
        )

    return "\n".join(sections)
