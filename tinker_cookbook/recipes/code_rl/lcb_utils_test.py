"""TEST_CODE artifact cleanup with a stub grader and a fake os.unlink; no submission runs."""

import os
import sys
import types
from pathlib import Path

import pytest

from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, error: OSError
) -> tuple[object, list[str]]:
    for artifact in ("test_cases.txt", "testing_util.py"):
        (tmp_path / artifact).write_text("{}")
    (tmp_path / "code.py").write_text("raise SystemExit('submission must not run')")
    graded: list[str] = []

    def stub_run_test(sample: dict, test: str, debug: bool, timeout: int) -> tuple[list, dict]:
        graded.append(test)
        return [True], {}

    stub = types.ModuleType("testing_util")
    stub.__dict__["run_test"] = stub_run_test
    monkeypatch.setitem(sys.modules, "testing_util", stub)
    real_unlink = os.unlink

    def fake_unlink(path: str) -> None:
        if path == name:
            raise error
        real_unlink(path)

    monkeypatch.setattr(os, "unlink", fake_unlink)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        exec(TEST_CODE % {"timeout": 6}, {"__name__": "__main__"})
    return exc.value.code, graded


@pytest.mark.parametrize("artifact", ["test_cases.txt", "testing_util.py"])
def test_undeletable_artifact_aborts_before_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    code, graded = _run(tmp_path, monkeypatch, artifact, PermissionError(13, "denied", artifact))
    assert code not in (0, None)
    assert graded == []
    assert (tmp_path / artifact).exists()


def test_already_absent_artifact_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, graded = _run(tmp_path, monkeypatch, "testing_util.py", FileNotFoundError(2, "gone"))
    assert code == 0
    assert len(graded) == 1
    assert not (tmp_path / "test_cases.txt").exists()
