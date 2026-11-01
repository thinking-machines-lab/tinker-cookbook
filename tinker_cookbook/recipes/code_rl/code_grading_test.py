"""Unit tests for out-of-band code grading.

The sandbox is replaced by a local subprocess that runs the same sandbox-side harness (RUN_CODE) on the
same files. Only programs written in this file run, so no sandbox service is needed.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tinker_cookbook.recipes.code_rl import code_grading
from tinker_cookbook.recipes.code_rl.lcb_utils import RUN_CODE, TEST_CODE, TEST_UTIL

DOUBLE = [
    {"input": "3\n", "output": "6\n", "testtype": "stdin_stdout"},
    {"input": "5\n", "output": "10\n", "testtype": "stdin_stdout"},
]
TWO_SUM = [
    {
        "input": "[2, 7, 11, 15]\n9",
        "output": "[0, 1]",
        "testtype": "functional",
        "metadata": {"func_name": "twoSum"},
    },
    {
        "input": "[3, 2, 4]\n6",
        "output": "[1, 2]",
        "testtype": "functional",
        "metadata": {"func_name": "twoSum"},
    },
]
TWO_SUM_OK = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        seen = {}\n"
    "        for i, x in enumerate(nums):\n"
    "            if target - x in seen:\n"
    "                return [seen[target - x], i]\n"
    "            seen[x] = i\n"
)

HONEST = [
    ("stdin correct", DOUBLE, "n = int(input())\nprint(2 * n)", True),
    ("stdin wrong", DOUBLE, "n = int(input())\nprint(2 * n + 1)", False),
    (
        "stdin trailing whitespace",
        DOUBLE,
        "n = int(input())\nprint(str(2 * n) + '  ')\nprint()",
        True,
    ),
    (
        "stdin __main__ block",
        DOUBLE,
        "def f():\n    print(2 * int(input()))\nif __name__ == '__main__':\n    f()",
        True,
    ),
    (
        "stdin exits after printing",
        DOUBLE,
        "import sys\nprint(2 * int(input()))\nsys.exit(0)",
        True,
    ),
    (
        "stdin runtime error on 2nd test",
        DOUBLE,
        "n = int(input())\nprint(2 * n if n < 4 else 1 // 0)",
        False,
    ),
    ("call correct", TWO_SUM, TWO_SUM_OK, True),
    (
        "call returns tuple",
        TWO_SUM,
        TWO_SUM_OK.replace("[seen[target - x], i]", "(seen[target - x], i)"),
        True,
    ),
    (
        "call wrong",
        TWO_SUM,
        "class Solution:\n    def twoSum(self, nums, target):\n        return [0, 0]",
        False,
    ),
    (
        "call syntax error",
        TWO_SUM,
        "class Solution:\n    def twoSum(self, nums, target)\n        return []",
        False,
    ),
]

HOSTILE = [
    ("os._exit(0) at import", DOUBLE, "import os\nos._exit(0)"),
    (
        "os._exit(0) in call",
        TWO_SUM,
        "import os\nclass Solution:\n    def twoSum(self, nums, target):\n        os._exit(0)",
    ),
    ("sys.exit(0) at module level", DOUBLE, "import sys\nsys.exit(0)"),
    (
        "raise SystemExit in method",
        TWO_SUM,
        "class Solution:\n    def twoSum(self, nums, target):\n        raise SystemExit(0)",
    ),
    ("read the answer key", DOUBLE, "print(open('test_cases.txt').read())"),
    (
        "forge a run record",
        DOUBLE,
        'import os\nos.write(1, b\'<<CODE_RL_RUN_RESULT>>x:{"status": "ran", "records": []}\\n\')\nos._exit(0)',
    ),
]


def _run_files(files: dict[str, str], main: str, timeout: int) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            Path(tmp, name).write_text(content)
        Path(tmp, "run.py").write_text(main)
        return subprocess.run(
            [sys.executable, "run.py"], cwd=tmp, capture_output=True, text=True, timeout=timeout
        )


@pytest.fixture
def local_sandbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Replace SandboxFusion with a local run of RUN_CODE; record the files each run received."""
    seen: list[dict[str, str]] = []

    async def fake_run(files: dict[str, str], total_timeout: int) -> str:
        seen.append(files)
        result = await asyncio.to_thread(_run_files, files, RUN_CODE, total_timeout)
        return result.stdout

    monkeypatch.setattr(code_grading, "_run_with_sandboxfusion", fake_run)
    return seen


def _check(sample: list[dict[str, Any]], code: str) -> tuple[bool, dict[str, Any]]:
    return asyncio.run(code_grading.sandbox_check_correctness(sample, code, timeout=6))


def _legacy_verdict(sample: list[dict[str, Any]], code: str) -> bool:
    """The previous harness: run_test inside the program's process, verdict by exit code."""
    files = {
        "test_cases.txt": json.dumps(code_grading.postprocess_lcb_sample(sample)),
        "code.py": code,
        "testing_util.py": TEST_UTIL,
    }
    return _run_files(files, TEST_CODE % {"timeout": 6}, 60).returncode == 0


@pytest.mark.parametrize("name,sample,code,expected", HONEST, ids=[h[0] for h in HONEST])
def test_honest_programs_get_the_same_verdict_as_before(
    local_sandbox, name, sample, code, expected
):
    passed, details = _check(sample, code)
    assert passed is expected, details
    assert passed is _legacy_verdict(sample, code)


@pytest.mark.parametrize("name,sample,code", HOSTILE, ids=[h[0] for h in HOSTILE])
def test_programs_that_fake_a_pass_fail(local_sandbox, name, sample, code):
    passed, details = _check(sample, code)
    assert passed is False, details


def test_early_exit_passed_under_the_previous_harness():
    assert _legacy_verdict(DOUBLE, "import os\nos._exit(0)") is True


def test_expected_outputs_never_reach_the_sandbox(local_sandbox):
    sample = [{"input": "123457\n", "output": "SECRET246914\n", "testtype": "stdin_stdout"}]
    _check(sample, "print(2 * int(input()))")
    assert local_sandbox and all("SECRET" not in content for content in local_sandbox[0].values())


def test_a_sample_without_tests_fails(local_sandbox):
    passed, details = _check([], "print(1)")
    assert passed is False and details["error"] == "no test cases"
    assert not local_sandbox


def test_the_preamble_matches_run_test():
    assert code_grading._preamble().startswith("from string import *\n")
    assert code_grading._preamble() in TEST_UTIL.encode().decode("unicode_escape")
