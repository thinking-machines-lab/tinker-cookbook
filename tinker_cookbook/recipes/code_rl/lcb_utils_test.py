"""Security regression tests for the code_rl grader (TEST_UTIL / TEST_CODE).

These tests stage the sandbox workdir exactly as code_grading.py does
(test_cases.txt / code.py / testing_util.py + run.py built from TEST_CODE) and
execute it in a subprocess, driving the real run_test/call_method path with no
sandbox service required. They pin two properties:

1. Honest solutions keep passing and wrong answers keep failing.
2. Reward-hacking submissions (reading the answer key from the workdir,
   patching the grader module, or smuggling zero test cases) fail closed.

The frame-walking variant (reading expected outputs from run_test's stack
frame) cannot be closed while the submission executes in the grader's process;
it is a documented residual risk, not a test here.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tinker_cookbook.recipes.code_rl.code_grading import (
    postprocess_lcb_sample,
    sandbox_check_correctness,
)
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL

STDIN_CASES = [
    {"input": "3\n1 2 3\n", "output": "6\n", "testtype": "stdin_stdout", "metadata": {}},
    {"input": "4\n5 6 7 8\n", "output": "26\n", "testtype": "stdin_stdout", "metadata": {}},
]

# One JSON value per line == one positional argument per line (run_test contract).
CALLBASED_CASES = [
    {"input": "1\n2", "output": "3", "testtype": "functional", "metadata": {"func_name": "add"}},
    {"input": "10\n20", "output": "30", "testtype": "functional", "metadata": {"func_name": "add"}},
]

HONEST_STDIN = """
_n = int(input())
_vals = input().split()
print(sum(int(x) for x in _vals))
"""

HONEST_CALLBASED = """
def add(a, b):
    return a + b
"""

# Reads the answer key with the plain open() builtin: never guarded on the
# call-based path, and the sol preamble's `from builtins import *` shadows
# call_method's mock_open for stdin problems.
ECHO_STDIN_BUILTIN_OPEN = """
import json
with open("test_cases.txt") as _f:
    _io = json.loads(json.loads(_f.read())["input_output"])
_c = getattr(json, "_ctr", -1) + 1
json._ctr = _c
print(_io["outputs"][_c], end="")
"""

# Same via the os-level primitives that reliability_guard used to leave live.
ECHO_STDIN_OS_OPEN = """
import json
import os
_fd = os.open("test_cases.txt", os.O_RDONLY)
_data = b""
while True:
    _chunk = os.read(_fd, 65536)
    if not _chunk:
        break
    _data += _chunk
os.close(_fd)
_io = json.loads(json.loads(_data)["input_output"])
_c = getattr(json, "_ctr", -1) + 1
json._ctr = _c
print(_io["outputs"][_c], end="")
"""

# Same via io.open, which bypasses any builtins.open guard.
ECHO_STDIN_IO_OPEN = """
import json
from io import open as _iopen
with _iopen("test_cases.txt") as _f:
    _io = json.loads(json.loads(_f.read())["input_output"])
_c = getattr(json, "_ctr", -1) + 1
json._ctr = _c
print(_io["outputs"][_c], end="")
"""

ECHO_CALLBASED_OPEN = """
import json
with open("test_cases.txt") as _f:
    _outs = [json.loads(o) for o in json.loads(json.loads(_f.read())["input_output"])["outputs"]]
class Solution:
    def __init__(self):
        self._i = 0
    def add(self, a, b):
        _r = _outs[self._i]
        self._i += 1
        return _r
"""

# The grader module is staged in the workdir; importing it yields the live
# module object, so overwriting its comparison hook used to pass any output.
PATCH_TESTING_UTIL = """
import testing_util
testing_util.custom_compare_ = lambda output, ground_truth: True
print("garbage that still passes")
"""

EMPTY_CASES = {"input_output": json.dumps({"inputs": [], "outputs": []})}

EMPTY_SUBMISSION = """
print("no tests, no problem")
"""

ATTACKS: dict[str, tuple[list[dict] | None, str, dict | None]] = {
    "echo_stdin_builtin_open": (STDIN_CASES, ECHO_STDIN_BUILTIN_OPEN, None),
    "echo_stdin_os_open": (STDIN_CASES, ECHO_STDIN_OS_OPEN, None),
    "echo_stdin_io_open": (STDIN_CASES, ECHO_STDIN_IO_OPEN, None),
    "echo_callbased_open": (CALLBASED_CASES, ECHO_CALLBASED_OPEN, None),
    "patch_testing_util": (STDIN_CASES, PATCH_TESTING_UTIL, None),
    "empty_inputs": (None, EMPTY_SUBMISSION, EMPTY_CASES),
}


def run_grader(
    sample: list[dict] | None,
    generation: str,
    raw_cases: dict | None = None,
) -> tuple[int, list[str]]:
    """Stage the workdir like code_grading.py and run TEST_CODE in a subprocess.

    Returns (exit code, names of files left in the workdir after the run).
    """
    test_cases = raw_cases if raw_cases is not None else postprocess_lcb_sample(sample or [])
    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        (workdir / "testing_util.py").write_text(TEST_UTIL)
        (workdir / "test_cases.txt").write_text(json.dumps(test_cases))
        (workdir / "code.py").write_text(generation)
        (workdir / "run.py").write_text(TEST_CODE % {"timeout": 6})
        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        remaining = sorted(p.name for p in workdir.iterdir())
    return proc.returncode, remaining


def test_honest_stdin_solution_passes() -> None:
    code, remaining = run_grader(STDIN_CASES, HONEST_STDIN)
    assert code == 0
    # Answer key and grader module must not survive in the workdir.
    assert "test_cases.txt" not in remaining
    assert "testing_util.py" not in remaining
    assert "__pycache__" not in remaining


def test_honest_stdin_wrong_answer_fails() -> None:
    code, _ = run_grader(STDIN_CASES, "print(999)")
    assert code != 0


def test_honest_callbased_solution_passes() -> None:
    code, _ = run_grader(CALLBASED_CASES, HONEST_CALLBASED)
    assert code == 0


@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_reward_hacking_submissions_fail_closed(attack: str) -> None:
    sample, generation, raw_cases = ATTACKS[attack]
    code, _ = run_grader(sample, generation, raw_cases=raw_cases)
    assert code != 0, f"attack {attack!r} fooled the grader into awarding a passing reward"


def test_sandbox_check_correctness_rejects_empty_sample() -> None:
    # Empty samples must never yield a pass. The wrapper currently raises
    # AssertionError before touching any sandbox backend (callers catch it and
    # score 0); returning (False, ...) would be equally acceptable.
    try:
        passed, _details = asyncio.run(sandbox_check_correctness([], "print(1)"))
    except AssertionError:
        passed = False
    assert passed is False
