"""
Code grading utilities for RL training.

Supports two execution backends:
- sandboxfusion: Local Docker-based sandbox (default)
- modal: Cloud-based Modal sandbox

Grading is out of band: the sandbox only runs the program on the test inputs and returns its raw
outputs (lcb_utils.RUN_CODE); the expected outputs never enter the sandbox, and the comparison runs
locally, through the same run_test, on a trusted replay of those outputs. A verdict decided inside the
sandbox (the exit code of a harness that shares a process with the program) can be faked: a program
that calls os._exit(0) before the tests run would score as passing.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from typing import Any

from tinker_cookbook.recipes.code_rl.lcb_utils import RUN_CODE, TEST_UTIL
from tinker_cookbook.sandbox import SandboxBackend, SandboxFusionClient

# Global sandbox backend clients (lazily initialized)
_sandboxfusion_client: SandboxFusionClient | None = None
_modal_pool: Any = None  # ModalSandboxPool, but avoid import at module level


def _get_sandboxfusion_client() -> SandboxFusionClient:
    """Get or create the SandboxFusion client."""
    global _sandboxfusion_client
    if _sandboxfusion_client is None:
        _sandboxfusion_client = SandboxFusionClient()
    return _sandboxfusion_client


def _get_modal_pool():
    """Get or create the Modal sandbox pool."""
    global _modal_pool
    if _modal_pool is None:
        import modal

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandboxPool

        image = modal.Image.debian_slim().pip_install("numpy")
        _modal_pool = ModalSandboxPool(image=image)
    return _modal_pool


def extract_code_from_model(model_response: str) -> str | None:
    """Extract the last fenced code block from a model response."""
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    """Convert test cases to LiveCodeBench format for the test runner."""
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]

    sample_dict: dict[str, Any] = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is None:
            raise AssertionError(f"Function name missing in metadata: {metadata}. Sample: {sample}")
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


_MARKER = "<<CODE_RL_RUN_RESULT>>"

# Runs locally in a throwaway subprocess (run_test's reliability_guard disables os functions in the
# process that calls it). Only trusted code runs here: TEST_UTIL and the replay program.
_GRADE_CODE = r"""
import json, sys
p = json.loads(sys.stdin.read())
util = {}
exec(p["test_util"], util)
results, meta = util["run_test"]({"input_output": p["input_output"]}, test=p["replay"], timeout=p["timeout"])
record = json.dumps({"results": [r is True for r in results], "meta": meta}, default=str)
sys.__stdout__.write("\n" + p["marker"] + record + "\n")
"""

_DECODE = """
def _dec(x):
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if isinstance(x, dict) and "__t" in x:
        return tuple(_dec(v) for v in x["__t"])
    if isinstance(x, dict) and "__d" in x:
        return {_dec(k): _dec(v) for k, v in x["__d"]}
    return x
"""


def _preamble() -> str:
    """The import preamble run_test prepends to every program, taken from its source."""
    start = TEST_UTIL.index('sol = "') + len('sol = "')
    return json.loads('"' + TEST_UTIL[start : TEST_UTIL.index('"\n', start)] + '"')


def _replay_program(records: list[dict[str, Any]], fn_name: str | None, class_based: bool) -> str:
    """A trusted stand-in for the program: returns (call-based) or prints (stdin) the recorded outputs in
    order, raising where the program raised. The records are embedded as a JSON string literal."""
    data = repr(json.dumps(records))
    if fn_name is not None:
        body = (
            f"_R = json.loads({data})\n" + _DECODE + "def _next():\n    r = _R.pop(0)\n"
            "    if not r['ok']:\n        raise RuntimeError(r['error'])\n    return _dec(r['out'])\n"
        )
        if class_based:
            method = f"class Solution:\n    def {fn_name}(self, *a):\n        return _next()\n"
        else:
            method = f"def {fn_name}(*a):\n    return _next()\n"
        return "import json\n" + body + method
    # run_test rewraps a stdin program into code(), called once per test: keep the cursor on sys.
    return (
        "import json\nimport sys\n"
        f"_r = json.loads({data})[getattr(sys, '_ri', 0)]; sys._ri = getattr(sys, '_ri', 0) + 1\n"
        "if not _r['ok']:\n    raise RuntimeError(_r['error'])\nsys.stdout.write(_r['out'])\n"
    )


def _last_record(stdout: str, marker: str) -> dict[str, Any] | None:
    """The last record printed after `marker`, or None."""
    i = stdout.rfind(marker)
    if i < 0:
        return None
    try:
        record = json.loads(stdout[i + len(marker) :].strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return record if isinstance(record, dict) else None


async def _run_with_sandboxfusion(files: dict[str, str], total_timeout: int) -> str:
    """Run RUN_CODE with `files` in SandboxFusion; return its stdout."""
    client = _get_sandboxfusion_client()
    _success, data = await client.run(code=RUN_CODE, files=files, timeout=total_timeout)
    return str((data.get("run_result") or {}).get("stdout") or "")


async def _run_with_modal(files: dict[str, str], total_timeout: int) -> str:
    """Run RUN_CODE with `files` in a Modal sandbox; return its stdout."""
    pool = _get_modal_pool()
    result = await pool.run_in_workdir(
        files={**files, "run.py": RUN_CODE},
        command=["python", "run.py"],
        timeout=total_timeout,
    )
    return result.stdout or ""


async def _grade_locally(
    records: list[dict[str, Any]], input_output: str, generation: str, timeout: int, marker: str
) -> tuple[bool, dict[str, Any]]:
    """Compare recorded outputs with the expected ones through run_test, in a local subprocess."""
    fn_name = json.loads(input_output).get("fn_name")
    payload = {
        "test_util": TEST_UTIL,
        "replay": _replay_program(records, fn_name, "class Solution" in generation),
        "input_output": input_output,
        "timeout": timeout,
        "marker": marker,
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        _GRADE_CODE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(json.dumps(payload).encode()), 120)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, {"error": "local grading timed out"}
    graded = _last_record(out.decode("utf-8", "replace"), marker)
    if graded is None:
        return False, {"error": "local grading failed"}
    results = graded.get("results") or []
    n = len(json.loads(input_output)["inputs"])
    passed = bool(results) and len(results) == n and all(results)
    return passed, {"results": results, "meta": graded.get("meta")}


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code using sandbox execution.

    The sandbox runs the program on the test inputs only and returns its outputs; the comparison with
    the expected outputs happens here, outside the sandbox. No valid run record (early exit, crash,
    timeout, garbage) is a failure, and so is a sample without test cases.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds
        backend: Sandbox backend to use (defaults to "sandboxfusion")

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    if not sample:
        return False, {"error": "no test cases"}

    try:
        input_output = postprocess_lcb_sample(sample)["input_output"]
        in_outs = json.loads(input_output)
        test_cnt = len(in_outs["inputs"])
        if test_cnt == 0:
            return False, {"error": "no test cases"}
        total_timeout = (timeout + 1) * test_cnt + 5

        # Inputs only: the expected outputs never enter the sandbox.
        inputs_only: dict[str, Any] = {"inputs": in_outs["inputs"]}
        if "fn_name" in in_outs:
            inputs_only["fn_name"] = in_outs["fn_name"]
        marker = f"{_MARKER}{uuid.uuid4().hex}:"
        files = {
            "payload.json": json.dumps(
                {
                    "program": generation,
                    "input_output": json.dumps(inputs_only),
                    "timeout": timeout,
                    "marker": marker,
                    "preamble": _preamble(),
                }
            ),
            "testing_util.py": TEST_UTIL,
        }

        use_backend = backend or SandboxBackend.SANDBOXFUSION
        if use_backend == SandboxBackend.MODAL:
            stdout = await _run_with_modal(files, total_timeout)
        elif use_backend == SandboxBackend.SANDBOXFUSION:
            stdout = await _run_with_sandboxfusion(files, total_timeout)
        else:
            raise ValueError(f"Invalid sandbox backend: {use_backend}")

        run = _last_record(stdout, marker)
        if run is None or run.get("status") != "ran" or not isinstance(run.get("records"), list):
            return False, {"error": "no valid run record", "status": (run or {}).get("status")}
        records = run["records"]
        if not all(isinstance(r, dict) and isinstance(r.get("ok"), bool) for r in records):
            return False, {"error": "malformed run record"}
        return await _grade_locally(records, input_output, generation, timeout, marker)

    except Exception as e:
        return False, {"error": str(e)}


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert TACO-style tests to LiveCodeBench format."""
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    n = max(len(inputs), len(outputs))

    test_cases: list[dict[str, Any]] = []
    for i in range(n):
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        if isinstance(out, list):
            out = out[0] if out else ""
        case: dict[str, Any] = {
            "input": inp,
            "output": out,
            "metadata": {},
        }
        if "fn_name" in tests:
            case["testtype"] = "functional"
            case["metadata"]["func_name"] = tests["fn_name"]
        else:
            case["testtype"] = "stdin_stdout"
        test_cases.append(case)

    return test_cases
