"""
Code grading utilities for RL training.

Supports two execution backends:
- sandboxfusion: Local Docker-based sandbox (default)
- modal: Cloud-based Modal sandbox

Set CODE_SANDBOX_BACKEND environment variable to switch backends.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from typing import Any, Literal

from tinker_cookbook.execution import SandboxFusionClient
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL

# Backend selection
BackendType = Literal["sandboxfusion", "modal"]
SANDBOX_BACKEND: BackendType = os.getenv("CODE_SANDBOX_BACKEND", "sandboxfusion")  # type: ignore[assignment]

# Global clients (lazily initialized)
_sandboxfusion_client: SandboxFusionClient | None = None
_modal_sandbox: Any = None  # ModalSandbox, but avoid import at module level


def _get_sandboxfusion_client() -> SandboxFusionClient:
    """Get or create the SandboxFusion client."""
    global _sandboxfusion_client
    if _sandboxfusion_client is None:
        _sandboxfusion_client = SandboxFusionClient()
    return _sandboxfusion_client


def _get_modal_sandbox():
    """Get or create the Modal sandbox."""
    global _modal_sandbox
    if _modal_sandbox is None:
        from tinker_cookbook.execution.modal_sandbox import ModalSandbox

        _modal_sandbox = ModalSandbox()
    return _modal_sandbox


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
            raise AssertionError(
                f"Function name missing in metadata: {metadata}. Sample: {sample}"
            )
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


async def _check_with_sandboxfusion(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using SandboxFusion backend."""
    client = _get_sandboxfusion_client()

    return await client.run(
        code=TEST_CODE % {"timeout": timeout},
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
        },
        timeout=total_timeout,
    )


def _check_with_modal_sync(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using Modal sandbox backend (sync implementation)."""
    sandbox = _get_modal_sandbox()

    # Use unique workdir to avoid concurrent call collisions
    workdir = f"/workspace/lcb/{uuid.uuid4().hex[:12]}"

    try:
        sandbox.mkdir(workdir)

        # Write files to sandbox
        sandbox.write_file(f"{workdir}/test_cases.txt", json.dumps(test_cases))
        sandbox.write_file(f"{workdir}/code.py", generation)
        sandbox.write_file(f"{workdir}/testing_util.py", TEST_UTIL)
        sandbox.write_file(f"{workdir}/run.py", TEST_CODE % {"timeout": timeout})

        # Execute test runner
        exit_code, stdout, stderr = sandbox.exec(
            "python", "run.py", workdir=workdir, timeout=total_timeout
        )

        success = exit_code == 0
        details = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
        return success, details
    finally:
        # TODO(tgriggs): Improve the modal sandbox cleanup and error handling
        # Clean up workdir
        try:
            sandbox.cleanup(workdir)
        except Exception:
            pass  # Best effort cleanup


async def _check_with_modal(
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests using Modal sandbox backend (async wrapper)."""
    # Run sync Modal operations in thread pool to avoid blocking event loop
    return await asyncio.to_thread(
        _check_with_modal_sync, test_cases, generation, timeout, total_timeout
    )


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: BackendType | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code using sandbox execution.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds
        backend: Override default backend ("sandboxfusion" or "modal")

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        # Select backend
        use_backend = backend or SANDBOX_BACKEND

        if use_backend == "modal":
            return await _check_with_modal(
                test_cases, generation, timeout, total_timeout
            )
        else:
            return await _check_with_sandboxfusion(
                test_cases, generation, timeout, total_timeout
            )

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
            out = out[0]
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


async def close_clients() -> None:
    """Close all sandbox clients. Call at end of training."""
    global _sandboxfusion_client, _modal_sandbox

    if _sandboxfusion_client is not None:
        await _sandboxfusion_client.close()
        _sandboxfusion_client = None

    if _modal_sandbox is not None:
        _modal_sandbox.close()
        _modal_sandbox = None
