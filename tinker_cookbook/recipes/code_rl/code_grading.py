"""
Code grading utilities for RL training.

Supports three execution backends:
- sandboxfusion: Local Docker-based sandbox (default)
- modal: Cloud-based Modal sandbox
- hyperbrowser: Cloud-based Hyperbrowser sandbox
"""

from __future__ import annotations

import json
import re
from typing import Any

from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL
from tinker_cookbook.sandbox import SandboxBackend, SandboxFusionClient

# Global sandbox backend clients (lazily initialized)
_sandboxfusion_client: SandboxFusionClient | None = None
_modal_pool: Any = None  # ModalSandboxPool, but avoid import at module level
_hyperbrowser_pool: Any = None  # HyperbrowserSandboxPool, likewise


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


def _get_hyperbrowser_pool():
    """Get or create the Hyperbrowser sandbox pool."""
    global _hyperbrowser_pool
    if _hyperbrowser_pool is None:
        from tinker_cookbook.sandbox.hyperbrowser_sandbox import (
            HyperbrowserImage,
            HyperbrowserSandboxPool,
        )

        # Layers on a base image are applied at sandbox startup, so this needs
        # no local Docker daemon.
        image = HyperbrowserImage.base("python").pip_install("numpy")
        _hyperbrowser_pool = HyperbrowserSandboxPool(image=image)
    return _hyperbrowser_pool


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


async def _check_with_pool(
    pool: Any,
    test_cases: dict[str, str],
    generation: str,
    timeout: int,
    total_timeout: int,
) -> tuple[bool, dict[str, Any]]:
    """Execute tests in a cloud sandbox pool (Modal or Hyperbrowser).

    Both pools expose the same ``run_in_workdir`` contract, so the grading logic
    is shared.
    """
    result = await pool.run_in_workdir(
        files={
            "test_cases.txt": json.dumps(test_cases),
            "code.py": generation,
            "testing_util.py": TEST_UTIL,
            "run.py": TEST_CODE % {"timeout": timeout},
        },
        command=["python", "run.py"],
        timeout=total_timeout,
    )
    return result.exit_code == 0, {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def sandbox_check_correctness(
    sample: list[dict[str, Any]],
    generation: str,
    timeout: int = 6,
    backend: SandboxBackend | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Check correctness of generated code using sandbox execution.

    Args:
        sample: List of test cases in LiveCodeBench format
        generation: Generated code to test
        timeout: Per-test timeout in seconds
        backend: Sandbox backend to use — "sandboxfusion" (default), "modal",
            or "hyperbrowser"

    Returns:
        Tuple of (all_passed: bool, details: dict)
    """
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)
    use_backend = backend or SandboxBackend.SANDBOXFUSION

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        if use_backend == SandboxBackend.MODAL:
            return await _check_with_pool(
                _get_modal_pool(), test_cases, generation, timeout, total_timeout
            )
        elif use_backend == SandboxBackend.HYPERBROWSER:
            return await _check_with_pool(
                _get_hyperbrowser_pool(), test_cases, generation, timeout, total_timeout
            )
        elif use_backend == SandboxBackend.SANDBOXFUSION:
            return await _check_with_sandboxfusion(test_cases, generation, timeout, total_timeout)
        else:
            raise ValueError(f"Invalid sandbox backend: {use_backend}")

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
