"""
Code grading utilities for RL training.

Uses Sandbox Fusion for safe code execution.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any

import aiohttp

from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_UTIL, TEST_CODE

# Sandbox configuration
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
SANDBOX_MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))

# Sandbox session management
_SANDBOX_SESSION: aiohttp.ClientSession | None = None
_SANDBOX_SESSION_LOCK = asyncio.Lock()


async def _get_sandbox_session() -> aiohttp.ClientSession:
    """
    Get or create a shared aiohttp session with connection limits.

    The TCPConnector limits concurrent connections to SANDBOX_MAX_CONCURRENCY.
    When all connections are busy, additional requests automatically wait in a queue
    until a connection becomes available.
    """
    global _SANDBOX_SESSION

    async with _SANDBOX_SESSION_LOCK:
        if _SANDBOX_SESSION is None or _SANDBOX_SESSION.closed:
            connector = aiohttp.TCPConnector(
                limit=SANDBOX_MAX_CONCURRENCY,
                limit_per_host=SANDBOX_MAX_CONCURRENCY,
            )
            timeout = aiohttp.ClientTimeout(total=6000)
            _SANDBOX_SESSION = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return _SANDBOX_SESSION


def _b64encode(content: str) -> str:
    return base64.b64encode(content.encode("utf-8")).decode("utf-8")


def extract_code_from_model(model_response: str) -> str | None:
    """
    Extract the last fenced code block from a model response.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
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


async def sandbox_check_correctness(
    sample: list[dict[str, Any]], generation: str, timeout: int = 6
) -> tuple[bool, dict[str, Any]]:
    """Check correctness of generated code using sandbox execution."""
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)

    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        test_code = TEST_CODE % {"timeout": timeout}
        asset = {
            "test_cases.txt": _b64encode(json.dumps(test_cases)),
            "code.py": _b64encode(generation),
            "testing_util.py": _b64encode(TEST_UTIL),
        }

        payload = {
            "code": test_code,
            "language": "python",
            "run_timeout": total_timeout,
            "files": asset,
        }

        session = await _get_sandbox_session()
        async with session.post(SANDBOX_URL, json=payload) as result:
            if result.status != 200:
                raise Exception(
                    f"Sandbox API responded with code {result.status}: {await result.text()}"
                )
            resp = await result.json()
            if resp.get("status") == "SandboxError":
                raise Exception(f"Sandbox responded with error: {resp.get('message')}")

            # Check if all tests passed
            all_passed = resp.get("status") == "Success"
            return all_passed, resp
    except Exception as e:
        return False, {"error": str(e)}


def taco_to_lcb_format(tests: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert TACO-style tests to LiveCodeBench format.
    """
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
