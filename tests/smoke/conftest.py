"""Shared fixtures for recipe smoke tests."""

import os
import re
import signal
import subprocess
import tempfile

import pytest

# Timeout for each recipe (seconds). Override with SMOKE_TEST_TIMEOUT env var.
DEFAULT_TIMEOUT = int(os.environ.get("SMOKE_TEST_TIMEOUT", "300"))

# Patterns that indicate step 1 has started (meaning step 0 completed)
STEP_1_PATTERNS = re.compile(r"Step 1|Sampling batch 1|batch_idx=1")

# Patterns that indicate step 0 ran (for recipes that complete quickly)
STEP_0_PATTERNS = re.compile(r"Step 0|Sampling batch 0|batch_idx=0|step.*=.*0")


def _skip_if_no_api_key():
    if not os.environ.get("TINKER_API_KEY"):
        pytest.skip("TINKER_API_KEY not set")


def run_recipe(module: str, args: list[str] | None = None, timeout: int = DEFAULT_TIMEOUT):
    """Run a recipe module until step 1 is detected, then kill it.

    Args:
        module: Python module path (e.g., "tinker_cookbook.recipes.chat_sl.train")
        args: CLI arguments to pass to the module
        timeout: Maximum seconds to wait for step 1
    """
    _skip_if_no_api_key()

    cmd = ["uv", "run", "python", "-m", module] + (args or [])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as logfile:
        logpath = logfile.name

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=open(logpath, "w"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        import time

        elapsed = 0
        step1_seen = False
        while proc.poll() is None and elapsed < timeout:
            time.sleep(2)
            elapsed += 2
            try:
                with open(logpath) as f:
                    content = f.read()
                if STEP_1_PATTERNS.search(content):
                    step1_seen = True
                    break
            except OSError:
                pass

        # Kill the process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        proc.wait(timeout=10)

        # Read final log
        with open(logpath) as f:
            content = f.read()

        if step1_seen:
            return  # Success

        # Process exited on its own — check if step 0 ran
        if proc.returncode is not None and STEP_0_PATTERNS.search(content):
            return  # Success — recipe completed quickly

        # Failure
        last_lines = "\n".join(content.splitlines()[-30:])
        pytest.fail(
            f"Recipe {module} did not reach step 1 within {timeout}s "
            f"(exit code: {proc.returncode})\n\nLast 30 lines:\n{last_lines}"
        )
    finally:
        os.unlink(logpath)
