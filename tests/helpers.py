"""Shared helpers for recipe smoke tests."""

import os
import re
import signal
import subprocess
import time

import pytest

# Timeout for each recipe (seconds). Override with SMOKE_TEST_TIMEOUT env var.
DEFAULT_TIMEOUT = int(os.environ.get("SMOKE_TEST_TIMEOUT", "900"))

# Patterns that indicate step 1 has started (meaning step 0 completed)
STEP_1_PATTERNS = re.compile(r"Step 1|Sampling batch 1|batch_idx=1")

# Patterns that indicate step 0 ran (for recipes that complete quickly)
STEP_0_PATTERNS = re.compile(r"Step 0|Sampling batch 0|batch_idx=0|step.*=.*0")


def run_recipe(module: str, args: list[str] | None = None, timeout: int = DEFAULT_TIMEOUT):
    """Run a recipe module until step 1 is detected, then kill it.

    Output is streamed to stdout in real time for debuggability in CI.

    Args:
        module: Python module path (e.g., "tinker_cookbook.recipes.chat_sl.train")
        args: CLI arguments to pass to the module
        timeout: Maximum seconds to wait for step 1
    """
    cmd = ["uv", "run", "python", "-m", module] + (args or [])
    print(f"\n>>> {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        process_group=0,
    )

    output_lines: list[str] = []
    step1_seen = False
    start_time = time.monotonic()

    try:
        assert proc.stdout is not None
        while True:
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                break

            # Check if process exited
            if proc.poll() is not None:
                # Drain remaining output
                for line in proc.stdout:
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                    output_lines.append(decoded)
                    print(decoded, flush=True)
                break

            # Read available output (non-blocking via readline with implicit buffer)
            line = proc.stdout.readline()
            if line:
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                output_lines.append(decoded)
                print(decoded, flush=True)
                if STEP_1_PATTERNS.search(decoded):
                    step1_seen = True
                    break
    finally:
        # Kill the process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        proc.wait(timeout=10)

    content = "\n".join(output_lines)
    elapsed = time.monotonic() - start_time

    if step1_seen:
        print(f"\n>>> PASSED: step 1 reached after {elapsed:.0f}s", flush=True)
        return  # Success

    # Process exited on its own — check if step 0 ran
    if proc.returncode is not None and STEP_0_PATTERNS.search(content):
        print(f"\n>>> PASSED: recipe completed (step 0 seen, exit={proc.returncode})", flush=True)
        return  # Success — recipe completed quickly

    # Failure
    last_lines = "\n".join(output_lines[-30:])
    pytest.fail(
        f"Recipe {module} did not reach step 1 within {timeout}s "
        f"(exit code: {proc.returncode})\n\nLast 30 lines:\n{last_lines}"
    )
