"""Shared helpers for recipe smoke tests."""

import os
import select
import subprocess
import time

import pytest

# Timeout for each recipe (seconds). Override with SMOKE_TEST_TIMEOUT env var.
DEFAULT_TIMEOUT = int(os.environ.get("SMOKE_TEST_TIMEOUT", "1800"))

# Default number of training steps for smoke tests.
DEFAULT_MAX_STEPS = 2


def run_recipe(
    module: str,
    args: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Run a recipe module for a limited number of steps and verify clean exit.

    Passes max_steps to the recipe so it exits naturally after N training steps.
    Output is streamed to stdout in real time for debuggability in CI.

    Args:
        module: Python module path (e.g., "tinker_cookbook.recipes.chat_sl.train")
        args: CLI arguments to pass to the module
        timeout: Maximum seconds to wait for the recipe to complete
        max_steps: Number of training steps to run (passed as CLI arg)
    """
    cmd = ["uv", "run", "python", "-m", module] + (args or []) + [f"max_steps={max_steps}"]
    print(f"\n>>> {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    output_lines: list[str] = []
    start_time = time.monotonic()

    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                proc.terminate()
                proc.wait(timeout=10)
                last_lines = "\n".join(output_lines[-30:])
                pytest.fail(
                    f"Recipe {module} did not complete within {timeout}s "
                    f"(exit code: {proc.returncode})\n\nLast 30 lines:\n{last_lines}"
                )

            # Check if process exited
            if proc.poll() is not None:
                # Drain remaining output
                for line in proc.stdout:
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                    output_lines.append(decoded)
                    print(decoded, flush=True)
                break

            # Wait up to 5s for output, then re-check timeout
            ready, _, _ = select.select([fd], [], [], 5.0)
            if not ready:
                continue

            line = proc.stdout.readline()
            if line:
                decoded = line.decode("utf-8", errors="replace").rstrip("\n")
                output_lines.append(decoded)
                print(decoded, flush=True)
    except Exception:
        # Ensure cleanup on unexpected errors
        proc.kill()
        proc.wait(timeout=10)
        raise

    elapsed = time.monotonic() - start_time

    if proc.returncode == 0:
        print(f"\n>>> PASSED: recipe completed cleanly in {elapsed:.0f}s", flush=True)
        return

    # Non-zero exit code
    last_lines = "\n".join(output_lines[-30:])
    pytest.fail(
        f"Recipe {module} failed with exit code {proc.returncode}\n\nLast 30 lines:\n{last_lines}"
    )
