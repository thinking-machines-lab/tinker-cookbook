"""MBPP benchmark -- Mostly Basic Python Programming via code execution.

Dataset: ``google-research-datasets/mbpp`` (sanitized) on HuggingFace.
Metric: Pass@1 -- fraction of problems where generated code passes assertion tests.
Pattern: Single-turn ``MessageEnv`` (with sandbox) + ``EnvFromMessageEnv`` wrapper.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    SandboxMixin,
    build_messages,
    extract_python_code,
    get_sandbox_factory,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MessageEnv implementation
# ---------------------------------------------------------------------------


class MBPPMessageEnv(SandboxMixin, MessageEnv):
    """Single-turn message env for one MBPP problem with sandboxed execution-based grading.

    The sandbox is created in ``initial_observation`` and used in ``step``
    to run the generated code against assertion tests. Wrapped by
    ``EnvFromMessageEnv`` which handles rendering and thinking extraction.
    """

    def __init__(
        self,
        prompt: str,
        task_prompt: str,
        test_list: list[str],
        sandbox_factory,
        example_id: str = "",
        system_prompt: str | None = None,
    ):
        self.prompt = prompt
        self.task_prompt = task_prompt
        self.test_list = test_list
        self.sandbox_factory = sandbox_factory
        self.example_id = example_id
        self.system_prompt = system_prompt

    async def initial_observation(self) -> list[Message]:
        # Create sandbox for code execution
        self.sandbox = await self.sandbox_factory()
        return build_messages(self.prompt, self.system_prompt)

    async def step(self, message: Message) -> MessageStepResult:
        assert self.sandbox is not None
        response = get_text_content(message)
        code = extract_python_code(response)

        # Build test code: solution + assertion tests
        test_code = code + "\n\n" + "\n".join(self.test_list)

        passed = False
        try:
            await self.sandbox.write_file("/tmp/test_code.py", test_code)
            result = await self.sandbox.run_command("python3 /tmp/test_code.py", timeout=15)
            passed = result.exit_code == 0
        except Exception:
            passed = False

        # Cleanup sandbox -- single-turn, episode is done
        await self.cleanup()

        return MessageStepResult(
            reward=1.0 if passed else 0.0,
            episode_done=True,
            next_messages=[],
            metrics={"correct": float(passed)},
            logs={
                "example_id": self.example_id,
                "input": self.task_prompt[:200],
                "output": response[:500],
                "code": code[:500],
            },
        )


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class MBPPBenchmarkBuilder(BenchmarkBuilder):
    """MBPP: Mostly Basic Python Programming with execution-based testing."""

    name = "mbpp"
    requires_sandbox = True

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(
            Dataset, load_benchmark_dataset("google-research-datasets/mbpp", name="sanitized")
        )

        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = get_sandbox_factory(config)

        envs: list[Env] = []
        for row in ds:
            row = dict(row)
            task_prompt = row.get("prompt", row.get("text", ""))
            test_list = row.get("test_list", [])
            if not task_prompt or not test_list:
                continue

            example_test = test_list[0] if test_list else ""
            prompt = (
                f"{task_prompt}\n\n"
                f"Example test: `{example_test}`\n\n"
                "Write a Python function that satisfies the requirements. "
                "Provide ONLY the function definition in a ```python code block."
            )
            task_id = row.get("task_id")
            if task_id is not None:
                example_id = f"mbpp_{task_id}"
            else:
                example_id = make_example_id("mbpp", task_prompt)
            msg_env = MBPPMessageEnv(
                prompt,
                task_prompt,
                test_list,
                sandbox_factory,
                example_id=example_id,
                system_prompt=config.system_prompt,
            )
            envs.append(
                EnvFromMessageEnv(
                    renderer=renderer,
                    message_env=msg_env,
                    failed_parse_reward=0.0,
                )
            )
        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(MBPPBenchmarkBuilder())
