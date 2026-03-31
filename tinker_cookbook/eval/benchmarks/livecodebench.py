"""LiveCodeBench benchmark -- competitive programming with execution-based testing.

Dataset: ``livecodebench/code_generation_lite`` on HuggingFace.
Metric: Pass@1 -- fraction of problems where generated code passes all test cases.
Pattern: Single-turn generate + sandboxed execution (stdin/stdout).

LiveCodeBench continuously collects problems from LeetCode, AtCoder, and Codeforces.
Each problem has a description, starter code (optional), and JSON-encoded test cases
(stdin/stdout pairs).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import cast

import tinker
from datasets import Dataset

from tinker_cookbook.eval.benchmarks._common import (
    SandboxMixin,
    decode_response,
    extract_python_code,
    get_sandbox_factory,
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class LiveCodeBenchEnv(SandboxMixin, Env):
    """Single-turn env for one LiveCodeBench problem with sandboxed execution-based grading."""

    def __init__(
        self,
        prompt: str,
        question: str,
        test_cases_json: str,
        difficulty: str,
        renderer: Renderer,
        sandbox_factory,
        example_id: str = "",
    ):
        self.prompt = prompt
        self.question = question
        self.test_cases_json = test_cases_json
        self.difficulty = difficulty
        self.renderer = renderer
        self.sandbox_factory = sandbox_factory
        self.example_id = example_id

    async def initial_observation(self):
        # Create sandbox for code execution
        self.sandbox = await self.sandbox_factory()

        messages: list[Message] = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        response = decode_response(action, self.renderer)
        code = extract_python_code(response)

        passed = await self._check_solution(code)

        # Cleanup sandbox -- single-turn, episode is done
        await self.cleanup()

        return StepResult(
            reward=1.0 if passed else 0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"correct": float(passed), "difficulty": self.difficulty},
            logs={
                "example_id": self.example_id,
                "input": self.question[:200],
                "output": response[:500],
                "code": code[:500],
                "difficulty": self.difficulty,
            },
        )

    async def _check_solution(self, code: str) -> bool:
        """Run the solution against all test cases in the sandbox."""
        try:
            tests = json.loads(self.test_cases_json)
        except (json.JSONDecodeError, TypeError):
            return False

        inputs = tests.get("inputs", tests.get("input", []))
        outputs = tests.get("outputs", tests.get("output", []))

        if not inputs or not outputs or len(inputs) != len(outputs):
            return False

        # Write solution to sandbox once
        await self.sandbox.write_file("/tmp/solution.py", code)

        for inp, expected_out in zip(inputs, outputs):
            stdin_str = inp if isinstance(inp, str) else str(inp)
            expected_str = (
                expected_out.strip() if isinstance(expected_out, str) else str(expected_out).strip()
            )

            try:
                await self.sandbox.write_file("/tmp/stdin.txt", stdin_str)
                result = await self.sandbox.run_command(
                    "python3 /tmp/solution.py < /tmp/stdin.txt",
                    timeout=30,
                )
                if result.exit_code != 0 or result.stdout.strip() != expected_str:
                    return False
            except Exception:
                return False

        return True


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class LiveCodeBenchBenchmarkBuilder(BenchmarkBuilder):
    """LiveCodeBench: competitive programming with execution-based Pass@1."""

    name = "livecodebench"
    requires_sandbox = True

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        try:
            ds = cast(Dataset, load_benchmark_dataset("livecodebench/code_generation_lite"))
        except Exception:
            # The dataset has a legacy loading script that newer `datasets` versions reject.
            # Fall back to downloading the JSONL directly via huggingface_hub.
            try:
                from datasets import Dataset as HFDataset
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    "livecodebench/code_generation_lite",
                    "test6.jsonl",
                    repo_type="dataset",
                )
                ds = cast(Dataset, HFDataset.from_json(path))
                logger.info(f"Loaded LiveCodeBench from JSONL fallback ({len(ds)} examples)")
            except Exception as exc:
                logger.warning(f"Could not load LiveCodeBench dataset: {exc}.")
                return []

        ds = limit_dataset(ds, config.max_examples)

        sandbox_factory = get_sandbox_factory(config)

        envs = []
        for row in ds:
            question = row.get("question_content", row.get("question", ""))
            starter_code = row.get("starter_code", "")
            # Test cases may be under "input_output" (old format) or
            # "public_test_cases"/"private_test_cases" (JSONL format)
            test_cases_json = row.get("input_output", "")
            if not test_cases_json:
                public_tests = row.get("public_test_cases", [])
                # Normalize to {"inputs": [...], "outputs": [...]} format
                if isinstance(public_tests, str) and public_tests:
                    try:
                        public_tests = json.loads(public_tests)
                    except json.JSONDecodeError:
                        public_tests = []
                if isinstance(public_tests, list) and public_tests:
                    test_cases_json = json.dumps(
                        {
                            "inputs": [t["input"] for t in public_tests],
                            "outputs": [t["output"] for t in public_tests],
                        }
                    )
            difficulty = row.get("difficulty", "unknown")

            if not question or not test_cases_json:
                continue

            prompt_parts = [question]
            if starter_code:
                prompt_parts.append(f"\nStarter code:\n```python\n{starter_code}\n```")
            prompt_parts.append(
                "\nWrite a complete Python solution. Read input from stdin and write output to stdout. "
                "Provide your solution in a ```python code block."
            )
            prompt = "\n".join(prompt_parts)

            example_id = make_example_id("livecodebench", question)
            envs.append(
                LiveCodeBenchEnv(
                    prompt,
                    question,
                    test_cases_json,
                    difficulty,
                    renderer,
                    sandbox_factory,
                    example_id=example_id,
                )
            )
        return envs

    def aggregate(self, rewards: list[float], metrics_list: list[dict]) -> BenchmarkResult:
        """Aggregate with per-difficulty breakdown."""
        num_correct = sum(1 for r in rewards if r > 0)
        accuracy = num_correct / len(rewards) if rewards else 0.0

        difficulty_results: dict[str, list[bool]] = {}
        for r, m in zip(rewards, metrics_list):
            d = m.get("difficulty", "unknown")
            if isinstance(d, str):
                difficulty_results.setdefault(d, []).append(r > 0)

        metrics: dict[str, float] = {"livecodebench/pass_at_1": accuracy}
        for d, d_results in sorted(difficulty_results.items()):
            metrics[f"livecodebench/{d}/pass_at_1"] = (
                sum(d_results) / len(d_results) if d_results else 0.0
            )

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=metrics,
        )


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(LiveCodeBenchBenchmarkBuilder())
