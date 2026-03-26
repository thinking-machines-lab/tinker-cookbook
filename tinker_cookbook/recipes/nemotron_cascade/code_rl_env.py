"""
Code RL environment for Nemotron-Cascade-2 replication.

Trains on competitive programming problems (AtCoder, Codeforces) with binary
test execution rewards. The model generates a Python or C++ solution, which is
executed against test cases in a Modal sandbox.

Paper hyperparameters (Code RL stage):
  - Data: 3.5K hard competitive programming problems with test cases
  - Reward: Strict binary (pass all tests = 1, else = 0)
  - Batch: 128, Group: 16, LR: 3e-6, Max tokens: 118K, Top-p: 0.95
  - Steps: ~22
  - Async reward verification across 384 CPU cores
"""

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import chz
import tinker
from datasets import Dataset

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


def extract_code(response: str) -> tuple[str | None, str]:
    """Extract code from a model response.

    Looks for ```python or ```cpp fenced code blocks.
    Returns (code, language) where language is "python" or "cpp".
    Falls back to the last fenced code block if no language is specified.
    """
    # Try python first
    match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip(), "python"

    # Try C++ variants
    for lang_tag in ("cpp", "c++", "cxx"):
        match = re.search(rf'```{re.escape(lang_tag)}\s*\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1).strip(), "cpp"

    # Try C
    match = re.search(r'```c\s*\n(.*?)\n```', response, re.DOTALL)
    if match:
        return match.group(1).strip(), "cpp"

    # Fall back to any fenced code block
    matches = re.findall(r'```(?:\w*)\s*\n(.*?)\n```', response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        # Heuristic: if it has #include it's C++, otherwise assume Python
        if "#include" in code:
            return code, "cpp"
        return code, "python"

    return None, "unknown"


def build_execution_script(code: str, language: str, test_cases: list[dict]) -> str:
    """Build a bash script that runs the solution against all test cases.

    Supports two formats:
    - {"input": ..., "expected_output": ...} — stdin/stdout comparison
    - {"assertion": "assert func(x) == y"} — MBPP-style assertion tests

    The script exits 0 only if ALL test cases pass (strict binary reward).
    """
    import base64

    # Check if these are assertion-based tests (MBPP format)
    has_assertions = any("assertion" in tc for tc in test_cases)

    if has_assertions and language == "python":
        code_b64 = base64.b64encode(code.encode()).decode()
        assertions = [tc["assertion"] for tc in test_cases if "assertion" in tc]
        test_code = "\n".join(assertions)
        test_b64 = base64.b64encode(test_code.encode()).decode()
        return "\n".join([
            f"echo '{code_b64}' | base64 -d > /tmp/solution.py",
            f"echo '{test_b64}' | base64 -d > /tmp/tests.py",
            # Combine solution + tests and run
            "cat /tmp/solution.py /tmp/tests.py > /tmp/run.py",
            "timeout 30 python3 /tmp/run.py 2>/dev/null",
        ])

    if language == "python":
        code_b64 = base64.b64encode(code.encode()).decode()
        script_parts = [
            "set -e",
            f"echo '{code_b64}' | base64 -d > /tmp/solution.py",
            "PASS=0",
            f"TOTAL={len(test_cases)}",
        ]
        for i, tc in enumerate(test_cases):
            stdin_b64 = base64.b64encode(tc.get("input", "").encode()).decode()
            expected_b64 = base64.b64encode(tc.get("expected_output", "").strip().encode()).decode()
            script_parts.append(
                f"echo '{stdin_b64}' | base64 -d > /tmp/input_{i}.txt"
            )
            script_parts.append(
                f"echo '{expected_b64}' | base64 -d > /tmp/expected_{i}.txt"
            )
            script_parts.append(
                f"timeout 10 python3 /tmp/solution.py < /tmp/input_{i}.txt > /tmp/actual_{i}.txt 2>/dev/null"
            )
            script_parts.append(
                f'if diff <(sed "s/[[:space:]]*$//" /tmp/expected_{i}.txt) '
                f'<(sed "s/[[:space:]]*$//" /tmp/actual_{i}.txt) > /dev/null 2>&1; '
                f"then PASS=$((PASS + 1)); fi"
            )
        script_parts.append('echo "RESULT: $PASS/$TOTAL"')
        script_parts.append('test "$PASS" -eq "$TOTAL"')
        return "\n".join(script_parts)

    elif language == "cpp":
        code_b64 = base64.b64encode(code.encode()).decode()
        script_parts = [
            "set -e",
            f"echo '{code_b64}' | base64 -d > /tmp/solution.cpp",
            "g++ -O2 -std=c++17 -o /tmp/solution /tmp/solution.cpp 2>/dev/null || exit 1",
            "PASS=0",
            f"TOTAL={len(test_cases)}",
        ]
        for i, tc in enumerate(test_cases):
            stdin_b64 = base64.b64encode(tc["input"].encode()).decode()
            expected_b64 = base64.b64encode(tc["expected_output"].strip().encode()).decode()
            script_parts.append(
                f"echo '{stdin_b64}' | base64 -d > /tmp/input_{i}.txt"
            )
            script_parts.append(
                f"echo '{expected_b64}' | base64 -d > /tmp/expected_{i}.txt"
            )
            script_parts.append(
                f"timeout 10 /tmp/solution < /tmp/input_{i}.txt > /tmp/actual_{i}.txt 2>/dev/null"
            )
            script_parts.append(
                f'if diff <(sed "s/[[:space:]]*$//" /tmp/expected_{i}.txt) '
                f'<(sed "s/[[:space:]]*$//" /tmp/actual_{i}.txt) > /dev/null 2>&1; '
                f"then PASS=$((PASS + 1)); fi"
            )
        script_parts.append('echo "RESULT: $PASS/$TOTAL"')
        script_parts.append('test "$PASS" -eq "$TOTAL"')
        return "\n".join(script_parts)

    else:
        return "echo 'Unsupported language'; exit 1"


async def run_code_in_modal(
    code: str,
    language: str,
    test_cases: list[dict],
    timeout: int = 30,
) -> tuple[bool, str]:
    """Execute a code solution against test cases in a Modal sandbox.

    Returns (all_passed, details).
    """
    try:
        import modal
    except ImportError:
        return False, "Modal not installed"

    if not test_cases:
        return False, "No test cases"

    script = build_execution_script(code, language, test_cases)

    # Choose image based on language
    if language == "cpp":
        image = modal.Image.debian_slim().apt_install("g++")
    else:
        image = modal.Image.debian_slim().pip_install("numpy")

    try:
        app = await modal.App.lookup.aio("nemotron-cascade-code-rl", create_if_missing=True)
        sb = await modal.Sandbox.create.aio(
            "bash", "-c", script,
            image=image,
            app=app,
            timeout=timeout,
        )
        await sb.wait.aio()
        stdout = await sb.stdout.read.aio()
        stderr = await sb.stderr.read.aio()
        passed = sb.returncode == 0
        output = (stdout + "\n" + stderr)[-500:]
        return passed, output
    except Exception as e:
        return False, f"Modal error: {str(e)[:200]}"


class CodeRLEnv(Env):
    """Single-turn competitive programming environment.

    The model receives a problem description and generates a solution in
    Python or C++. The solution is executed against test cases in a Modal
    sandbox. Reward is strict binary: 1 if all tests pass, 0 otherwise.
    """

    def __init__(
        self,
        problem_id: str,
        problem_description: str,
        test_cases: list[dict],
        renderer: renderers.Renderer,
        use_modal: bool = True,
    ):
        self.problem_id = problem_id
        self.problem_description = problem_description
        self.test_cases = test_cases
        self.renderer = renderer
        self.use_modal = use_modal

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Show sample test cases in the prompt (first 2) so the model
        # understands the I/O format; hold back the rest for evaluation.
        sample_tests = self.test_cases[:2]
        examples_str = ""
        for i, tc in enumerate(sample_tests):
            examples_str += f"\n**Example {i + 1}:**\n"
            if "assertion" in tc:
                examples_str += f"Test: `{tc['assertion']}`\n"
            else:
                examples_str += f"Input:\n```\n{tc.get('input', '')}\n```\n"
                examples_str += f"Output:\n```\n{tc.get('expected_output', '')}\n```\n"

        # Adapt prompt based on test type
        has_assertions = any("assertion" in tc for tc in self.test_cases)
        if has_assertions:
            task_desc = "Write a Python function that satisfies the following requirements."
            instructions = "Provide your solution as a Python function in a fenced code block."
        else:
            task_desc = "Solve the following competitive programming problem. Read input from stdin and write output to stdout."
            instructions = "Provide your solution in a single fenced code block using "

        prompt = (
            f"{task_desc}\n\n"
            f"## Problem\n{self.problem_description}\n"
            f"{examples_str}\n"
            f"## Instructions\n"
            f"{instructions}"
        )
        messages: list[Message] = [{"role": "user", "content": prompt}]
        return self.renderer.build_generation_prompt(messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

        stop_reason = (extra or {}).get("stop_reason")
        if stop_reason == "length":
            reward = 0.0
            details = "overlong"
            has_code = False
            language = "unknown"
        else:
            code, language = extract_code(content)
            has_code = code is not None

            if not code:
                reward = 0.0
                details = "no code extracted"
            elif self.use_modal and self.test_cases:
                passed, details = await run_code_in_modal(
                    code, language, self.test_cases,
                )
                reward = 1.0 if passed else 0.0
            else:
                # Fallback: reward for producing valid-looking code
                reward = 0.5 if code else 0.0
                details = "code produced (no execution)"

        # Logging
        with logtree.scope_header("Problem"):
            logtree.log_text(f"ID: {self.problem_id}")
            logtree.log_text(f"Description: {self.problem_description[:500]}...")
            logtree.log_text(f"Test cases: {len(self.test_cases)}")
        with logtree.scope_header("Response"):
            logtree.log_text(content[:1000])
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reward": f"{reward:.1f}",
                    "language": language,
                    "has_code": str(has_code),
                    "details": str(details)[:200],
                },
                caption="Code RL reward",
            )

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "correct": reward,
                "has_code": float(has_code),
                "overlong": float(stop_reason == "length") if stop_reason else 0.0,
            },
        )


@dataclass(frozen=True)
class CodeRLGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], CodeRLEnv]
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def logging_tags(self) -> list[str]:
        return ["code_rl"]


class CodeRLDataset(RLDataset):
    """Competitive programming dataset for Code RL.

    By default, loads from the LiveCodeBench code generation dataset on
    HuggingFace. Also supports loading from a local JSONL file where each
    line has: problem_id, problem_description, test_cases (list of
    {input, expected_output} dicts).
    """

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        data_path: str | None = None,
        use_modal: bool = True,
        seed: int = 0,
    ):
        if data_path:
            logger.info(f"Loading Code RL data from {data_path}")
            import json
            rows = []
            with open(data_path) as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("test_cases") and row.get("problem_description"):
                        rows.append(row)
            self.ds = Dataset.from_list(rows).shuffle(seed=seed)
        else:
            logger.info("Loading MBPP code data from HuggingFace...")
            from datasets import load_dataset
            try:
                ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            except Exception:
                ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
            ds = cast(Dataset, ds)
            self.ds = ds.shuffle(seed=seed)

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.use_modal = use_modal
        self.data_path = data_path
        logger.info(
            f"Code RL dataset: {len(self.ds)} problems, "
            f"batch_size={batch_size}, group_size={group_size}"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _parse_test_cases(self, row: dict) -> list[dict]:
        """Parse test cases from various dataset formats.

        Supports:
        - Direct list of {input, expected_output} dicts (local JSONL)
        - LiveCodeBench format (input_output field with JSON-encoded cases)
        - MBPP format (test_list with assertion strings)
        """
        import json

        # Direct format from local JSONL
        if "test_cases" in row and isinstance(row["test_cases"], list):
            return row["test_cases"]

        # MBPP format: test_list contains assertion strings
        if "test_list" in row and isinstance(row["test_list"], list):
            return [{"assertion": test} for test in row["test_list"]]

        # LiveCodeBench format: input_output is a JSON string
        if "input_output" in row:
            io_data = row["input_output"]
            if isinstance(io_data, str):
                try:
                    io_data = json.loads(io_data)
                except json.JSONDecodeError:
                    return []

            if isinstance(io_data, dict):
                inputs = io_data.get("inputs", [])
                outputs = io_data.get("outputs", [])
                return [
                    {"input": str(inp), "expected_output": str(out)}
                    for inp, out in zip(inputs, outputs)
                ]
            elif isinstance(io_data, list):
                return [
                    {"input": str(tc.get("input", "")), "expected_output": str(tc.get("output", ""))}
                    for tc in io_data
                    if isinstance(tc, dict)
                ]

        return []

    def _make_builder(self, row: dict) -> CodeRLGroupBuilder | None:
        try:
            # Handle different dataset formats
            if self.data_path:
                # Local JSONL format
                problem_id = str(row.get("problem_id", "unknown"))
                problem_description = row["problem_description"]
                test_cases = self._parse_test_cases(row)
            else:
                # MBPP / LiveCodeBench format
                problem_id = str(row.get("task_id", row.get("question_id", "unknown")))
                problem_description = row.get("prompt", row.get("question_content", ""))
                test_cases = self._parse_test_cases(row)

            if not problem_description or not test_cases:
                return None

            return CodeRLGroupBuilder(
                env_thunk=lambda pid=problem_id, pd=problem_description, tc=test_cases: CodeRLEnv(
                    problem_id=pid,
                    problem_description=pd,
                    test_cases=tc,
                    renderer=self.renderer,
                    use_modal=self.use_modal,
                ),
                num_envs=self.group_size,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Code RL row: {e}")
            return None


@chz.chz
class CodeRLDatasetBuilder(RLDatasetBuilder):
    """Builder for the Code RL competitive programming environment.

    Paper hyperparameters:
      - batch_size: 128
      - group_size: 16
      - max_tokens: 118K
      - LR: 3e-6
      - top_p: 0.95
      - ~22 steps
    """

    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    data_path: str | None = None
    use_modal: bool = True
    seed: int = 0

    async def __call__(self) -> tuple[CodeRLDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        return CodeRLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            data_path=self.data_path,
            use_modal=self.use_modal,
            seed=self.seed,
        ), None
