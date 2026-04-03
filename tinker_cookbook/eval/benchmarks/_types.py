"""Core types for the benchmark evaluation framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import tinker

from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env


@dataclass
class BenchmarkConfig:
    """Runtime configuration for benchmark evaluation.

    Controls concurrency, timeouts, generation parameters, storage, and
    optional customization hooks (system prompt, custom grading).

    Example::

        # Basic usage — run all examples with defaults
        config = BenchmarkConfig()

        # Production eval with storage and higher timeout for thinking models
        config = BenchmarkConfig(
            save_dir="evals/checkpoint_500",
            timeout_seconds=1800,
            max_tokens=65536,
        )

        # Custom grading — override built-in answer extraction
        config = BenchmarkConfig(
            grade_fn=lambda response, logs: 1.0 if logs["expected"] in response else 0.0,
        )

        # Pass@k evaluation — run each example 4 times
        config = BenchmarkConfig(num_samples=4, save_dir="evals/pass_at_k")
    """

    # Limits
    max_examples: int | None = None
    """Maximum number of examples to evaluate. ``None`` = all."""

    # Concurrency
    concurrency: int = 64
    """Maximum concurrent rollouts for single-turn benchmarks."""
    agent_concurrency: int = 8
    """Maximum concurrent rollouts for multi-turn/sandbox benchmarks (heavier)."""

    # Timeouts
    timeout_seconds: float = 300
    """Per-example timeout in seconds. Default 5 min for single-turn.
    Multi-turn/sandbox benchmarks should increase this (e.g., 1800 for 30 min).
    Timed-out examples are recorded as failures with ``error="timeout"``
    and ``reward=0``. They count toward ``num_errors`` in BenchmarkResult,
    not silently dropped."""

    # Context management (multi-turn)
    max_trajectory_tokens: int | None = None
    """Maximum total tokens in the conversation before terminating the episode.
    When set, multi-turn benchmarks (terminal_bench, swe_bench) will end the
    episode gracefully instead of crashing with a context overflow error.
    Should be set below the model's context window (e.g., 60000 for 65K context)."""

    max_generation_tokens: int | None = None
    """Maximum tokens per generation step. Used with ``max_trajectory_tokens``
    to ensure prompt + generation fits in the context window."""

    # Generation
    max_tokens: int = 32768
    """Maximum tokens per model generation. The runner will automatically
    cap this so that prompt + max_tokens fits the model's context window.
    For thinking models that generate long reasoning chains, increase this
    (MathArena uses 64K for AIME)."""
    temperature: float = 0.6
    """Sampling temperature for model generation."""
    context_window: int | None = None
    """If set, dynamically cap max_tokens per request to fit in context."""

    # Storage
    save_dir: str | None = None
    """Directory for saving trajectories and results. ``None`` = no saving."""

    # Pass@k sampling
    num_samples: int = 1
    """Number of samples per example for pass@k evaluation. When > 1, each
    example is run ``num_samples`` times and pass@k metrics are computed."""

    # Judge model (for benchmarks that need LLM-as-judge)
    judge_sampling_client: tinker.SamplingClient | None = None
    """Sampling client for LLM judge. Required for arena_hard, swe_bench, etc."""
    judge_renderer: Renderer | None = None
    """Renderer for the judge model. If None, uses the candidate renderer."""

    # Sandbox (for benchmarks that execute code: mbpp, livecodebench, terminal_bench, swe_bench)
    sandbox_factory: Callable[..., Any] | None = None
    """Async callable returning a :class:`~tinker_cookbook.sandbox.SandboxInterface`.
    Called once per eval example to create an isolated execution environment.
    When ``None``, defaults to Modal (requires ``pip install 'tinker-cookbook[modal]'``).

    Example::

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
        config = BenchmarkConfig(sandbox_factory=ModalSandbox.create)
    """

    # Customization hooks
    system_prompt: str | None = None
    """System prompt to prepend to benchmark examples. All stable benchmarks
    support this. Single-turn benchmarks prepend it via ``build_messages()``,
    multi-turn benchmarks (terminal_bench, swe_bench) use it as the system
    message in the agent conversation.

    Example::

        config = BenchmarkConfig(
            system_prompt="You are a helpful math assistant. Always put your final answer in \\\\boxed{}."
        )
    """

    grade_fn: Callable[[str, dict], float] | None = None
    """Custom grading function: ``(response, logs) -> reward``.
    If set, overrides the benchmark's built-in grading logic.
    ``response`` is the decoded model output (thinking stripped).
    ``logs`` contains benchmark-specific fields like ``expected``, ``example_id``.

    Example::

        def my_grader(response: str, logs: dict) -> float:
            expected = logs["expected"]
            # Custom extraction logic
            extracted = my_extract(response)
            return 1.0 if extracted == expected else 0.0

        config = BenchmarkConfig(grade_fn=my_grader)
    """

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be > 0, got {self.concurrency}")
        if self.agent_concurrency <= 0:
            raise ValueError(f"agent_concurrency must be > 0, got {self.agent_concurrency}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError(f"max_examples must be > 0 when set, got {self.max_examples}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError(f"context_window must be > 0 when set, got {self.context_window}")


@dataclass
class StoredTurn:
    """A single turn in a stored trajectory — human-readable for visualization."""

    role: str
    """``"user"``, ``"assistant"``, ``"environment"``, ``"grader"``."""
    content: str
    """Decoded text content of this turn."""
    token_count: int = 0
    """Number of tokens in this turn."""
    metadata: dict = field(default_factory=dict)
    """Arbitrary per-turn data (timing, tool calls, etc.)."""


@dataclass
class StoredTrajectory:
    """A complete eval trajectory stored for visualization and analysis.

    Written to ``trajectories.jsonl`` — one per line, self-contained.
    """

    idx: int
    """Index in the benchmark's example list (for resumability)."""
    benchmark: str
    """Benchmark name."""
    example_id: str = ""
    """Stable identifier for this example across runs. Deterministic from the
    dataset (e.g., question hash, dataset-provided ID). Used for cross-run
    comparison — the same question always gets the same example_id regardless
    of which checkpoint is being evaluated. If empty, falls back to ``idx``."""
    turns: list[StoredTurn] = field(default_factory=list)
    """Full conversation history (decoded text, not tokens)."""
    reward: float = 0.0
    """Total reward (sum of per-step rewards)."""
    metrics: dict = field(default_factory=dict)
    """Per-example metrics from Env.step() (e.g., correct, overlong)."""
    logs: dict = field(default_factory=dict)
    """Per-example logs from Env.step() (e.g., input, expected, extracted)."""
    error: str | None = None
    """Error message if this example failed."""
    time_seconds: float = 0.0
    """Wall time for this example."""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "idx": self.idx,
            "benchmark": self.benchmark,
            "example_id": self.example_id,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "token_count": t.token_count,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "reward": self.reward,
            "metrics": self.metrics,
            "logs": self.logs,
            "error": self.error,
            "time_seconds": self.time_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StoredTrajectory:
        """Deserialize from a dict."""
        return cls(
            idx=d["idx"],
            benchmark=d["benchmark"],
            example_id=d.get("example_id", ""),
            turns=[StoredTurn(**t) for t in d.get("turns", [])],
            reward=d.get("reward", 0.0),
            metrics=d.get("metrics", {}),
            logs=d.get("logs", {}),
            error=d.get("error"),
            time_seconds=d.get("time_seconds", 0.0),
        )


@dataclass
class BenchmarkResult:
    """Aggregated result from running a benchmark.

    Attributes:
        name: Benchmark name (e.g. ``"gsm8k"``).
        score: Primary metric normalized to 0–1.
        num_examples: Total examples evaluated (excluding errors).
        num_correct: Examples graded as correct (reward > 0).
        num_errors: Examples that failed with an error.
        metrics: Benchmark-specific additional metrics.
        time_seconds: Total wall time for the benchmark.
    """

    name: str
    score: float
    num_examples: int
    num_correct: int
    num_errors: int = 0
    metrics: dict = field(default_factory=dict)
    time_seconds: float = 0.0
    pass_at_k: dict[int, float] = field(default_factory=dict)
    """Maps k to pass@k score. Only populated when ``num_samples > 1``.
    E.g., ``{1: 0.45, 5: 0.72, 10: 0.85}``."""


class BenchmarkBuilder(ABC):
    """Defines a benchmark as a list of Env instances.

    Subclass this to add new benchmarks. The framework handles everything
    else: rollouts, concurrency, storage, aggregation.

    A BenchmarkBuilder creates one :class:`Env` per evaluation example.
    Each Env is single-use — ``initial_observation()`` provides the prompt,
    ``step()`` grades the model's response and returns the reward.

    For single-turn benchmarks (most common), ``step()`` sets
    ``episode_done=True`` after one turn. For multi-turn benchmarks
    (terminal-bench, tau2-bench), ``step()`` drives the conversation.

    The same Env implementation can be used for both evaluation and RL
    training — no separate eval code needed.

    Example::

        class MyBenchmark(BenchmarkBuilder):
            name = "my_benchmark"

            def make_envs(self, renderer, config):
                ds = load_dataset("my/dataset", split="test")
                return [MyEnv(row, renderer) for row in ds]
    """

    name: str
    """Unique benchmark name used in the registry and file paths."""

    multi_turn: bool = False
    """If True, uses agent_concurrency (lower) instead of concurrency."""

    recommended_timeout: float = 300
    """Recommended per-example timeout in seconds. Users can override via
    ``BenchmarkConfig.timeout_seconds``. Guidelines:
    - Single-turn programmatic grading (gsm8k, mmlu): 60-300s
    - Single-turn with LLM judge (arena_hard): 300-600s
    - Code execution (mbpp, livecodebench): 300-600s
    - Multi-turn agent (terminal_bench, tau2): 600-1800s
    """

    experimental: bool = False
    """If True, the runner logs a warning that this benchmark is experimental
    and may not match published scores."""

    # Requirements — validated by the runner before calling make_envs().
    requires_sandbox: bool = False
    """If True, this benchmark executes code in a sandbox. The runner
    validates that ``config.sandbox_factory`` is set or Modal is importable."""
    requires_judge: bool = False
    """If True, this benchmark uses an LLM judge. The runner validates
    that ``config.judge_sampling_client`` is set."""

    @abstractmethod
    def make_envs(
        self,
        renderer: Renderer,
        config: BenchmarkConfig,
    ) -> Sequence[Env]:
        """Create one Env per evaluation example.

        Args:
            renderer: Renderer for tokenization and prompt building.
            config: Runtime configuration (limits, tokens, etc.).

        Returns:
            Sequence of single-use Env instances.
        """
        ...

    def aggregate(
        self,
        rewards: list[float],
        metrics_list: list[dict],
    ) -> BenchmarkResult:
        """Aggregate per-example rewards into a BenchmarkResult.

        Override for custom aggregation (e.g., per-category breakdowns).
        Default: accuracy = fraction with reward > 0.

        Args:
            rewards: Total reward per example.
            metrics_list: Per-example metrics from Env.step().

        Returns:
            Aggregated BenchmarkResult.
        """
        num_correct = sum(1 for r in rewards if r > 0)
        return BenchmarkResult(
            name=self.name,
            score=num_correct / len(rewards) if rewards else 0.0,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics={},
        )
