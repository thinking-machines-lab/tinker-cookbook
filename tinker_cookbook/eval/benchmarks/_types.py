"""Core types for the benchmark evaluation framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import tinker

from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env

# ---------------------------------------------------------------------------
# Type aliases — reuse RL conventions where possible
# ---------------------------------------------------------------------------

Metrics = dict[str, float | int]
"""Numeric values aggregated across examples (e.g., ``{"correct": 1.0}``).
Matches :data:`tinker_cookbook.rl.types.Metrics`."""

Logs = dict[str, Any]
"""Per-example diagnostic data for display/debugging (e.g., expected answer,
extracted answer, example_id). Not aggregated — preserved per trajectory."""

METRIC_MAX_TOKENS_REACHED = "max_tokens_reached"
"""Metric key set by ``EnvFromMessageEnv`` when the model hits ``max_tokens``."""

METRIC_CONTEXT_OVERFLOW = "context_overflow"
"""Metric key set by ``EnvFromMessageEnv`` when the conversation exceeds context."""

PassAtKScores = dict[int, float]
"""Maps k values to pass@k probabilities.

Keys are k values (e.g., 1, 5, 10), values are the unbiased pass@k
estimate in [0, 1] using the Codex paper formula.
E.g., ``{1: 0.45, 5: 0.72, 10: 0.85}``.

Only populated when ``BenchmarkConfig.num_samples > 1``.
"""


# ---------------------------------------------------------------------------
# Serialization TypedDicts — define the JSON schema for stored data
# ---------------------------------------------------------------------------


TurnRole = Literal["user", "assistant", "environment", "grader"]
"""Valid roles for turns in stored trajectories."""


class StoredTurnDict(TypedDict):
    """JSON schema for a single turn in a stored trajectory."""

    role: TurnRole
    content: str | list[dict[str, Any]]
    """Text string or structured content parts (from renderer's content_to_jsonable)."""
    token_count: int
    metadata: dict[str, Any]


class StoredTrajectoryDict(TypedDict):
    """JSON schema for a stored trajectory (JSONL format)."""

    idx: int
    benchmark: str
    example_id: str
    turns: list[StoredTurnDict]
    reward: float
    metrics: Metrics
    logs: Logs
    error: str | None
    time_seconds: float


class BenchmarkResultDict(TypedDict):
    """JSON schema for a saved BenchmarkResult.

    Note: ``pass_at_k`` uses string keys in JSON (e.g., ``{"1": 0.45}``).
    The dataclass :class:`BenchmarkResult` uses int keys (e.g., ``{1: 0.45}``).
    Conversion happens in :func:`_save_result` and :func:`load_result`.
    """

    name: str
    score: float
    num_examples: int
    num_correct: int
    num_errors: int
    num_truncated: int
    metrics: Metrics
    time_seconds: float
    pass_at_k: dict[str, float]  # string keys for JSON; int keys in PassAtKScores


# ---------------------------------------------------------------------------
# Model-specific eval defaults
# ---------------------------------------------------------------------------

# Maps model ID to recommended eval config defaults.
# max_tokens is set to the full context window — the runner dynamically caps
# per request via context_window so prompt + max_tokens never exceeds it.
# See https://tinker-docs.thinkingmachines.ai/tinker/models/
_MODEL_EVAL_DEFAULTS: dict[str, dict[str, int | float]] = {
    # Qwen3.5 — Hybrid (thinking), 64K context
    "Qwen/Qwen3.5-397B-A17B": {
        "max_tokens": 65536,
        "context_window": 65536,
        "timeout_seconds": 1800,
    },
    "Qwen/Qwen3.5-397B-A17B:peft:262144": {
        "max_tokens": 262144,
        "context_window": 262144,
        "timeout_seconds": 1800,
    },
    "Qwen/Qwen3.5-35B-A3B": {"max_tokens": 65536, "context_window": 65536, "timeout_seconds": 1800},
    "Qwen/Qwen3.5-27B": {"max_tokens": 65536, "context_window": 65536, "timeout_seconds": 1800},
    "Qwen/Qwen3.5-4B": {"max_tokens": 65536, "context_window": 65536, "timeout_seconds": 1800},
    # Qwen3 — Hybrid (thinking), 32K context
    "Qwen/Qwen3-30B-A3B": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    "Qwen/Qwen3-32B": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    "Qwen/Qwen3-8B": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    # Qwen3 — Instruction (non-thinking), 32K context
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    # Nemotron — Hybrid (thinking), 64K context
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {
        "max_tokens": 65536,
        "context_window": 65536,
        "timeout_seconds": 1800,
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {
        "max_tokens": 65536,
        "context_window": 65536,
        "timeout_seconds": 1800,
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144": {
        "max_tokens": 262144,
        "context_window": 262144,
        "timeout_seconds": 1800,
    },
    # GPT-OSS — Reasoning, 32K context
    "openai/gpt-oss-120b": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    "openai/gpt-oss-120b:peft:131072": {
        "max_tokens": 131072,
        "context_window": 131072,
        "timeout_seconds": 1800,
    },
    "openai/gpt-oss-20b": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    # DeepSeek — Hybrid (thinking), 32K context
    "deepseek-ai/DeepSeek-V3.1": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 1800,
    },
    # Kimi — Reasoning, 32K context
    "moonshotai/Kimi-K2-Thinking": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 1800,
    },
    "moonshotai/Kimi-K2.5": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    "moonshotai/Kimi-K2.5:peft:131072": {
        "max_tokens": 131072,
        "context_window": 131072,
        "timeout_seconds": 1800,
    },
    "moonshotai/Kimi-K2.6": {"max_tokens": 32768, "context_window": 32768, "timeout_seconds": 1800},
    "moonshotai/Kimi-K2.6:peft:131072": {
        "max_tokens": 131072,
        "context_window": 131072,
        "timeout_seconds": 1800,
    },
    # Llama — Instruction (non-thinking), 32K context
    "meta-llama/Llama-3.3-70B-Instruct": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    # Llama — Base, 32K context
    "meta-llama/Llama-3.1-70B": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "meta-llama/Llama-3.1-8B": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "meta-llama/Llama-3.2-3B": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
    "meta-llama/Llama-3.2-1B": {
        "max_tokens": 32768,
        "context_window": 32768,
        "timeout_seconds": 300,
    },
}


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

    # Context management (multi-turn only)
    max_trajectory_tokens: int | None = None
    """**Multi-turn only.** Maximum total tokens accumulated across all turns
    before terminating the episode. Prevents context overflow in long agent
    conversations (terminal_bench, swe_bench). Set below the model's context
    window (e.g., 60000 for a 65K model). Ignored for single-turn benchmarks."""

    max_generation_tokens: int | None = None
    """**Multi-turn only.** Maximum tokens per generation step within a
    multi-turn episode. Used with ``max_trajectory_tokens`` to dynamically
    shrink generation limits as the conversation grows. Ignored for
    single-turn benchmarks (use ``max_tokens`` instead)."""

    # Generation (single-turn)
    max_tokens: int = 32768
    """Maximum tokens per model generation for single-turn benchmarks.
    For thinking models that generate long reasoning chains, increase this
    (e.g., 65536 for full context). This is the ``max_tokens`` passed to
    the sampling API."""
    temperature: float = 0.6
    """Sampling temperature for model generation."""
    context_window: int | None = None
    """Model's total context window size (e.g., 65536). When set, the runner
    dynamically caps ``max_tokens`` per request so that prompt + max_tokens
    fits in the context window. Prevents context overflow errors."""

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
    sandbox_factory: Callable[[], Any] | None = None
    """Async callable returning a :class:`~tinker_cookbook.sandbox.SandboxInterface`.
    Signature: ``async def factory() -> SandboxInterface``.
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

    grade_fn: Callable[[str, Logs], float] | None = None
    """Custom grading function: ``(response, logs) -> reward``.
    If set, overrides the benchmark's built-in grading logic.
    ``response`` is the decoded model output (thinking stripped).
    ``logs`` contains benchmark-specific fields like ``expected``, ``example_id``.

    Example::

        def my_grader(response: str, logs: Logs) -> float:
            expected = logs["expected"]
            # Custom extraction logic
            extracted = my_extract(response)
            return 1.0 if extracted == expected else 0.0

        config = BenchmarkConfig(grade_fn=my_grader)
    """

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> BenchmarkConfig:
        """Create a config with recommended defaults for a specific model.

        Looks up ``max_tokens`` and ``timeout_seconds`` from a built-in
        table of Tinker-supported models. Any keyword argument overrides
        the defaults.

        Example::

            config = BenchmarkConfig.for_model(
                "Qwen/Qwen3.5-35B-A3B",
                save_dir="evals/my_model",
            )
            result = await run_benchmark("gsm8k", client, renderer, config)

        Args:
            model_name: Model ID (e.g., ``"Qwen/Qwen3.5-35B-A3B"``).
                See https://tinker-docs.thinkingmachines.ai/tinker/models/
            **kwargs: Override any :class:`BenchmarkConfig` field.

        Raises:
            ValueError: If the model is not in the built-in table.
        """
        if model_name not in _MODEL_EVAL_DEFAULTS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {sorted(_MODEL_EVAL_DEFAULTS.keys())}. "
                f"Use BenchmarkConfig() directly and set max_tokens/timeout_seconds manually."
            )
        model_defaults = _MODEL_EVAL_DEFAULTS[model_name]
        return cls(
            max_tokens=kwargs.pop("max_tokens", int(model_defaults["max_tokens"])),
            timeout_seconds=kwargs.pop("timeout_seconds", float(model_defaults["timeout_seconds"])),
            context_window=kwargs.pop(
                "context_window", int(model_defaults.get("context_window", 0)) or None
            ),
            **kwargs,
        )

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

    role: TurnRole
    """``"user"``, ``"assistant"``, ``"environment"``, ``"grader"``."""
    content: str | list[dict[str, Any]]
    """Text string or structured content parts (from renderer's content_to_jsonable).
    Structured form is a list of dicts like ``[{"type": "text", "text": "..."}, ...]``."""
    token_count: int = 0
    """Number of tokens in this turn."""
    metadata: dict[str, Any] = field(default_factory=dict)
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
    """Stable identifier for this example across runs. Typically a content hash
    from ``make_example_id(prefix, question_text)`` — deterministic from the
    dataset, not positional. Used for cross-run comparison, resumability, and
    pass@k per-example tracking."""
    turns: list[StoredTurn] = field(default_factory=list)
    """Full conversation history (decoded text, not tokens)."""
    reward: float = 0.0
    """Total reward (sum of per-step rewards)."""
    metrics: Metrics = field(default_factory=dict)
    """Per-example metrics from Env.step() (e.g., ``{"correct": 1.0}``)."""
    logs: Logs = field(default_factory=dict)
    """Per-example logs from Env.step() (e.g., input, expected, extracted)."""
    error: str | None = None
    """Error message if this example failed."""
    time_seconds: float = 0.0
    """Wall time for this example."""

    def to_dict(self) -> StoredTrajectoryDict:
        """Serialize to a JSON-compatible dict."""
        return StoredTrajectoryDict(
            idx=self.idx,
            benchmark=self.benchmark,
            example_id=self.example_id,
            turns=[
                StoredTurnDict(
                    role=t.role,
                    content=t.content,
                    token_count=t.token_count,
                    metadata=t.metadata,
                )
                for t in self.turns
            ],
            reward=self.reward,
            metrics=self.metrics,
            logs=self.logs,
            error=self.error,
            time_seconds=self.time_seconds,
        )

    @classmethod
    def from_dict(cls, d: StoredTrajectoryDict | dict[str, Any]) -> StoredTrajectory:
        """Deserialize from a dict (e.g., loaded from JSONL).

        Accepts plain dicts for backward compatibility — old data may be
        missing newer fields like ``example_id``.
        """
        return cls(
            idx=d["idx"],
            benchmark=d["benchmark"],
            example_id=d.get("example_id", ""),
            turns=[
                StoredTurn(
                    role=t["role"],
                    content=t["content"],
                    token_count=t.get("token_count", 0),
                    metadata=t.get("metadata", {}),
                )
                for t in d.get("turns", [])
            ],
            reward=d.get("reward", 0.0),
            metrics=d.get("metrics", {}),
            logs=d.get("logs", {}),
            error=d.get("error"),
            time_seconds=d.get("time_seconds", 0.0),
        )


@dataclass
class BenchmarkResult:
    """Aggregated result from running a benchmark.

    Every evaluated example falls into exactly one of four categories::

        num_examples = num_correct + num_wrong + num_truncated + num_errors

    Where ``num_wrong = num_examples - num_correct - num_truncated - num_errors``.

    Two score metrics are provided:

    - ``score``: raw accuracy (``num_correct / num_examples``). Truncated and
      errored examples are scored as 0, dragging the score down.
    - ``score_completed``: accuracy on examples that actually completed
      (``num_correct / num_completed``). Excludes truncated and errored
      examples from the denominator.

    For thinking models that often hit ``max_tokens``, ``score_completed``
    is typically the more meaningful comparison against published scores.

    Example::

        result = await run_benchmark("gsm8k", client, renderer, config)
        print(f"Raw: {result.score:.1%}")                # 81.7%
        print(f"Completed: {result.score_completed:.1%}") # 95.6%
        print(f"{result.num_truncated} truncated, {result.num_errors} errors")
    """

    name: str
    """Benchmark name (e.g. ``"gsm8k"``)."""
    score: float
    """Raw accuracy: ``num_correct / num_examples``. Includes truncated and
    errored examples as 0 in the denominator."""
    num_examples: int
    """Total examples evaluated."""
    num_correct: int
    """Examples graded as correct (reward > 0)."""
    num_errors: int = 0
    """Examples that failed with an error (timeout, crash). Scored as 0."""
    num_truncated: int = 0
    """Examples where the model hit ``max_tokens`` or exceeded the context
    window before producing an answer. These are never graded — the episode
    terminates before the grading function runs. Scored as 0."""
    metrics: Metrics = field(default_factory=dict)
    """Benchmark-specific additional metrics (e.g., per-category scores)."""
    time_seconds: float = 0.0
    """Wall time for the entire benchmark run in seconds."""
    pass_at_k: PassAtKScores = field(default_factory=dict)
    """Pass@k scores. Only populated when ``BenchmarkConfig.num_samples > 1``."""

    @property
    def num_completed(self) -> int:
        """Examples that completed without error or truncation.

        Equal to ``num_examples - num_errors - num_truncated``.
        """
        return self.num_examples - self.num_errors - self.num_truncated

    @property
    def score_completed(self) -> float:
        """Accuracy on completed examples only.

        Excludes errored and truncated examples from both numerator and
        denominator: ``num_correct / num_completed``.

        This is the metric to compare against published model card scores,
        which typically don't penalize for context overflow.
        """
        return self.num_correct / self.num_completed if self.num_completed > 0 else 0.0


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

    recommended_system_prompt: str | None = None
    """System prompt that improves this benchmark's scores. Applied automatically
    when ``BenchmarkConfig.system_prompt`` is ``None``. For example, math
    benchmarks set this to instruct the model to use ``\\boxed{}``."""

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
        metrics_list: list[Metrics],
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
