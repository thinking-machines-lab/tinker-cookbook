"""Benchmark modules for Nemotron-Cascade-2 evaluation.

Each module exposes an ``evaluate`` coroutine with the signature::

    async def evaluate(
        sampling_client: tinker.SamplingClient,
        renderer: Renderer,
        max_tokens: int = 32768,
        max_examples: int | None = None,
    ) -> EvalResult
"""

from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.aime import evaluate as eval_aime
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gpqa import evaluate as eval_gpqa
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import evaluate as eval_gsm8k
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.ifeval import evaluate as eval_ifeval
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.longbench import evaluate as eval_longbench
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.math500 import evaluate as eval_math500
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mbpp import evaluate as eval_mbpp
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mmlu_pro import evaluate as eval_mmlu_pro

# Tier 2 benchmarks
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.arena_hard import evaluate as eval_arena_hard
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import evaluate as eval_bfcl
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.ifbench import evaluate as eval_ifbench
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.livecodebench import evaluate as eval_livecodebench
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mmlu_redux import evaluate as eval_mmlu_redux
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import evaluate as eval_swe_bench
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.tau2_bench import evaluate as eval_tau2_bench
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import evaluate as eval_terminal_bench

BENCHMARKS = {
    # Tier 1 (original)
    "gsm8k": eval_gsm8k,
    "ifeval": eval_ifeval,
    "mmlu_pro": eval_mmlu_pro,
    "math500": eval_math500,
    "gpqa": eval_gpqa,
    "aime": eval_aime,
    "mbpp": eval_mbpp,
    "longbench": eval_longbench,
    # Tier 2 (new)
    "livecodebench": eval_livecodebench,
    "mmlu_redux": eval_mmlu_redux,
    "arena_hard": eval_arena_hard,
    "bfcl": eval_bfcl,
    "ifbench": eval_ifbench,
    # Tier 3 (agentic / SWE)
    "swe_bench": eval_swe_bench,
    "tau2_bench": eval_tau2_bench,
    "terminal_bench": eval_terminal_bench,
}

__all__ = ["BENCHMARKS"]
