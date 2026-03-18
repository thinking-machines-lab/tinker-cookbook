"""Evaluation interfaces for training loops.

Provides evaluator base classes that integrate with the supervised and RL
training loops, plus the Inspect AI integration.

Example::

    from tinker_cookbook.eval import (
        SamplingClientEvaluator,
        TrainingClientEvaluator,
        EvaluatorBuilder,
    )
"""

from tinker_cookbook.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
    TrainingClientEvaluator,
)

__all__ = [
    # Evaluator base classes
    "TrainingClientEvaluator",
    "SamplingClientEvaluator",
    # Type aliases
    "Evaluator",
    "EvaluatorBuilder",
    "SamplingClientEvaluatorBuilder",
]
