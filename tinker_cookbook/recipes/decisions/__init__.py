"""Decision models: score one answer token per choice, and train on those
probabilities. See ``README.md``."""

from tinker_cookbook.recipes.decisions.decision import (
    ChoiceRenderer,
    ChoiceRendering,
    Decision,
    DecisionRenderer,
    DecisionRendering,
    PrefixedChoiceRenderer,
    sample_decision,
)
from tinker_cookbook.recipes.decisions.train import (
    ProbabilityLossFn,
    decision_choice_datums,
    decision_dist_datums,
    probability_loss,
)

__all__ = [
    "ChoiceRenderer",
    "ChoiceRendering",
    "Decision",
    "DecisionRenderer",
    "DecisionRendering",
    "PrefixedChoiceRenderer",
    "ProbabilityLossFn",
    "decision_choice_datums",
    "decision_dist_datums",
    "probability_loss",
    "sample_decision",
]
