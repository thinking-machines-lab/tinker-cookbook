"""Interactive demo: route a question about Tinker to the team that can answer it.

Prints the probability of each team as JSON; an empty line or Ctrl-D quits.

    uv run -m tinker_cookbook.recipes.decisions.demo
    echo 'How do I pick a learning rate for LoRA?' | uv run -m tinker_cookbook.recipes.decisions.demo

    # Other models, or fine-tuned weights saved with save_weights_for_sampler
    uv run -m tinker_cookbook.recipes.decisions.demo --model_name=thinkingmachines/Inkling-Small
    uv run -m tinker_cookbook.recipes.decisions.demo --model_path=tinker://.../sampler_weights/final

Requires ``TINKER_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json

import chz
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.decisions import Decision, DecisionRenderer, sample_decision
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.git_rev import recipe_user_metadata

TEAMS = [
    ("billing", "Charges, invoices, credits, payment methods and pricing"),
    ("sdk", "The tinker Python package: installation, API errors, training and sampling calls"),
    ("console", "The web console: signing in, API keys, viewing runs and checkpoints"),
    ("cookbook", "The Tinker Cookbook recipes: training loops, renderers, evaluations and docs"),
]


@chz.chz
class Config:
    model_name: str = "thinkingmachines/Inkling-Small"
    """Base model to use as the decision model."""

    model_path: str | None = None
    """Optional ``tinker://`` path to fine-tuned sampler weights for ``model_name``."""


async def main(config: Config) -> None:
    service_client = tinker.ServiceClient(
        user_metadata=recipe_user_metadata("recipe_decisions_demo")
    )
    sampling_client = await service_client.create_sampling_client_async(
        model_path=config.model_path, base_model=config.model_name
    )
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(config.model_name),
        get_tokenizer(config.model_name),
    )
    decision_renderer = DecisionRenderer(renderer)

    while True:
        try:
            question = input("Ask a question about Tinker (empty to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        decision = Decision(
            f"A user asked the following question about Tinker:\n\n{question}\n\n"
            "Which team should answer it?",
            TEAMS,
        )
        probabilities = await sample_decision(sampling_client, decision_renderer.render(decision))
        print(json.dumps({team: round(p, 4) for team, p in probabilities.items()}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config, allow_hyphens=True)))
