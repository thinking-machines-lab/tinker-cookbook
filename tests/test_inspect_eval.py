"""Smoke test for inspect evaluation integration.

Tests both include_reasoning=False (default) and include_reasoning=True
using a thinking model (Qwen3) to verify reasoning content is correctly
preserved or stripped.
"""

import asyncio

import pytest
import tinker
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ContentReasoning as InspectAIContentReasoning
from inspect_ai.model import ContentText as InspectAIContentText
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

# Use a thinking model so we can verify reasoning is produced
MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3"

DATASET = MemoryDataset(
    name="smoke_test",
    samples=[
        Sample(input="What is 1 + 1? Reply with just the number.", target="2"),
    ],
)


@scorer(metrics=[accuracy()])
def check_has_reasoning():
    """Scorer that checks whether the model response contains reasoning content."""

    async def score(state: TaskState, target: Target) -> Score:
        content = state.output.choices[0].message.content
        if isinstance(content, list):
            has_reasoning = any(isinstance(c, InspectAIContentReasoning) for c in content)
            has_text = any(isinstance(c, InspectAIContentText) for c in content)
        else:
            has_reasoning = False
            has_text = isinstance(content, str) and len(content) > 0

        return Score(
            value=1 if has_text else 0,
            metadata={"has_reasoning": has_reasoning, "has_text": has_text},
        )

    return score


@task
def smoke_task() -> Task:
    return Task(
        name="smoke_test",
        dataset=DATASET,
        solver=generate(),
        scorer=check_has_reasoning(),
    )


def _create_api(include_reasoning: bool) -> InspectAPIFromTinkerSampling:
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    return InspectAPIFromTinkerSampling(
        renderer_name=RENDERER_NAME,
        model_name=MODEL_NAME,
        sampling_client=sampling_client,
        include_reasoning=include_reasoning,
    )


@pytest.mark.integration
def test_inspect_eval_without_reasoning():
    """Default behavior: reasoning is stripped, content is a plain string."""
    api = _create_api(include_reasoning=False)
    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(temperature=0.6, max_tokens=1024),
    )

    from inspect_ai import eval_async

    results = asyncio.run(
        eval_async(
            tasks=[smoke_task()],
            model=[model],
            log_dir="/tmp/tinker-smoke-test/inspect-eval",
        )
    )
    assert len(results) == 1
    result = results[0]
    assert result.results is not None
    assert result.results.scores is not None

    # Content should be a plain string (no reasoning)
    assert result.samples is not None
    sample = result.samples[0]
    content = sample.output.choices[0].message.content
    assert isinstance(content, str), f"Expected string content, got {type(content)}"


@pytest.mark.integration
def test_inspect_eval_with_reasoning():
    """With include_reasoning=True, content should include ContentReasoning."""
    api = _create_api(include_reasoning=True)
    model = InspectAIModel(
        api=api,
        config=InspectAIGenerateConfig(temperature=0.6, max_tokens=1024),
    )

    from inspect_ai import eval_async

    results = asyncio.run(
        eval_async(
            tasks=[smoke_task()],
            model=[model],
            log_dir="/tmp/tinker-smoke-test/inspect-eval",
        )
    )
    assert len(results) == 1
    result = results[0]
    assert result.results is not None
    assert result.results.scores is not None

    # Content should be a list with ContentReasoning and ContentText
    assert result.samples is not None
    sample = result.samples[0]
    content = sample.output.choices[0].message.content
    assert isinstance(content, list), f"Expected list content, got {type(content)}"

    has_reasoning = any(isinstance(c, InspectAIContentReasoning) for c in content)
    has_text = any(isinstance(c, InspectAIContentText) for c in content)
    assert has_reasoning, "Expected ContentReasoning in response from thinking model"
    assert has_text, "Expected ContentText in response"

    # The scorer metadata should confirm reasoning was detected
    score = result.results.scores[0]
    assert score.metrics["accuracy"].value == 1.0
