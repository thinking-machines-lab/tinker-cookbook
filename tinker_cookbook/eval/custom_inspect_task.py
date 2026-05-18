"""
Example of using LLM-as-a-judge with inspect_ai.

To run this task, use:
python -m tinker_cookbook.eval.run_inspect_evals \
    model_path=tinker://your-model-path \
    tasks=tinker_cookbook.eval.custom_inspect_task:example_lm_as_judge \
    renderer_name=role_colon \
    model_name=Qwen/Qwen3.5-9B-Base
"""

import tinker
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

QA_DATASET = MemoryDataset(
    name="qa_dataset",
    samples=[
        Sample(
            input="What is the capital of France?",
            target="Paris",
        ),
        Sample(
            input="What is the capital of Italy?",
            target="Rome",
        ),
    ],
)

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3.5-9B")

api = InspectAPIFromTinkerSampling(
    renderer_name="qwen3_5_disable_thinking",  # pyright: ignore[reportCallIssue]
    model_name="Qwen/Qwen3.5-9B",
    sampling_client=sampling_client,  # pyright: ignore[reportCallIssue]
    verbose=False,  # pyright: ignore[reportCallIssue]
)

GRADER_MODEL = InspectAIModel(api=api, config=InspectAIGenerateConfig())


@task
def example_lm_as_judge() -> Task:
    """
    Example task using LLM-as-a-judge scoring.

    Note: The grader model defaults to the model being evaluated.
    To use a different grader model, specify it with --model-grader when using inspect directly.
    """
    return Task(
        name="llm_as_judge",
        dataset=QA_DATASET,
        solver=generate(),
        scorer=model_graded_qa(
            instructions="Grade strictly against the target text as general answer key and rubric. "
            "Respond 'GRADE: C' if correct or 'GRADE: I' otherwise.",
            partial_credit=False,
            # model parameter is optional - if not specified, uses the model being evaluated
            model=GRADER_MODEL,
        ),
    )
