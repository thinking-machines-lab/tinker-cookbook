import logging
import os
from typing import Optional

import chz
import tinker
from inspect_ai import Tasks, eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

# Set up logger
logger = logging.getLogger(__name__)


@chz.chz
class InspectEvaluatorBuilder:
    """
    Configuration for inspect evaluation.
    This class provides a structured way to configure inspect evaluation
    parameters that can be used both in training configs and evaluator builders.
    """

    # Required parameters
    tasks: Tasks
    renderer_name: str
    model_name: str | None = None
    seed: int | None = None
    verbose: bool = False

    # Generation parameters
    temperature: float = 1.0
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int = -1
    num_choices: int = 1

    # Evaluation parameters
    limit: Optional[int] = None
    debug_errors: bool = True
    log_dir: Optional[str] = None
    max_connections: int = 512
    log_level: str = "INFO"

    def __call__(self) -> SamplingClientEvaluator:
        return InspectEvaluator(self)


class InspectEvaluator(SamplingClientEvaluator):
    """
    A SamplingClientEvaluator that runs inspect tasks and returns their metrics.
    """

    def __init__(self, config: InspectEvaluatorBuilder):
        """
        Initialize the InspectEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.config = config

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run inspect evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """
        # Create the inspect API wrapper
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.config.renderer_name,  # pyright: ignore[reportCallIssue]
            model_name=self.config.model_name,
            sampling_client=sampling_client,  # pyright: ignore[reportCallIssue]
            verbose=self.config.verbose,  # pyright: ignore[reportCallIssue]
        )
        # Create the inspect model
        model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
                num_choices=self.config.num_choices,
            ),
        )

        # Run evaluation
        results = await eval_async(
            tasks=self.config.tasks,
            model=[model],
            limit=self.config.limit,
            debug_errors=self.config.debug_errors,
            # Never retry - the tinker SDK is doing this for us already
            retry_on_error=0,
            # Although Tinker sampling tries very hard to only throw unrecoverable failures,
            # the inspect evaluation can still fail if e.g. the parser returns an error for
            # a given sample.
            fail_on_error=False,
            log_dir=self.config.log_dir or os.path.expanduser("~/inspect-logs"),
            max_connections=self.config.max_connections,
            log_level=self.config.log_level,
            log_realtime=False,
            log_buffer=1000,
        )

        # Extract metrics from results
        metrics = {}
        for task_result in results:
            if task_result.results is not None and task_result.results.scores is not None:
                for task_name, score in task_result.results.scores[0].metrics.items():
                    if task_result.eval.dataset is not None:
                        dataset_name = task_result.eval.dataset.name
                    else:
                        dataset_name = "unknown"
                    metrics[dataset_name + "/" + task_name] = score.value  # pyright: ignore[reportOptionalOperand]

        logger.info(f"Inspect evaluation completed. Metrics: {metrics}")
        return metrics
