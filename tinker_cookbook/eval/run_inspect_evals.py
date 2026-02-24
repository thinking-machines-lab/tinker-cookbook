import asyncio
import logging

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluator, InspectEvaluatorBuilder

logger = logging.getLogger(__name__)


@chz.chz
class Config(InspectEvaluatorBuilder):
    model_path: str | None = None


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    # Create a sampling client from the model path
    service_client = tinker.ServiceClient()

    if config.model_path is None and config.model_name is None:
        raise ValueError("model_path or model_name must be provided")

    if config.model_path is not None:
        model_path = config.model_path
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if config.model_name:
            if config.model_name != training_run.base_model:
                raise ValueError(
                    f"Model name {config.model_name} does not match training run base model {training_run.base_model}"
                )
        else:
            config = chz.replace(config, model_name=training_run.base_model)

        if config.renderer_name is None:
            renderer_name = await checkpoint_utils.get_renderer_name_from_checkpoint_async(
                service_client, model_path
            )
            if renderer_name is None:
                assert config.model_name is not None
                renderer_name = model_info.get_recommended_renderer_name(config.model_name)
            config = chz.replace(config, renderer_name=renderer_name)
    elif config.renderer_name is None:
        # No checkpoint metadata available; fall back to model default.
        assert config.model_name is not None
        config = chz.replace(
            config, renderer_name=model_info.get_recommended_renderer_name(config.model_name)
        )

    logger.info(f"Using base model: {config.model_name}")
    logger.info(f"Using renderer: {config.renderer_name}")

    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.model_name
    )

    # Run the evaluation
    logger.info(f"Running inspect evaluation for tasks: {config.tasks}")

    # Create the inspect evaluator
    evaluator = InspectEvaluator(config)
    metrics = await evaluator(sampling_client)

    # Print results
    logger.info("Inspect evaluation completed!")
    logger.info("Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
