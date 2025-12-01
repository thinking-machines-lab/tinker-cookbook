import logging
import os
from datetime import datetime

import chz
from gepa.api import optimize as gepa_optimize
import tinker

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.recipes.gepa.adapter import (
    TinkerDataInst,
    TinkerGEPAAdapter,
    TinkerReflectionLM,
)
from tinker_cookbook.recipes.gepa.tasks import GEPADataInstance, get_task, list_tasks
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Task
    task_name: str = "gsm8k"
    seed_prompt_override: str | None = None

    # Model configuration
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    reflection_model: str | None = "deepseek-ai/DeepSeek-V3.1"
    renderer_name: str | None = None
    reflection_renderer_name: str | None = None

    # GEPA optimization
    max_metric_calls: int = 150
    seed: int = 42
    eval_test: bool = False

    # Sampling configuration
    max_tokens: int = 2048
    temperature: float = 0.7

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    use_wandb: bool = False
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Tinker
    base_url: str | None = None


def main(config: CLIConfig) -> None:
    logger.info(f"Available tasks: {list_tasks()}")

    task = get_task(config.task_name)

    model_tag = config.model_name.replace("/", "-")
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"gepa-{config.task_name}-{model_tag}-{date_str}"

    log_path = config.log_path or f"/tmp/tinker-examples/gepa/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name or run_name,
        config=config,
    )

    logger.info(f"Starting GEPA optimization: {config}")
    logger.info(f"Task: {task.name}")

    trainset, valset, testset = task.load_data(seed=config.seed)
    logger.info(
        f"Loaded {task.name}: {len(trainset)} train, {len(valset)} val, {len(testset)} test"
    )

    seed_prompt = config.seed_prompt_override or task.seed_prompt
    logger.info(f"Seed prompt: {seed_prompt[:100]}...")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    adapter = TinkerGEPAAdapter(
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
        scorer=task.score,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        component_name=task.prompt_component_name,
    )

    reflection_model_name = config.reflection_model or config.model_name
    logger.info(f"Using reflection model: {reflection_model_name}")

    if reflection_model_name == config.model_name:
        reflection_client = sampling_client
        reflection_renderer = renderer
        reflection_tokenizer = tokenizer
    else:
        reflection_client = service_client.create_sampling_client(base_model=reflection_model_name)
        reflection_renderer_name = (
            config.reflection_renderer_name
            or model_info.get_recommended_renderer_name(reflection_model_name)
        )
        reflection_tokenizer = get_tokenizer(reflection_model_name)
        reflection_renderer = renderers.get_renderer(reflection_renderer_name, reflection_tokenizer)
        logger.info(f"Using reflection renderer: {reflection_renderer_name}")

    reflection_lm = TinkerReflectionLM(
        sampling_client=reflection_client,
        renderer=reflection_renderer,
        tokenizer=reflection_tokenizer,
    )

    def to_gepa_format(instances: list[GEPADataInstance]) -> list[TinkerDataInst]:
        return [
            TinkerDataInst(
                input=inst["input"],
                answer=inst["answer"],
                metadata=inst.get("metadata", {}),
            )
            for inst in instances
        ]

    gepa_trainset = to_gepa_format(trainset)
    gepa_valset = to_gepa_format(valset)
    gepa_testset = to_gepa_format(testset) if testset else []

    logger.info("Starting GEPA optimization loop...")
    result = gepa_optimize(
        seed_candidate={task.prompt_component_name: seed_prompt},
        trainset=gepa_trainset,
        valset=gepa_valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=config.max_metric_calls,
        run_dir=log_path,
        use_wandb=config.use_wandb,
        wandb_init_kwargs={"project": config.wandb_project, "name": config.wandb_name or run_name}
        if config.use_wandb
        else None,
        seed=config.seed,
        display_progress_bar=True,
    )

    best_score = result.val_aggregate_scores[result.best_idx]
    total_calls = result.total_metric_calls or 0
    best_candidate = result.best_candidate

    logger.info("=" * 60)
    logger.info("GEPA Optimization Complete!")
    logger.info(f"Best score: {best_score:.4f}")
    logger.info(f"Best prompt:\n{best_candidate}")
    logger.info(f"Total metric calls: {total_calls}")
    logger.info(f"Results saved to: {log_path}")
    logger.info("=" * 60)

    prompt_path = os.path.join(log_path, "best_prompt.txt")
    with open(prompt_path, "w") as f:
        for name, text in best_candidate.items():
            f.write(f"=== {name} ===\n{text}\n\n")

    test_score = None
    if config.eval_test and gepa_testset:
        logger.info("Evaluating best prompt on test set...")
        test_batch = adapter.evaluate(
            batch=gepa_testset,
            candidate=best_candidate,
            capture_traces=False,
        )
        test_score = sum(test_batch.scores) / len(test_batch.scores)
        logger.info(f"Test score: {test_score:.4f}")

    ml_metrics = {
        "final/best_score": best_score,
        "final/num_candidates": result.num_candidates,
        "final/total_metric_calls": total_calls,
    }
    if test_score is not None:
        ml_metrics["final/test_score"] = test_score

    ml_logger.log_metrics(ml_metrics, step=total_calls)
    ml_logger.close()


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    main(cli_config)
