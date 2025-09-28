import asyncio
import logging
import time

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.math_rl.math_env import MathDatasetBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import remove_mask
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-loop"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 64
    group_size: int = 32
    learning_rate: float = 1e-4
    max_length: int = 32768
    lora_rank: int = 32
    save_every: int = 20


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    dataset_builder = MathDatasetBuilder(
        batch_size=config.batch_size,  # 64 problem groups per training batch
        group_size=config.group_size,  # 32 solution attempts per problem
        model_name_for_tokenizer=config.model_name,
        renderer_name=model_info.get_recommended_renderer_name(config.model_name),
        convo_prefix="standard",  # Includes few-shot examples
    )

    train_dataset, _ = asyncio.run(dataset_builder())

    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    num_batches = len(train_dataset)
    logger.info(f"Training for {num_batches} batches")

    async def get_policy(step: int):
        # Save a checkpoint that you can use for sampling
        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path

        # Create a sampling client with that checkpoint
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)  #

        # Wrap in completer interface for RL algorithms
        return TinkerTokenCompleter(sampling_client=sampling_client, max_tokens=512)

    async def generate_rollouts(step: int, policy: TokenCompleter):
        # Creates 64 builders, each building 32 MathEnv instances
        env_group_builders_P = train_dataset.get_batch(step)
        # Generate rollouts for each group of 32 environments
        trajectory_groups_P = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in env_group_builders_P]
        )
        taglist_P = [builder.logging_tags() for builder in env_group_builders_P]
        return trajectory_groups_P, taglist_P

    def process_trajectory_groups(trajectory_groups_P: list[TrajectoryGroup]):
        # (Optionally) Remove groups with all successes or all failures
        filtered_trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)
        # Compute advantages for each trajectory in each group
        advantages_P = compute_advantages(filtered_trajectory_groups_P)
        # Convert trajectories to token-level training examples
        data_D, _ = assemble_training_data(filtered_trajectory_groups_P, advantages_P)
        return data_D

    async def train_step(
        config: Config,
        metrics: dict[str, float],
        i_batch: int,
        training_client: tinker.TrainingClient,
    ):
        # 1. Create policy with current weights
        policy = await get_policy(i_batch)

        # 2. Generate rollouts
        with timed("generate_rollouts", metrics):
            # Generate rollouts: 64 groups × 32 environments = 2,048 total rollouts
            trajectory_groups_P, taglist_P = await generate_rollouts(i_batch, policy)

        # Compute trajectory metrics
        metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

        # 3. Process trajectory data into training examples
        data_D = process_trajectory_groups(trajectory_groups_P)

        # 4. Train the model on collected trajectories
        # Forward-backward pass on all data
        fwd_bwd_future = await training_client.forward_backward_async(
            list(map(remove_mask, data_D)), loss_fn="importance_sampling"
        )
        # Optimizer step
        adam_params = tinker.AdamParams(
            learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        optim_step_future = await training_client.optim_step_async(adam_params)
        # Wait for results
        _fwd_bwd_result = await fwd_bwd_future.result_async()
        _optim_step_result = await optim_step_future.result_async()
        return

    #  Main training loop
    for batch_idx in range(start_batch, num_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / num_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        asyncio.run(train_step(config, metrics, batch_idx, training_client))
        # Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": num_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
