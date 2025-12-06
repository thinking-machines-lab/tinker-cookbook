"""
Minimal supervised fine-tuning script without abstractions.
Uses existing modules but with a simple, flat training loop.
"""

import logging
import time

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/sl-loop"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    lora_rank: int = 32
    save_every: int = 20  # 0 = disabled


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load No Robots dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("HuggingFaceH4/no_robots")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Train batches: {n_train_batches}")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resuming
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    # Training loop (single epoch)
    logger.info(f"Training for {n_train_batches} steps")

    # Shuffle dataset
    train_dataset = train_dataset.shuffle(seed=0)

    for batch_idx in range(start_batch, n_train_batches):
        start_time = time.time()
        step = batch_idx
        metrics = {}

        # Save checkpoint
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Linear learning rate schedule
        lr_mult = max(0.0, 1.0 - step / n_train_batches)
        current_lr = config.learning_rate * lr_mult
        adam_params = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        batch = [
            conversation_to_datum(
                row["messages"],  # type: ignore
                renderer,
                config.max_length,
                config.train_on_what,
            )
            for row in batch_rows
        ]

        # Training step
        fwd_bwd_future = training_client.forward_backward(batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Compute train metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        # Log metrics
        metrics.update(
            num_sequences=len(batch),
            num_tokens=sum(d.model_input.length for d in batch),
            learning_rate=current_lr,
            train_mean_nll=train_nll,
            progress=step / n_train_batches,
            time_total=time.time() - start_time,
        )
        ml_logger.log_metrics(metrics=metrics, step=step)

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
