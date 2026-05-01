import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig

from fireworks.training.sdk import (
    TrainerJobManager,
)
from tinker_cookbook.fireworks.utils import ReconnectableClient, create_trainer_job
from tinker_cookbook.fireworks.utils.config import InfraConfig

logger = logging.getLogger(__name__)


def _to_infra_config(cfg_section: DictConfig) -> InfraConfig:
    """Convert an OmegaConf infra section to an ``InfraConfig`` dataclass."""
    return InfraConfig(
        training_shape_id=cfg_section.get("training_shape_id"),
        ref_training_shape_id=cfg_section.get("ref_training_shape_id"),
        region=cfg_section.get("region"),
        custom_image_tag=cfg_section.get("custom_image_tag"),
        accelerator_type=cfg_section.get("accelerator_type"),
        accelerator_count=cfg_section.get("accelerator_count"),
        node_count=cfg_section.get("node_count", 1),
        skip_validations=cfg_section.get("skip_validations", False),
        extra_args=list(cfg_section.get("extra_args") or []),
    )


def init_fireworks_infra(cfg: DictConfig) -> tuple:
    """Create Fireworks TrainerJobManager, DeploymentManager,
    ReconnectableClient, WeightSyncer, and DeploymentSampler.

    Expects a fully-resolved ``DictConfig`` matching the schema in
    ``fireworks.yaml``.  Typically called from a ``@hydra.main`` entry point.
    """
    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = cfg.get("fireworks_base_url", "https://api.fireworks.ai")

    rlor_mgr = TrainerJobManager(api_key=api_key, base_url=base_url)

    infra = _to_infra_config(cfg.training_infra)

    # Resolve training shape profile and auto-derive config values
    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        if profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = profile.max_supported_context_length
            logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)


    with ThreadPoolExecutor(max_workers=2) as pool:
        pol_fut = pool.submit(
            create_trainer_job,
            rlor_mgr,
            base_model=cfg.model.name,
            infra=infra,
            profile=profile,
            lora_rank=cfg.model.get("lora_rank", 0),
            max_seq_len=cfg.training.max_length,
            learning_rate=cfg.training.learning_rate,
            display_name=cfg.get("train_display_name", "policy"),
            job_id=cfg.training_infra.training_job_id,
        )
        policy_ep = pol_fut.result()

    policy_rc = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.model.name,
        lora_rank=cfg.model.get("lora_rank", 0),
    )

    return policy_ep


@hydra.main(config_path=".", config_name="fireworks", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_ep = init_fireworks_infra(cfg)
    logger.info("Fireworks policy endpoint ready (policy=%s)", policy_ep.base_url)


if __name__ == "__main__":
    main()
