import logging
import os
from concurrent.futures import ThreadPoolExecutor

import hydra
from omegaconf import DictConfig

from fireworks.training.sdk import (
    DeploymentManager,
    DeploymentSampler,
    TrainerJobManager,
    WeightSyncer,
)
from tinker_cookbook.fireworks.utils import ReconnectableClient, create_trainer_job, setup_deployment
from tinker_cookbook.fireworks.utils.config import InfraConfig, DeployConfig
from transformers import AutoTokenizer

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


def _to_deploy_config(cfg_section: DictConfig) -> DeployConfig:
    """Convert an OmegaConf ``deployment`` section to a ``DeployConfig`` dataclass."""
    return DeployConfig(
        deployment_id=cfg_section.get("deployment_id"),
        deployment_shape=cfg_section.get("deployment_shape"),
        deployment_region=cfg_section.get("deployment_region"),
        replica_count=cfg_section.get("replica_count"),
        deployment_accelerator_type=cfg_section.get("deployment_accelerator_type"),
        hot_load_bucket_type=cfg_section.get("hot_load_bucket_type", "FW_HOSTED"),
        deployment_timeout_s=cfg_section.get("deployment_timeout_s", 5400),
        deployment_extra_args=list(cfg_section.get("deployment_extra_args") or []) or None,
        tokenizer_model=cfg_section.get("tokenizer_model"),
        sample_timeout=cfg_section.get("sample_timeout", 600),
        disable_speculative_decoding=cfg_section.get("disable_speculative_decoding", True),
        extra_values=dict(cfg_section.get("extra_values") or {}) or None,
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
    deploy_mgr = DeploymentManager(api_key=api_key, base_url=base_url)

    infra = _to_infra_config(cfg.training_infra)
    reference_infra_cfg = cfg.get("reference_training_infra") or cfg.training_infra
    reference_infra = _to_infra_config(reference_infra_cfg)
    deploy = _to_deploy_config(cfg.deployment)

    # Resolve training shape profile and auto-derive config values
    profile = None
    if infra.training_shape_id:
        profile = rlor_mgr.resolve_training_profile(infra.training_shape_id)
        dep_shape = getattr(profile, "deployment_shape", None) or getattr(profile, "deployment_shape_version", None)
        if dep_shape and not deploy.deployment_shape:
            deploy.deployment_shape = dep_shape
            logger.info("Auto-derived deployment_shape from training shape: %s", dep_shape)
        if profile.max_supported_context_length and not cfg.training.get("max_length"):
            cfg.training.max_length = profile.max_supported_context_length
            logger.info("Auto-derived max_length from training shape: %d", cfg.training.max_length)

    dep_info = setup_deployment(deploy_mgr, deploy, cfg.model.name, infra)
    deployment_id = dep_info.deployment_id
    use_reference = cfg.algorithm.get("kl_beta", 0.0) > 0

    ref_profile = None
    if use_reference:
        if reference_infra.training_shape_id:
            ref_profile = rlor_mgr.resolve_training_profile(reference_infra.training_shape_id)
        elif infra.ref_training_shape_id:
            ref_profile = rlor_mgr.resolve_training_profile(infra.ref_training_shape_id)
        elif profile is not None:
            ref_profile = profile

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
            hot_load_deployment_id=deployment_id,
        )
        if use_reference:
            ref_fut = pool.submit(
                create_trainer_job,
                rlor_mgr,
                base_model=cfg.model.name,
                infra=reference_infra,
                profile=ref_profile,
                lora_rank=cfg.model.get("lora_rank", 0),
                max_seq_len=cfg.training.max_length,
                learning_rate=cfg.training.learning_rate,
                display_name=cfg.get("ref_display_name", "reference"),
                job_id=reference_infra_cfg.get("training_job_id"),
                forward_only=True,
            )
        policy_ep = pol_fut.result()
        reference_ep = ref_fut.result() if use_reference else None

    # policy_job_id = policy_ep.job_id
    # reference_job_id = reference_ep.job_id if reference_ep else None

    policy_rc = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, cfg.model.name,
        lora_rank=cfg.model.get("lora_rank", 0),
    )
    reference_rc = (
        ReconnectableClient(
            rlor_mgr, reference_ep.job_id, cfg.model.name,
            lora_rank=cfg.model.get("lora_rank", 0),
        )
        if reference_ep else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        deploy.tokenizer_model or cfg.model.name,
        trust_remote_code=True,
    )
    inference_model = dep_info.inference_model if dep_info else cfg.model.name
    sampling_client = DeploymentSampler(
        inference_url=deploy_mgr.inference_url,
        model=inference_model,
        api_key=api_key,
        tokenizer=tokenizer,
    )
    weight_syncer = WeightSyncer(
        policy_client=policy_rc.inner,
        deploy_mgr=deploy_mgr,
        deployment_id=deployment_id,
        base_model=cfg.model.name,
        hotload_timeout=cfg.hotload.hot_load_timeout,
    )

    return policy_ep, reference_ep, sampling_client, weight_syncer


@hydra.main(config_path=".", config_name="fireworks", version_base=None)
def main(cfg: DictConfig) -> None:
    policy_ep, reference_ep, sampling_client, weight_syncer = init_fireworks_infra(cfg)
    logger.info("Fireworks policy endpoint ready (policy=%s)", policy_ep.base_url)

    logger.info("Fireworks reference endpoint ready (reference=%s)", reference_ep.base_url if reference_ep else None)
    logger.info("Fireworks sampling client ready (sampling_client=%s)", sampling_client.model)
    # logger.info("Fireworks weight syncer ready (weight_syncer=%s)", weight_syncer.base_url)


if __name__ == "__main__":
    main()
