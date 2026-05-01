"""Infrastructure setup utilities: RLOR jobs, deployments, training clients.

One high-level entry point: :func:`setup_infra`. Recipes pass a config + a
couple of flags (``needs_reference``, ``needs_inference``) and get back an
:class:`Infra` bundle of trainer clients. All control flow — shape
auto-selection, parallel trainer provisioning, LoRA shared-reference
branching, deployment setup — lives inside :func:`setup_infra`.

Inference-side objects (sampler, weight syncer, concurrency controller) are
intentionally *not* created here — each recipe builds them after calling
:func:`setup_infra` so the construction is visible and customisable in the
loop.

Generic across loop types:

* RL (``rl_loop.py``)   → ``needs_reference=(kl_beta > 0)``, ``needs_inference=True``
* DPO (``dpo_loop.py``) → ``needs_reference=True``, ``needs_inference=False``
* SFT                   → ``needs_reference=False``, ``needs_inference=False``
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlencode

from fireworks.training.sdk.client import (
    FiretitanServiceClient,
    FiretitanTrainingClient,
)
from fireworks.training.sdk import (
    TrainerJobConfig,
    TrainerJobManager,
    TrainerServiceEndpoint,
    TrainingShapeProfile,
)
try:
    from fireworks.training.sdk import CreatedTrainerJob
except ImportError:
    CreatedTrainerJob = Any
from fireworks.training.sdk.deployment import (
    DeploymentConfig,
    DeploymentInfo,
    DeploymentManager,
)
from tinker_cookbook.fireworks.utils.config import InfraConfig, DeployConfig, WeightSyncScope
from tinker_cookbook.fireworks.utils.training_shapes import auto_select_training_shape
from tinker_cookbook.fireworks.utils.client import ReconnectableClient, DEFAULT_TIMEOUT_S

TrainerHandle = CreatedTrainerJob | TrainerServiceEndpoint
"""Return type of :func:`request_trainer_job`.

``CreatedTrainerJob`` for a freshly-POSTed job (caller must
:func:`wait_trainer_job` to block until READY); ``TrainerServiceEndpoint``
for the reuse path (already polled)."""

StatusCallback = Callable[[str], None]
"""Invoked with a human-readable provisioning status message at each
lifecycle step (creating, waiting, ready, failed)."""

logger = logging.getLogger(__name__)

_TRAINER_JOB_CONFIG_SUPPORTS_SKIP_VALIDATIONS = (
    "skip_validations" in inspect.signature(TrainerJobConfig).parameters
)
_TRAINER_JOB_CONFIG_SUPPORTS_TRAINING_SHAPE_REF = (
    "training_shape_ref" in inspect.signature(TrainerJobConfig).parameters
)

_DEPLOYMENT_ACCELERATOR_REGION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("NVIDIA_H200", "US_VIRGINIA_1"),
    ("NVIDIA_B200", "US_OHIO_1"),
    ("NVIDIA_B300", "NA_BRITISHCOLUMBIA_1"),
)

_TRAINER_CANCEL_GRACE_ENV = "FW_TRAINER_CANCEL_GRACE_PERIOD_S"
_TRAINER_DELETE_GRACE_ENV = "FW_TRAINER_DELETE_GRACE_PERIOD_S"

_FIREWORKS_API_EXTRA_HEADERS_ENV = "FIREWORKS_API_EXTRA_HEADERS"


def read_api_extra_headers_env() -> dict[str, str] | None:
    """Parse ``FIREWORKS_API_EXTRA_HEADERS`` into a header dict.

    The env var, when set, must be a JSON object whose values are strings.
    Used to pass additional HTTP headers (e.g. routing, auth, correlation IDs)
    to every Fireworks API request. Returns ``None`` when the var is unset or
    empty.
    """
    raw = os.environ.get(_FIREWORKS_API_EXTRA_HEADERS_ENV, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Invalid %s (not valid JSON); ignoring: %s",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
            exc,
        )
        return None
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        logger.warning(
            "%s must be a JSON object of string->string; ignoring",
            _FIREWORKS_API_EXTRA_HEADERS_ENV,
        )
        return None
    return parsed or None


def _default_trainer_cancel_grace_period_s() -> float:
    raw = os.environ.get(_TRAINER_CANCEL_GRACE_ENV)
    source_env = _TRAINER_CANCEL_GRACE_ENV
    if raw is None:
        raw = os.environ.get(_TRAINER_DELETE_GRACE_ENV, "30")
        source_env = _TRAINER_DELETE_GRACE_ENV
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to 30s trainer cancel grace period",
            source_env,
            raw,
        )
        return 30.0


class ResourceCleanup:
    """Register resource IDs for automatic cleanup on scope exit.

    Wrap recipe logic in ``with ResourceCleanup(...) as cleanup:`` and
    call ``cleanup.trainer(job_id)`` / ``cleanup.deployment(dep_id)``
    after creating each resource.  On scope exit (including exceptions),
    registered trainers are cancelled and deployments are cleaned in
    reverse creation order.

    Pre-created resources that should survive simply aren't registered.
    """

    def __init__(
        self,
        rlor_mgr: TrainerJobManager,
        deploy_mgr: DeploymentManager | None = None,
        trainer_cancel_grace_period_s: float | None = None,
        trainer_delete_grace_period_s: float | None = None,
    ):
        # Back-compat: accept trainer_delete_grace_period_s as an alias.
        if trainer_cancel_grace_period_s is None and trainer_delete_grace_period_s is not None:
            trainer_cancel_grace_period_s = trainer_delete_grace_period_s
        self._rlor_mgr = rlor_mgr
        self._deploy_mgr = deploy_mgr
        self._jobs: list[str] = []
        self._deployments: list[tuple[str, str]] = []
        self._trainer_cancel_grace_period_s = (
            _default_trainer_cancel_grace_period_s()
            if trainer_cancel_grace_period_s is None
            else max(0.0, float(trainer_cancel_grace_period_s))
        )

    def trainer(self, job_id: str) -> None:
        """Register a trainer job for cancellation on exit."""
        self._jobs.append(job_id)

    def cancel_trainer(self, job_id: str) -> None:
        """Cancel a trainer job now and unregister it from cleanup."""
        self._cancel_trainer_with_grace(job_id)
        try:
            self._jobs.remove(job_id)
        except ValueError:
            pass

    def delete_trainer(self, job_id: str) -> None:
        """Deprecated alias for :meth:`cancel_trainer`. Will be removed in a future release."""
        import warnings
        warnings.warn(
            "ResourceCleanup.delete_trainer is deprecated; use cancel_trainer.",
            DeprecationWarning, stacklevel=2,
        )
        self.cancel_trainer(job_id)

    def deployment(self, dep_id: str, action: str = "delete") -> None:
        """Register a deployment for cleanup on exit.

        *action*: ``"delete"`` (default) or ``"scale_to_zero"``.
        """
        self._deployments.append((dep_id, action))

    def __enter__(self) -> ResourceCleanup:
        return self

    def _cancel_trainer(self, job_id: str) -> None:
        cancel = getattr(self._rlor_mgr, "cancel", None)
        if callable(cancel):
            cancel(job_id)
            return

        post = getattr(self._rlor_mgr, "_post", None)
        account_id = getattr(self._rlor_mgr, "account_id", None)
        if callable(post) and account_id is not None:
            path = f"/v1/accounts/{account_id}/rlorTrainerJobs/{job_id}:cancel"
            resp = post(path, timeout=60)
            resp.raise_for_status()
            return

        raise AttributeError(
            "TrainerJobManager does not expose trainer cancellation support"
        )

    def _cancel_trainer_with_grace(self, job_id: str) -> None:
        if self._trainer_cancel_grace_period_s > 0:
            logger.info(
                "Cleanup: waiting %.1fs before canceling trainer job %s",
                self._trainer_cancel_grace_period_s,
                job_id,
            )
            time.sleep(self._trainer_cancel_grace_period_s)
        logger.info("Cleanup: canceling trainer job %s", job_id)
        self._cancel_trainer(job_id)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for jid in reversed(self._jobs):
            try:
                self._cancel_trainer_with_grace(jid)
            except Exception as e:
                logger.warning("Cleanup: failed to cancel trainer %s: %s", jid, e)
        for did, action in reversed(self._deployments):
            try:
                if action == "scale_to_zero":
                    logger.info("Cleanup: scaling deployment %s to zero", did)
                    self._deploy_mgr.scale_to_zero(did)
                else:
                    logger.info("Cleanup: deleting deployment %s", did)
                    self._deploy_mgr.delete(did)
            except Exception as e:
                logger.warning("Cleanup: failed to clean deployment %s: %s", did, e)


def _fetch_job_failure_reason(
    rlor_mgr: TrainerJobManager,
    job_id: str,
) -> str | None:
    """Try to retrieve a detailed failure reason from the server.

    Returns a human-readable string when the job has error details,
    ``None`` when the server doesn't expose them or the fetch fails.
    """
    try:
        job = rlor_mgr.get(job_id)
        state = job.get("state", "")
        reason_parts: list[str] = []
        if state:
            reason_parts.append(f"state={state}")
        error_msg = (
            job.get("error", {}).get("message")
            or job.get("errorMessage")
            or job.get("statusMessage")
        )
        if error_msg:
            reason_parts.append(error_msg)
        return "; ".join(reason_parts) if reason_parts else None
    except Exception:
        return None


def request_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    profile: TrainingShapeProfile | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    grad_accum: int = 1,
    display_name: str = "trainer",
    hot_load_deployment_id: str | None = None,
    extra_args: list[str] | None = None,
    job_id: str | None = None,
    forward_only: bool = False,
    cleanup: ResourceCleanup | None = None,
    on_status: StatusCallback | None = None,
) -> TrainerHandle:
    """POST a trainer job creation request; return immediately without waiting for READY.

    Returns a ``CreatedTrainerJob`` for a freshly-POSTed job (callers must
    pass the result to :func:`wait_trainer_job` to block until READY), or a
    ``TrainerServiceEndpoint`` for the reuse path (already polled by
    ``_reuse_or_resume_job``, so :func:`wait_trainer_job` is a no-op).

    Two launch paths:

    * **Shape path** (profile provided): sends ``training_shape_ref``
      plus algorithm fields only.  The backend populates accelerator,
      image tag, node count, sharding, etc. from the validated shape.
    * **Manual path** (no profile): sends all ``InfraConfig`` fields
      as-is; the server skips shape validation.

    *cleanup*, when provided, registers the created job for cancellation
    on scope exit. Thread-safe: ``ResourceCleanup.trainer`` uses a lock.
    """
    trainer_role = "reference" if forward_only else "policy"

    def _emit(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    if job_id:
        _emit(f"reusing existing {trainer_role} trainer {job_id}")
        return _reuse_or_resume_job(rlor_mgr, job_id)

    extra_trainer_args: dict[str, Any] = {}
    if _TRAINER_JOB_CONFIG_SUPPORTS_SKIP_VALIDATIONS:
        extra_trainer_args["skip_validations"] = infra.skip_validations
    if profile is not None and _TRAINER_JOB_CONFIG_SUPPORTS_TRAINING_SHAPE_REF:
        extra_trainer_args["training_shape_ref"] = profile.training_shape_version

    if profile is not None:
        config = TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len or profile.max_supported_context_length,
            learning_rate=learning_rate,
            gradient_accumulation_steps=grad_accum,
            display_name=display_name,
            hot_load_deployment_id=hot_load_deployment_id,
            region=infra.region,
            extra_args=extra_args or infra.extra_args,
            forward_only=forward_only,
            **extra_trainer_args,
        )
        if not _TRAINER_JOB_CONFIG_SUPPORTS_TRAINING_SHAPE_REF:
            config.training_shape_ref = profile.training_shape_version
        config.node_count = None
    else:
        config = TrainerJobConfig(
            base_model=base_model,
            lora_rank=lora_rank,
            max_context_length=max_seq_len,
            learning_rate=learning_rate,
            gradient_accumulation_steps=grad_accum,
            node_count=infra.node_count,
            display_name=display_name,
            hot_load_deployment_id=hot_load_deployment_id,
            region=infra.region,
            custom_image_tag=infra.custom_image_tag,
            extra_args=extra_args or infra.extra_args,
            accelerator_type=infra.accelerator_type,
            accelerator_count=infra.accelerator_count,
            forward_only=forward_only,
        )
        if not _TRAINER_JOB_CONFIG_SUPPORTS_TRAINING_SHAPE_REF:
            config.training_shape_ref = None

    if infra.purpose:
        config.purpose = infra.purpose
    config.managed_by = infra.managed_by

    logger.info(
        "Creating %s trainer job '%s' (forward_only=%s)...",
        trainer_role,
        display_name,
        forward_only,
    )
    _emit(f"creating {trainer_role} trainer '{display_name}'")

    try:
        created_job = rlor_mgr.create(config)
    except Exception as e:
        error_msg = (
            f"Failed to create {trainer_role} trainer job '{display_name}' "
            f"(forward_only={forward_only}): {e}"
        )
        logger.error(error_msg)
        _emit(error_msg)
        raise RuntimeError(error_msg) from e

    if cleanup:
        cleanup.trainer(created_job.job_id)

    _emit(
        f"waiting for {trainer_role} trainer '{display_name}' "
        f"to become ready (job {created_job.job_id})"
    )
    return created_job


def wait_trainer_job(
    rlor_mgr: TrainerJobManager,
    created: TrainerHandle,
    *,
    infra: InfraConfig,
    display_name: str = "trainer",
    forward_only: bool = False,
    on_status: StatusCallback | None = None,
) -> TrainerServiceEndpoint:
    """Wait for a previously-requested trainer job to become READY.

    When *created* is already a ``TrainerServiceEndpoint`` (reuse path),
    returns it unchanged — ``_reuse_or_resume_job`` already polled.

    May be called from worker threads; the *on_status* callback and logging
    are both thread-safe.
    """
    if isinstance(created, TrainerServiceEndpoint):
        return created

    trainer_role = "reference" if forward_only else "policy"

    def _emit(msg: str) -> None:
        if on_status is not None:
            try:
                on_status(msg)
            except Exception:
                pass

    created_job_id = created.job_id
    try:
        endpoint = rlor_mgr.wait_for_ready(
            created.job_id,
            job_name=created.job_name,
            timeout_s=infra.trainer_timeout_s,
        )
    except Exception as e:
        detail = str(e)
        server_reason = _fetch_job_failure_reason(rlor_mgr, created_job_id)
        if server_reason:
            detail = f"{detail} — server: {server_reason}"
        logger.error(
            "Failed to create %s trainer job '%s' (forward_only=%s): %s",
            trainer_role,
            display_name,
            forward_only,
            detail,
        )
        error_msg = (
            f"Failed to create {trainer_role} trainer job '{display_name}' "
            f"(forward_only={forward_only}): {detail}"
        )
        _emit(error_msg)
        raise RuntimeError(error_msg) from e

    logger.info(
        "Created %s trainer job '%s': %s",
        trainer_role,
        display_name,
        endpoint.job_id,
    )
    _emit(f"{trainer_role} trainer '{display_name}' is ready (job {endpoint.job_id})")
    return endpoint


def create_trainer_job(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    infra: InfraConfig,
    profile: TrainingShapeProfile | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    grad_accum: int = 1,
    display_name: str = "trainer",
    hot_load_deployment_id: str | None = None,
    extra_args: list[str] | None = None,
    job_id: str | None = None,
    forward_only: bool = False,
    base_url_override: str | None = None,
    cleanup: ResourceCleanup | None = None,
    on_status: StatusCallback | None = None,
) -> TrainerServiceEndpoint:
    """Create a trainer job and block until it is READY.

    One-shot wrapper around :func:`request_trainer_job` + :func:`wait_trainer_job`
    for recipes that only need a single trainer (SFT, ORPO, and tools).
    RL recipes go through :func:`setup_infra` instead, which fans out
    policy + reference trainers in parallel.
    """
    if base_url_override is not None:
        logger.warning(
            "create_trainer_job(base_url_override=...) is ignored; the gateway "
            "now routes all trainer traffic. This kwarg is kept for back-compat "
            "and will be removed in a future release.",
        )
    created = request_trainer_job(
        rlor_mgr,
        base_model=base_model,
        infra=infra,
        profile=profile,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        grad_accum=grad_accum,
        display_name=display_name,
        hot_load_deployment_id=hot_load_deployment_id,
        extra_args=extra_args,
        job_id=job_id,
        forward_only=forward_only,
        cleanup=cleanup,
        on_status=on_status,
    )
    return wait_trainer_job(
        rlor_mgr,
        created,
        infra=infra,
        display_name=display_name,
        forward_only=forward_only,
        on_status=on_status,
    )


def request_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
) -> DeploymentInfo:
    """POST the deployment creation request; return immediately without waiting for READY.

    Auto-generates ``deploy_cfg.deployment_id`` when unset. Returns the
    post-create :class:`DeploymentInfo` (typically ``state=CREATING``).
    Callers pass the result to :func:`wait_deployment` to block until READY.
    """
    if not deploy_cfg.deployment_id:
        model_short = base_model.rsplit("/", 1)[-1]
        deploy_cfg.deployment_id = f"{model_short}-{int(time.time())}"

    info = deploy_mgr.get(deploy_cfg.deployment_id)
    if info:
        return info

    dep_config = deploy_cfg.to_deployment_config(base_model, infra)
    if dep_config.region is None and dep_config.deployment_shape:
        dep_config.region = _infer_region_from_deployment_shape(
            deploy_mgr, dep_config.deployment_shape
        )
    if dep_config.region is None:
        return _create_deployment_via_cookbook(deploy_mgr, dep_config, purpose=infra.purpose)
    return deploy_mgr.create_or_get(dep_config)


def wait_deployment(
    deploy_mgr: DeploymentManager,
    info: DeploymentInfo,
    deploy_cfg: DeployConfig,
) -> DeploymentInfo:
    """Wait for a previously-requested deployment to become READY.

    When *info* is already READY or UPDATING, returns it unchanged.
    May be called from worker threads; SDK polling and logging are thread-safe.
    """
    if info.state not in ("READY", "UPDATING"):
        info = deploy_mgr.wait_for_ready(
            deploy_cfg.deployment_id,
            timeout_s=deploy_cfg.deployment_timeout_s,
        )
    return info


def setup_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
) -> DeploymentInfo:
    """Create an inference deployment and block until it is READY.

    One-shot wrapper around :func:`request_deployment` + :func:`wait_deployment`.

    * If ``deploy_cfg.deployment_id`` is set, the existing deployment is
      fetched and waited on until READY.
    * If ``deploy_cfg.deployment_id`` is ``None``, a new deployment is
      created with an auto-generated ID derived from the model name.
    """
    info = request_deployment(deploy_mgr, deploy_cfg, base_model, infra)
    return wait_deployment(deploy_mgr, info, deploy_cfg)


def setup_or_reattach_deployment(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    trainer_job_name: str,
    weight_syncer: "WeightSyncer | None" = None,
    reattach_settle_timeout_s: int = 600,
) -> DeploymentInfo:
    """Standalone helper to create or re-attach a single deployment.

    If ``deploy_cfg.deployment_id`` names a live deployment, PATCHes its
    ``hot_load_trainer_job`` to *trainer_job_name* and waits for the pod
    rolling restart to settle. Otherwise creates a fresh deployment with
    the trainer reference baked in at creation.

    :func:`setup_infra` wraps the same logic end-to-end via
    :class:`WeightSyncScope`; use this helper only when you need the
    single-resource operation outside the RL / DPO recipes.
    """
    existing = (
        deploy_mgr.get(deploy_cfg.deployment_id)
        if deploy_cfg.deployment_id
        else None
    )
    if not existing or existing.state in ("FAILED", "DELETED", "DELETING"):
        fresh = dataclasses.replace(deploy_cfg, hot_load_trainer_job=trainer_job_name)
        return setup_deployment(deploy_mgr, fresh, base_model, infra)

    prev_identity = _read_replica_identity(
        deploy_mgr, deploy_cfg.deployment_id, base_model,
    )
    deploy_mgr.update(
        deploy_cfg.deployment_id,
        body={"hotLoadTrainerJob": trainer_job_name},
        update_mask="hot_load_trainer_job",
    )
    logger.info(
        "Re-attached deployment %s to trainer %s (prev_pod=%s)",
        deploy_cfg.deployment_id, trainer_job_name, prev_identity,
    )
    _wait_for_reattach_settled(
        deploy_mgr, deploy_cfg.deployment_id, base_model,
        prev_identity=prev_identity, timeout_s=reattach_settle_timeout_s,
    )
    if weight_syncer is not None:
        weight_syncer.reset_delta_chain()
    return existing


def _read_replica_identity(
    deploy_mgr: DeploymentManager,
    deployment_id: str,
    base_model: str,
) -> str | None:
    """Return the first replica's pod identity from the hotload status, or None."""
    try:
        status = deploy_mgr.hotload_check_status(deployment_id, base_model)
    except Exception as e:  # noqa: BLE001
        logger.debug("hotload_check_status failed for identity probe: %s", e)
        return None
    replicas = status.get("replicas") or []
    if not replicas:
        return None
    return replicas[0].get("identity")


def _wait_for_reattach_settled(
    deploy_mgr: DeploymentManager,
    deployment_id: str,
    base_model: str,
    *,
    prev_identity: str | None,
    timeout_s: int,
    poll_interval_s: float = 5.0,
) -> None:
    """Block until a fresh pod is exposing the hotload manager.

    PATCHing ``hot_load_trainer_job`` triggers a rolling restart of the
    serving container. We need to wait for two things:

    1. The old pod has gone (its replica identity is no longer reported).
    2. A new pod has come up *and* its hotload manager is initialized
       (the replicas list is non-empty with a different identity).

    Without this wait the SDK's first post-reattach hotload polls the
    *old* pod (still ready, but about to die) or polls during the gap
    where no replica is reporting, and bails out with stage=error.
    """
    deadline = time.time() + max(timeout_s, 1)
    saw_pod_gone = prev_identity is None
    while time.time() < deadline:
        current = _read_replica_identity(deploy_mgr, deployment_id, base_model)
        if prev_identity is None:
            if current is not None:
                logger.info(
                    "Re-attach settled: hotload manager up on pod %s",
                    current,
                )
                return
        else:
            if current is None:
                if not saw_pod_gone:
                    logger.info(
                        "Old pod %s has gone; waiting for new pod...",
                        prev_identity,
                    )
                saw_pod_gone = True
            elif current != prev_identity:
                logger.info(
                    "Re-attach settled: new pod %s replaced %s",
                    current,
                    prev_identity,
                )
                return
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Re-attach for deployment {deployment_id!r} did not produce a "
        f"fresh pod within {timeout_s}s (prev_identity={prev_identity!r})."
    )


def _infer_region_from_deployment_shape(
    deploy_mgr: DeploymentManager,
    deployment_shape: str,
) -> str | None:
    """Infer a rollout region from a validated deployment shape snapshot."""
    version = _get_deployment_shape_version(deploy_mgr, deployment_shape)
    snapshot = version.get("snapshot", {}) or {}
    accelerator = snapshot.get("acceleratorType", "")
    for prefix, region in _DEPLOYMENT_ACCELERATOR_REGION_PREFIXES:
        if accelerator.startswith(prefix):
            logger.info(
                "Inferred deployment region %s from deployment shape %s (accelerator=%s)",
                region,
                deployment_shape,
                accelerator,
            )
            return region
    if accelerator:
        logger.info(
            "No cookbook deployment-region override for deployment shape %s (accelerator=%s); using auto placement",
            deployment_shape,
            accelerator,
        )
    else:
        logger.info(
            "Deployment shape %s returned no accelerator type; using auto placement",
            deployment_shape,
        )
    return None


def _get_deployment_shape_version(
    deploy_mgr: DeploymentManager,
    deployment_shape: str,
) -> dict[str, Any]:
    """Fetch an exact or latest-validated deployment-shape version."""
    if "/versions/" in deployment_shape:
        path = f"/v1/{deployment_shape}"
    else:
        path = (
            f"/v1/{deployment_shape}/versions?"
            f"{urlencode({'filter': 'latest_validated=true', 'pageSize': 1})}"
        )
    resp = deploy_mgr._get(path, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "/versions/" in deployment_shape:
        return data
    versions = data.get("deploymentShapeVersions", []) or []
    if not versions:
        raise RuntimeError(
            f"No latest validated deployment-shape version was returned for '{deployment_shape}'"
        )
    return versions[0]


def get_deployment_gpu_count(
    deploy_mgr: DeploymentManager,
    deployment_shape: str | None,
    replica_count: int = 1,
) -> int:
    """Return total GPU count (``acceleratorCount × replica_count``) for a deployment.

    Reads ``acceleratorCount`` from the deployment shape snapshot; falls back to
    ``replica_count`` if the shape is missing or the field isn't exposed.
    """
    # Back-compat: old signature accepted a DeployConfig as arg 2. Unpack
    # it into the new (shape, replica_count) form and warn; fall through to
    # the single-path body below.
    if hasattr(deployment_shape, "deployment_shape"):
        logger.warning(
            "get_deployment_gpu_count: passing a DeployConfig is deprecated; "
            "pass deployment_shape (str) and replica_count separately.",
        )
        cfg = deployment_shape
        deployment_shape = cfg.deployment_shape
        replica_count = cfg.replica_count or replica_count

    if not deployment_shape:
        return replica_count  # no shape, assume 1 GPU per replica

    try:
        version = _get_deployment_shape_version(deploy_mgr, deployment_shape)
        snapshot = version.get("snapshot", {}) or {}
        accel_count = snapshot.get("acceleratorCount", 1) or 1
        total = accel_count * replica_count
        logger.info(
            "Deployment GPU count: %d (accelerator_count=%d × replica_count=%d)",
            total, accel_count, replica_count,
        )
        return total
    except Exception as e:
        logger.warning("Could not determine GPU count from shape: %s", e)
        return replica_count


def _create_deployment_via_cookbook(
    deploy_mgr: DeploymentManager,
    config: DeploymentConfig,
    purpose: str | None = None,
) -> DeploymentInfo:
    """Create a deployment while leaving placement selection to the control plane."""
    path = f"/v1/accounts/{deploy_mgr.account_id}/deployments?deploymentId={config.deployment_id}"
    if config.skip_shape_validation:
        path += "&skipShapeValidation=true"
    if config.disable_speculative_decoding:
        path += "&disableSpeculativeDecoding=true"

    body: dict[str, Any] = {
        "baseModel": config.base_model,
        "minReplicaCount": config.min_replica_count,
        "maxReplicaCount": config.max_replica_count,
        "enableHotLoad": True,
    }
    if config.hot_load_bucket_type:
        body["hotLoadBucketType"] = config.hot_load_bucket_type
    if config.deployment_shape:
        body["deploymentShape"] = config.deployment_shape
    if config.accelerator_type:
        body["acceleratorType"] = config.accelerator_type
    if config.extra_args:
        flat: list[str] = []
        for arg in config.extra_args:
            flat.extend(arg.split()) if " " in arg else flat.append(arg)
        body["extraArgs"] = flat
    if config.extra_values:
        body["extraValues"] = config.extra_values
    if purpose:
        body["annotations"] = {
            "internal/purpose": purpose.removeprefix("PURPOSE_").lower(),
        }

    logger.info("Creating deployment: %s", config.deployment_id)
    resp = deploy_mgr._post(path, json=body, timeout=60)
    resp.raise_for_status()
    return deploy_mgr._parse_deployment_info(config.deployment_id, resp.json())


def setup_training_client(
    endpoint: TrainerServiceEndpoint,
    base_model: str,
    api_key: str = "tml-local",
    lora_rank: int = 0,
) -> tuple[FiretitanServiceClient, FiretitanTrainingClient]:
    """Create a ServiceClient + TrainingClient for a trainer endpoint."""
    svc = FiretitanServiceClient(base_url=endpoint.base_url, api_key=api_key)
    cli = svc.create_training_client(base_model=base_model, lora_rank=lora_rank)
    return svc, cli


def _reuse_or_resume_job(
    rlor_mgr: TrainerJobManager, job_id: str
) -> TrainerServiceEndpoint:
    job = rlor_mgr.get(job_id)
    state = job.get("state", "")
    resumable = (
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
        "JOB_STATE_COMPLETED",
    )
    if state in resumable:
        logger.info("Job %s is %s, resuming...", job_id, state)
        return rlor_mgr.resume_and_wait(job_id)
    return rlor_mgr.wait_for_existing(job_id)


# ---------------------------------------------------------------------------
# Bundled infrastructure setup (setup_infra) — generic across RL, DPO, SFT
# ---------------------------------------------------------------------------


@dataclass
class _ReattachHandle:
    """Pending re-attach: PATCH has been issued; pod-roll settle is still pending.

    *dep_info* is the existing ``DeploymentInfo`` captured before the PATCH.
    *prev_identity* is the replica identity before the PATCH so
    :func:`_wait_for_reattach_settled` can detect the pod roll.
    *deployment_id* mirrors ``dep_info`` for convenience.
    *timeout_s* is forwarded to the settle function.
    """

    dep_info: DeploymentInfo
    prev_identity: str | None
    deployment_id: str
    timeout_s: int


@dataclass
class Infra:
    """Wired-up infrastructure handed back to the recipe.

    Inference-side objects (sampler, weight syncer, concurrency controller)
    are *not* included — each recipe constructs them after calling
    :func:`setup_infra` so they remain visible and customisable in the loop.
    """

    policy: ReconnectableClient
    """Policy trainer client. Always set."""
    reference: ReconnectableClient | None
    """Reference trainer client: a separate client (full-param) or a base-only
    handle on the policy session (LoRA shared). ``None`` when no reference is needed."""
    policy_profile: TrainingShapeProfile
    """Resolved policy training-shape profile."""
    policy_job_id: str
    """Policy trainer job ID."""
    reference_job_id: str | None
    """Reference trainer job ID; ``None`` for LoRA shared or ``needs_reference=False``."""
    inference_model: str | None
    """Inference model name (may differ from ``base_model`` if an adapter is mounted);
    ``None`` when ``needs_inference=False``."""
    boot_metrics: dict[str, float]
    """One-shot boot metrics suitable for ``wandb_log``."""
    closeables: list[Any] = field(default_factory=list)
    """Objects with ``.close()`` the caller should register onto an ``ExitStack``."""
    max_seq_len: int | None = None
    """Resolved max sequence length (auto-populated from shape when not provided)."""
    training_shape_id: str | None = None
    """Resolved policy training-shape ID (auto-selected if ``InfraConfig.training_shape_id`` was None)."""
    ref_training_shape_id: str | None = None
    """Resolved reference training-shape ID (auto-selected for full-param + ``needs_reference``)."""
    deployment_id: str | None = None
    """Final deployment ID, including auto-generated ones when ``DeployConfig.deployment_id`` was unset."""
    deployment_shape: str | None = None
    """Resolved deployment-shape version (auto-populated from the policy profile when
    ``DeployConfig.deployment_shape`` was unset). ``None`` when ``needs_inference=False``."""
    deployment_gpu_count: int = 1
    """Total GPU count for the deployment (``acceleratorCount × replica_count``),
    computed once from the resolved shape. ``1`` when ``needs_inference=False`` or
    the shape doesn't expose ``acceleratorCount``. Recipes use this to size
    concurrency windows, etc."""


def setup_infra(
    *,
    rlor_mgr: TrainerJobManager,
    deploy_mgr: DeploymentManager | None = None,
    base_model: str,
    infra_cfg: InfraConfig,
    deploy_cfg: DeployConfig | None = None,
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    learning_rate: float = 1e-5,
    step_timeout: float | None = None,
    policy_job_id: str | None = None,
    reference_job_id: str | None = None,
    needs_reference: bool = False,
    needs_inference: bool = False,
    role_prefix: str,
    api_key: str,
    cleanup: ResourceCleanup | None = None,
    on_status: StatusCallback | None = None,
) -> Infra:
    """Build all training-side infrastructure in one call.

    Generic across loop types (RL, DPO, SFT). Neither ``infra_cfg`` nor
    ``deploy_cfg`` is mutated; both are shape-resolved into local copies.

    Args:
        rlor_mgr: Trainer job manager.
        deploy_mgr: Deployment manager. Required when ``needs_inference=True``.
        base_model: Base model resource name.
        infra_cfg: Infrastructure config (shape IDs, region, etc.).
        deploy_cfg: Deployment config. Required when ``needs_inference=True``.
        lora_rank: LoRA rank; 0 means full-parameter.
        max_seq_len: Max sequence length. Auto-populated from shape when ``None``.
        learning_rate: Optimizer learning rate forwarded to trainer jobs.
        step_timeout: Per-step RPC timeout. Falls back to SDK default when ``None``.
        policy_job_id: Reuse an existing policy trainer job instead of creating one.
        reference_job_id: Reuse an existing reference trainer job.
        needs_reference: Provision a reference trainer (or shared LoRA session).
        needs_inference: Provision an inference deployment.
        role_prefix: Display-name prefix for trainer jobs (e.g. ``"grpo"``, ``"dpo"``).
        api_key: Fireworks API key forwarded to trainer clients.
        cleanup: When provided, register created trainers for cancellation and
            deployments for scale-to-zero on scope exit. Pass ``None`` to skip
            registration (e.g. when resuming and wanting resources to outlive the run).
        on_status: Optional callback receiving human-readable status messages.

    Returns:
        An :class:`Infra` bundle. Caller registers ``infra.closeables`` on an
        ``ExitStack``.
    """
    if needs_inference and deploy_mgr is None:
        raise ValueError("deploy_mgr is required when needs_inference=True")

    emit: StatusCallback = on_status or (lambda _: None)
    boot_start = time.time()

    policy_profile, resolved_max_seq_len, resolved_shape_id, resolved_deploy_shape = (
        _resolve_policy_shape(
            rlor_mgr,
            base_model=base_model,
            training_shape_id=infra_cfg.training_shape_id,
            deploy_shape=deploy_cfg.deployment_shape if deploy_cfg else None,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            needs_inference=needs_inference,
        )
    )
    ref_profile, resolved_ref_shape_id = _resolve_reference_shape(
        rlor_mgr,
        base_model=base_model,
        ref_training_shape_id=infra_cfg.ref_training_shape_id,
        lora_rank=lora_rank,
        max_seq_len=resolved_max_seq_len,
        needs_reference=needs_reference,
    )
    logger.info(
        "Policy shape=%s  ref shape=%s  deployment_shape=%s",
        resolved_shape_id,
        resolved_ref_shape_id,
        resolved_deploy_shape if needs_inference else None,
    )

    # Internal copy carrying resolved shape IDs for downstream request_trainer_job calls.
    resolved_infra = dataclasses.replace(
        infra_cfg,
        training_shape_id=resolved_shape_id,
        ref_training_shape_id=resolved_ref_shape_id,
    )
    # Build a local deploy_cfg copy with deployment_shape filled in; never mutates caller's object.
    local_deploy_cfg: DeployConfig | None = None
    if deploy_cfg is not None:
        local_deploy_cfg = (
            dataclasses.replace(deploy_cfg, deployment_shape=resolved_deploy_shape)
            if resolved_deploy_shape and not deploy_cfg.deployment_shape
            else deploy_cfg
        )

    weight_sync_scope = (
        local_deploy_cfg.weight_sync_scope if local_deploy_cfg else WeightSyncScope.PER_TRAINER
    )

    # The two scopes differ only in who is created first: whichever side owns
    # the bucket needs its identifier before the other side can be wired up.
    trainer_kwargs = dict(
        rlor_mgr=rlor_mgr, base_model=base_model, infra=resolved_infra,
        policy_profile=policy_profile, ref_profile=ref_profile,
        lora_rank=lora_rank, max_seq_len=resolved_max_seq_len,
        learning_rate=learning_rate,
        policy_job_id=policy_job_id, reference_job_id=reference_job_id,
        role_prefix=role_prefix, cleanup=cleanup, on_status=emit,
    )

    dep_info: DeploymentInfo | _ReattachHandle | None = None
    resolved_deployment_id: str | None = (
        local_deploy_cfg.deployment_id if local_deploy_cfg else None
    )
    if not needs_inference:
        policy_handle, ref_handle = _request_trainers(**trainer_kwargs, hot_load_deployment_id=None)
    elif weight_sync_scope == WeightSyncScope.PER_DEPLOYMENT:
        # Deployment first: its ID feeds each trainer's hot_load_deployment_id.
        dep_info, resolved_deployment_id = _provision_deployment_owned(
            deploy_mgr, local_deploy_cfg, base_model, resolved_infra, cleanup=cleanup,
        )
        policy_handle, ref_handle = _request_trainers(
            **trainer_kwargs, hot_load_deployment_id=resolved_deployment_id,
        )
    else:  # PER_TRAINER
        # Trainer first: its job_name feeds the deployment's hot_load_trainer_job.
        policy_handle, ref_handle = _request_trainers(**trainer_kwargs, hot_load_deployment_id=None)
        dep_info, resolved_deployment_id = _provision_trainer_owned(
            deploy_mgr, local_deploy_cfg, base_model, resolved_infra,
            trainer_job_name=policy_handle.job_name, cleanup=cleanup,
        )

    # Ensure the deploy_cfg fed into wait_deployment carries the resolved
    # deployment_id (request_deployment auto-generates it on an inner-scope
    # copy; without this propagation wait_deployment polls /deployments/None).
    if local_deploy_cfg is not None and resolved_deployment_id is not None:
        local_deploy_cfg = dataclasses.replace(local_deploy_cfg, deployment_id=resolved_deployment_id)

    # Wait for all pending resources in parallel.
    policy_ep, reference_ep, ready_dep_info = _await_in_parallel(
        rlor_mgr=rlor_mgr,
        policy_handle=policy_handle,
        ref_handle=ref_handle,
        dep_info=dep_info,
        deploy_mgr=deploy_mgr if needs_inference else None,
        infra=resolved_infra,
        deploy_cfg=local_deploy_cfg,
        base_model=base_model,
        role_prefix=role_prefix,
        on_status=emit,
    )

    inference_model = None
    if needs_inference:
        inference_model = ready_dep_info.inference_model or base_model

    closeables: list[Any] = []
    policy = _make_policy_client(
        rlor_mgr, policy_ep,
        base_model=base_model,
        lora_rank=lora_rank,
        step_timeout=step_timeout,
        api_key=api_key,
        closeables=closeables,
    )
    reference = _make_reference_client(
        rlor_mgr=rlor_mgr,
        reference_ep=reference_ep,
        policy=policy,
        base_model=base_model,
        lora_rank=lora_rank,
        step_timeout=step_timeout,
        needs_reference=needs_reference,
        api_key=api_key,
        closeables=closeables,
    )

    boot_metrics = _make_boot_metrics(boot_start, deploy_mgr if needs_inference else None)

    deployment_gpu_count = 1
    if needs_inference and resolved_deploy_shape:
        deployment_gpu_count = get_deployment_gpu_count(
            deploy_mgr,
            resolved_deploy_shape,
            replica_count=local_deploy_cfg.replica_count or 1,
        )

    return Infra(
        policy=policy,
        reference=reference,
        policy_profile=policy_profile,
        policy_job_id=policy_ep.job_id,
        reference_job_id=reference_ep.job_id if reference_ep is not None else None,
        inference_model=inference_model,
        boot_metrics=boot_metrics,
        closeables=closeables,
        max_seq_len=resolved_max_seq_len,
        training_shape_id=resolved_shape_id,
        ref_training_shape_id=resolved_ref_shape_id,
        deployment_id=resolved_deployment_id,
        deployment_shape=resolved_deploy_shape if needs_inference else None,
        deployment_gpu_count=deployment_gpu_count,
    )


def _register_closeable(obj: Any, closeables: list[Any]) -> None:
    if hasattr(obj, "close"):
        closeables.append(obj)


def _resolve_policy_shape(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    training_shape_id: str | None,
    deploy_shape: str | None,
    lora_rank: int,
    max_seq_len: int | None,
    needs_inference: bool,
) -> tuple[Any, int, str | None, str | None]:
    """Return (profile, resolved_max_seq_len, resolved_shape_id, resolved_deploy_shape)."""
    if training_shape_id is None:
        training_shape_id = auto_select_training_shape(
            rlor_mgr,
            base_model=base_model,
            trainer_role="policy",
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
        )
        logger.info("Auto-selected policy training shape: %s", training_shape_id)
    profile = rlor_mgr.resolve_training_profile(training_shape_id)

    resolved_deploy_shape = deploy_shape
    if needs_inference and not deploy_shape and profile.deployment_shape_version:
        resolved_deploy_shape = profile.deployment_shape_version

    if max_seq_len is None:
        max_seq_len = profile.max_supported_context_length
    if max_seq_len is None:
        raise ValueError(
            "max_seq_len is required. Set it in Config, or use a training shape "
            "(InfraConfig.training_shape_id) to auto-populate it."
        )
    return profile, max_seq_len, training_shape_id, resolved_deploy_shape


def _resolve_reference_shape(
    rlor_mgr: TrainerJobManager,
    *,
    base_model: str,
    ref_training_shape_id: str | None,
    lora_rank: int,
    max_seq_len: int,
    needs_reference: bool,
) -> tuple[Any | None, str | None]:
    """Return (ref_profile, resolved_ref_shape_id)."""
    if ref_training_shape_id:
        if lora_rank > 0:
            logger.info(
                "Using explicit reference shape %s alongside lora_rank=%d. The ref "
                "trainer will run forward-only on a LoRA-capable shape (its own GPUs). "
                "To save GPUs via the shared-session reference "
                "(policy.create_base_reference()), leave ref_training_shape_id unset.",
                ref_training_shape_id, lora_rank,
            )
        return rlor_mgr.resolve_training_profile(ref_training_shape_id), ref_training_shape_id
    if not (needs_reference and lora_rank == 0):
        return None, None
    ref_training_shape_id = auto_select_training_shape(
        rlor_mgr,
        base_model=base_model,
        trainer_role="reference",
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
    )
    logger.info("Auto-selected reference training shape: %s", ref_training_shape_id)
    return rlor_mgr.resolve_training_profile(ref_training_shape_id), ref_training_shape_id


def _strip_args(infra: InfraConfig, flags: set[str]) -> InfraConfig:
    """Return a copy of *infra* with the named flags removed from extra_args."""
    if not infra.extra_args:
        return infra
    stripped = [a for a in infra.extra_args if a not in flags]
    if stripped == infra.extra_args:
        return infra
    return dataclasses.replace(infra, extra_args=stripped or None)


def _request_trainers(
    *,
    rlor_mgr: TrainerJobManager,
    base_model: str,
    infra: InfraConfig,
    policy_profile: Any,
    ref_profile: Any | None,
    lora_rank: int,
    max_seq_len: int,
    learning_rate: float,
    policy_job_id: str | None,
    reference_job_id: str | None,
    role_prefix: str,
    hot_load_deployment_id: str | None,
    cleanup: ResourceCleanup | None,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None]:
    on_status("provisioning policy trainer")
    policy_handle = request_trainer_job(
        rlor_mgr,
        base_model=base_model,
        infra=infra,
        profile=policy_profile,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        display_name=f"{role_prefix}-policy",
        forward_only=False,
        hot_load_deployment_id=hot_load_deployment_id,
        job_id=policy_job_id,
        cleanup=cleanup,
        on_status=on_status,
    )

    ref_handle = None
    if ref_profile is not None:
        on_status("provisioning reference trainer")
        # Reference trainer runs forward-only: strip --full-oom-check, which
        # triggers a backward warmup that OOMs on smaller reference shapes.
        ref_infra = _strip_args(infra, {"--full-oom-check"})
        # Propagate lora_rank so the gateway infers the correct trainer_mode:
        #   lora_rank > 0  -> LORA_TRAINER (matches LoRA-capable ref shape)
        #   lora_rank == 0 -> FORWARD_ONLY (full-param base-weight forward)
        # A forward-only LoRA ref still loads the adapter on top of base
        # weights; the adapter is configured upstream and only the forward
        # pass runs here.
        ref_handle = request_trainer_job(
            rlor_mgr,
            base_model=base_model,
            infra=ref_infra,
            profile=ref_profile,
            lora_rank=lora_rank,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            display_name=f"{role_prefix}-reference",
            forward_only=True,
            hot_load_deployment_id=hot_load_deployment_id,
            job_id=reference_job_id,
            cleanup=cleanup,
            on_status=on_status,
        )

    return policy_handle, ref_handle


def _get_alive_deployment(
    deploy_mgr: DeploymentManager, dep_id: str | None,
) -> DeploymentInfo | None:
    """Return an existing, non-terminal deployment or ``None``."""
    if not dep_id:
        return None
    existing = deploy_mgr.get(dep_id)
    if existing is None or existing.state in ("FAILED", "DELETED", "DELETING"):
        return None
    return existing


def _register_deployment_cleanup(
    cleanup: ResourceCleanup | None, dep_id: str,
) -> None:
    if cleanup is not None:
        cleanup.deployment(dep_id, action="scale_to_zero")


def _provision_deployment_owned(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    *,
    cleanup: ResourceCleanup | None,
) -> tuple[DeploymentInfo, str]:
    """PER_DEPLOYMENT: reuse an existing deployment or POST a fresh one.

    The deployment owns a stable bucket (auto-filled by the server from the
    deployment ID). Trainers will be created later with ``hot_load_deployment_id``
    pointing here — no PATCH or re-attach needed.
    """
    existing = _get_alive_deployment(deploy_mgr, deploy_cfg.deployment_id)
    if existing is not None:
        logger.info("Re-using deployment %s (PER_DEPLOYMENT)", deploy_cfg.deployment_id)
        _register_deployment_cleanup(cleanup, deploy_cfg.deployment_id)
        return existing, deploy_cfg.deployment_id

    local = dataclasses.replace(deploy_cfg)
    info = request_deployment(deploy_mgr, local, base_model, infra)
    _register_deployment_cleanup(cleanup, local.deployment_id)
    return info, local.deployment_id


def _provision_trainer_owned(
    deploy_mgr: DeploymentManager,
    deploy_cfg: DeployConfig,
    base_model: str,
    infra: InfraConfig,
    *,
    trainer_job_name: str,
    cleanup: ResourceCleanup | None,
) -> tuple[DeploymentInfo | _ReattachHandle, str]:
    """PER_TRAINER: re-attach an existing deployment to *trainer_job_name* or POST a fresh one.

    Fresh deployments inherit the trainer's bucket URL at creation via
    ``hot_load_trainer_job``. Existing deployments are re-pointed via a PATCH,
    which rolls the serving pod — the returned :class:`_ReattachHandle` lets
    the await step wait for the new pod in parallel with trainer readiness.
    """
    existing = _get_alive_deployment(deploy_mgr, deploy_cfg.deployment_id)
    if existing is not None:
        dep_id = deploy_cfg.deployment_id
        prev_identity = _read_replica_identity(deploy_mgr, dep_id, base_model)
        deploy_mgr.update(
            dep_id,
            body={"hotLoadTrainerJob": trainer_job_name},
            update_mask="hot_load_trainer_job",
        )
        logger.info(
            "Re-attached deployment %s to trainer %s (prev_pod=%s) — "
            "settling in parallel with trainer waits",
            dep_id, trainer_job_name, prev_identity,
        )
        _register_deployment_cleanup(cleanup, dep_id)
        timeout_s = getattr(deploy_cfg, "reattach_settle_timeout_s", None) or 600
        return _ReattachHandle(
            dep_info=existing,
            prev_identity=prev_identity,
            deployment_id=dep_id,
            timeout_s=timeout_s,
        ), dep_id

    local = dataclasses.replace(deploy_cfg, hot_load_trainer_job=trainer_job_name)
    info = request_deployment(deploy_mgr, local, base_model, infra)
    _register_deployment_cleanup(cleanup, local.deployment_id)
    return info, local.deployment_id


def _await_in_parallel(
    *,
    rlor_mgr: TrainerJobManager,
    policy_handle: Any,
    ref_handle: Any | None,
    dep_info: "DeploymentInfo | _ReattachHandle | None",
    deploy_mgr: "DeploymentManager | None",
    infra: InfraConfig,
    deploy_cfg: "DeployConfig | None",
    base_model: str,
    role_prefix: str,
    on_status: Callable[[str], None],
) -> tuple[Any, Any | None, "DeploymentInfo | None"]:
    """Wait for all pending resources in parallel using a thread pool."""
    max_workers = 1 + (1 if ref_handle is not None else 0) + (1 if dep_info is not None else 0)

    policy_ep = ref_ep = final_dep_info = None
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max(max_workers, 1)) as pool:
        policy_future = pool.submit(
            wait_trainer_job,
            rlor_mgr,
            policy_handle,
            infra=infra,
            display_name=f"{role_prefix}-policy",
            forward_only=False,
            on_status=on_status,
        )
        ref_future = (
            pool.submit(
                wait_trainer_job,
                rlor_mgr,
                ref_handle,
                infra=infra,
                display_name=f"{role_prefix}-reference",
                forward_only=True,
                on_status=on_status,
            )
            if ref_handle is not None
            else None
        )
        if dep_info is None:
            dep_future = None
        elif isinstance(dep_info, _ReattachHandle):
            dep_future = pool.submit(
                _wait_for_reattach_settled,
                deploy_mgr,
                dep_info.deployment_id,
                base_model,
                prev_identity=dep_info.prev_identity,
                timeout_s=dep_info.timeout_s,
            )
        else:
            dep_future = pool.submit(wait_deployment, deploy_mgr, dep_info, deploy_cfg)

        try:
            policy_ep = policy_future.result()
        except Exception as e:
            errors.append(f"Policy trainer: {e}")

        if ref_future is not None:
            try:
                ref_ep = ref_future.result()
            except Exception as e:
                errors.append(f"Reference trainer: {e}")

        if dep_future is not None:
            try:
                settle_result = dep_future.result()
                final_dep_info = (
                    dep_info.dep_info
                    if isinstance(dep_info, _ReattachHandle)
                    else settle_result
                )
            except Exception as e:
                errors.append(f"Deployment: {e}")

    if errors:
        raise RuntimeError("Infrastructure provisioning failed:\n" + "\n".join(errors))
    assert policy_ep is not None
    return policy_ep, ref_ep, final_dep_info


def _make_policy_client(
    rlor_mgr: TrainerJobManager,
    policy_ep: Any,
    *,
    base_model: str,
    lora_rank: int,
    step_timeout: float | None,
    api_key: str,
    closeables: list[Any],
) -> Any:
    policy = ReconnectableClient(
        rlor_mgr, policy_ep.job_id, base_model,
        lora_rank=lora_rank,
        fw_api_key=api_key,
        default_timeout=step_timeout or DEFAULT_TIMEOUT_S,
    )
    _register_closeable(policy, closeables)
    return policy


def _make_reference_client(
    *,
    rlor_mgr: TrainerJobManager,
    reference_ep: Any | None,
    policy: Any,
    base_model: str,
    lora_rank: int,
    step_timeout: float | None,
    needs_reference: bool,
    api_key: str,
    closeables: list[Any],
) -> Any | None:
    if reference_ep is not None:
        # Client lora_rank must match the ref trainer's server config:
        # a LoRA-capable ref pod expects create_training_client(lora_rank>0),
        # a full-param forward-only pod expects lora_rank=0.
        reference = ReconnectableClient(
            rlor_mgr, reference_ep.job_id, base_model,
            lora_rank=lora_rank,
            fw_api_key=api_key,
            default_timeout=step_timeout or DEFAULT_TIMEOUT_S,
        )
        _register_closeable(reference, closeables)
        return reference
    if needs_reference and lora_rank > 0:
        reference = policy.create_base_reference()
        _register_closeable(reference, closeables)
        return reference
    return None


def _make_boot_metrics(boot_start: float, deploy_mgr: DeploymentManager | None) -> dict:
    metrics: dict = {
        "train/step": 0,
        "infra/total_boot_time": time.time() - boot_start,
    }
    if deploy_mgr is not None and deploy_mgr.boot_time_s is not None:
        metrics["infra/deploy_boot_time"] = deploy_mgr.boot_time_s
    return metrics


