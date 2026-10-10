import pytest

from tests.helpers import run_recipe
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preflight import PreflightConfig, capture_preflight_snapshot
from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.stores.training_store import TrainingRunStore

MODULE = "tinker_cookbook.recipes.chat_sl.train"


@pytest.mark.integration
def test_chat_sl_resume(tmp_path):
    """Train from scratch, then load an intermediate checkpoint and continue."""
    log_path = str(tmp_path / "chat_sl_resume")
    run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=delete",
            f"log_path={log_path}",
            "save_every=1",
        ],
        preflight=PreflightConfig(
            log_path=log_path,
            required_metric_keys=("train_mean_nll",),
            require_sampler_checkpoint=True,
            minimum_metric_step=1,
        ),
    )

    # Re-select step 1 so the second command resumes instead of treating the
    # bounded run's final checkpoint as a completed epoch.
    store = TrainingRunStore(storage_from_uri(log_path))
    periodic_checkpoint = next(
        record for record in store.read_checkpoint_records() if record.name == "000001"
    )
    store.write_checkpoint(periodic_checkpoint.to_dict())
    resume_checkpoint = checkpoint_utils.get_last_checkpoint(log_path)
    assert resume_checkpoint is not None
    assert resume_checkpoint == periodic_checkpoint
    assert resume_checkpoint.state_path is not None

    before_resume = capture_preflight_snapshot(log_path)
    assert before_resume.metric_records == 2
    assert before_resume.checkpoint_records == 3
    resume_output = run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=resume",
            f"log_path={log_path}",
            "save_every=1",
        ],
        max_steps=4,
        preflight=PreflightConfig(
            log_path=log_path,
            required_metric_keys=("train_mean_nll",),
            require_sampler_checkpoint=True,
            minimum_metric_step=3,
        ),
        preflight_after=before_resume,
    )

    assert f"Resumed training from {resume_checkpoint.state_path}" in resume_output
