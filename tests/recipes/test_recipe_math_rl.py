import json
import tempfile
from pathlib import Path

import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.math_rl.train"

_COMMON_ARGS = [
    "model_name=Qwen/Qwen3.5-4B",
    "groups_per_batch=8",
    "group_size=4",
    "max_tokens=5",
]


@pytest.mark.integration
def test_math_rl_sync():
    run_recipe(
        MODULE,
        [*_COMMON_ARGS, "behavior_if_log_dir_exists=delete"],
    )


@pytest.mark.integration
def test_math_rl_async():
    run_recipe(
        MODULE,
        [*_COMMON_ARGS, "max_steps_off_policy=2", "behavior_if_log_dir_exists=delete"],
    )


@pytest.mark.integration
def test_math_rl_stream_minibatch():
    run_recipe(
        MODULE,
        [
            *_COMMON_ARGS,
            "stream_minibatch_config.groups_per_batch=8",
            "stream_minibatch_config.num_minibatches=2",
            "behavior_if_log_dir_exists=delete",
        ],
    )


@pytest.mark.integration
def test_math_rl_async_resume():
    """Verify async RL training can resume after a crash without hitting a 409 Conflict."""
    with tempfile.TemporaryDirectory() as log_path:
        # Step 1: Run training for 3 steps with save_every=1
        run_recipe(
            MODULE,
            [
                *_COMMON_ARGS,
                "max_steps_off_policy=2",
                "save_every=1",
                "eval_every=0",
                f"log_path={log_path}",
                "behavior_if_log_dir_exists=delete",
            ],
            max_steps=3,
        )

        # Step 2: Trim checkpoints.jsonl to simulate a crash at batch 2
        ckpt_file = Path(log_path) / "checkpoints.jsonl"
        records = [json.loads(line) for line in ckpt_file.read_text().splitlines()]
        # Keep only records up to batch 2 (discard batch 3 and "final")
        trimmed = [r for r in records if r.get("name") not in ("000003", "final")]
        ckpt_file.write_text("\n".join(json.dumps(r) for r in trimmed) + "\n")

        # Step 3: Resume — should reuse existing checkpoint, not 409
        run_recipe(
            MODULE,
            [
                *_COMMON_ARGS,
                "max_steps_off_policy=2",
                "save_every=1",
                "eval_every=0",
                f"log_path={log_path}",
                "behavior_if_log_dir_exists=resume",
            ],
            max_steps=3,
        )
