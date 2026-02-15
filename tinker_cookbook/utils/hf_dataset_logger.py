"""HuggingFace Dataset logger for tinker-cookbook RL training.

Pushes trajectory data (prompts, completions, rewards, advantages) as parquet
files to a HuggingFace Hub dataset repo using CommitScheduler for background
uploads.
"""

import logging
from pathlib import Path
from typing import Any

from tinker_cookbook.utils.ml_log import Logger, dump_config

logger = logging.getLogger(__name__)

_SCHEMA_DOCS = """\
## Schema

Each parquet file corresponds to one training iteration:

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Training iteration |
| `group_idx` | int | Group index within the batch |
| `traj_idx` | int | Trajectory index within the group |
| `prompt` | str | Decoded prompt/observation text |
| `completion` | str | Decoded model completion text |
| `reward` | float | Total reward for this trajectory |
| `advantage` | float | Computed advantage |
| `prompt_length` | int | Number of prompt tokens |
| `response_length` | int | Number of completion tokens |
| `final_reward` | float | Group-computed final reward |
| `episode_done` | bool | Whether the episode terminated naturally |
| `num_steps` | int | Number of environment steps |
| `step_observations` | list[str] | Decoded observation text at each step |
| `step_completions` | list[str] | Decoded model completion at each step |
| `step_rewards` | list[float] | Immediate reward at each step |
| `step_logprobs` | list[list[float]] | Per-token log-probabilities at each step |
| `step_response_lengths` | list[int] | Number of completion tokens at each step |
"""

# Keys to skip when auto-generating the training details table
# (internal/path-like fields that aren't useful in the card)
_SKIP_KEYS = {
    "log_path",
    "base_url",
    "enable_trace",
    "load_checkpoint_path",
    "hf_dataset_repo",
    "hf_dataset_private",
    "wandb_project",
    "wandb_name",
}


def _build_dataset_card(config_dict: dict[str, Any], repo_id: str) -> str:
    """Build a dataset card from any on-policy trainer config."""
    model_name = str(config_dict.get("model_name", "unknown"))
    loss_fn = str(config_dict.get("loss_fn", "unknown"))

    # --- tags ---
    tags = [
        "tinker",
        "tinker-cookbook",
        "rl-completions",
        "on-policy",
        "reinforcement-learning",
        loss_fn.lower().replace("_", "-"),
    ]
    if model_name and model_name != "unknown":
        tags.append(model_name.lower().replace("/", "-"))
    tags_yaml = "\n".join(f"- {t}" for t in tags)

    # --- training details table (auto-generated from config) ---
    detail_rows: list[str] = []
    for key, value in config_dict.items():
        if key in _SKIP_KEYS:
            continue
        # Format the value: link HF model names, stringify the rest
        display_key = key.replace("_", " ").title()
        if key == "model_name" and value and value != "unknown":
            display_val = f"[{value}](https://huggingface.co/{value})"
        elif isinstance(value, (dict, list)):
            # Compact repr for complex nested values
            display_val = f"`{_compact_repr(value)}`"
        else:
            display_val = str(value)
        detail_rows.append(f"| **{display_key}** | {display_val} |")
    details_table = "\n".join(detail_rows)

    return f"""\
---
pretty_name: "tinker-cookbook RL completion logs"
tags:
{tags_yaml}
---

# tinker-cookbook Completion Logs

This dataset contains on-policy generations produced during training
with [tinker-cookbook](https://github.com/thinking-machines-lab/tinker).

## Training details

| Key | Value |
|-----|-------|
{details_table}

{_SCHEMA_DOCS}

## Loading

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}")
```
"""


def _compact_repr(value: Any, max_len: int = 120) -> str:
    """Return a compact string representation, truncating if needed."""
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len // 2] + " ... " + s[-(max_len // 2) :]
    return s


class HfDatasetLogger(Logger):
    """Logger that pushes trajectory data as parquet files to HuggingFace Hub.

    Creates a private dataset repo and uses CommitScheduler for background uploads.
    """

    def __init__(
        self,
        repo_id: str,
        log_dir: str | Path,
        private: bool = True,
        commit_every: int = 2,  # minutes
    ):
        from huggingface_hub import CommitScheduler, HfApi

        self.repo_id = repo_id
        self.parquet_dir = Path(log_dir) / "hf_completions"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        self._api = api

        self.scheduler = CommitScheduler(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(self.parquet_dir),
            every=commit_every,
            allow_patterns=["*.parquet"],
        )
        logger.info("HfDatasetLogger: repo=%s, dir=%s", repo_id, self.parquet_dir)

    def log_hparams(self, config: Any) -> None:
        """Push dataset card with training config."""
        config_dict = dump_config(config)
        card = _build_dataset_card(config_dict, self.repo_id)
        self._api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        logger.info("HfDatasetLogger: pushed dataset card to %s", self.repo_id)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """No-op -- metrics are handled by other loggers (WandB, JSON, etc.)."""
        pass

    def log_trajectories(
        self,
        step: int,
        trajectory_groups,  # list[TrajectoryGroup]
        tokenizer,
    ) -> None:
        """Extract trajectory data, decode tokens, and save as parquet."""
        import tinker
        from datasets import Dataset
        from tinker_cookbook.rl.data_processing import compute_advantages

        advantages_P = compute_advantages(trajectory_groups)

        rows: dict[str, list] = {
            "step": [],
            "group_idx": [],
            "traj_idx": [],
            "prompt": [],
            "completion": [],
            "reward": [],
            "advantage": [],
            "prompt_length": [],
            "response_length": [],
            "final_reward": [],
            "episode_done": [],
            "num_steps": [],
            # Per-step parallel lists for multi-turn debugging
            "step_observations": [],
            "step_completions": [],
            "step_rewards": [],
            "step_logprobs": [],
            "step_response_lengths": [],
        }
        for i_group, (traj_group, advantages_G) in enumerate(zip(trajectory_groups, advantages_P)):
            total_rewards = traj_group.get_total_rewards()
            for i_traj, (traj, total_reward, advantage) in enumerate(
                zip(traj_group.trajectories_G, total_rewards, advantages_G)
            ):
                # Decode prompt (first observation) for the top-level column
                first_ob = traj.transitions[0].ob
                prompt_tokens: list[int] = []
                for chunk in first_ob.chunks:
                    if isinstance(chunk, tinker.EncodedTextChunk):
                        prompt_tokens.extend(chunk.tokens)
                prompt_text = tokenizer.decode(prompt_tokens)

                # Collect per-step detail and aggregate totals
                total_action_tokens = 0
                per_step_observations: list[str] = []
                per_step_completions: list[str] = []
                per_step_rewards: list[float] = []
                per_step_logprobs: list[list[float]] = []
                per_step_response_lengths: list[int] = []

                for transition in traj.transitions:
                    # Decode observation at this step
                    ob_tokens: list[int] = []
                    for chunk in transition.ob.chunks:
                        if isinstance(chunk, tinker.EncodedTextChunk):
                            ob_tokens.extend(chunk.tokens)
                    per_step_observations.append(tokenizer.decode(ob_tokens))

                    # Decode action/completion at this step
                    per_step_completions.append(tokenizer.decode(transition.ac.tokens))
                    per_step_response_lengths.append(len(transition.ac.tokens))
                    total_action_tokens += len(transition.ac.tokens)

                    per_step_rewards.append(transition.reward)
                    per_step_logprobs.append(
                        list(transition.ac.maybe_logprobs)
                        if transition.ac.maybe_logprobs is not None
                        else []
                    )

                # Full completion = concatenation of all action texts
                completion_text = "".join(per_step_completions)

                # Episode done from last transition
                episode_done = traj.transitions[-1].episode_done

                rows["step"].append(step)
                rows["group_idx"].append(i_group)
                rows["traj_idx"].append(i_traj)
                rows["prompt"].append(prompt_text)
                rows["completion"].append(completion_text)
                rows["reward"].append(float(total_reward))
                rows["advantage"].append(float(advantage))
                rows["prompt_length"].append(len(prompt_tokens))
                rows["response_length"].append(total_action_tokens)
                rows["final_reward"].append(float(traj_group.final_rewards_G[i_traj]))
                rows["episode_done"].append(episode_done)
                rows["num_steps"].append(len(traj.transitions))
                rows["step_observations"].append(per_step_observations)
                rows["step_completions"].append(per_step_completions)
                rows["step_rewards"].append(per_step_rewards)
                rows["step_logprobs"].append(per_step_logprobs)
                rows["step_response_lengths"].append(per_step_response_lengths)

        if rows["step"]:
            ds = Dataset.from_dict(rows)
            path = self.parquet_dir / f"completions_{step:06d}.parquet"
            ds.to_parquet(str(path))
            logger.info("HfDatasetLogger: wrote %d rows to %s", len(ds), path)

    def close(self) -> None:
        """Trigger final commit."""
        if self.scheduler is not None:
            self.scheduler.trigger()
            logger.info("HfDatasetLogger: triggered final Hub commit")

    def get_logger_url(self) -> str | None:
        return f"https://huggingface.co/datasets/{self.repo_id}"
