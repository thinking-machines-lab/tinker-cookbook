import argparse
import os

import pandas

from tinker_cookbook import model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm


def json_already_exists(log_relpath: str) -> bool:
    metrics_path = os.path.expanduser(f"~/experiments/{log_relpath}/metrics.jsonl")
    if not os.path.exists(metrics_path):
        return False
    df = pandas.read_json(metrics_path, lines=True)
    return len(df) > 0


def build_rl_basic_config(max_steps_off_policy: int, name: str) -> rl_train.Config:
    model_name = "meta-llama/Llama-3.1-8B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = Gsm8kDatasetBuilder(
        batch_size=128,
        group_size=16,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )
    return rl_train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/async_rl_sweep_{name}",
        dataset_builder=builder,
        learning_rate=4e-5,
        max_tokens=256,
        eval_every=2,
        async_config=rl_train.AsyncConfig(
            max_steps_off_policy=max_steps_off_policy,
            groups_per_batch=16,
        )
        if max_steps_off_policy > 0
        else None,
        enable_trace=True,
    )


def async_rl_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry_run", action="store_true", help="If set, perform a dry run (do not launch jobs)"
    )
    parser.add_argument("--verbose", action="store_true", help="If set, print verbose output")
    args = parser.parse_args()

    log_relpath_base = "async_rl_sweep"
    job_specs = []
    for max_steps_off_policy in [0, 1, 2, 4, 8, 16]:
        tmux_window_name = (
            f"off_policy_{max_steps_off_policy}" if max_steps_off_policy > 0 else "on_policy"
        )
        rl_config = build_rl_basic_config(
            max_steps_off_policy=max_steps_off_policy,
            name=tmux_window_name,
        )
        log_relpath = os.path.expanduser(f"~/experiments/{log_relpath_base}/{tmux_window_name}")

        if json_already_exists(log_relpath):
            print(f"Skipping {log_relpath} because it already exists")
            continue
        job_specs.append(
            JobSpec(
                main_fn=rl_train.main,
                log_relpath=log_relpath,
                entrypoint_config=rl_config,
                tmux_window_name=tmux_window_name,
            )
        )

    if job_specs:
        print(f"Launching {len(job_specs)} sweep experiments with xmux")
        config = SwarmConfig(
            sweep_name=log_relpath_base,
            max_panes_per_window=5,
            debug=False,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        launch_swarm(job_specs, config)
    else:
        print("No experiments to launch (all already exist)")


if __name__ == "__main__":
    async_rl_sweep()
