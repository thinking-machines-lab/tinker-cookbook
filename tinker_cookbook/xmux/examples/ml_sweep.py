#!/usr/bin/env python
"""Example ML sweep using xmux with different grouping strategies"""

import os
import random
import shutil
import sys

from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm
from tinker_cookbook.xmux.examples.fake_train import main as fake_train_model


def demo_individual_windows():
    """Demo: Each experiment gets its own window"""
    print("\n" + "=" * 60)
    print("DEMO 1: Individual Windows (no grouping)")
    print("=" * 60 + "\n")

    # Simulate a learning rate sweep
    job_specs = []
    for i, (model, lr) in enumerate(
        [
            ("small", 0.001),
            ("small", 0.01),
            ("small", 0.1),
            ("medium", 0.001),
            ("medium", 0.01),
            ("medium", 0.1),
            ("large", 0.001),
            ("large", 0.01),
            ("large", 0.1),
        ]
    ):
        log_relpath = f"demo/lr-sweep/{model}/lr{lr}"
        abspath = os.path.join(os.path.expanduser("~/experiments"), log_relpath)
        if os.path.exists(abspath):
            shutil.rmtree(abspath)

        # Make jobs run faster and with varying success rates
        # First few jobs succeed, middle ones have mixed results, last ones fail
        if i < 3:
            failure_rate = 0.0  # First 3 always succeed
        elif i < 6:
            failure_rate = 0.5  # Middle 3 have 50% chance
        else:
            failure_rate = 1.0  # Last 3 always fail

        job_specs.append(
            JobSpec(
                main_fn=fake_train_model,
                log_relpath=log_relpath,
                entrypoint_config={
                    "model": model,
                    "lr": lr,
                    "duration": random.randint(5, 15),  # Much faster: 5-15 seconds
                    "failure_rate": failure_rate,
                },
            )
        )

    config = SwarmConfig(sweep_name="lr-sweep-individual", dry_run="--dry-run" in sys.argv)

    launch_swarm(job_specs, config)


def demo_grouped_by_model():
    """Demo: Group experiments by model type"""
    print("\n" + "=" * 60)
    print("DEMO 2: Grouped by Model")
    print("=" * 60 + "\n")

    job_specs = []
    for model in ["bert-base", "bert-large", "gpt2", "t5-small"]:
        for lr in [1e-5, 5e-5, 1e-4]:
            log_relpath = f"demo/model-groups/{model}/lr{lr}"

            job_specs.append(
                JobSpec(
                    main_fn=fake_train_model,
                    log_relpath=log_relpath,
                    entrypoint_config={
                        "model": model,
                        "lr": lr,
                        "duration": random.randint(30, 90),
                        "failure_rate": 0.15,
                    },
                    tmux_window_name=model,  # Group by model
                )
            )

    config = SwarmConfig(
        sweep_name="model-grouped-sweep",
        max_panes_per_window=3,  # Max 3 learning rates per window
        dry_run="--dry-run" in sys.argv,
    )

    launch_swarm(job_specs, config)


def demo_mixed_grouping():
    """Demo: Mix of grouped and individual experiments"""
    print("\n" + "=" * 60)
    print("DEMO 3: Mixed Grouping Strategy")
    print("=" * 60 + "\n")

    job_specs = []

    # Quick experiments - group together
    for i in range(6):
        log_relpath = f"demo/mixed/quick/exp{i}"

        job_specs.append(
            JobSpec(
                main_fn=fake_train_model,
                log_relpath=log_relpath,
                entrypoint_config={
                    "exp_id": i,
                    "model": f"quick-model-{i}",
                    "duration": random.randint(10, 30),
                    "failure_rate": 0.1,
                },
                tmux_window_name="quick-exps",
            )
        )

    # Long-running experiments - individual windows
    for dataset in ["imagenet", "coco", "wmt"]:
        for size in ["full", "sample"]:
            log_relpath = f"demo/mixed/long/{dataset}-{size}"

            job_specs.append(
                JobSpec(
                    main_fn=fake_train_model,
                    log_relpath=log_relpath,
                    entrypoint_config={
                        "dataset": dataset,
                        "size": size,
                        "model": f"{dataset}-model",
                        "duration": random.randint(180, 300),
                        "failure_rate": 0.05,  # Lower failure rate for long runs
                    },
                    # No tmux_window_name = individual window
                )
            )

    config = SwarmConfig(
        sweep_name="mixed-strategy-demo", max_panes_per_window=4, dry_run="--dry-run" in sys.argv
    )

    launch_swarm(job_specs, config)


def demo_large_scale():
    """Demo: Large scale sweep with many experiments"""
    print("\n" + "=" * 60)
    print("DEMO 4: Large Scale Sweep")
    print("=" * 60 + "\n")

    job_specs = []

    # Grid search over many hyperparameters
    models = ["model-v1", "model-v2", "model-v3"]
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
    batch_sizes = [16, 32, 64]
    optimizers = ["adam", "sgd"]

    for model in models:
        for lr in learning_rates:
            for bs in batch_sizes:
                for opt in optimizers:
                    log_relpath = f"demo/grid/{model}/lr{lr}-bs{bs}-{opt}"

                    # Group by model and optimizer
                    window_name = f"{model}-{opt}"

                    job_specs.append(
                        JobSpec(
                            main_fn=fake_train_model,
                            log_relpath=log_relpath,
                            entrypoint_config={
                                "model": model,
                                "lr": lr,
                                "batch_size": bs,
                                "optimizer": opt,
                                "duration": random.randint(60, 180),
                                "failure_rate": 0.1,
                            },
                            tmux_window_name=window_name,
                        )
                    )

    print(f"Total experiments: {len(job_specs)}")

    config = SwarmConfig(
        sweep_name="large-grid-search", max_panes_per_window=4, dry_run="--dry-run" in sys.argv
    )

    launch_swarm(job_specs, config)


def demo_real_usage():
    """Demo: How you would use xmux with real training code"""
    print(f"""
{"=" * 60}
DEMO: Real Usage Pattern
{"=" * 60}

In real usage, you would import your actual training function:

```python
from my_project.training import train_model
from my_project.config import TrainingConfig

job_specs = []
for lr in [1e-4, 5e-4, 1e-3]:
    config = TrainingConfig(
        model_name='bert-base',
        learning_rate=lr,
        batch_size=32,
        num_epochs=10
    )

    job_specs.append(JobSpec(
        main_fn=train_model,
        log_relpath=f'experiments/bert/lr{{lr}}',
        entrypoint_config=config
    ))

launch_swarm(job_specs, SwarmConfig('bert-lr-sweep'))
```
""")


def main():
    """Run demo based on command line argument"""
    if len(sys.argv) < 2 or sys.argv[1] not in ["1", "2", "3", "4", "real", "all"]:
        print("""
Usage: python ml_sweep.py <demo_number> [--dry-run]

Demos:
  1 - Individual windows (no grouping)
  2 - Grouped by model
  3 - Mixed grouping strategy
  4 - Large scale sweep
  real - Show real usage pattern
  all - Run all demos

Add --dry-run to see what would be executed without running
""")
        sys.exit(1)

    demo = sys.argv[1]

    # Run requested demo(s)
    if demo == "1":
        demo_individual_windows()
    elif demo == "2":
        demo_grouped_by_model()
    elif demo == "3":
        demo_mixed_grouping()
    elif demo == "4":
        demo_large_scale()
    elif demo == "real":
        demo_real_usage()
    elif demo == "all":
        demo_individual_windows()
        input("\nPress Enter to continue to next demo...")
        demo_grouped_by_model()
        input("\nPress Enter to continue to next demo...")
        demo_mixed_grouping()
        input("\nPress Enter to continue to next demo...")
        demo_large_scale()
        input("\nPress Enter to continue to next demo...")
        demo_real_usage()


if __name__ == "__main__":
    main()
