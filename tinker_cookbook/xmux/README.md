# xmux - TMUX-based Experiment Launcher

xmux is a tool for launching and managing hierarchical ML experiments using TMUX. It provides an interactive control window for monitoring and managing large numbers of concurrent experiments.

## Key Features

- **Hierarchical Organization**: Session = Sweep, with a control window for management
- **Smart Grouping**: Group related experiments in the same window as panes
- **Interactive Control**: Navigate, monitor, and kill experiments from the control window
- **Smart Naming**: Automatic abbreviation of long experiment names
- **Multi-line Status Bar**: Clear overview of all running experiments

## Quick Start

```python
from tinker_cookbook.xmux import JobSpec, SwarmConfig, launch_swarm

# Define your experiments
job_specs = [
    JobSpec(
        main_fn=train_model,  # Your training function
        log_relpath="sweep/model1/lr0.001",
        entrypoint_config={"model": "bert", "lr": 0.001}
    ),
    # ... more experiments
]

# Launch the swarm
config = SwarmConfig(sweep_name="my-lr-sweep")
launch_swarm(job_specs, config)
```

## Grouping Experiments

You can group related experiments into the same window:

```python
# Group by model type
JobSpec(
    main_fn=train_model,
    log_relpath="sweep/bert/lr0.001",
    entrypoint_config=config,
    tmux_window_name="bert",  # Groups all BERT experiments
    pane_title="lr0.001"      # Shows in the pane
)
```

## Using the Control Window

After launching, attach to the TMUX session:

```bash
tmux attach-session -t my-lr-sweep
```

Control window commands:
- **0-9**: Jump to window by number
- **↑↓**: Navigate job list
- **k**: Kill selected job
- **K**: Kill entire window group
- **r**: Refresh status
- **q**: Quit control window

## Adding to an Existing Experiment

If you already have an existing session, you can add
additional jobs to the experiment by using the same
sweep name.

## Examples

See `examples/ml_sweep.py` for complete examples:

```bash
# Run demo with dry-run to see what would happen
python examples/ml_sweep.py 1 --dry-run

# Run actual experiments
python examples/ml_sweep.py 2

# Demo options:
# 1 - Individual windows (no grouping)
# 2 - Grouped by model
# 3 - Mixed grouping strategy
# 4 - Large scale sweep (72 experiments)
```

## Tips

1. **Kill entire sweep**: `tmux kill-session -t sweep-name`
2. **List xmux sessions**: Look for sessions with metadata in `~/experiments/.xmux/`
3. **Window limit**: Use grouping for large sweeps to avoid too many windows
4. **Pane limit**: Set `max_panes_per_window` to control pane overflow
