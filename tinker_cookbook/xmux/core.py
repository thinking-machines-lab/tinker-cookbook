"""Core data structures and main launch function for xmux"""

import os
import shlex
import subprocess
import sys
from collections.abc import Callable

import cloudpickle
from pydantic import BaseModel
from termcolor import colored

from .control import PaneJobInfo, SessionMetadata, WindowJobInfo, load_existing_metadata
from .utils import generate_unique_names, get_symbol_path


class JobSpec(BaseModel):
    """Specification for a single job in the swarm"""

    main_fn: Callable[..., object]  # function to run
    log_relpath: str  # path to log directory
    entrypoint_config: object  # argument to pass to main_fn
    tmux_window_name: str | None = None  # If set, groups jobs together

    def get_window_name(self, default_name: str) -> str:
        """Get the window name for this job"""
        return self.tmux_window_name or default_name


class SwarmConfig(BaseModel):
    """Configuration for launching a swarm of jobs"""

    sweep_name: str  # Becomes tmux session name
    max_panes_per_window: int = 4  # When grouping, split into multiple windows if needed
    use_pickle: bool = True  # Whether to use pickle for config serialization
    dry_run: bool = False  # If set, will create the session but not launch any jobs
    control_window_cmd: str | None = None  # Custom control window command
    status_format: str | None = None  # Custom status bar format
    debug: bool = False  # Whether to run jobs with pdb debugger
    verbose: bool = False  # Whether to enable verbose logging

    def get_session_name(self) -> str:
        """Get sanitized session name"""
        # Sanitize session name for tmux
        return "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in self.sweep_name)[:50]


class JobConfig(BaseModel):
    """Internal job configuration"""

    log_relpath: str
    entrypoint: str
    entrypoint_config: object
    num_gpus: int = 0


def _execute_command(cmdlist: list[str], verbose: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute a command and return the result"""
    cmd_str = shlex.join(cmdlist)
    if verbose:
        print(colored(f"Executing: {cmd_str}", "green"))
    return subprocess.run(cmdlist, check=True, stdout=sys.stderr, stderr=sys.stderr, text=True)


def _save_job_config(job_spec: JobSpec, use_pickle: bool, dry_run: bool) -> str:
    """Save job configuration and return the config path"""
    # Create log directory
    log_dir = os.path.expanduser(f"~/experiments/{job_spec.log_relpath}")
    os.makedirs(log_dir, exist_ok=True)

    # Get entrypoint path
    entrypoint = get_symbol_path(job_spec.main_fn)

    # Create JobConfig
    job_config = JobConfig(
        log_relpath=job_spec.log_relpath,
        entrypoint=entrypoint,
        entrypoint_config=job_spec.entrypoint_config,
        num_gpus=0,
    )

    # Save config
    if use_pickle:
        config_path = os.path.join(log_dir, "job_config.pickle")
        if not dry_run:
            with open(config_path, "wb") as f:
                cloudpickle.dump(job_config, f)
    else:
        config_path = os.path.join(log_dir, "job_config.json")
        if not dry_run:
            with open(config_path, "w") as f:
                _ = f.write(job_config.model_dump_json(indent=2))

    return config_path


class JobCommand(BaseModel):
    """Command to run a job"""

    command: str
    env: dict[str, str]


def _get_tmux_env_flags(env: dict[str, str]) -> list[str]:
    """Get the flags to pass to tmux to set the environment"""
    key_value_pairs = [["-e", f"{key}={value}"] for key, value in env.items()]
    return sum(key_value_pairs, [])


def _create_job_command(config_path: str, debug: bool = False, dry_run: bool = False) -> JobCommand:
    """Create the command to run a job"""
    # Use the same Python interpreter that's running xmux
    log_dir = os.path.dirname(config_path)
    if debug:
        job_cmd_base = f"{sys.executable} -m pdb -c continue -m tinker_cookbook.xmux.run_job {shlex.quote(config_path)}"
    else:
        job_cmd_base = (
            f"{sys.executable} -m tinker_cookbook.xmux.run_job {shlex.quote(config_path)}"
        )

    if dry_run:
        job_cmd_base = 'echo "dry run enabled, sleeping for 10 seconds" && sleep 10'

    # Wrap with marker creation - tmux will run this in a shell, so we can use shell syntax directly
    job_cmd_with_markers = (
        f"{job_cmd_base} && touch {shlex.quote(os.path.join(log_dir, '.completed'))} || "
        f"touch {shlex.quote(os.path.join(log_dir, '.failed'))}"
    )
    return JobCommand(command=job_cmd_with_markers, env=os.environ.copy())


def _session_exists(session_name: str) -> bool:
    """Check if a tmux session exists"""
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name], capture_output=True, check=False
    )
    return result.returncode == 0


def _get_next_window_index(metadata: SessionMetadata | None) -> int:
    """Get the next available window index for new jobs"""
    if not metadata or not metadata.job_mapping:
        return 1  # Start from 1 (0 is control window)

    # Find highest existing window index
    max_index = 0
    for window_idx in metadata.job_mapping:
        max_index = max(max_index, int(window_idx))

    return max_index + 1


def _merge_metadata(
    existing_metadata: SessionMetadata | None, new_metadata: SessionMetadata
) -> SessionMetadata:
    """Merge new job metadata with existing metadata"""
    if not existing_metadata:
        return new_metadata

    # Merge job_mapping
    merged_job_mapping: dict[str, WindowJobInfo] = {}
    if existing_metadata.job_mapping:
        merged_job_mapping.update(existing_metadata.job_mapping)
    if new_metadata.job_mapping:
        merged_job_mapping.update(new_metadata.job_mapping)

    # Merge window_groups
    merged_window_groups: dict[str, int] = {}
    if existing_metadata.window_groups:
        merged_window_groups.update(existing_metadata.window_groups)
    if new_metadata.window_groups:
        merged_window_groups.update(new_metadata.window_groups)

    # Merge pane_titles
    merged_pane_titles: dict[str, list[str]] = {}
    if existing_metadata.pane_titles:
        merged_pane_titles.update(existing_metadata.pane_titles)
    if new_metadata.pane_titles:
        merged_pane_titles.update(new_metadata.pane_titles)

    # Calculate new totals
    existing_total = existing_metadata.total_jobs
    new_total = new_metadata.total_jobs

    existing_ungrouped = existing_metadata.ungrouped_jobs
    new_ungrouped = new_metadata.ungrouped_jobs

    return SessionMetadata(
        session_name=existing_metadata.session_name or new_metadata.session_name,
        sweep_name=existing_metadata.sweep_name or new_metadata.sweep_name,
        total_jobs=existing_total + new_total,
        window_groups=merged_window_groups if merged_window_groups else None,
        ungrouped_jobs=existing_ungrouped + new_ungrouped,
        pane_titles=merged_pane_titles if merged_pane_titles else None,
        job_mapping=merged_job_mapping if merged_job_mapping else None,
    )


def _enable_pane_logging(
    session_name: str,
    window_index: int,
    pane_index: int,
    log_path: str,
    verbose: bool = False,
):
    """Enable logging for a specific pane"""
    pane_target = f"{session_name}:{window_index}.{pane_index}"
    # -o means "output" mode, -a would mean append (but pipe-pane appends by default)
    _ = _execute_command(
        ["tmux", "pipe-pane", "-t", pane_target, "-o", f"cat >> {shlex.quote(log_path)}"],
        verbose=verbose,
    )


def _configure_status_bar(session_name: str, sweep_name: str, verbose: bool = False) -> None:
    """Configure a multi-line status bar for the session"""

    # Status bar content
    # Line 1: Session info
    status_left = f"[{sweep_name}] "
    status_right = "Jobs: #{window_index}/#{session_windows} | #{host} | %H:%M"

    # Line 2 will be handled by status-format

    commands = [
        # Enable status bar
        ["tmux", "set-option", "-t", session_name, "status", "on"],
        # Make it 2 lines tall
        ["tmux", "set-option", "-t", session_name, "status", "2"],
        # Configure first line
        ["tmux", "set-option", "-t", session_name, "status-left", status_left],
        ["tmux", "set-option", "-t", session_name, "status-right", status_right],
        ["tmux", "set-option", "-t", session_name, "status-left-length", "40"],
        ["tmux", "set-option", "-t", session_name, "status-right-length", "60"],
        # Window list format (shortened names)
        ["tmux", "set-option", "-t", session_name, "window-status-format", "#I:#W"],
        [
            "tmux",
            "set-option",
            "-s",
            "-t",
            session_name,
            "window-status-current-format",
            "#[bold]#I:#W#[nobold]",
        ],
        # Colors
        [
            "tmux",
            "set-option",
            "-s",
            "-t",
            session_name,
            "status-style",
            "bg=colour235,fg=colour248",
        ],
        ["tmux", "set-option", "-t", session_name, "status-left-style", "fg=colour39,bold"],
        ["tmux", "set-option", "-t", session_name, "status-right-style", "fg=colour248"],
        # Window colors
        ["tmux", "set-option", "-t", session_name, "window-status-style", "fg=colour248"],
        [
            "tmux",
            "set-option",
            "-s",
            "-t",
            session_name,
            "window-status-current-style",
            "fg=colour39,bold",
        ],
        # Refresh interval
        ["tmux", "set-option", "-t", session_name, "status-interval", "5"],
    ]

    if verbose:
        print(colored(f"Configuring status bar for session '{session_name}'", "cyan"))
    for cmd in commands:
        _ = _execute_command(cmd, verbose=verbose)


def launch_swarm(job_specs: list[JobSpec], config: SwarmConfig) -> None:
    """Launch a swarm of experiments with a control window"""
    session_name = config.get_session_name()

    # Check if session already exists
    session_exists = _session_exists(session_name) and not config.dry_run
    existing_metadata: SessionMetadata | None = None
    starting_window_index = 1

    if session_exists:
        response = input(
            colored(
                f"Session '{session_name}' already exists. Add new jobs to existing session? (y/N): ",
                "yellow",
            )
        )
        if response.lower() not in ["y", "yes"]:
            print(colored("Aborted.", "red"))
            return
        if config.verbose:
            print(
                colored(
                    f"Adding new jobs to existing session '{session_name}'.",
                    "cyan",
                )
            )
        existing_metadata = load_existing_metadata(session_name)
        starting_window_index = _get_next_window_index(existing_metadata)
        if config.verbose:
            print(colored(f"Starting new jobs from window index {starting_window_index}", "cyan"))
    else:
        if config.verbose:
            print(colored(f"Creating new tmux session '{session_name}'", "cyan"))

    # Group jobs by window name
    window_groups: dict[str, list[JobSpec]] = {}
    ungrouped_jobs: list[JobSpec] = []

    for job_spec in job_specs:
        if job_spec.tmux_window_name:
            if job_spec.tmux_window_name not in window_groups:
                window_groups[job_spec.tmux_window_name] = []
            window_groups[job_spec.tmux_window_name].append(job_spec)
        else:
            ungrouped_jobs.append(job_spec)

    # Split large groups if they exceed max_panes_per_window
    final_window_groups: dict[str, list[JobSpec]] = {}
    for window_name, jobs in window_groups.items():
        if len(jobs) <= config.max_panes_per_window:
            final_window_groups[window_name] = jobs
        else:
            # Split into multiple windows
            for i in range(0, len(jobs), config.max_panes_per_window):
                chunk = jobs[i : i + config.max_panes_per_window]
                chunk_name = f"{window_name}-{i // config.max_panes_per_window + 1}"
                final_window_groups[chunk_name] = chunk

    ending_window_index = starting_window_index + len(final_window_groups) + len(ungrouped_jobs)

    # Create control window command
    control_script_path = os.path.join(os.path.dirname(__file__), "control.py")
    # Use sys.executable to use the same Python interpreter that's running this script
    control_cmd = (
        config.control_window_cmd or f"{sys.executable} {control_script_path} {session_name}"
    )

    # Create session only if it doesn't exist
    if not session_exists:
        # Create session with a placeholder window first
        _ = _execute_command(
            ["tmux", "new-session", "-d", "-s", session_name, "-n", "placeholder", "sleep 1"],
            verbose=config.verbose,
        )
        _ = _execute_command(
            ["tmux", "set-option", "-t", session_name, "key-table", session_name],
            verbose=config.verbose,
        )

        # Set session options immediately after creation
        # Set remain-on-exit as a global option for this session
        # This ensures all windows and panes in this session won't close on exit
        _ = _execute_command(
            ["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"],
            verbose=config.verbose,
        )

        # Enable mouse support - session-specific setting
        _ = _execute_command(
            ["tmux", "set-option", "-t", session_name, "mouse", "on"], verbose=config.verbose
        )

        # Add key binding to detach from the session
        _ = _execute_command(
            ["tmux", "bind-key", "-T", session_name, "q", "detach-client"],
            verbose=config.verbose,
        )

        # Add key binding to return to control window from any pane
        # 0 key will switch to the control window (window 0) - no prefix needed
        _ = _execute_command(
            [
                "tmux",
                "bind-key",
                "-T",
                session_name,
                "0",
                "select-window",
                "-t",
                f"{session_name}:0",
            ],
            verbose=config.verbose,
        )

        # Configure status bar
        _configure_status_bar(session_name, config.sweep_name, verbose=config.verbose)

    # If a new session is added, we need to add bindings for the new windows too
    for window_index in range(starting_window_index, ending_window_index):
        # We can only add single digit hotkeys
        if window_index < 0 or window_index > 9:
            continue
        _ = _execute_command(
            [
                "tmux",
                "bind-key",
                "-T",
                session_name,
                str(window_index),
                "select-window",
                "-t",
                f"{session_name}:{window_index}",
            ],
            verbose=config.verbose,
        )

    # Launch grouped jobs
    window_index = starting_window_index
    for window_name, jobs in final_window_groups.items():
        if config.verbose:
            print(colored(f"Creating window '{window_name}' with {len(jobs)} panes", "blue"))

        # Create first pane in new window
        first_job = jobs[0]
        config_path = _save_job_config(first_job, config.use_pickle, config.dry_run)
        job_cmd = _create_job_command(config_path, config.debug, config.dry_run)

        _ = _execute_command(
            [
                "tmux",
                "new-window",
                "-t",
                f"{session_name}:{window_index}",
            ]
            + _get_tmux_env_flags(job_cmd.env)
            + [
                "-n",
                window_name,
                job_cmd.command,
            ],
            verbose=config.verbose,
        )
        _ = _execute_command(
            [
                "tmux",
                "set-option",
                "-t",
                f"{session_name}:{window_index}",
                "remain-on-exit",
                "on",
            ],
            verbose=config.verbose,
        )

        # Enable logging for the first pane
        log_path = os.path.expanduser(f"~/experiments/{first_job.log_relpath}/log.txt")
        _enable_pane_logging(session_name, window_index, 0, log_path, config.verbose)

        # Add remaining panes
        for i, job in enumerate(jobs[1:], 1):
            config_path = _save_job_config(job, config.use_pickle, config.dry_run)
            job_cmd: JobCommand = _create_job_command(config_path, config.debug, config.dry_run)

            # Don't specify percentage, let tmux handle it
            _ = _execute_command(
                [
                    "tmux",
                    "split-window",
                    "-t",
                    f"{session_name}:{window_index}",
                    "-h",  # Split horizontally for better layout
                ]
                + _get_tmux_env_flags(job_cmd.env)
                + [
                    job_cmd.command,
                ],
                verbose=config.verbose,
            )

            # Enable logging for this pane
            log_path = os.path.expanduser(f"~/experiments/{job.log_relpath}/log.txt")
            _enable_pane_logging(session_name, window_index, i, log_path, config.verbose)

        # Even out the panes
        if len(jobs) > 1:
            _ = _execute_command(
                ["tmux", "select-layout", "-t", f"{session_name}:{window_index}", "tiled"],
                verbose=config.verbose,
            )

        window_index += 1

    # Generate smart abbreviated names for all jobs
    all_log_relpaths = [job.log_relpath for job in job_specs]
    abbreviated_names = generate_unique_names(all_log_relpaths, max_length=20)
    name_map = dict(zip(all_log_relpaths, abbreviated_names, strict=True))

    # Launch ungrouped jobs (each in its own window)
    for job_spec in ungrouped_jobs:
        # Use smart abbreviated name
        window_name = name_map[job_spec.log_relpath]

        if config.verbose:
            print(colored(f"Creating window '{window_name}' for individual job", "blue"))

        config_path = _save_job_config(job_spec, config.use_pickle, config.dry_run)
        job_cmd = _create_job_command(config_path, config.debug, config.dry_run)

        _ = _execute_command(
            [
                "tmux",
                "new-window",
                "-t",
                f"{session_name}:{window_index}",
            ]
            + _get_tmux_env_flags(job_cmd.env)
            + [
                "-n",
                window_name,
                job_cmd.command,
            ],
            verbose=config.verbose,
        )
        _ = _execute_command(
            [
                "tmux",
                "set-option",
                "-t",
                f"{session_name}:{window_index}",
                "remain-on-exit",
                "on",
            ],
            verbose=config.verbose,
        )

        # Enable logging for this window (only one pane, so pane index is 0)
        log_path = os.path.expanduser(f"~/experiments/{job_spec.log_relpath}/log.txt")
        _enable_pane_logging(session_name, window_index, 0, log_path, config.verbose)

        window_index += 1

    # Save swarm metadata
    metadata_path = os.path.expanduser(f"~/experiments/.xmux/{session_name}.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Build pane titles mapping using smart naming
    pane_titles: dict[str, list[str]] = {}
    for window_name, jobs in final_window_groups.items():
        # Always generate smart abbreviated names for panes based on log paths
        job_paths = [job.log_relpath for job in jobs]
        pane_titles[window_name] = generate_unique_names(job_paths, max_length=15)

    # Build complete job mapping - this is what control.py needs!
    job_mapping: dict[str, WindowJobInfo] = {}

    # Add grouped jobs
    window_idx = starting_window_index  # Start from calculated starting index
    for window_name, jobs in final_window_groups.items():
        panes: dict[str, PaneJobInfo] = {}
        for pane_idx, job in enumerate(jobs):
            display_name = (
                pane_titles[window_name][pane_idx]
                if pane_idx < len(pane_titles.get(window_name, []))
                else f"pane-{pane_idx}"
            )
            panes[str(pane_idx)] = PaneJobInfo(
                log_relpath=job.log_relpath,
                display_name=display_name,
            )
        job_mapping[str(window_idx)] = WindowJobInfo(
            window_name=window_name,
            panes=panes,
        )
        window_idx += 1

    # Add ungrouped jobs
    for job_spec in ungrouped_jobs:
        window_name = name_map[job_spec.log_relpath]
        job_mapping[str(window_idx)] = WindowJobInfo(
            window_name=window_name,
            panes={"0": PaneJobInfo(log_relpath=job_spec.log_relpath, display_name=window_name)},
        )
        window_idx += 1

    new_metadata = SessionMetadata(
        session_name=session_name,
        sweep_name=config.sweep_name,
        total_jobs=len(job_specs),
        window_groups={k: len(v) for k, v in final_window_groups.items()},
        ungrouped_jobs=len(ungrouped_jobs),
        pane_titles=pane_titles,
        job_mapping=job_mapping,
    )

    # Merge with existing metadata if session exists
    if session_exists and existing_metadata:
        final_metadata = _merge_metadata(existing_metadata, new_metadata)
    else:
        final_metadata = new_metadata

    with open(metadata_path, "w") as f:
        _ = f.write(final_metadata.model_dump_json(exclude_none=True, indent=2))

    # Create the control window only for new sessions
    if not session_exists:
        if config.verbose:
            print(colored("Creating control window", "cyan"))

        # Kill the placeholder window
        _ = _execute_command(
            ["tmux", "kill-window", "-t", f"{session_name}:0"], verbose=config.verbose
        )

        # Create the actual control window at index 0
        _ = _execute_command(
            [
                "tmux",
                "new-window",
                "-t",
                f"{session_name}:0",
                "-n",
                "control",
                control_cmd,
            ],
            verbose=config.verbose,
        )

        # Set remain-on-exit for the control window
        _ = _execute_command(
            ["tmux", "set-option", "-w", "-t", f"{session_name}:0", "remain-on-exit", "on"],
            verbose=config.verbose,
        )

    # Switch to control window
    _ = _execute_command(
        ["tmux", "select-window", "-t", f"{session_name}:0"], verbose=config.verbose
    )

    # Print summary
    print(colored("\n" + "=" * 60, "green"))
    if session_exists:
        print(
            colored(f"Jobs added to existing swarm '{config.sweep_name}'!", "green", attrs=["bold"])
        )
        print(colored(f"Session: {session_name}", "green"))
        print(colored(f"New jobs added: {len(job_specs)}", "green"))
        total_jobs_now = (existing_metadata.total_jobs if existing_metadata else 0) + len(job_specs)
        print(colored(f"Total jobs in session: {total_jobs_now}", "green"))
    else:
        print(
            colored(f"Swarm '{config.sweep_name}' launched successfully!", "green", attrs=["bold"])
        )
        print(colored(f"Session: {session_name}", "green"))
        print(colored(f"Total jobs: {len(job_specs)}", "green"))
    print(colored(f"New windows created: {window_index - starting_window_index}", "green"))
    print(colored("\nTo attach to the session:", "cyan"))
    print(colored(f"  tmux attach-session -t {session_name}", "cyan", attrs=["bold"]))
    print(colored("\nTo kill the entire swarm:", "yellow"))
    print(colored(f"  tmux kill-session -t {session_name}", "yellow", attrs=["bold"]))
    print(colored("=" * 60 + "\n", "green"))
