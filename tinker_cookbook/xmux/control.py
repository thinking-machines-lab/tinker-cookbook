#!/usr/bin/env python
"""Control window for xmux - provides interactive interface for managing experiments"""

import contextlib
import curses
import os
import subprocess
import sys
import time
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    """Job status enumeration"""

    UNKNOWN = "unknown"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PaneJobInfo(BaseModel):
    """Information about a pane job from metadata"""

    log_relpath: str
    display_name: str


class WindowJobInfo(BaseModel):
    """Information about a window job from metadata"""

    window_name: str
    panes: dict[str, PaneJobInfo]


class SessionMetadata(BaseModel):
    """Session metadata structure"""

    session_name: str
    sweep_name: str | None = None
    total_jobs: int = 0
    window_groups: dict[str, int] | None = None
    ungrouped_jobs: int = 0
    pane_titles: dict[str, list[str]] | None = None
    job_mapping: dict[str, WindowJobInfo] | None = None


class PaneInfo(BaseModel):
    """Information about a tmux pane"""

    index: int
    pid: int | None
    dead: bool


class JobInfo(BaseModel):
    """Information about a job"""

    # Required fields (no defaults) must come first
    window_index: int
    window_name: str
    log_relpath: str

    # Optional fields (with defaults) come after
    pane_index: int | None = None
    status: JobStatus = JobStatus.UNKNOWN
    pid: int | None = None


def load_existing_metadata(session_name: str) -> SessionMetadata | None:
    """Load existing session metadata"""
    metadata_path = os.path.expanduser(f"~/experiments/.xmux/{session_name}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return SessionMetadata.model_validate_json(f.read())
    return None


class ControlWindow:
    """Interactive control window for managing xmux sessions"""

    def __init__(self, session_name: str):
        self.session_name: str = session_name
        self.jobs: list[JobInfo] = []
        self.selected_index: int = 0
        self.last_refresh: float = time.time()
        self.start_time: datetime = datetime.now()

        metadata = load_existing_metadata(self.session_name)
        assert metadata is not None
        self.metadata: SessionMetadata = metadata

        # Set up debug log file
        self.debug_log: str = os.path.expanduser(
            f"~/experiments/.xmux/{session_name}_control_debug.log"
        )
        os.makedirs(os.path.dirname(self.debug_log), exist_ok=True)

    def debug_print(self, msg: str) -> None:
        """Write debug messages to log file"""
        with open(self.debug_log, "a") as f:
            _ = f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    def _load_metadata(self) -> SessionMetadata:
        """Load session metadata"""
        metadata_path = os.path.expanduser(f"~/experiments/.xmux/{self.session_name}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return SessionMetadata.model_validate_json(f.read())
        return SessionMetadata(session_name=self.session_name)

    def _get_window_list(self) -> list[tuple[int, str]]:
        """Get list of windows in the session"""
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "list-windows",
                    "-t",
                    self.session_name,
                    "-F",
                    "#{window_index}:#{window_name}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            windows: list[tuple[int, str]] = []
            for line in result.stdout.strip().split("\n"):
                if line and ":" in line:
                    idx, name = line.split(":", 1)
                    if name != "control":  # Skip control window
                        windows.append((int(idx), name))
            return windows
        except subprocess.CalledProcessError:
            return []

    def _get_pane_info(self, window_index: int) -> list[PaneInfo]:
        """Get information about panes in a window"""
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-t",
                    f"{self.session_name}:{window_index}",
                    "-F",
                    "#{pane_index}:#{pane_pid}:#{pane_dead}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            panes: list[PaneInfo] = []
            for line in result.stdout.strip().split("\n"):
                if line and ":" in line:
                    parts = line.split(":")
                    panes.append(
                        PaneInfo(
                            index=int(parts[0]),
                            pid=int(parts[1]) if parts[1] else None,
                            dead=parts[2] == "1",
                        )
                    )
            return panes
        except subprocess.CalledProcessError:
            return []

    def _check_job_status(self, job: JobInfo) -> JobStatus:
        """Check the status of a job by looking at its log files"""
        log_dir = os.path.expanduser(f"~/experiments/{job.log_relpath}")

        # First check if process is still running
        is_dead = False
        if job.window_index:
            panes = self._get_pane_info(job.window_index)
            if job.pane_index is not None:
                # Specific pane
                for pane in panes:
                    if pane.index == job.pane_index:
                        is_dead = pane.dead
                        break
            else:
                # Window with single pane
                if panes:
                    is_dead = panes[0].dead

        # If still running, return immediately
        if not is_dead:
            return JobStatus.RUNNING

        self.debug_print(f"Looking in {log_dir}")
        # Process is dead - check for completion markers
        completed_path = os.path.join(log_dir, ".completed")
        failed_path = os.path.join(log_dir, ".failed")

        if os.path.exists(completed_path):
            self.debug_print(f"Found completed marker: {completed_path}")
            return JobStatus.COMPLETED
        elif os.path.exists(failed_path):
            self.debug_print(f"Found failed marker: {failed_path}")
            return JobStatus.FAILED
        else:
            # No markers found - this means the marker creation failed
            # or we're checking too soon. Treat as failed.
            self.debug_print(
                f"No markers found in {log_dir}. Files: {os.listdir(log_dir) if os.path.exists(log_dir) else 'DIR NOT FOUND'}"
            )
            return JobStatus.UNKNOWN

    def refresh_jobs(self) -> None:
        """Refresh the list of jobs and their statuses"""
        self.jobs = []
        windows = self._get_window_list()

        # Get job mapping from metadata
        self.metadata = self._load_metadata()
        job_mapping = self.metadata.job_mapping or {}

        for window_index, window_name in windows:
            panes = self._get_pane_info(window_index)

            # Get job info for this window from metadata
            window_job_info = job_mapping.get(str(window_index))
            if not window_job_info:
                continue
            pane_info = window_job_info.panes

            if len(panes) <= 1:
                # Single job in window
                # Get log_relpath from metadata
                pane_0 = pane_info.get("0")
                if not pane_0:
                    self.debug_print(
                        f"WARNING: No pane info found for window {window_index} ({window_name})"
                    )
                    # Skip this job if we don't have a valid path
                    continue
                log_relpath = pane_0.log_relpath
                if not log_relpath:
                    self.debug_print(
                        f"WARNING: No log_relpath found for window {window_index} ({window_name})"
                    )
                    # Skip this job if we don't have a valid path
                    continue

                job = JobInfo(
                    window_index=window_index,
                    window_name=window_name,
                    log_relpath=log_relpath,
                    pid=panes[0].pid if panes else None,
                )
                job.status = self._check_job_status(job)
                self.jobs.append(job)
            else:
                # Multiple jobs in window (grouped)
                for i, pane in enumerate(panes):
                    # Get info for this specific pane
                    pane_job_info = pane_info.get(str(i))
                    if not pane_job_info:
                        continue
                    log_relpath = pane_job_info.log_relpath

                    if not log_relpath:
                        self.debug_print(
                            f"WARNING: No log_relpath found for window {window_index} pane {i} ({window_name})"
                        )
                        # Skip this pane if we don't have a valid path
                        continue

                    display_name = pane_job_info.display_name or f"{window_name}[{i}]"

                    # Use full name with window prefix for grouped panes
                    full_display_name = f"{window_name}/{display_name}"

                    job = JobInfo(
                        window_index=window_index,
                        window_name=full_display_name,
                        log_relpath=log_relpath,
                        pane_index=pane.index,
                        pid=pane.pid,
                    )
                    job.status = self._check_job_status(job)
                    self.jobs.append(job)

        self.last_refresh = time.time()

    def draw_header(self, stdscr: curses.window, height: int, width: int) -> None:
        """Draw the header with session info"""
        # Title
        _ = height  # Parameter required for interface consistency
        sweep_name = self.metadata.sweep_name or self.session_name
        title = f"XMUX CONTROL: {sweep_name}"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(1))

        # Stats
        total = len(self.jobs)
        running = sum(1 for j in self.jobs if j.status == JobStatus.RUNNING)
        completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self.jobs if j.status == JobStatus.FAILED)

        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)

        stats = f"Jobs: {total} | Running: {running} | Completed: {completed} | Failed: {failed} | Uptime: {hours}h{minutes}m"
        stdscr.addstr(2, 2, stats)

        # Separator
        stdscr.addstr(3, 0, "=" * width, curses.color_pair(1))

    def draw_jobs(self, stdscr: curses.window, height: int, width: int) -> None:
        """Draw the job list"""
        # Column headers
        headers = f"{'Win':>4} {'Name':<30} {'Status':<12} {'Command':<20}"
        stdscr.addstr(5, 2, headers, curses.A_BOLD)
        stdscr.addstr(6, 0, "-" * width)

        # Job list (with scrolling)
        list_start = 7
        list_height = height - list_start - 6  # Leave room for footer

        # Calculate scroll position
        if self.selected_index >= list_height:
            scroll_offset = self.selected_index - list_height + 1
        else:
            scroll_offset = 0

        for i, job in enumerate(self.jobs[scroll_offset : scroll_offset + list_height]):
            y = list_start + i
            idx = scroll_offset + i

            # Highlight selected
            attr = curses.A_REVERSE if idx == self.selected_index else 0

            # Status color
            if job.status == JobStatus.RUNNING:
                status_attr = curses.color_pair(2)  # Green
            elif job.status == JobStatus.COMPLETED:
                status_attr = curses.color_pair(4)  # Cyan
            elif job.status == JobStatus.FAILED:
                status_attr = curses.color_pair(3)  # Red
            else:
                status_attr = 0

            # Format job info
            win_str = f"{job.window_index:>4}"
            name_str = job.window_name[:30].ljust(30)
            status_str = str(job.status.value).upper()[:12].ljust(12)

            # Draw line
            stdscr.addstr(y, 2, win_str, attr)
            stdscr.addstr(y, 7, name_str, attr)
            stdscr.addstr(y, 38, status_str, attr | status_attr)

    def draw_footer(self, stdscr: curses.window, height: int, width: int) -> None:
        """Draw the footer with commands"""
        y = height - 4
        stdscr.addstr(y, 0, "-" * width)

        commands = [
            "[0] Control window",
            "[1-9] Go to job group window",
            "[k] Kill job",
            "[K] Kill group",
            "[r] Refresh",
            "[q] Detach",
            "[↑↓] Navigate",
            "[Enter] Select window",
        ]

        y += 1
        cmd_str = " | ".join(commands)
        stdscr.addstr(y, 2, cmd_str[: width - 4])

        # Status line
        y += 2
        status = f"Last refresh: {int(time.time() - self.last_refresh)}s ago"
        stdscr.addstr(y, 2, status)

    def handle_input(self, _stdscr: curses.window, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord("r"):
            self.refresh_jobs()

        elif key == curses.KEY_UP:
            self.selected_index = max(0, self.selected_index - 1)

        elif key == curses.KEY_DOWN:
            self.selected_index = min(len(self.jobs) - 1, self.selected_index + 1)

        elif ord("0") <= key <= ord("9"):
            # Jump to window
            window_num = key - ord("0")
            for job in self.jobs:
                if job.window_index == window_num:
                    # Switch to window
                    _ = subprocess.run(
                        ["tmux", "select-window", "-t", f"{self.session_name}:{window_num}"]
                    )
                    break

        elif key == ord("\n") or key == ord(" "):
            # Select job
            job = self.jobs[self.selected_index]
            _ = subprocess.run(
                ["tmux", "select-window", "-t", f"{self.session_name}:{job.window_index}"]
            )

        elif key == ord("k") and self.jobs:
            # Kill selected job
            job = self.jobs[self.selected_index]
            if job.pane_index is not None:
                # Kill specific pane
                _ = subprocess.run(
                    [
                        "tmux",
                        "kill-pane",
                        "-t",
                        f"{self.session_name}:{job.window_index}.{job.pane_index}",
                    ]
                )
            else:
                # Kill window
                _ = subprocess.run(
                    ["tmux", "kill-window", "-t", f"{self.session_name}:{job.window_index}"]
                )
            self.refresh_jobs()

        elif key == ord("K") and self.jobs:
            # Kill entire window group
            job = self.jobs[self.selected_index]
            _ = subprocess.run(
                ["tmux", "kill-window", "-t", f"{self.session_name}:{job.window_index}"]
            )
            self.refresh_jobs()

        elif key == ord("q") or key == 27:  # q or ESC key
            # Detach from the tmux session but keep control window alive
            _ = subprocess.run(["tmux", "detach-client"])
            # Don't return False - keep the control window running

        return True

    def run(self, stdscr: curses.window) -> None:
        """Main UI loop"""
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Header
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Running
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)  # Failed
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)  # Completed

        # Configure terminal
        _ = curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(1000)  # Refresh every second

        # Initial refresh
        self.refresh_jobs()

        while True:
            try:
                height, width = stdscr.getmaxyx()
                stdscr.clear()

                # Draw UI
                self.draw_header(stdscr, height, width)
                self.draw_jobs(stdscr, height, width)
                self.draw_footer(stdscr, height, width)

                stdscr.refresh()

                # Handle input
                key = stdscr.getch()
                if key != -1 and not self.handle_input(stdscr, key):
                    break

                # Auto-refresh every 3 seconds
                if time.time() - self.last_refresh > 3:
                    self.refresh_jobs()

            except curses.error:
                # Handle terminal resize or other curses errors
                time.sleep(0.1)
                continue
            except KeyboardInterrupt:
                # Ignore Ctrl-C gracefully
                pass


def main() -> None:
    """Entry point for control window"""
    if len(sys.argv) < 2:
        print("Usage: control.py <session_name>")
        sys.exit(1)

    session_name = sys.argv[1]
    control = ControlWindow(session_name)

    with contextlib.suppress(KeyboardInterrupt):
        curses.wrapper(control.run)


if __name__ == "__main__":
    main()
