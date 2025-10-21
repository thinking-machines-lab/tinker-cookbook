"""Simplified logging utilities for tinker-cookbook."""

import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import chz
from rich.console import Console
from rich.table import Table
from tinker_cookbook.utils.code_state import code_state

logger = logging.getLogger(__name__)

# Check WandB availability
_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    wandb = None

# Check Neptune availability
_neptune_available = False
try:
    from neptune_scale import Run as NeptuneRun

    _neptune_available = True
except ImportError:
    NeptuneRun = None

# Check Trackio availability
_trackio_available = False
try:
    import trackio

    _trackio_available = True
except ImportError:
    trackio = None


def dump_config(config: Any) -> Any:
    """Convert configuration object to JSON-serializable format."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif chz.is_chz(config):
        return chz.asdict(config)
    elif is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    elif isinstance(config, dict):
        return {k: dump_config(v) for k, v in config.items()}
    elif isinstance(config, (list, tuple)):
        return [dump_config(item) for item in config]
    elif isinstance(config, Enum):
        return config.value
    elif hasattr(config, "__dict__"):
        # Handle simple objects with __dict__
        return {
            k: dump_config(v) for k, v in config.__dict__.items() if not k.startswith(("_", "X_"))
        }
    elif callable(config):
        # For callables, return their string representation
        return f"{config.__module__}.{config.__name__}"
    else:
        return config


class Logger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters/configuration."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics dictionary with optional step number."""
        pass

    def log_long_text(self, key: str, text: str) -> None:
        """Log long text content (optional to implement)."""
        pass

    def close(self) -> None:
        """Cleanup when done (optional to implement)."""
        pass

    def sync(self) -> None:
        """Force synchronization (optional to implement)."""
        pass

    def get_logger_url(self) -> str | None:
        """Get a permalink to view this logger's results."""
        return None


class _PermissiveJSONEncoder(json.JSONEncoder):
    """A JSON encoder that handles non-encodable objects by converting them to their type string."""

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            # Only handle the truly non-encodable objects
            return str(type(o))


class JsonLogger(Logger):
    """Logger that writes metrics to a JSONL file."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self._logged_hparams = False

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to a separate config.json file."""
        if not self._logged_hparams:
            config_dict = dump_config(config)
            config_file = self.log_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2, cls=_PermissiveJSONEncoder)
            diff_file = code_state()
            with open(self.log_dir / "code.diff", "w") as f:
                f.write(diff_file)
            self._logged_hparams = True

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Append metrics to JSONL file."""
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            logger.info("Wrote metrics to %s", self.metrics_file)


class PrettyPrintLogger(Logger):
    """Logger that displays metrics in a formatted table in the console."""

    def __init__(self):
        self.console = Console()
        self._last_step = None

    def log_hparams(self, config: Any) -> None:
        """Print configuration summary."""
        config_dict = dump_config(config)
        with _rich_console_use_logger(self.console):
            self.console.print("[bold cyan]Configuration:[/bold cyan]")
            for key, value in config_dict.items():
                self.console.print(f"  {key}: {_maybe_truncate_repr(value)}")

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Display metrics in console."""
        if not metrics:
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")

        if step is not None:
            table.title = f"Step {step}"

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        with _rich_console_use_logger(self.console):
            self.console.print(table)


def _maybe_truncate_repr(value: Any) -> str:
    repr_value = repr(value)
    if len(repr_value) > 256:
        return repr_value[:128] + " ... " + repr_value[-128:]
    return repr_value


@contextmanager
def _rich_console_use_logger(console: Console):
    with console.capture() as capture:
        yield
    logger.info("\n" + capture.get().rstrip())
    # ^^^ add a leading newline so things like table formatting work properly


class WandbLogger(Logger):
    """Logger for Weights & Biases."""

    def __init__(
        self,
        project: str | None = None,
        config: Any | None = None,
        log_dir: str | Path | None = None,
        wandb_name: str | None = None,
    ):
        if not _wandb_available:
            raise ImportError(
                "wandb is not installed. Please install it with: "
                "pip install wandb (or uv add wandb)"
            )

        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY environment variable not set")

        # Initialize wandb run
        assert wandb is not None  # For type checker
        self.run = wandb.init(
            project=project,
            config=dump_config(config) if config else None,
            dir=str(log_dir) if log_dir else None,
            name=wandb_name,
        )

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to wandb."""
        if self.run and wandb is not None:
            wandb.config.update(dump_config(config))

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb."""
        if self.run and wandb is not None:
            wandb.log(metrics, step=step)
            logger.info("Logging to: %s", self.run.url)

    def close(self) -> None:
        """Close wandb run."""
        if self.run and wandb is not None:
            wandb.finish()

    def get_logger_url(self) -> str | None:
        """Get the URL of the wandb run."""
        if self.run and wandb is not None:
            return self.run.url
        return None


class NeptuneLogger(Logger):
    """Logger for Neptune."""

    def __init__(
        self,
        project: str | None = None,
        config: Any | None = None,
        log_dir: str | Path | None = None,
        neptune_name: str | None = None,
    ):
        if not _neptune_available:
            raise ImportError(
                "neptune-scale is not installed. Please install it with: "
                "pip install neptune-scale (or uv add neptune-scale)"
            )

        if not os.environ.get("NEPTUNE_API_TOKEN"):
            raise ValueError("NEPTUNE_API_TOKEN environment variable not set")

        # Initialize neptune run
        assert NeptuneRun is not None  # For type checker
        self.run = NeptuneRun(
            project=project,
            log_directory=str(log_dir) if log_dir else None,
            experiment_name=neptune_name,
        )
        self.run.log_configs(dump_config(config) if config else None, flatten=True)

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to neptune."""
        if self.run and NeptuneRun is not None:
            self.run.log_configs(dump_config(config) if config else None, flatten=True)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: float | int | None = None,
    ) -> None:
        """Log metrics to neptune."""
        if self.run and NeptuneRun is not None:
            assert step is not None, "step is required to be int or float for Neptune logging."
            self.run.log_metrics(metrics, step=step)
            logger.info("Logging to: %s", self.run.get_run_url())

    def close(self) -> None:
        """Close neptune run."""
        if self.run and NeptuneRun is not None:
            self.run.close()


class TrackioLogger(Logger):
    """Logger for Trackio."""

    def __init__(
        self,
        project: str | None = None,
        config: Any | None = None,
        log_dir: str | Path | None = None,
        trackio_name: str | None = None,
    ):
        if not _trackio_available:
            raise ImportError(
                "trackio is not installed. Please install it with: "
                "pip install trackio (or uv add trackio)"
            )

        assert trackio is not None
        self.run = trackio.init(
            project=project or "default",
            name=trackio_name,
            config=dump_config(config) if config else None,
        )

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to trackio."""
        if self.run and trackio is not None:
            pass

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics to trackio."""
        if self.run and trackio is not None:
            trackio.log(metrics, step=step)
            logger.info("Logged metrics to Trackio project: %s", self.run.project)

    def close(self) -> None:
        """Close trackio run."""
        if self.run and trackio is not None:
            trackio.finish()


class MultiplexLogger(Logger):
    """Logger that forwards operations to multiple child loggers."""

    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def log_hparams(self, config: Any) -> None:
        """Forward log_hparams to all child loggers."""
        for logger in self.loggers:
            logger.log_hparams(config)

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Forward log_metrics to all child loggers."""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_long_text(self, key: str, text: str) -> None:
        """Forward log_long_text to all child loggers."""
        for logger in self.loggers:
            if hasattr(logger, "log_long_text"):
                logger.log_long_text(key, text)

    def close(self) -> None:
        """Close all child loggers."""
        for logger in self.loggers:
            if hasattr(logger, "close"):
                logger.close()

    def sync(self) -> None:
        """Sync all child loggers."""
        for logger in self.loggers:
            if hasattr(logger, "sync"):
                logger.sync()

    def get_logger_url(self) -> str | None:
        """Get the first URL returned by the child loggers."""
        for logger in self.loggers:
            if url := logger.get_logger_url():
                return url
        return None


def setup_logging(
    log_dir: str,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    config: Any | None = None,
    do_configure_logging_module: bool = True,
) -> Logger:
    """
    Set up logging infrastructure with multiple backends.

    Args:
        log_dir: Directory for logs
        wandb_project: W&B project name (if None, W&B logging is skipped)
        wandb_name: W&B run name
        config: Configuration object to log
        do_configure_logging_module: Whether to configure the logging module

    Returns:
        MultiplexLogger that combines all enabled loggers
    """
    # Create log directory
    log_dir_path = Path(log_dir).expanduser()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize loggers
    loggers = []

    # Always add JSON logger
    loggers.append(JsonLogger(log_dir_path))

    # Always add pretty print logger
    loggers.append(PrettyPrintLogger())

    # Add W&B logger if available and configured
    if wandb_project:
        if not _wandb_available:
            print("WARNING: wandb is not installed. Skipping W&B logging.")
        elif not os.environ.get("WANDB_API_KEY"):
            print("WARNING: WANDB_API_KEY environment variable not set. Skipping W&B logging. ")
        else:
            loggers.append(
                WandbLogger(
                    project=wandb_project,
                    config=config,
                    log_dir=log_dir_path,
                    wandb_name=wandb_name,
                )
            )

    # Add Neptune logger if available and configured
    # - MZ 10/8/25: Hack, but before doing bigger logger-agnostic refactor,
    #   allow Neptune to use the same W&B project and name.
    # - Project_name should be `workspace-name/project-name`.
    # - Also allow logging to both W&B and Neptune
    if wandb_project and _neptune_available:
        # if not _neptune_available:
        #     print("WARNING: neptune-scale is not installed. Skipping Neptune logging.")
        if not os.environ.get("NEPTUNE_API_TOKEN"):
            print(
                "WARNING: NEPTUNE_API_TOKEN environment variable not set. "
                "Skipping Neptune logging. "
            )
        else:
            loggers.append(
                NeptuneLogger(
                    project=wandb_project,
                    config=config,
                    log_dir=log_dir_path,
                    neptune_name=wandb_name,
                )
            )

    if wandb_project and _trackio_available:
        loggers.append(
            TrackioLogger(
                project=wandb_project,
                config=config,
                log_dir=log_dir_path,
                trackio_name=wandb_name,
            )
        )
        print(f"Trackio logging enabled for project: {wandb_project}")

    # Create multiplex logger
    ml_logger = MultiplexLogger(loggers)

    # Log initial configuration
    if config is not None:
        ml_logger.log_hparams(config)

    if do_configure_logging_module:
        configure_logging_module(str(log_dir_path / "logs.log"))

    logger.info(f"Logging to: {log_dir_path}")
    return ml_logger


def configure_logging_module(path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to console (color) and file (plain), forcing override of prior config."""
    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    class ColorFormatter(logging.Formatter):
        """Colorized log formatter for console output that doesn't mutate record.levelname."""

        def format(self, record: logging.LogRecord) -> str:
            color = COLORS.get(record.levelname, "")
            # add a separate attribute for the colored level name
            record.levelname_colored = f"{color}{record.levelname}{RESET}"
            return super().format(record)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColorFormatter("%(name)s:%(lineno)d [%(levelname_colored)s] %(message)s")
    )

    # File handler without colors
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(name)s:%(lineno)d [%(levelname)s] %(message)s"))

    # Force override like basicConfig(..., force=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    return root
