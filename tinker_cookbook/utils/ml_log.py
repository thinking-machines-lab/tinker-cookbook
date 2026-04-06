"""Simplified logging utilities for tinker-cookbook."""

import json
import logging
import os
import shlex
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import chz
from rich.console import Console
from rich.table import Table

from tinker_cookbook.exceptions import ConfigurationError
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
    """Recursively convert a configuration object to a JSON-serializable format.

    Handles ``chz`` dataclasses, standard dataclasses, dicts, lists, enums,
    callables, and plain objects with ``__dict__``.

    Args:
        config (Any): Configuration object to convert.

    Returns:
        Any: A JSON-serializable representation (dict, list, str, number, etc.).
    """
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif chz.is_chz(config):
        # Recursively dump values to handle nested non-serializable fields
        return {k: dump_config(v) for k, v in chz.asdict(config).items()}
    elif is_dataclass(config) and not isinstance(config, type):
        # Recursively dump values to handle nested non-serializable fields
        return {k: dump_config(v) for k, v in asdict(config).items()}
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
    """Abstract base class for metric/experiment loggers.

    Subclasses must implement :meth:`log_hparams` and :meth:`log_metrics`.
    Other methods have default no-op implementations and can be overridden
    as needed.
    """

    @abstractmethod
    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters/configuration.

        Args:
            config (Any): Configuration object (will be passed through
                :func:`dump_config` by callers).
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of metrics with an optional step number.

        Args:
            metrics (dict[str, Any]): Metric name-to-value mapping.
            step (int | None): Training step (or ``None`` for step-less logging).
        """
        pass

    def log_long_text(self, key: str, text: str) -> None:
        """Log long text content (optional to implement).

        Args:
            key (str): Identifier for the text entry.
            text (str): The text content to log.
        """
        pass

    def close(self) -> None:
        """Release resources and flush pending data (optional to implement)."""
        pass

    def sync(self) -> None:
        """Force synchronization of buffered data to the backend (optional to implement)."""
        pass

    def get_logger_url(self) -> str | None:
        """Return a permalink to view this logger's results, or ``None``.

        Returns:
            str | None: URL string if the backend provides one.
        """
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
    """Logger that writes metrics to a JSONL file and config to JSON.

    On first :meth:`log_hparams` call, writes ``config.json`` and
    ``code.diff`` (via :func:`code_state`) into *log_dir*.  Subsequent
    :meth:`log_metrics` calls append one JSON object per line to
    ``metrics.jsonl``.

    All file I/O goes through a :class:`~tinker_cookbook.stores.TrainingRunStore`,
    enabling cloud storage backends.

    Args:
        log_dir (str | Path): Directory for output files (created if missing).
        store (TrainingRunStore | None): Optional pre-configured store.
            If ``None`` (default), a ``LocalStorage``-backed store is created.
    """

    def __init__(self, log_dir: str | Path, store: "TrainingRunStore | None" = None) -> None:
        from tinker_cookbook.stores.storage import LocalStorage
        from tinker_cookbook.stores.training_store import TrainingRunStore as _TRS

        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or _TRS(LocalStorage(self.log_dir))
        self._logged_hparams = False

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to config.json and code diff."""
        if not self._logged_hparams:
            config_dict = dump_config(config)
            # Use _PermissiveJSONEncoder as safety net for non-serializable values
            sanitized = json.loads(json.dumps(config_dict, cls=_PermissiveJSONEncoder))
            self.store.write_config(sanitized)
            self.store.write_code_diff(code_state())
            self._logged_hparams = True

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append metrics to JSONL file."""
        self.store.write_metrics(metrics, step)
        logger.info("Wrote metrics to %s", self.store._storage.url("metrics.jsonl"))


class PrettyPrintLogger(Logger):
    """Logger that displays metrics as a Rich-formatted table in the console.

    Noisy aggregate keys (e.g. ``*:total``, ``*:count``) are hidden from
    console output but still available in other logger sinks.
    """

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

    # Metric suffixes to hide from console output. These are still logged to
    # metrics.jsonl and other sinks — only the pretty-print table is filtered.
    _HIDDEN_SUFFIXES = (":total", ":count")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Display metrics in console."""
        if not metrics:
            return

        # Filter out noisy aggregate keys from the console display
        display_items = [
            (k, v)
            for k, v in sorted(metrics.items())
            if not any(k.endswith(s) for s in self._HIDDEN_SUFFIXES)
        ]
        if not display_items:
            return

        # Adapt column width to the longest metric name
        max_key_len = max(len(k) for k, _ in display_items)
        metric_col_width = min(max(max_key_len, 20), 60)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=metric_col_width)
        table.add_column("Value", style="green")

        if step is not None:
            table.title = f"Step {step}"

        for key, value in display_items:
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
    """Logger that streams metrics and config to Weights & Biases.

    Requires ``wandb`` to be installed and ``WANDB_API_KEY`` to be set.

    Args:
        project (str | None): W&B project name.
        config (Any | None): Initial configuration to log.
        log_dir (str | Path | None): Local directory for W&B files.
        wandb_name (str | None): Display name for the W&B run.

    Raises:
        ImportError: If ``wandb`` is not installed.
        ConfigurationError: If ``WANDB_API_KEY`` is not set.
    """

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
            raise ConfigurationError("WANDB_API_KEY environment variable not set")

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

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
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
    """Logger that streams metrics and config to Neptune.

    Requires ``neptune-scale`` to be installed and ``NEPTUNE_API_TOKEN``
    to be set.

    Args:
        project (str | None): Neptune project name (``workspace/project``).
        config (Any | None): Initial configuration to log.
        log_dir (str | Path | None): Local log directory for Neptune files.
        neptune_name (str | None): Experiment display name.

    Raises:
        ImportError: If ``neptune-scale`` is not installed.
        ConfigurationError: If ``NEPTUNE_API_TOKEN`` is not set.
    """

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
            raise ConfigurationError("NEPTUNE_API_TOKEN environment variable not set")

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
        metrics: dict[str, Any],
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
    """Logger that streams metrics and config to Trackio.

    Requires ``trackio`` to be installed.

    Args:
        project (str | None): Trackio project name (defaults to ``"default"``).
        config (Any | None): Initial configuration to log.
        log_dir (str | Path | None): Local log directory (unused by Trackio
            but accepted for interface consistency).
        trackio_name (str | None): Display name for the run.

    Raises:
        ImportError: If ``trackio`` is not installed.
    """

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

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to trackio."""
        if self.run and trackio is not None:
            trackio.log(metrics, step=step)
            logger.info("Logged metrics to Trackio project: %s", self.run.project)

    def close(self) -> None:
        """Close trackio run."""
        if self.run and trackio is not None:
            trackio.finish()


class MultiplexLogger(Logger):
    """Logger that fans out every operation to multiple child loggers.

    This is the logger returned by :func:`setup_logging` and is the primary
    interface callers use to log metrics, hyperparameters, and text.

    Args:
        loggers (list[Logger]): Child loggers to forward calls to.
    """

    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log_hparams(self, config: Any) -> None:
        """Forward log_hparams to all child loggers."""
        for logger in self.loggers:
            logger.log_hparams(config)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
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

    @property
    def store(self) -> "TrainingRunStore | None":
        """Return the TrainingRunStore from the JsonLogger child, if any."""
        for lg in self.loggers:
            if isinstance(lg, JsonLogger):
                return lg.store
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


def _get_command_line_invocation() -> str:
    """Return the current command line in a shell-safe form."""
    if not sys.argv:
        return "<empty sys.argv>"
    return shlex.join(sys.argv)


def configure_logging_module(path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure the Python ``logging`` module with coloured console and plain file handlers.

    Replaces any previously installed root handlers (like ``basicConfig(..., force=True)``).
    The console handler uses ANSI colours for level names; the file handler
    writes plain text.

    Args:
        path (str): File path for the log file (appended to, created if missing).
        level (int): Root logger level (default ``logging.INFO``).

    Returns:
        logging.Logger: The root logger instance.
    """
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
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(name)s:%(lineno)d [%(levelname)s] %(message)s"))

    # Force override like basicConfig(..., force=True)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.info("Command line invocation: %s", _get_command_line_invocation())

    return root
