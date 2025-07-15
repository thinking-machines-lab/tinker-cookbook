"""Simplified logging utilities for tinker-cookbook."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import chz
from rich.console import Console
from rich.table import Table

_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    wandb = None


def dump_config(config: Any) -> Any:
    """Convert configuration object to JSON-serializable format."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
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
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        # Create new file (don't append)
        self.metrics_file.write_text("")
        self._logged_hparams = False

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to a separate config.json file."""
        if not self._logged_hparams:
            config_dict = dump_config(config)
            config_file = self.log_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2, cls=_PermissiveJSONEncoder)
            self._logged_hparams = True

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Append metrics to JSONL file."""
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class PrettyPrintLogger(Logger):
    """Logger that displays metrics in a formatted table in the console."""

    def __init__(self):
        self.console = Console()
        self._last_step = None

    def log_hparams(self, config: Any) -> None:
        """Print configuration summary."""
        config_dict = chz.asdict(config)
        self.console.print("[bold cyan]Configuration:[/bold cyan]")
        for key, value in config_dict.items():
            self.console.print(f"  {key}: {value}")

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

        self.console.print(table)


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
            raise ImportError("wandb is not installed. Please install it with: pip install wandb")

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

    def close(self) -> None:
        """Close wandb run."""
        if self.run and wandb is not None:
            wandb.finish()


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


def setup_logging(
    log_dir: str,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    config: Any | None = None,
) -> Logger:
    """
    Set up logging infrastructure with multiple backends.

    Args:
        log_dir: Directory for logs
        wandb_project: W&B project name (if None, W&B logging is skipped)
        config: Configuration object to log

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
    if wandb_project and _wandb_available and os.environ.get("WANDB_API_KEY"):
        loggers.append(
            WandbLogger(
                project=wandb_project, config=config, log_dir=log_dir_path, wandb_name=wandb_name
            )
        )

    # Create multiplex logger
    ml_logger = MultiplexLogger(loggers)

    # Log initial configuration
    if config is not None:
        ml_logger.log_hparams(config)

    print(f"Logging to: {log_dir_path}")
    return ml_logger
