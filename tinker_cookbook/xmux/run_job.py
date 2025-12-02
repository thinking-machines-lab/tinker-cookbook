"""Minimal job runner for xmux"""

import argparse
import asyncio
import importlib
import inspect
import pickle
from collections.abc import Callable

from .core import JobConfig


def get_module_member(path_with_colon: str) -> Callable[..., object]:
    """Import a module member from a colon-separated path"""
    module_path, member_name = path_with_colon.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, member_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("config_path")
    args = parser.parse_args()

    # Load configuration
    config: JobConfig
    config_path: str = str(args.config_path)
    if config_path.endswith(".pickle"):
        with open(config_path, "rb") as f:
            loaded = pickle.load(f)  # type: ignore[assignment]
            if not isinstance(loaded, JobConfig):
                raise ValueError("Pickle file does not contain a JobConfig object")
            config = loaded
    elif config_path.endswith(".json"):
        with open(config_path, "r") as f:
            config = JobConfig.model_validate_json(f.read())
    else:
        raise ValueError(f"Unknown file extension: {config_path}")

    # Get and run the function
    function: Callable[..., object] = get_module_member(config.entrypoint)
    result: object = function(config.entrypoint_config)

    # Handle async functions
    if inspect.iscoroutine(result):
        _ = asyncio.run(result)


if __name__ == "__main__":
    main()
