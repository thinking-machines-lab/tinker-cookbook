import logging
from typing import Literal

from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.utils.misc_utils import is_uri

logger = logging.getLogger(__name__)

LogdirBehavior = Literal["delete", "resume", "ask", "raise"]

def check_log_dir(log_dir: str, behavior_if_exists: LogdirBehavior):
    """
    Call this at the beginning of CLI entrypoint to training scripts. This handles
    cases that occur if we're trying to log to a directory or URI that already
    exists. The user might want to resume, overwrite, or delete it.

    Args:
        log_dir: The directory to check.
        behavior_if_exists: What to do if the log directory already exists.

        "ask": Ask user if they want to delete the log directory.
        "resume": Continue to the training loop, which means we'll try to resume from the last checkpoint.
        "delete": Delete the log directory and start logging there.
        "raise": Raise an error if the log directory already exists.

    Returns:
        None
    """
    storage = storage_from_uri(log_dir, mkdir=False)
    label = f"Log URI {log_dir}" if is_uri(log_dir) else f"Log directory {log_dir}"
    if storage.exists_tree():
        if behavior_if_exists == "delete":
            logger.info(f"{label} already exists. Will delete it and start logging there.")
            storage.remove_tree()
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"{label} already exists. What do you want to do? [delete, resume, exit]: "
                )
                if user_input == "delete":
                    storage.remove_tree()
                    return
                elif user_input == "resume":
                    return
                elif user_input == "exit":
                    exit(0)
                else:
                    logger.warning(
                        f"Invalid input: {user_input}. Please enter 'delete', 'resume', or 'exit'."
                    )
        elif behavior_if_exists == "resume":
            return
        elif behavior_if_exists == "raise":
            raise ConfigurationError(f"{label} already exists. Will not delete it.")
        else:
            raise AssertionError(f"Invalid behavior_if_exists: {behavior_if_exists}")
    else:
        logger.info(f"{label} does not exist. Will create it and start logging there.")
