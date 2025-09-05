import logging
import os
import shutil
from typing import Literal

logger = logging.getLogger(__name__)

LogdirBehavior = Literal["delete", "resume", "ask", "raise"]


def check_log_dir(log_dir: str, behavior_if_exists: LogdirBehavior):
    """
    Call this at the beginning of CLI entrypoint to training scripts. This handles
    cases that occur if we're trying to log to a directory that already exists.
    The user might want to resume, overwrite, or delete it.

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
    if os.path.exists(log_dir):
        if behavior_if_exists == "delete":
            logger.info(
                f"Log directory {log_dir} already exists. Will delete it and start logging there."
            )
            shutil.rmtree(log_dir)
        elif behavior_if_exists == "ask":
            while True:
                user_input = input(
                    f"Log directory {log_dir} already exists. What do you want to do? [delete, resume, exit]: "
                )
                if user_input == "delete":
                    shutil.rmtree(log_dir)
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
            raise ValueError(f"Log directory {log_dir} already exists. Will not delete it.")
        else:
            raise AssertionError(f"Invalid behavior_if_exists: {behavior_if_exists}")
    else:
        logger.info(
            f"Log directory {log_dir} does not exist. Will create it and start logging there."
        )
