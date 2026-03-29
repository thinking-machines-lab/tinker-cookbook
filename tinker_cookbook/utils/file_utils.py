"""File I/O utilities for tinker-cookbook."""

import json


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL (JSON Lines) file and return a list of parsed dictionaries.

    Each line in the file is expected to be a valid JSON object.

    Args:
        path (str): Path to the JSONL file.

    Returns:
        list[dict]: List of dictionaries, one per line in the file.

    Example::

        records = read_jsonl("metrics.jsonl")
        for record in records:
            print(record["step"], record["loss"])
    """
    with open(path) as f:
        return [json.loads(line) for line in f]
