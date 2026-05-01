"""Dataset loading, tokenization, and advantage computation."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Iterator, List

import torch
import requests

from fireworks.training.sdk.errors import request_with_retries

logger = logging.getLogger(__name__)


class RLPromptDataset:
    """Batch-indexed prompt dataset for RL training.

    Follows tinker_cookbook's dataset pattern (``get_batch`` / ``__len__``)
    but returns raw row dicts instead of ``EnvGroupBuilder``.
    """

    def __init__(self, rows: list[dict], prompts_per_step: int):
        self.rows = rows
        self.prompts_per_step = prompts_per_step

    def get_batch(self, index: int) -> list[dict]:
        start = index * self.prompts_per_step
        end = min(start + self.prompts_per_step, len(self.rows))
        return self.rows[start:end]

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.prompts_per_step) if self.rows else 0


def load_jsonl_dataset(path_or_url: str, max_rows: int | None = None) -> List[Dict[str, Any]]:
    """Load a JSONL dataset from a local path or URL."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = request_with_retries(requests.get, path_or_url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
    else:
        with open(path_or_url) as f:
            lines = f.readlines()

    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if max_rows is not None and len(rows) >= max_rows:
            break
    logger.info("Loaded %d examples from dataset", len(rows))
    return rows


def _to_msgs(v: Any) -> List[Dict[str, Any]]:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [{"role": "assistant", "content": v}]
    return []


def normalize_preference_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize one JSONL row to ``{"chosen": ..., "rejected": ...}`` or None.

    Supports three on-disk formats:

      * ``{"chosen": ..., "rejected": ...}`` — pass through.
      * ``{"samples": [...]}`` with per-sample ``evals.score`` (or ``score``)
        of 1.0 / 0.0 — derive chosen/rejected.
      * ``{"input": ..., "preferred_output": ..., "non_preferred_output": ...}``
        (OpenAI-style preference SFT) — derive chosen/rejected.
    """
    if "chosen" in row and "rejected" in row:
        return row
    if "samples" in row:
        chosen = rejected = None
        for s in row["samples"]:
            score = s.get("evals", {}).get("score", s.get("score"))
            if score == 1.0:
                chosen = s
            elif score == 0.0:
                rejected = s
        if chosen and rejected:
            return {"chosen": chosen, "rejected": rejected}
        return None
    if "preferred_output" in row and "non_preferred_output" in row:
        inp = row.get("input", {})
        if isinstance(inp, dict) and "messages" in inp:
            input_msgs = inp["messages"]
        elif isinstance(inp, list):
            input_msgs = inp
        elif isinstance(inp, str):
            input_msgs = [{"role": "user", "content": inp}]
        else:
            input_msgs = []
        return {
            "chosen": {"messages": input_msgs + _to_msgs(row["preferred_output"])},
            "rejected": {"messages": input_msgs + _to_msgs(row["non_preferred_output"])},
        }
    return None


def iter_preference_examples(
    path: str, max_pairs: int | None = None,
) -> Iterator[Dict[str, Any]]:
    """Stream normalized preference examples from a JSONL file.

    Yields one ``{"chosen": ..., "rejected": ...}`` dict at a time without
    materialising the full list. Skips blank lines and rows that don't
    match any of the supported preference schemas. Stops after ``max_pairs``
    *valid* pairs when set.

    See :func:`load_preference_dataset` for the eager equivalent and the
    SFT v2 streaming render fix (fw-ai/cookbook#371) for the motivating
    OOM context.
    """
    yielded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = normalize_preference_row(json.loads(line))
            if pair is None:
                continue
            yield pair
            yielded += 1
            if max_pairs is not None and yielded >= max_pairs:
                return


def load_preference_dataset(path: str, max_pairs: int | None = None) -> List[dict[str, Any]]:
    """Load preference dataset (chosen/rejected pairs).

    Supports three input shapes per row:

    - ``{"chosen": ..., "rejected": ...}`` -- Fireworks / OpenAI-compatible.
    - ``{"samples": [{"messages": ..., "score": 0.0 | 1.0}, ...]}`` -- our
      legacy preference-sample format. Scores are *strictly* binary: 1.0
      marks the chosen sample, 0.0 marks the rejected sample. Graded scores
      (e.g. 0.5, 0.8) and missing scores raise ``ValueError`` instead of
      being silently dropped.
    - ``{"input": ..., "preferred_output": ..., "non_preferred_output": ...}``
      -- OpenAI fine-tuning DPO format.

    Unlike :func:`iter_preference_examples`, this eager loader is strict:
    malformed rows fail fast with file:line context so ORPO / older DPO
    callers surface customer dataset issues early.
    """
    data: list[dict[str, Any]] = []
    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "chosen" in row and "rejected" in row:
                data.append(row)
            elif "samples" in row:
                samples = row["samples"]
                if not isinstance(samples, list):
                    raise ValueError(
                        f"{path}:{line_no}: 'samples' must be a list, got "
                        f"{type(samples).__name__}."
                    )
                chosen = rejected = None
                for i, sample in enumerate(samples):
                    if not isinstance(sample, dict):
                        raise ValueError(
                            f"{path}:{line_no}: samples[{i}] must be a dict, "
                            f"got {type(sample).__name__}."
                        )
                    evals = sample.get("evals")
                    if evals is not None and not isinstance(evals, dict):
                        raise ValueError(
                            f"{path}:{line_no}: samples[{i}].evals must be a "
                            f"dict if present, got {type(evals).__name__}."
                        )
                    score = (evals or {}).get("score", sample.get("score"))
                    if score == 1.0:
                        if chosen is not None:
                            raise ValueError(
                                f"{path}:{line_no}: row has multiple samples "
                                f"with score=1.0; preference is ambiguous. "
                                f"Each samples row must have exactly one "
                                f"chosen (score=1.0) and one rejected "
                                f"(score=0.0) sample."
                            )
                        chosen = sample
                    elif score == 0.0:
                        if rejected is not None:
                            raise ValueError(
                                f"{path}:{line_no}: row has multiple samples "
                                f"with score=0.0; preference is ambiguous. "
                                f"Each samples row must have exactly one "
                                f"chosen (score=1.0) and one rejected "
                                f"(score=0.0) sample."
                            )
                        rejected = sample
                    else:
                        raise ValueError(
                            f"{path}:{line_no}: invalid preference score "
                            f"{score!r} in samples[{i}]. Samples-format "
                            f"preference rows must use exactly score=1.0 "
                            f"(chosen) or score=0.0 (rejected); graded/"
                            f"missing scores are not supported."
                        )
                if chosen is None or rejected is None:
                    raise ValueError(
                        f"{path}:{line_no}: samples row does not yield both a "
                        f"chosen (score=1.0) and a rejected (score=0.0) "
                        f"sample (chosen={chosen is not None}, "
                        f"rejected={rejected is not None}). Each preference "
                        f"row must contain at least one of each."
                    )
                data.append({"chosen": chosen, "rejected": rejected})
            elif "preferred_output" in row and "non_preferred_output" in row:
                inp = row.get("input", {})
                if isinstance(inp, dict) and "messages" in inp:
                    input_msgs = inp["messages"]
                elif isinstance(inp, list):
                    input_msgs = inp
                elif isinstance(inp, str):
                    input_msgs = [{"role": "user", "content": inp}]
                else:
                    input_msgs = []
                data.append(
                    {
                        "chosen": {"messages": input_msgs + _to_msgs(row["preferred_output"])},
                        "rejected": {"messages": input_msgs + _to_msgs(row["non_preferred_output"])},
                    }
                )
            else:
                raise ValueError(
                    f"{path}:{line_no}: row does not match any supported "
                    f"preference format. Expected one of:\n"
                    f"  - {{'chosen': ..., 'rejected': ...}}\n"
                    f"  - {{'samples': [{{'messages': ..., 'score': 0.0|1.0}}, ...]}}\n"
                    f"  - {{'input': ..., 'preferred_output': ..., 'non_preferred_output': ...}}\n"
                    f"Got keys: {sorted(row.keys())}"
                )
            if max_pairs is not None and len(data) >= max_pairs:
                break
    return data


def extract_text(item: dict[str, Any]) -> str:
    """Extract text from a chosen/rejected preference item."""
    if "text" in item:
        return item["text"]
    if "messages" in item:
        parts = []
        for msg in item["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)
    return ""


def prepare_sampling_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preserve multi-turn chat history, dropping only trailing assistant turns.

    RL prompt datasets should contain the full prior conversation. If the row
    accidentally includes a final assistant completion, strip only that tail so
    sampling resumes from the latest non-assistant turn.
    """
    prepared = list(messages)
    while prepared and prepared[-1].get("role") == "assistant":
        prepared.pop()
    return prepared


def find_common_prefix_length(tokens1: List[int], tokens2: List[int]) -> int:
    """Find the length of the longest common prefix between two token lists."""
    min_len = min(len(tokens1), len(tokens2))
    for i in range(min_len):
        if tokens1[i] != tokens2[i]:
            return i
    return min_len


def encode_text(
    base_url: str,
    text: str,
    timeout: int = 60,
    max_wait_time: float = 300,
) -> List[int]:
    """Encode text to tokens using the RLOR trainer's tokenizer endpoint.

    Uses ``request_with_retries`` to tolerate trainer pods that are still
    initializing (model/checkpoint loading) after the job is reported ready.
    """
    resp = request_with_retries(
        requests.post,
        f"{base_url}/api/v1/tokenizer/encode",
        json={"text": text},
        timeout=timeout,
        verify=False,
        max_wait_time=max_wait_time,
    )
    resp.raise_for_status()
    return resp.json()["tokens"]


def compute_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    """Compute normalised advantages from a group of rewards."""
    t = torch.tensor(rewards, dtype=torch.float32)
    mean_r = t.mean()
    std_r = t.std()
    if std_r < eps:
        std_r = torch.tensor(1.0)
    return ((t - mean_r) / (std_r + eps)).tolist()
