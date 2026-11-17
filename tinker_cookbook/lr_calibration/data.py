"""Pinned plain-text corpora for LR calibration.

Each example is a window of about ``window_chars`` characters cut from one
pretraining document. Windows are text, not tokens, so every model trains on
byte-identical data and bits per byte are comparable across tokenizers.
Documents are streamed from the start of one pinned shard, and the built file is
checked against a SHA-256 so that every calibration uses exactly the same text.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import huggingface_hub

from tinker_cookbook.exceptions import DataError
from tinker_cookbook.utils.misc_utils import safezip


@dataclass(frozen=True)
class CorpusSpec:
    """A pinned shard of a HuggingFace dataset of plain documents.

    Attributes:
        name (str): Short name used in file names and records.
        repo (str): HuggingFace dataset repo.
        revision (str): Git revision of the repo.
        path (str): Shard path inside the repo (``.jsonl.gz`` or ``.parquet``).
        sha256 (str): Expected SHA-256 of the built JSONL.
        n_windows (int): Number of windows to cut.
        window_chars (int): Target window length in characters.
        seed (int): Seed for window offsets and shuffling.
    """

    name: str
    repo: str
    revision: str
    path: str
    sha256: str
    n_windows: int = 8000
    window_chars: int = 4000
    seed: int = 0


CORPORA: dict[str, CorpusSpec] = {
    spec.name: spec
    for spec in [
        CorpusSpec(
            name="arxiv",
            repo="common-pile/arxiv_papers_filtered",
            revision="033cf7f53f9b348deec868c1a5a48484f3ee9e52",
            path="arxiv-papers-0000.json.gz",
            sha256="e64cfdae724a19f1bfea0283ea8477b0376fe9971297530cd2ebf9c58b7ef9c8",
        ),
        CorpusSpec(
            name="caselaw",
            repo="common-pile/caselaw_access_project_filtered",
            revision="50e1961a5ed8fdddcee64c9e66e1338de8953b63",
            path="CAP-Dolma-0000.json.gz",
            sha256="36597c6cc64a2f9f587f39c6158869cbefcca78e199de494615058ff9cb065c3",
        ),
        CorpusSpec(
            name="python",
            repo="common-pile/stackv2_edu_filtered",
            revision="c354dbe88469a1153e97c6a63ac50591849654de",
            path="stack-edu-0080.json.gz",
            sha256="2874846128c406bcf3d660d2ac599aa4793bca25e22316dab3e6947829f383eb",
        ),
        CorpusSpec(
            name="fineweb_edu",
            repo="HuggingFaceFW/fineweb-edu",
            revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            path="sample/10BT/000_00000.parquet",
            sha256="31106ed77810dfd872d927d0a57abed64914ea8e3ea921e98b9f9a6cb29d3afc",
        ),
    ]
}


def _iter_texts(spec: CorpusSpec) -> Iterator[str]:
    fs = huggingface_hub.HfFileSystem()
    url = f"datasets/{spec.repo}@{spec.revision}/{spec.path}"
    with fs.open(url, "rb", block_size=2**22) as f:
        if spec.path.endswith(".json.gz") or spec.path.endswith(".jsonl.gz"):
            for line in io.TextIOWrapper(
                gzip.GzipFile(fileobj=cast(BinaryIO, f)), encoding="utf-8"
            ):
                yield json.loads(line)["text"]
        elif spec.path.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                yield from pf.read_row_group(i, columns=["text"]).column("text").to_pylist()
        else:
            raise DataError(f"unsupported shard format: {spec.path}")


def cut_window(text: str, window_chars: int, rng: random.Random) -> str:
    """Cut a window that starts at a line (or word) boundary and ends at whitespace."""
    if len(text) <= window_chars:
        raise ValueError("document shorter than window")
    start = rng.randint(0, len(text) - window_chars)
    if start > 0:
        nl = text.find("\n", start, start + 400)
        if nl != -1:
            start = nl + 1
        else:
            sp = text.find(" ", start, start + 100)
            start = sp + 1 if sp != -1 else start
    end = min(start + window_chars, len(text))
    best = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
    if end < len(text) and best > start + window_chars // 2:
        end = best
    return text[start:end].strip()


def build_windows(spec: CorpusSpec) -> list[str]:
    """Stream the shard and cut ``spec.n_windows`` windows, shuffled with ``spec.seed``."""
    rng = random.Random(f"{spec.seed}:{spec.name}")
    rows: list[dict] = []
    for doc_idx, text in enumerate(_iter_texts(spec)):
        if len(text) < spec.window_chars + 200:
            continue
        window = cut_window(text, spec.window_chars, rng)
        if len(window) < spec.window_chars // 2:
            continue
        rows.append({"source_doc_index": doc_idx, "text": window})
        if len(rows) >= spec.n_windows:
            break
    if len(rows) < spec.n_windows:
        raise DataError(f"{spec.name}: only {len(rows)} windows, wanted {spec.n_windows}")
    random.Random(spec.seed).shuffle(rows)
    return [json.dumps(r) for r in rows]


def load_windows(spec: CorpusSpec, cache_dir: Path) -> list[str]:
    """Return the corpus windows, building and caching them on first use.

    Raises:
        DataError: If the built file does not match ``spec.sha256``.
    """
    path = (
        cache_dir / f"{spec.name}-{spec.revision[:12]}-{spec.n_windows}x{spec.window_chars}.jsonl"
    )
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        lines = build_windows(spec)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(line + "\n" for line in lines))
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.sha256:
        raise DataError(
            f"{path}: sha256 {digest[:12]} does not match pinned {spec.sha256}; "
            "the upstream shard or the windowing code changed"
        )
    with path.open() as f:
        return [json.loads(line)["text"] for line in f]


MIXTURES: dict[str, tuple[str, ...]] = {
    "pretrain_mix_v1": ("arxiv", "caselaw", "python", "fineweb_edu"),
}
"""Round-robin interleavings of corpora: every run of ``len(components)`` consecutive
windows has one window from each component, so batches and the held-out set are balanced."""


def load_corpus(name: str, cache_dir: Path) -> list[str]:
    """Return the windows of a corpus or mixture by name."""
    if name in CORPORA:
        return load_windows(CORPORA[name], cache_dir)
    if name not in MIXTURES:
        raise DataError(f"unknown corpus {name!r}; known: {sorted(CORPORA) + sorted(MIXTURES)}")
    parts = [load_windows(CORPORA[c], cache_dir) for c in MIXTURES[name]]
    return [window for group in safezip(*parts) for window in group]
