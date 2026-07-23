"""One-time Expresso download + data preparation.

Turns the raw 36 GB Expresso release into everything the recipes read at
runtime, so the training/eval code never touches raw audio or split logic:

- downloads the official tar (resumable; verified against its published md5)
  and extracts it -- both steps are skipped when already done,
- joins the read-speech transcriptions with the official train/dev/test
  splits (only the short "base" read clips have transcriptions; longform and
  singing are skipped),
- splits train into ``sft_train`` -- one style-balanced rendition per
  distinct sentence (Expresso reads each sentence ~4.5x across styles and
  speakers) -- and ``rl_train``, drawn from the remaining renditions, so the
  two training stages see clip-disjoint data; both are capped at what the
  reference runs consume (see ``N_SFT_CLIPS`` / ``N_RL_CLIPS``) to keep the
  one-time transcode quick,
- transcodes the selected clips with ffmpeg to 16 kHz / 16-bit WAVs (the
  DMel audio decoder cannot read Expresso's 48 kHz / 24-bit originals, and
  16 kHz is the encoder's native rate),
- writes one JSONL manifest per split (``sft_train``, ``rl_train``, ``dev``,
  ``test``) with everything the recipes need per clip: id, transcript,
  style, wav path (relative to the manifest), frame count, sample rate.

Requires ``ffmpeg`` (transcode) and ``curl`` (download) on PATH. Idempotent
-- the download, extraction, and already-transcoded clips are all skipped on
re-runs.

    uv run python -m tinker_cookbook.recipes.audio.emotion.prepare_data \
        data_path=<data-dir>

writes ``<data-dir>/expresso`` (the raw release; deletable afterwards) and
``<data-dir>/expresso_16khz`` (what the recipes read: pass it to them as
``data_dir=<data-dir>/expresso_16khz``).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chz

from tinker_cookbook.recipes.audio.grading import normalize_text

EXPRESSO_URL = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar"
EXPRESSO_MD5 = "6bf580a4cd4392ae2473626147b5307c"

SAMPLE_RATE = 16_000  # the DMel audio encoder's native rate
SPLITS = ("train", "dev", "test")
SHUFFLE_SEED = 0  # split files are ordered by style; manifests are shuffled

# Training manifests are sized for the reference recipe (100 SFT steps +
# 50 RL steps, both at batch/groups_per_batch 16) with nothing to spare;
# dev and test are kept whole for evaluation.
N_SFT_CLIPS = 1_600  # 100 steps x batch_size 16
N_RL_CLIPS = 800  # 50 steps x groups_per_batch 16


def _balanced_unique_indices(texts: list[str], styles: list[str]) -> list[int]:
    """One index per unique text, greedily picking the rendition whose style
    is least represented so far (ties: first seen). Expresso reads the same
    sentence in several styles, so a uniform pick would skew the style mix."""
    from collections import Counter

    by_text: dict[str, list[int]] = {}
    for i, text in enumerate(texts):
        by_text.setdefault(text, []).append(i)
    style_counts: Counter[str] = Counter()
    picked: list[int] = []
    for renditions in by_text.values():
        best = min(renditions, key=lambda i: style_counts[styles[i]])
        style_counts[styles[best]] += 1
        picked.append(best)
    return picked


def _transcode(src: Path, dst: Path) -> int:
    """ffmpeg ``src`` -> 16 kHz / 16-bit mono WAV at ``dst`` (idempotent);
    return the frame count."""
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src)]
            + ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
            check=True,
        )
        tmp.replace(dst)  # atomic, so an interrupted run never leaves torn files
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE
        return w.getnframes()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(data_path: Path) -> Path:
    """Download + extract the Expresso release under ``data_path`` (skipping
    whatever already exists); return the extracted ``expresso/`` root."""
    root = data_path / "expresso"
    if (root / "read_transcriptions.txt").exists():
        print(f"Expresso already extracted at {root}.")
        return root

    tar_path = data_path / "expresso.tar"
    data_path.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl") is None:
        sys.exit(f"curl is required to download Expresso (or place the tar at {tar_path}).")
    print(f"Downloading {EXPRESSO_URL} -> {tar_path} (36 GB, resumable)...")
    subprocess.run(
        ["curl", "-L", "-C", "-", "-o", str(tar_path), EXPRESSO_URL],
        check=True,
    )

    print("Verifying md5...")
    if (digest := _md5(tar_path)) != EXPRESSO_MD5:
        sys.exit(f"md5 mismatch for {tar_path}: got {digest}, expected {EXPRESSO_MD5}")

    print(f"Extracting into {data_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(data_path, filter="data")
    print(f"Extracted: {root} (the tar can be deleted to reclaim 36 GB).")
    return root


def prepare(expresso_root: Path, out_dir: Path) -> None:
    """Build the 16 kHz clips + per-split JSONL manifests under ``out_dir``."""
    transcripts = dict(
        line.rstrip("\n").split("\t", 1)
        for line in (expresso_root / "read_transcriptions.txt").open(encoding="utf-8")
    )
    # read/{speaker}/{style}/base/{id}.wav; substyles (e.g. default_emphasis)
    # live under their parent style directory, so style = directory name.
    wavs = {p.stem: p for p in (expresso_root / "audio_48khz" / "read").glob("*/*/base/*.wav")}

    # Split id lists -> usable (clip_id, source wav) pairs, shuffled.
    by_split: dict[str, list[tuple[str, Path]]] = {}
    for split in SPLITS:
        pairs = []
        for line in (expresso_root / "splits" / f"{split}.txt").open(encoding="utf-8"):
            clip_id = line.rstrip("\n").split("\t")[0]
            path = wavs.get(clip_id)
            if path is not None and clip_id in transcripts:
                pairs.append((clip_id, path))
        random.Random(SHUFFLE_SEED).shuffle(pairs)
        by_split[split] = pairs

    # train -> sft_train (style-balanced sentence dedup) + rl_train (the rest).
    train = by_split.pop("train")
    picked = set(
        _balanced_unique_indices(
            [normalize_text(transcripts[clip_id]) for clip_id, _ in train],
            [path.parent.parent.name for _, path in train],
        )
    )
    by_split["sft_train"] = [p for i, p in enumerate(train) if i in picked][:N_SFT_CLIPS]
    by_split["rl_train"] = [p for i, p in enumerate(train) if i not in picked][:N_RL_CLIPS]

    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for split, pairs in by_split.items():
        with ThreadPoolExecutor(max_workers=16) as executor:
            num_frames = list(
                executor.map(
                    _transcode,
                    [src for _, src in pairs],
                    [wav_dir / f"{cid}.wav" for cid, _ in pairs],
                )
            )
        manifest = out_dir / f"{split}.jsonl"
        manifest.write_text(
            "".join(
                json.dumps(
                    {
                        "id": clip_id,
                        "text": transcripts[clip_id].strip(),
                        "emotion": src.parent.parent.name,
                        "path": f"wav/{clip_id}.wav",
                        "num_frames": frames,
                        "sample_rate": SAMPLE_RATE,
                    }
                )
                + "\n"
                for (clip_id, src), frames in zip(pairs, num_frames)
            ),
            encoding="utf-8",
        )
        print(f"{manifest}: {len(pairs)} clips")


@chz.chz
class Config:
    # Holds the raw expresso/ download and the prepared expresso_16khz/ output.
    data_path: str = "/tmp/tinker-examples/audio-data"


def cli_main(cfg: Config) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required (transcodes clips to 16 kHz).")
    data_path = Path(cfg.data_path).expanduser()
    expresso_root = download_and_extract(data_path)
    prepare(expresso_root, data_path / "expresso_16khz")


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
