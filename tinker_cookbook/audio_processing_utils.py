"""Audio helpers for tinker-cookbook messages."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tinker_cookbook.renderers.base import AudioPart

_AUDIO_FORMAT_BY_MIME = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}
_SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac")


def _decode_audio_data_uri(source: str, explicit_format: str | None) -> tuple[bytes, str]:
    header, separator, payload = source.partition(",")
    if not separator:
        raise ValueError("audio data URI must contain a comma-separated payload")
    if ";base64" not in header.lower():
        raise ValueError("audio data URI must use base64 encoding")

    mime = header[5:].split(";", 1)[0].lower()
    inferred_format = _AUDIO_FORMAT_BY_MIME.get(mime)
    if explicit_format and inferred_format and explicit_format != inferred_format:
        raise ValueError(
            f"AudioPart.format {explicit_format!r} disagrees with data URI MIME type {mime!r}"
        )
    audio_format = explicit_format or inferred_format
    if audio_format is None:
        raise ValueError(
            f"cannot infer audio format from data URI MIME type {mime!r}; "
            "set AudioPart.format explicitly"
        )
    try:
        return base64.b64decode(payload, validate=True), audio_format
    except ValueError as exc:
        raise ValueError("audio data URI payload is not valid base64") from exc


def _read_local_audio(source: str, explicit_format: str | None) -> tuple[bytes, str]:
    parsed = urlparse(source)
    if parsed.scheme == "":
        path = Path(source).expanduser()
    elif parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        path = Path(unquote(parsed.path)).expanduser()
    else:
        raise ValueError(
            f"tml_v0 does not fetch remote audio URLs (scheme {parsed.scheme!r}); "
            "provide encoded bytes, a local path, or a base64 data: URI"
        )

    suffix = path.suffix.lower().removeprefix(".")
    audio_format = explicit_format or {"wave": "wav", "mpeg": "mp3"}.get(suffix, suffix)
    if not audio_format:
        raise ValueError(f"cannot infer audio format from path {str(path)!r}; set AudioPart.format")
    return path.read_bytes(), audio_format


def audio_part_to_input_audio(part: AudioPart) -> dict[str, Any]:
    """Read a cookbook ``AudioPart`` as the ``input_audio`` object of OpenAI and tml-renderers.

    The object holds base64 ``data``, the ``format``, and ``num_frames`` and ``sample_rate``
    when the part gives them.
    """
    source = part["audio"]
    explicit_format = part.get("format")

    if isinstance(source, bytes):
        raw, audio_format = source, explicit_format or "wav"
    elif isinstance(source, str) and source.startswith("data:"):
        raw, audio_format = _decode_audio_data_uri(source, explicit_format)
    elif isinstance(source, str):
        raw, audio_format = _read_local_audio(source, explicit_format)
    else:
        raise TypeError(f"audio must be bytes or str; got {type(source)!r}")

    if audio_format not in _SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"unsupported audio format {audio_format!r}; expected 'wav', 'mp3', or 'flac'"
        )

    input_audio: dict[str, Any] = {
        "data": base64.b64encode(raw).decode("ascii"),
        "format": audio_format,
    }
    has_num_frames = "num_frames" in part
    has_sample_rate = "sample_rate" in part
    if has_num_frames != has_sample_rate:
        raise ValueError("AudioPart must provide num_frames and sample_rate together")
    if audio_format != "wav" and not has_num_frames:
        raise ValueError(f"{audio_format} AudioPart must provide num_frames and sample_rate")
    if has_num_frames and has_sample_rate:
        num_frames = part["num_frames"]
        sample_rate = part["sample_rate"]
        if num_frames <= 0 or sample_rate <= 0:
            raise ValueError("AudioPart num_frames and sample_rate must be positive")
        input_audio.update(num_frames=num_frames, sample_rate=sample_rate)
    return input_audio
