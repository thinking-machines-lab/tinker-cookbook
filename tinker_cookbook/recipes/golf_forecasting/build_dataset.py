from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import aiohttp
import chz

from tinker_cookbook.recipes.golf_forecasting.data import (
    DatasetManifest,
    GolfForecastExample,
    PlayerSnapshot,
    SourceArtifact,
    normalize_player_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    format: Literal["json", "jsonl", "csv"]
    records_path: str | None = None
    enabled: bool = True
    request_headers: dict[str, str] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SourceSpec":
        return SourceSpec(
            name=str(data["name"]),
            url=str(data["url"]),
            format=cast(Literal["json", "jsonl", "csv"], data["format"]),
            records_path=data.get("records_path"),
            enabled=bool(data.get("enabled", True)),
            request_headers=cast(dict[str, str] | None, data.get("request_headers")),
        )


@dataclass(frozen=True)
class PriorSpec:
    name: str
    path: str
    key_field: str = "player_name"
    value_field: str = "prior_win_prob"

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PriorSpec":
        return PriorSpec(
            name=str(data["name"]),
            path=str(data["path"]),
            key_field=str(data.get("key_field", "player_name")),
            value_field=str(data.get("value_field", "prior_win_prob")),
        )


@dataclass(frozen=True)
class BuildDatasetManifest:
    sources: tuple[SourceSpec, ...]
    priors: tuple[PriorSpec, ...] = ()

    @staticmethod
    def from_path(path: str) -> "BuildDatasetManifest":
        payload = json.loads(Path(path).read_text())
        return BuildDatasetManifest(
            sources=tuple(SourceSpec.from_dict(item) for item in payload.get("sources", [])),
            priors=tuple(PriorSpec.from_dict(item) for item in payload.get("priors", [])),
        )


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_records_from_raw(path: Path, data_format: Literal["json", "jsonl", "csv"]) -> list[dict[str, Any]]:
    if data_format == "json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return cast(list[dict[str, Any]], payload)
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return cast(list[dict[str, Any]], records)
        raise ValueError(f"JSON source at {path} must be a list or contain a 'records' field.")
    if data_format == "jsonl":
        records: list[dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(cast(dict[str, Any], json.loads(stripped)))
        return records
    if data_format == "csv":
        with path.open(newline="") as handle:
            return [cast(dict[str, Any], row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported source format: {data_format}")


def _extract_records_path(payload: Any, dotted_path: str | None) -> list[dict[str, Any]]:
    if dotted_path is None:
        if isinstance(payload, list):
            return cast(list[dict[str, Any]], payload)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return cast(list[dict[str, Any]], payload["records"])
        raise ValueError("records_path is required when JSON payload is not a list or `records` object.")

    current = payload
    for component in dotted_path.split("."):
        if not isinstance(current, dict):
            raise ValueError(f"Could not traverse records_path={dotted_path}")
        current = current[component]
    if not isinstance(current, list):
        raise ValueError(f"records_path={dotted_path} did not resolve to a list")
    return cast(list[dict[str, Any]], current)


async def fetch_sources(
    *,
    manifest: BuildDatasetManifest,
    raw_dir: Path,
) -> list[SourceArtifact]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[SourceArtifact] = []
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for source in manifest.sources:
            if not source.enabled:
                continue
            logger.info("Fetching %s from %s", source.name, source.url)
            async with session.get(source.url, headers=source.request_headers) as response:
                response.raise_for_status()
                body = await response.read()
                suffix = {"json": ".json", "jsonl": ".jsonl", "csv": ".csv"}[source.format]
                output_path = raw_dir / f"{source.name}{suffix}"
                output_path.write_bytes(body)
                artifacts.append(
                    SourceArtifact(
                        name=source.name,
                        url=source.url,
                        local_path=str(output_path),
                        content_type=response.headers.get("content-type", "application/octet-stream"),
                        fetched_at=_timestamp(),
                    )
                )
    return artifacts


def _parse_players(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return cast(list[dict[str, Any]], value)
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return cast(list[dict[str, Any]], parsed)
    raise ValueError("players field must be a list or JSON-encoded list")


def _lookup(data: dict[str, Any], *aliases: str, default: Any = None) -> Any:
    for alias in aliases:
        if alias in data and data[alias] not in (None, ""):
            return data[alias]
    return default


def _normalize_player(player: dict[str, Any], priors: dict[str, float]) -> PlayerSnapshot:
    name = str(_lookup(player, "name", "player_name"))
    normalized_name = normalize_player_name(name)
    prior = _lookup(player, "prior_win_prob", "win_probability", default=priors.get(normalized_name))
    return PlayerSnapshot.from_dict(
        {
            "name": name,
            "position": str(_lookup(player, "position", "pos", default="")),
            "score_to_par": _lookup(player, "score_to_par", "score", default=0.0),
            "strokes_behind": _lookup(player, "strokes_behind", "behind", default=0.0),
            "holes_completed": _lookup(player, "holes_completed"),
            "current_hole": _lookup(player, "current_hole", "hole"),
            "holes_remaining": _lookup(player, "holes_remaining"),
            "round_score": _lookup(player, "round_score"),
            "tee_time": _lookup(player, "tee_time"),
            "prior_win_prob": prior,
            "recent_form_score": _lookup(player, "recent_form_score", "recent_form"),
            "metadata": cast(dict[str, Any], _lookup(player, "metadata", default={})),
        }
    )


def _load_priors(prior_specs: tuple[PriorSpec, ...]) -> dict[str, float]:
    priors: dict[str, float] = {}
    for spec in prior_specs:
        path = Path(spec.path)
        if not path.exists():
            logger.warning("Skipping prior file %s because it does not exist", path)
            continue
        records = _read_records_from_raw(path, "jsonl" if path.suffix == ".jsonl" else "json")
        for record in records:
            key = normalize_player_name(str(record[spec.key_field]))
            priors[key] = float(record[spec.value_field])
    return priors


def normalize_records(
    *,
    source_artifacts: list[SourceArtifact],
    build_manifest: BuildDatasetManifest,
) -> list[GolfForecastExample]:
    priors = _load_priors(build_manifest.priors)
    examples: list[GolfForecastExample] = []
    for artifact, source in zip(source_artifacts, [s for s in build_manifest.sources if s.enabled], strict=True):
        raw_path = Path(artifact.local_path)
        if source.format == "json":
            payload = json.loads(raw_path.read_text())
            records = _extract_records_path(payload, source.records_path)
        else:
            records = _read_records_from_raw(raw_path, source.format)

        for index, record in enumerate(records):
            tournament_name = str(_lookup(record, "tournament_name", "event_name"))
            tournament_id = str(
                _lookup(record, "tournament_id", default=f"{tournament_name.lower().replace(' ', '-')}")
            )
            snapshot_timestamp = str(_lookup(record, "snapshot_timestamp", "timestamp"))
            players = tuple(_normalize_player(player, priors) for player in _parse_players(record["players"]))
            example_id = str(
                _lookup(record, "example_id", default=f"{tournament_id}-{snapshot_timestamp}-{index}")
            )
            examples.append(
                GolfForecastExample(
                    example_id=example_id,
                    tournament_id=tournament_id,
                    tournament_name=tournament_name,
                    course_name=_lookup(record, "course_name"),
                    round_number=int(_lookup(record, "round_number", "round", default=1)),
                    event_day=_lookup(record, "event_day", "day"),
                    snapshot_timestamp=snapshot_timestamp,
                    players=players,
                    target_winner=str(_lookup(record, "target_winner", "winner")),
                    system_context=cast(dict[str, Any], _lookup(record, "system_context", default={})),
                    source_urls=tuple(
                        [
                            *cast(list[str], _lookup(record, "source_urls", default=[])),
                            artifact.url,
                        ]
                    ),
                    other_field_prior=(
                        float(_lookup(record, "other_field_prior"))
                        if _lookup(record, "other_field_prior") is not None
                        else None
                    ),
                )
            )
    return examples


def _split_tournaments(
    examples: list[GolfForecastExample],
    *,
    val_fraction: float,
    heldout_fraction: float,
    seed: int,
    existing_manifest: DatasetManifest | None = None,
) -> dict[str, list[GolfForecastExample]]:
    by_tournament: dict[str, list[GolfForecastExample]] = {}
    for example in examples:
        by_tournament.setdefault(example.tournament_id, []).append(example)

    tournament_ids = sorted(by_tournament)
    if existing_manifest is not None and existing_manifest.heldout_locked:
        heldout_ids = set(existing_manifest.split_tournament_ids.get("heldout", []))
        val_ids = set(existing_manifest.split_tournament_ids.get("val", []))
        train_ids = set(existing_manifest.split_tournament_ids.get("train", []))
        unseen_ids = [tid for tid in tournament_ids if tid not in heldout_ids | val_ids | train_ids]
        unseen_ids.sort()
        train_ids.update(unseen_ids)
    else:
        rng = random.Random(seed)
        shuffled_ids = tournament_ids[:]
        rng.shuffle(shuffled_ids)
        heldout_count = max(1, round(len(shuffled_ids) * heldout_fraction)) if shuffled_ids else 0
        val_count = max(1, round(len(shuffled_ids) * val_fraction)) if len(shuffled_ids) > 2 else 0
        heldout_ids = set(shuffled_ids[-heldout_count:]) if heldout_count else set()
        remaining_ids = [tid for tid in shuffled_ids if tid not in heldout_ids]
        val_ids = set(remaining_ids[-val_count:]) if val_count else set()
        train_ids = set(remaining_ids) - val_ids

    splits = {"train": [], "val": [], "heldout": []}
    for tournament_id, items in by_tournament.items():
        if tournament_id in heldout_ids:
            splits["heldout"].extend(items)
        elif tournament_id in val_ids:
            splits["val"].extend(items)
        else:
            splits["train"].extend(items)
    return splits


def _write_jsonl(path: Path, examples: list[GolfForecastExample]) -> None:
    with path.open("w") as handle:
        for example in examples:
            handle.write(example.to_json())
            handle.write("\n")


@chz.chz
class BuildDatasetConfig:
    source_manifest_path: str
    output_dir: str = "tinker_cookbook/example_data/golf_forecasting"
    fetch_online: bool = True
    val_fraction: float = 0.2
    heldout_fraction: float = 0.2
    seed: int = 0
    freeze_existing_heldout: bool = True


async def build_dataset(config: BuildDatasetConfig) -> DatasetManifest:
    build_manifest = BuildDatasetManifest.from_path(config.source_manifest_path)
    output_dir = Path(config.output_dir)
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_dir / "dataset_manifest.json"
    existing_manifest = (
        DatasetManifest.from_dict(json.loads(existing_manifest_path.read_text()))
        if existing_manifest_path.exists() and config.freeze_existing_heldout
        else None
    )

    if config.fetch_online:
        source_artifacts = await fetch_sources(manifest=build_manifest, raw_dir=raw_dir)
    else:
        source_artifacts = tuple(
            SourceArtifact(
                name=source.name,
                url=source.url,
                local_path=str(raw_dir / f"{source.name}.{source.format}"),
                content_type="application/octet-stream",
                fetched_at=_timestamp(),
            )
            for source in build_manifest.sources
            if source.enabled
        )
        missing = [artifact.local_path for artifact in source_artifacts if not Path(artifact.local_path).exists()]
        if missing:
            raise FileNotFoundError(f"Raw source files are missing; run with fetch_online=true first: {missing}")
        source_artifacts = list(source_artifacts)

    examples = normalize_records(source_artifacts=source_artifacts, build_manifest=build_manifest)
    splits = _split_tournaments(
        examples,
        val_fraction=config.val_fraction,
        heldout_fraction=config.heldout_fraction,
        seed=config.seed,
        existing_manifest=existing_manifest,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    heldout_path = output_dir / "heldout.jsonl"
    _write_jsonl(train_path, splits["train"])
    _write_jsonl(val_path, splits["val"])
    _write_jsonl(heldout_path, splits["heldout"])

    manifest = DatasetManifest(
        dataset_version=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        created_at=_timestamp(),
        output_dir=str(output_dir),
        train_path=str(train_path),
        val_path=str(val_path),
        heldout_path=str(heldout_path),
        heldout_locked=True,
        split_tournament_ids={
            split: sorted({example.tournament_id for example in split_examples})
            for split, split_examples in splits.items()
        },
        source_artifacts=tuple(source_artifacts),
    )
    existing_manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    return manifest


async def cli_main(config: BuildDatasetConfig) -> None:
    manifest = await build_dataset(config)
    logger.info("Wrote dataset manifest to %s", Path(manifest.output_dir) / "dataset_manifest.json")
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    cli_config = chz.entrypoint(BuildDatasetConfig)
    asyncio.run(cli_main(cli_config))
