"""Profiler suite discovery and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkippedProfileSuite:
    path: Path
    reason: str


@dataclass(frozen=True)
class SuiteMemberRun:
    seed: int
    run_dir: str
    profile_path: str
    avg_generation_ms: float
    run_total_ms: float
    generation_count: int
    disposition: str
    generation_times_ms: list[float]


@dataclass(frozen=True)
class ProfileSuite:
    path: Path
    name: str
    label: str
    suite_name: str
    config_path: str
    experiment_name: str
    generations: int
    config_fingerprint: str
    avg_generation_ms: float
    avg_generation_ms_stddev: float
    avg_run_total_ms: float
    aggregate_events: dict[str, dict[str, float]]
    aggregate_counters: dict[str, dict[str, float]]
    members: list[SuiteMemberRun]
    kept_members: list[SuiteMemberRun]
    dropped_members: list[SuiteMemberRun]
    raw: dict


def discover_profile_suites(
    input_dir: Path,
) -> tuple[list[ProfileSuite], list[SkippedProfileSuite]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"profiler output directory not found: {input_dir}")

    suites: list[ProfileSuite] = []
    skipped: list[SkippedProfileSuite] = []
    for path in sorted(input_dir.rglob("profile_suite.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            suites.append(_build_profile_suite(path, payload))
        except Exception as exc:
            skipped.append(SkippedProfileSuite(path, str(exc)))
    return suites, skipped


def _build_profile_suite(path: Path, payload: dict) -> ProfileSuite:
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported suite schema_version in {path}")

    suite_meta = payload.get("suite")
    runs = payload.get("runs")
    aggregate = payload.get("aggregate")
    if (
        not isinstance(suite_meta, dict)
        or not isinstance(runs, list)
        or not isinstance(aggregate, dict)
    ):
        raise ValueError(f"invalid suite structure in {path}")

    members = [
        SuiteMemberRun(
            seed=int(run.get("seed", 0)),
            run_dir=str(run.get("run_dir", "")),
            profile_path=str(run.get("profile_path", "")),
            avg_generation_ms=float(run.get("avg_generation_ms", 0.0)),
            run_total_ms=float(run.get("run_total_ms", 0.0)),
            generation_count=int(run.get("generation_count", 0)),
            disposition=str(run.get("disposition", "kept")),
            generation_times_ms=_load_generation_times(
                _resolve_profile_path(path.parent, str(run.get("profile_path", "")))
            ),
        )
        for run in runs
        if isinstance(run, dict)
    ]
    kept_members = [member for member in members if member.disposition == "kept"]
    dropped_members = [member for member in members if member.disposition != "kept"]

    return ProfileSuite(
        path=path.parent,
        name=path.parent.name,
        label=path.parent.name,
        suite_name=str(suite_meta.get("name", path.parent.name)),
        config_path=str(suite_meta.get("config_path", "")),
        experiment_name=str(suite_meta.get("experiment_name", "")),
        generations=int(suite_meta.get("generations", 0)),
        config_fingerprint=str(suite_meta.get("config_fingerprint", "")),
        avg_generation_ms=float(aggregate.get("avg_generation_ms", 0.0)),
        avg_generation_ms_stddev=float(aggregate.get("avg_generation_ms_stddev", 0.0)),
        avg_run_total_ms=float(aggregate.get("avg_run_total_ms", 0.0)),
        aggregate_events={
            name: {key: float(value) for key, value in values.items()}
            for name, values in aggregate.get("events", {}).items()
            if isinstance(values, dict)
        },
        aggregate_counters={
            name: {key: float(value) for key, value in values.items()}
            for name, values in aggregate.get("counters", {}).items()
            if isinstance(values, dict)
        },
        members=members,
        kept_members=kept_members,
        dropped_members=dropped_members,
        raw=payload,
    )


def _load_generation_times(profile_path: Path) -> list[float]:
    try:
        with profile_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    generations = payload.get("generations")
    if not isinstance(generations, list):
        return []

    values: list[float] = []
    for generation in generations:
        if not isinstance(generation, dict):
            continue
        events = generation.get("events_ms")
        if not isinstance(events, dict):
            continue
        values.append(float(events.get("generation_total", 0.0)))
    return values


def _resolve_profile_path(suite_dir: Path, profile_path: str) -> Path:
    candidate = Path(profile_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    return suite_dir / candidate
