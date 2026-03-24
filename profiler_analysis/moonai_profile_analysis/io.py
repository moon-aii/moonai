"""Profile run discovery and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SkippedProfile:
    path: Path
    reason: str


@dataclass(frozen=True)
class ProfileRun:
    path: Path
    name: str
    experiment_name: str
    mode: str
    generation_count: int
    avg_generation_ms: float
    top_event_name: str
    top_event_ms: float
    top_event_avg_ms: float
    top_event_nonzero_generation_count: int
    generations: pd.DataFrame
    raw: dict

    @property
    def label(self) -> str:
        return f"{self.experiment_name} [{self.mode}]"


def discover_profiles(input_dir: Path) -> tuple[list[ProfileRun], list[SkippedProfile]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"profiler output directory not found: {input_dir}")

    runs: list[ProfileRun] = []
    skipped: list[SkippedProfile] = []

    for path in sorted(input_dir.rglob("profile.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            runs.append(_build_profile_run(path, payload))
        except Exception as exc:
            skipped.append(SkippedProfile(path, str(exc)))

    return runs, skipped


def _build_profile_run(path: Path, payload: dict) -> ProfileRun:
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")

    run_meta = payload.get("run")
    generations = payload.get("generations")
    summary = payload.get("summary")
    if (
        not isinstance(run_meta, dict)
        or not isinstance(generations, list)
        or not isinstance(summary, dict)
    ):
        raise ValueError(f"invalid profile structure in {path}")

    frame_rows: list[dict] = []
    for generation in generations:
        if not isinstance(generation, dict):
            raise ValueError(f"invalid generation row in {path}")
        row = {
            "generation": generation.get("generation", 0),
            "cpu_used": bool(generation.get("cpu_used", False)),
            "gpu_used": bool(generation.get("gpu_used", False)),
            "predator_count": generation.get("predator_count", 0),
            "prey_count": generation.get("prey_count", 0),
            "species_count": generation.get("species_count", 0),
            "best_fitness": generation.get("best_fitness", 0.0),
            "avg_fitness": generation.get("avg_fitness", 0.0),
            "avg_complexity": generation.get("avg_complexity", 0.0),
        }
        events_ms = generation.get("events_ms", {})
        counters = generation.get("counters", {})
        if not isinstance(events_ms, dict) or not isinstance(counters, dict):
            raise ValueError(f"invalid generation event/counter payload in {path}")
        for key, value in events_ms.items():
            row[f"event::{key}"] = float(value)
        for key, value in counters.items():
            row[f"counter::{key}"] = float(value)
        frame_rows.append(row)

    if not frame_rows:
        raise ValueError(f"profile contains no generations: {path}")

    frame = pd.DataFrame(frame_rows).sort_values("generation").reset_index(drop=True)
    summary_events = summary.get("events", {})
    if not isinstance(summary_events, dict) or "generation_total" not in summary_events:
        raise ValueError(f"profile summary is missing generation_total event: {path}")

    cpu_count = int(summary.get("cpu_generation_count", 0) or 0)
    gpu_count = int(summary.get("gpu_generation_count", 0) or 0)
    if cpu_count > 0 and gpu_count > 0:
        mode = "mixed"
    elif gpu_count > 0:
        mode = "gpu"
    else:
        mode = "cpu"

    top_event_name = "generation_total"
    top_event_ms = 0.0
    top_event_avg_ms = 0.0
    top_event_nonzero_generation_count = 0
    for name, event_summary in summary_events.items():
        if name == "generation_total":
            continue
        avg_ms = float(event_summary.get("avg_ms_per_nonzero_generation", 0.0) or 0.0)
        total_ms = float(event_summary.get("total_ms", 0.0) or 0.0)
        nonzero_generation_count = int(
            event_summary.get("nonzero_generation_count", 0) or 0
        )
        if avg_ms > top_event_avg_ms:
            top_event_name = name
            top_event_ms = total_ms
            top_event_avg_ms = avg_ms
            top_event_nonzero_generation_count = nonzero_generation_count

    return ProfileRun(
        path=path.parent,
        name=path.parent.name,
        experiment_name=str(run_meta.get("experiment_name", path.parent.name)),
        mode=mode,
        generation_count=int(
            summary.get("generation_count", len(frame_rows)) or len(frame_rows)
        ),
        avg_generation_ms=float(
            summary_events["generation_total"].get("avg_ms_per_generation", 0.0) or 0.0
        ),
        top_event_name=top_event_name,
        top_event_ms=top_event_ms,
        top_event_avg_ms=top_event_avg_ms,
        top_event_nonzero_generation_count=top_event_nonzero_generation_count,
        generations=frame,
        raw=payload,
    )
