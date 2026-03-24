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
    total_generation_count: int
    generation_count: int
    avg_generation_ms: float
    top_event_name: str
    top_event_ms: float
    top_event_avg_ms: float
    top_event_nonzero_generation_count: int
    generations: pd.DataFrame
    trimmed_generations: pd.DataFrame
    trimmed_summary_events: dict[str, dict[str, float]]
    trimmed_summary_counters: dict[str, dict[str, float]]
    raw: dict

    @property
    def label(self) -> str:
        return f"{self.name} [{self.mode}]"


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

    # IQR trimming: discard first 25% and last 25%, keep middle 50%
    n = len(frame)
    q1_end = n // 4
    q3_end = n - n // 4
    if q3_end <= q1_end:
        # Too few generations to trim — use all
        trimmed = frame
    else:
        trimmed = frame.iloc[q1_end:q3_end].reset_index(drop=True)

    # Recompute averages from trimmed data
    gen_total_col = "event::generation_total"
    if gen_total_col in trimmed.columns and len(trimmed) > 0:
        avg_generation_ms = float(trimmed[gen_total_col].mean())
    else:
        avg_generation_ms = 0.0

    # Recompute top event from trimmed data
    trimmed_events = _compute_trimmed_event_summary(trimmed)
    trimmed_counters = _compute_trimmed_counter_summary(trimmed)

    top_event_name = "generation_total"
    top_event_ms = 0.0
    top_event_avg_ms = 0.0
    top_event_nonzero_generation_count = 0
    for name, stats in trimmed_events.items():
        if name == "generation_total":
            continue
        avg_ms = stats["avg_ms_per_nonzero_generation"]
        if avg_ms > top_event_avg_ms:
            top_event_name = name
            top_event_ms = stats["total_ms"]
            top_event_avg_ms = avg_ms
            top_event_nonzero_generation_count = int(stats["nonzero_generation_count"])

    return ProfileRun(
        path=path.parent,
        name=path.parent.name,
        experiment_name=str(run_meta.get("experiment_name", path.parent.name)),
        mode=mode,
        total_generation_count=n,
        generation_count=len(trimmed),
        avg_generation_ms=avg_generation_ms,
        top_event_name=top_event_name,
        top_event_ms=top_event_ms,
        top_event_avg_ms=top_event_avg_ms,
        top_event_nonzero_generation_count=top_event_nonzero_generation_count,
        generations=frame,
        trimmed_generations=trimmed,
        trimmed_summary_events=trimmed_events,
        trimmed_summary_counters=trimmed_counters,
        raw=payload,
    )


def _compute_trimmed_event_summary(
    trimmed: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    event_cols = [c for c in trimmed.columns if c.startswith("event::")]
    n = len(trimmed)
    for col in event_cols:
        name = col[len("event::") :]
        series = trimmed[col]
        total = float(series.sum())
        nonzero = int((series > 0).sum())
        result[name] = {
            "total_ms": total,
            "avg_ms_per_generation": total / n if n > 0 else 0.0,
            "nonzero_generation_count": float(nonzero),
            "avg_ms_per_nonzero_generation": total / nonzero if nonzero > 0 else 0.0,
        }
    return result


def _compute_trimmed_counter_summary(
    trimmed: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    counter_cols = [c for c in trimmed.columns if c.startswith("counter::")]
    n = len(trimmed)
    for col in counter_cols:
        name = col[len("counter::") :]
        series = trimmed[col]
        total = float(series.sum())
        nonzero = int((series > 0).sum())
        result[name] = {
            "total": total,
            "avg_per_generation": total / n if n > 0 else 0.0,
            "nonzero_generation_count": float(nonzero),
            "avg_per_nonzero_generation": total / nonzero if nonzero > 0 else 0.0,
        }
    return result
