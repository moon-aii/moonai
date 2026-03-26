"""Profiler suite discovery and data loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteMember:
    seed: int
    avg_window_ms: float
    run_total_ms: float
    window_count: int
    disposition: str
    window_times_ms: list[float]


@dataclass(frozen=True)
class ProfileSuite:
    path: Path
    name: str
    windows: int
    avg_window_ms: float
    stddev: float
    members: list[SuiteMember]
    kept: list[SuiteMember]
    dropped: list[SuiteMember]
    events: dict[str, dict[str, float]]
    raw: dict


def load_suites(input_dir: Path) -> list[ProfileSuite]:
    """Load all profile suite files from directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    suites = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text())
            suites.append(_parse_suite(path, data))
        except Exception as exc:
            print(f"Warning: Skipping {path.name}: {exc}")
    return suites


def _parse_suite(path: Path, data: dict) -> ProfileSuite:
    """Parse a single profile suite from JSON."""
    suite = data.get("suite", {})
    aggregate = data.get("aggregate", {})
    runs = data.get("runs", [])

    members = []
    for run in runs:
        profile = run.get("profile_data", {})
        windows = profile.get("windows", [])
        times = [w.get("events_ms", {}).get("window_total", 0.0) for w in windows]

        members.append(
            SuiteMember(
                seed=int(run.get("seed", 0)),
                avg_window_ms=float(run.get("avg_window_ms", 0.0)),
                run_total_ms=float(run.get("run_total_ms", 0.0)),
                window_count=int(run.get("window_count", 0)),
                disposition=str(run.get("disposition", "kept")),
                window_times_ms=times,
            )
        )

    kept = [m for m in members if m.disposition == "kept"]
    dropped = [m for m in members if m.disposition != "kept"]

    return ProfileSuite(
        path=path,
        name=path.stem,
        windows=int(suite.get("windows", 0)),
        avg_window_ms=float(aggregate.get("avg_window_ms", 0.0)),
        stddev=float(aggregate.get("avg_window_ms_stddev", 0.0)),
        members=members,
        kept=kept,
        dropped=dropped,
        events=_flatten_stats(aggregate.get("events", {})),
        raw=data,
    )


def _flatten_stats(stats: dict) -> dict[str, dict[str, float]]:
    """Convert nested JSON stats to flat float dict."""
    return {
        name: {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
        for name, values in stats.items()
        if isinstance(values, dict)
    }
