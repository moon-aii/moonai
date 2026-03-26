"""Profiler suite discovery and data loading."""

from __future__ import annotations

import json
import math
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


def _analyze_run(profile_data: dict) -> dict:
    """Compute per-run statistics from raw window data."""
    windows = profile_data.get("windows", [])
    if not windows:
        return {
            "window_count": 0,
            "run_total_ms": 0.0,
            "avg_window_ms": 0.0,
            "events": {},
            "window_times_ms": [],
        }

    # Extract window times for the main timing metric
    window_times_ms = [w.get("events_ms", {}).get("window_total", 0.0) for w in windows]
    run_total_ms = sum(window_times_ms)
    avg_window_ms = run_total_ms / len(windows) if windows else 0.0

    # Aggregate all events across windows
    events = {}
    for window in windows:
        for event_name, ms in window.get("events_ms", {}).items():
            if event_name not in events:
                events[event_name] = {"total_ms": 0.0, "count": 0}
            events[event_name]["total_ms"] += ms
            events[event_name]["count"] += 1

    # Calculate averages per event
    for stats in events.values():
        stats["avg_ms_per_window"] = (
            stats["total_ms"] / len(windows) if windows else 0.0
        )

    return {
        "window_count": len(windows),
        "run_total_ms": run_total_ms,
        "avg_window_ms": avg_window_ms,
        "events": events,
        "window_times_ms": window_times_ms,
    }


def _analyze_suite(runs: list[dict]) -> dict:
    """Compute suite-level statistics with outlier removal."""
    # Analyze each run from raw data
    analyzed = []
    for run in runs:
        profile = run.get("profile_data", {})
        stats = _analyze_run(profile)
        stats["seed"] = run.get("seed", 0)
        analyzed.append(stats)

    if not analyzed:
        return {
            "avg_window_ms": 0.0,
            "avg_window_ms_stddev": 0.0,
            "avg_run_total_ms": 0.0,
            "events": {},
            "kept": [],
            "dropped": [],
        }

    # Sort by avg_window_ms for outlier removal
    analyzed.sort(key=lambda x: x["avg_window_ms"])

    # Drop fastest and slowest (outlier removal)
    if len(analyzed) > 2:
        kept = analyzed[1:-1]
        dropped = [analyzed[0], analyzed[-1]]
    else:
        kept = analyzed
        dropped = []

    # Compute suite aggregates from kept runs
    avg_window_ms = sum(r["avg_window_ms"] for r in kept) / len(kept) if kept else 0.0
    avg_run_total_ms = sum(r["run_total_ms"] for r in kept) / len(kept) if kept else 0.0

    # Standard deviation
    if kept:
        variance = sum((r["avg_window_ms"] - avg_window_ms) ** 2 for r in kept) / len(
            kept
        )
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    # Aggregate events across kept runs
    event_totals = {}
    for run in kept:
        for event_name, stats in run["events"].items():
            if event_name not in event_totals:
                event_totals[event_name] = {"total_ms": 0.0, "count": 0}
            event_totals[event_name]["total_ms"] += stats["total_ms"]
            event_totals[event_name]["count"] += 1

    # Average across runs
    for stats in event_totals.values():
        stats["avg_ms_per_run"] = stats["total_ms"] / len(kept) if kept else 0.0

    return {
        "avg_window_ms": avg_window_ms,
        "avg_window_ms_stddev": stddev,
        "avg_run_total_ms": avg_run_total_ms,
        "events": event_totals,
        "kept": kept,
        "dropped": dropped,
    }


def _parse_suite(path: Path, data: dict) -> ProfileSuite:
    """Parse a single profile suite from JSON."""
    suite = data.get("suite", {})
    runs = data.get("runs", [])

    # Perform analysis on raw data
    analysis = _analyze_suite(runs)

    # Build SuiteMember objects
    members = []
    for run in runs:
        profile = run.get("profile_data", {})
        run_stats = _analyze_run(profile)

        # Determine disposition
        seed = run.get("seed", 0)
        is_dropped = any(d.get("seed") == seed for d in analysis.get("dropped", []))
        disposition = "dropped" if is_dropped else "kept"

        members.append(
            SuiteMember(
                seed=seed,
                avg_window_ms=run_stats["avg_window_ms"],
                run_total_ms=run_stats["run_total_ms"],
                window_count=run_stats["window_count"],
                disposition=disposition,
                window_times_ms=run_stats["window_times_ms"],
            )
        )

    kept = [m for m in members if m.disposition == "kept"]
    dropped = [m for m in members if m.disposition == "kept"]

    return ProfileSuite(
        path=path,
        name=path.stem,
        windows=int(suite.get("windows", 0)),
        avg_window_ms=analysis["avg_window_ms"],
        stddev=analysis["avg_window_ms_stddev"],
        members=members,
        kept=kept,
        dropped=dropped,
        events=_flatten_stats(analysis.get("events", {})),
        raw=data,
    )


def _flatten_stats(stats: dict) -> dict[str, dict[str, float]]:
    """Convert nested JSON stats to flat float dict."""
    return {
        name: {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
        for name, values in stats.items()
        if isinstance(values, dict)
    }
