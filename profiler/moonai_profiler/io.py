"""Profiler suite discovery and data loading."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScopeNode:
    """A single scope in the profiling tree."""

    name: str
    inclusive_ms: float
    exclusive_ms: float
    children: list[ScopeNode]


@dataclass(frozen=True)
class Frame:
    """A single frame with its timing tree."""

    root: ScopeNode

    @property
    def frame_total_ms(self) -> float:
        """Get total frame time from root node."""
        return self.root.inclusive_ms


@dataclass(frozen=True)
class SuiteMember:
    seed: int
    avg_frame_ms: float
    run_total_ms: float
    frame_count: int
    disposition: str
    frame_times_ms: list[float]
    frames: list[Frame]


@dataclass(frozen=True)
class ProfileSuite:
    path: Path
    name: str
    frames: int
    avg_frame_ms: float
    stddev: float
    members: list[SuiteMember]
    kept: list[SuiteMember]
    dropped: list[SuiteMember]
    # Event statistics: name -> {inclusive_ms, exclusive_ms, avg_inclusive_ms, avg_exclusive_ms}
    events: dict[str, dict[str, float]]
    # Averaged tree structure for hierarchical display
    tree: AveragedScopeNode | None
    # Metadata from the profiler run
    metadata: dict[str, Any]
    raw: dict


def _parse_scope_node(data: dict) -> ScopeNode:
    """Parse a scope node from JSON."""
    return ScopeNode(
        name=data["name"],
        inclusive_ms=data["inclusive_ms"],
        exclusive_ms=data["exclusive_ms"],
        children=[_parse_scope_node(child) for child in data.get("children", [])],
    )


def _parse_frame(data: dict) -> Frame:
    """Parse a frame from JSON."""
    return Frame(root=_parse_scope_node(data))


def _collect_scope_stats(node: ScopeNode, stats: dict[str, dict[str, float]]) -> None:
    """Recursively collect statistics from scope tree."""
    if node.name not in stats:
        stats[node.name] = {
            "total_inclusive_ms": 0.0,
            "total_exclusive_ms": 0.0,
            "count": 0,
        }

    stats[node.name]["total_inclusive_ms"] += node.inclusive_ms
    stats[node.name]["total_exclusive_ms"] += node.exclusive_ms
    stats[node.name]["count"] += 1

    for child in node.children:
        _collect_scope_stats(child, stats)


@dataclass
class AveragedScopeNode:
    """A scope node with averaged timing data."""

    name: str
    avg_inclusive_ms: float
    avg_exclusive_ms: float
    total_inclusive_ms: float
    total_exclusive_ms: float
    count: int
    children: list[AveragedScopeNode]


def _average_trees(frames: list[Frame]) -> AveragedScopeNode | None:
    """Build an averaged tree from all frames."""
    if not frames:
        return None

    def merge_nodes(nodes: list[ScopeNode]) -> AveragedScopeNode:
        """Merge multiple scope nodes with the same name/path."""
        if not nodes:
            raise ValueError("Cannot merge empty node list")

        name = nodes[0].name
        total_inclusive = sum(n.inclusive_ms for n in nodes)
        total_exclusive = sum(n.exclusive_ms for n in nodes)
        count = len(nodes)

        # Group children by name
        child_groups: dict[str, list[ScopeNode]] = {}
        for node in nodes:
            for child in node.children:
                if child.name not in child_groups:
                    child_groups[child.name] = []
                child_groups[child.name].append(child)

        # Recursively merge children
        merged_children = [merge_nodes(group) for group in child_groups.values()]

        return AveragedScopeNode(
            name=name,
            avg_inclusive_ms=total_inclusive / count,
            avg_exclusive_ms=total_exclusive / count,
            total_inclusive_ms=total_inclusive,
            total_exclusive_ms=total_exclusive,
            count=count,
            children=merged_children,
        )

    # Collect all roots (should all be frame_total)
    roots = [f.root for f in frames]
    return merge_nodes(roots)


def _analyze_run(run: dict) -> dict:
    """Compute per-run statistics from raw frame data."""
    frames_data = run.get("frames", [])
    if not frames_data:
        return {
            "frame_count": 0,
            "run_total_ms": 0.0,
            "avg_frame_ms": 0.0,
            "frame_times_ms": [],
            "frames": [],
            "scope_stats": {},
        }

    # Parse frames with tree structure
    frames = [_parse_frame(f) for f in frames_data]
    frame_times_ms = [f.frame_total_ms for f in frames]

    # Use pre-computed run_total_ms from JSON, fallback to sum if not present
    run_total_ms = run.get("run_total_ms", sum(frame_times_ms))
    avg_frame_ms = run_total_ms / len(frames) if frames else 0.0

    # Collect scope statistics from all frames
    scope_stats: dict[str, dict[str, float]] = {}
    for frame in frames:
        _collect_scope_stats(frame.root, scope_stats)

    return {
        "frame_count": len(frames),
        "run_total_ms": run_total_ms,
        "avg_frame_ms": avg_frame_ms,
        "frame_times_ms": frame_times_ms,
        "frames": frames,
        "scope_stats": scope_stats,
    }


def _analyze_suite(runs: list[dict]) -> dict:
    """Compute suite-level statistics with outlier removal."""
    # Analyze each run from raw data
    analyzed = []
    for run in runs:
        stats = _analyze_run(run)
        stats["seed"] = run.get("seed", 0)
        analyzed.append(stats)

    if not analyzed:
        return {
            "avg_frame_ms": 0.0,
            "avg_frame_ms_stddev": 0.0,
            "avg_run_total_ms": 0.0,
            "events": {},
            "kept": [],
            "dropped": [],
        }

    # Sort by avg_frame_ms for outlier removal
    analyzed.sort(key=lambda x: x["avg_frame_ms"])

    # Drop fastest and slowest (outlier removal)
    if len(analyzed) > 2:
        kept = analyzed[1:-1]
        dropped = [analyzed[0], analyzed[-1]]
    else:
        kept = analyzed
        dropped = []

    # Compute suite aggregates from kept runs
    avg_frame_ms = sum(r["avg_frame_ms"] for r in kept) / len(kept) if kept else 0.0
    avg_run_total_ms = sum(r["run_total_ms"] for r in kept) / len(kept) if kept else 0.0

    # Standard deviation
    if kept:
        variance = sum((r["avg_frame_ms"] - avg_frame_ms) ** 2 for r in kept) / len(
            kept
        )
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    # Aggregate scope statistics across all kept runs
    total_frames = sum(r["frame_count"] for r in kept)
    event_totals: dict[str, dict[str, float]] = {}

    for run in kept:
        for scope_name, stats in run["scope_stats"].items():
            if scope_name not in event_totals:
                event_totals[scope_name] = {
                    "total_inclusive_ms": 0.0,
                    "total_exclusive_ms": 0.0,
                    "count": 0,
                }
            event_totals[scope_name]["total_inclusive_ms"] += stats[
                "total_inclusive_ms"
            ]
            event_totals[scope_name]["total_exclusive_ms"] += stats[
                "total_exclusive_ms"
            ]
            event_totals[scope_name]["count"] += stats["count"]

    # Compute averages per frame
    for stats in event_totals.values():
        count = stats["count"]
        stats["avg_inclusive_ms"] = (
            stats["total_inclusive_ms"] / count if count > 0 else 0.0
        )
        stats["avg_exclusive_ms"] = (
            stats["total_exclusive_ms"] / count if count > 0 else 0.0
        )

    # Build averaged tree from all frames in kept runs
    all_frames: list[Frame] = []
    for run in kept:
        all_frames.extend(run["frames"])
    averaged_tree = _average_trees(all_frames)

    return {
        "avg_frame_ms": avg_frame_ms,
        "avg_frame_ms_stddev": stddev,
        "avg_run_total_ms": avg_run_total_ms,
        "events": event_totals,
        "tree": averaged_tree,
        "kept": kept,
        "dropped": dropped,
    }


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
    metadata = data.get("metadata", {})
    runs = data.get("runs", [])

    # Perform analysis on raw data
    analysis = _analyze_suite(runs)

    # Build SuiteMember objects
    members = []
    for run in runs:
        run_stats = _analyze_run(run)

        # Determine disposition
        seed = run.get("seed", 0)
        is_dropped = any(d.get("seed") == seed for d in analysis.get("dropped", []))
        disposition = "dropped" if is_dropped else "kept"

        members.append(
            SuiteMember(
                seed=seed,
                avg_frame_ms=run_stats["avg_frame_ms"],
                run_total_ms=run_stats["run_total_ms"],
                frame_count=run_stats["frame_count"],
                disposition=disposition,
                frame_times_ms=run_stats["frame_times_ms"],
                frames=run_stats["frames"],
            )
        )

    kept = [m for m in members if m.disposition == "kept"]
    dropped = [m for m in members if m.disposition != "kept"]

    # Get total frame count from metadata
    total_frames = int(metadata.get("frame_count", 0))

    return ProfileSuite(
        path=path,
        name=path.stem,
        frames=total_frames,
        avg_frame_ms=analysis["avg_frame_ms"],
        stddev=analysis["avg_frame_ms_stddev"],
        members=members,
        kept=kept,
        dropped=dropped,
        events=_flatten_stats(analysis.get("events", {})),
        tree=analysis.get("tree"),
        metadata=metadata,
        raw=data,
    )


def _flatten_stats(stats: dict) -> dict[str, dict[str, float]]:
    """Convert nested JSON stats to flat float dict."""
    return {
        name: {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
        for name, values in stats.items()
        if isinstance(values, dict)
    }
