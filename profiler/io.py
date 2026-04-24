"""Profiler suite discovery and data loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AveragedNode:
    """A scope node with averaged timing data."""

    name: str
    avg_ms: float
    total_ms: float
    count: int
    pct_of_parent: float
    children: list[AveragedNode]


@dataclass(frozen=True)
class SuiteMember:
    """A single run in the suite."""

    seed: int
    avg_frame_ms: float
    disposition: str  # kept, dropped_fastest, dropped_slowest


@dataclass(frozen=True)
class ProfileSuite:
    """Complete profiler suite with aggregated data."""

    path: Path
    name: str
    frames: int
    avg_frame_ms: float
    stddev: float
    members: list[SuiteMember]
    tree: AveragedNode | None
    frame_timeline_ms: list[float]
    metadata: dict[str, Any]


def _parse_averaged_node(data: dict) -> AveragedNode:
    """Parse an averaged node from JSON."""
    return AveragedNode(
        name=data["name"],
        avg_ms=data["avg_ms"],
        total_ms=data["total_ms"],
        count=data["count"],
        pct_of_parent=data["pct_of_parent"],
        children=[_parse_averaged_node(child) for child in data.get("children", [])],
    )


def _parse_suite(path: Path, data: dict) -> ProfileSuite:
    """Parse a profile suite from JSON."""
    metadata = data.get("metadata", {})
    summary = data.get("summary", {})
    runs = data.get("runs", [])
    tree_data = data.get("tree")
    timeline = data.get("frame_timeline_ms", [])

    # Parse members
    members = [
        SuiteMember(
            seed=run["seed"],
            avg_frame_ms=run["avg_frame_ms"],
            disposition=run["disposition"],
        )
        for run in runs
    ]

    # Parse tree
    tree = _parse_averaged_node(tree_data) if tree_data else None

    return ProfileSuite(
        path=path,
        name=path.stem,
        frames=metadata.get("frame_count", 0),
        avg_frame_ms=summary.get("avg_frame_ms", 0.0),
        stddev=summary.get("stddev_ms", 0.0),
        members=members,
        tree=tree,
        frame_timeline_ms=timeline,
        metadata=metadata,
    )


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
