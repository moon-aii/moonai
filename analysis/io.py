"""Run discovery and data loading for MoonAI analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_RUN_FILES = ("config.json", "stats.csv")
CONFIG_GROUP_IGNORE_KEYS = {"output_dir", "seed"}


@dataclass(frozen=True)
class SkippedRun:
    path: Path
    reason: str


@dataclass(frozen=True)
class RunData:
    path: Path
    name: str
    config: dict
    stats: pd.DataFrame
    final_step: int
    expected_steps: int | None
    seed: int | None
    config_signature: str


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path, *, required_columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#")
    if required_columns:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"missing columns {missing} in {path.name}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _normalize_value(value):
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def config_signature(config: dict) -> str:
    normalized = {
        key: _normalize_value(value)
        for key, value in config.items()
        if key not in CONFIG_GROUP_IGNORE_KEYS
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def expected_steps(config: dict) -> int | None:
    value = int(config.get("max_steps", 0) or 0)
    return value if value > 0 else None


def discover_runs(output_dir: Path) -> tuple[list[RunData], list[SkippedRun]]:
    runs: list[RunData] = []
    skipped: list[SkippedRun] = []

    if not output_dir.is_dir():
        raise FileNotFoundError(f"simulation output directory not found: {output_dir}")

    for path in sorted(output_dir.iterdir()):
        if not path.is_dir():
            continue

        missing_files = [
            name for name in REQUIRED_RUN_FILES if not (path / name).is_file()
        ]
        if missing_files:
            skipped.append(
                SkippedRun(path, f"missing required files: {', '.join(missing_files)}")
            )
            continue

        try:
            config = load_json(path / "config.json")
            stats = load_csv(
                path / "stats.csv",
                required_columns=[
                    "step",
                    "predator_species",
                    "prey_species",
                    "avg_complexity",
                    "predator_count",
                    "prey_count",
                ],
            )
        except Exception as exc:
            skipped.append(SkippedRun(path, str(exc)))
            continue

        final_step = int(stats["step"].iloc[-1])
        expected = expected_steps(config)
        if expected is not None and final_step < expected:
            skipped.append(
                SkippedRun(
                    path,
                    f"incomplete run: expected {expected} steps, found {final_step}",
                )
            )
            continue

        seed = config.get("seed")
        runs.append(
            RunData(
                path=path,
                name=path.name,
                config=config,
                stats=stats,
                final_step=final_step,
                expected_steps=expected,
                seed=int(seed) if isinstance(seed, int | float) else None,
                config_signature=config_signature(config),
            )
        )

    return runs, skipped


def load_optional_csv(
    path: Path, *, required_columns: list[str] | None = None
) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return load_csv(path, required_columns=required_columns)
    except Exception:
        return None
