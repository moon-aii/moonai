"""Structured summary data for MoonAI analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .io import SkippedRun
from .plots import COMPARISON_METRICS, ConditionAggregate


@dataclass(frozen=True)
class SummaryRow:
    condition: str
    run_count: int
    metrics: dict[str, str]


@dataclass(frozen=True)
class SummaryData:
    generation: int
    headers: list[str]
    rows: list[SummaryRow]
    skipped_runs: list[SkippedRun]


def build_summary(
    aggregates: list[ConditionAggregate], skipped_runs: list[SkippedRun]
) -> SummaryData:
    generation = min(
        int(aggregate.summary_frame["generation"].max()) for aggregate in aggregates
    )
    rows: list[SummaryRow] = []

    for aggregate in aggregates:
        row = aggregate.summary_frame[
            aggregate.summary_frame["generation"] == generation
        ].iloc[-1]
        metrics = {
            metric: f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"
            for metric in COMPARISON_METRICS
        }
        rows.append(
            SummaryRow(
                condition=aggregate.label,
                run_count=len(aggregate.runs),
                metrics=metrics,
            )
        )

    return SummaryData(
        generation=generation,
        headers=["condition", "runs", *COMPARISON_METRICS],
        rows=rows,
        skipped_runs=skipped_runs,
    )
