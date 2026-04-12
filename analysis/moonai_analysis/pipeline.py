"""Full analysis pipeline orchestration."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import shutil

from .genome import load_latest_genome, render_genome_plot
from .html_report import render_html_report
from .io import RunData, discover_runs
from .labels import LabelResolver
from .plots import (
    build_condition_aggregate,
    render_comparison_charts,
    render_condition_charts,
)
from .summary import build_summary


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    runs, skipped_runs = discover_runs(input_dir)
    if not runs:
        raise SystemExit(f"No qualifying runs found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_artifacts(output_dir)

    grouped_runs = _group_runs(runs)
    aggregates = [
        build_condition_aggregate(label, grouped_runs[label])
        for label in sorted(grouped_runs)
    ]

    generated_at = datetime.now()
    comparison_charts = render_comparison_charts(aggregates)
    condition_sections = []
    for aggregate in aggregates:
        charts = render_condition_charts(aggregate)
        genome = load_latest_genome(aggregate.representative_run.path)
        genome_chart = None
        if genome is not None:
            title = (
                f"{aggregate.label} - Representative Genome "
                f"({aggregate.representative_run.name}, Step {genome.get('step', '?')})"
            )
            genome_chart = {
                "title": "Genome",
                "image_uri": render_genome_plot(genome, title),
                "caption": f"Representative topology snapshot for `{aggregate.representative_run.name}`.",
            }
        condition_sections.append(
            {
                "label": aggregate.label,
                "run_count": len(aggregate.runs),
                "representative_run": aggregate.representative_run.name,
                "final_step": int(aggregate.summary_frame["step"].max()),
                "charts": [chart.__dict__ for chart in charts],
                "genome_chart": genome_chart,
            }
        )

    summary = build_summary(aggregates, skipped_runs)
    report_name = f"report_{generated_at.strftime('%Y%m%d_%H%M%S')}.html"
    report_path = output_dir / report_name
    report_html = render_html_report(
        {
            "generated_at": generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "report_name": report_name,
            "input_dir": str(input_dir),
            "run_count": len(runs),
            "condition_count": len(aggregates),
            "skipped_count": len(skipped_runs),
            "summary_step": summary.step,
            "summary_headers": summary.headers,
            "summary_rows": [
                {
                    "condition": row.condition,
                    "run_count": row.run_count,
                    "metrics": row.metrics,
                }
                for row in summary.rows
            ],
            "comparison_charts": [chart.__dict__ for chart in comparison_charts],
            "condition_sections": condition_sections,
            "skipped_runs": [
                {"name": skipped.path.name, "reason": skipped.reason}
                for skipped in skipped_runs
            ],
        }
    )
    report_path.write_text(report_html, encoding="utf-8")

    print(f"Analysed {len(runs)} runs across {len(aggregates)} conditions.")
    print(f"Wrote self-contained analysis report to {report_path}")


def _group_runs(runs: list[RunData]) -> dict[str, list[RunData]]:
    resolver = LabelResolver()
    grouped: dict[str, list[RunData]] = defaultdict(list)
    for run in runs:
        grouped[resolver.resolve(run)].append(run)
    return dict(grouped)


def _remove_legacy_artifacts(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.name in {"conditions", "comparisons"} and path.is_dir():
            shutil.rmtree(path)
            continue
        if path.suffix.lower() in {".md", ".png"}:
            path.unlink()
