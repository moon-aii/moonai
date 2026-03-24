"""Profiler analysis pipeline orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .html_report import render_html_report
from .io import discover_profiles
from .plots import render_comparison_charts, render_run_charts


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    runs, skipped = discover_profiles(input_dir)
    if not runs:
        raise SystemExit(f"No profile.json files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    comparison_charts = render_comparison_charts(runs)

    run_sections = []
    for run in runs:
        charts = render_run_charts(run)
        run_sections.append(
            {
                "name": run.name,
                "label": run.label,
                "experiment_name": run.experiment_name,
                "mode": run.mode,
                "generation_count": run.generation_count,
                "avg_generation_ms": f"{run.avg_generation_ms:.2f}",
                "top_event_name": run.top_event_name,
                "top_event_total_ms": f"{run.top_event_ms:.2f}",
                "top_event_avg_ms": f"{run.top_event_avg_ms:.2f}",
                "top_event_nonzero_generation_count": str(
                    run.top_event_nonzero_generation_count
                ),
                "path": str(run.path),
                "run_meta": run.raw["run"],
                "run_notes": run.raw.get("notes", []),
                "summary_meta": {
                    "cpu_generation_count": str(
                        run.raw["summary"].get("cpu_generation_count", 0)
                    ),
                    "gpu_generation_count": str(
                        run.raw["summary"].get("gpu_generation_count", 0)
                    ),
                    "path_count_note": str(
                        run.raw["summary"].get("path_count_note", "")
                    ),
                },
                "summary_events": _format_summary_mapping(
                    run.raw["summary"]["events"],
                    [
                        "total_ms",
                        "avg_ms_per_generation",
                        "nonzero_generation_count",
                        "avg_ms_per_nonzero_generation",
                    ],
                ),
                "summary_counters": _format_summary_mapping(
                    run.raw["summary"]["counters"],
                    [
                        "total",
                        "avg_per_generation",
                        "nonzero_generation_count",
                        "avg_per_nonzero_generation",
                    ],
                ),
                "charts": [chart.__dict__ for chart in charts],
            }
        )

    report_name = f"profile_report_{generated_at.strftime('%Y%m%d_%H%M%S')}.html"
    report_path = output_dir / report_name
    report_html = render_html_report(
        {
            "generated_at": generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "report_name": report_name,
            "input_dir": str(input_dir),
            "run_count": len(runs),
            "skipped_count": len(skipped),
            "summary_rows": [
                {
                    "label": run.label,
                    "generations": run.generation_count,
                    "avg_generation_ms": f"{run.avg_generation_ms:.2f}",
                    "top_event": run.top_event_name,
                    "top_event_avg_ms": f"{run.top_event_avg_ms:.2f}",
                    "top_event_nonzero_generation_count": str(
                        run.top_event_nonzero_generation_count
                    ),
                    "path": str(run.path),
                }
                for run in runs
            ],
            "comparison_charts": [chart.__dict__ for chart in comparison_charts],
            "run_sections": run_sections,
            "skipped_profiles": [
                {"name": skipped_run.path.name, "reason": skipped_run.reason}
                for skipped_run in skipped
            ],
        }
    )
    report_path.write_text(report_html, encoding="utf-8")

    print(f"Analysed {len(runs)} profile runs.")
    print(f"Wrote self-contained profiler report to {report_path}")


def _format_summary_mapping(data: dict, keys: list[str]) -> list[dict[str, str]]:
    rows = []
    for name, values in data.items():
        row = {"name": name}
        for key in keys:
            value = values.get(key, 0)
            if isinstance(value, float):
                row[key] = f"{value:.3f}"
            else:
                row[key] = str(value)
        rows.append(row)
    rows.sort(key=lambda row: float(row[keys[0]]), reverse=True)
    return rows
