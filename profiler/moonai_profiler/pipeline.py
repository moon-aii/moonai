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

        first_gen = int(run.trimmed_generations["generation"].iloc[0])
        last_gen = int(run.trimmed_generations["generation"].iloc[-1])
        trim_note = (
            f"Generations {first_gen}\u2013{last_gen} of "
            f"{run.total_generation_count} (IQR-trimmed, middle 50%)"
        )

        run_sections.append(
            {
                "name": run.name,
                "label": run.label,
                "experiment_name": run.experiment_name,
                "mode": run.mode,
                "total_generation_count": run.total_generation_count,
                "generation_count": run.generation_count,
                "trim_note": trim_note,
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
                "summary_events": _format_event_summary(run.trimmed_summary_events),
                "summary_counters": _format_counter_summary(
                    run.trimmed_summary_counters
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
                    "total_generations": run.total_generation_count,
                    "trimmed_generations": run.generation_count,
                    "avg_generation_ms": f"{run.avg_generation_ms:.2f}",
                    "top_event": run.top_event_name,
                    "top_event_avg_ms": f"{run.top_event_avg_ms:.2f}",
                    "top_event_nonzero_generation_count": str(
                        run.top_event_nonzero_generation_count
                    ),
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


def _format_event_summary(events: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    rows = []
    for name, values in events.items():
        rows.append(
            {
                "name": name,
                "total_ms": f"{values['total_ms']:.3f}",
                "avg_ms_per_generation": f"{values['avg_ms_per_generation']:.3f}",
                "nonzero_generation_count": str(
                    int(values["nonzero_generation_count"])
                ),
                "avg_ms_per_nonzero_generation": f"{values['avg_ms_per_nonzero_generation']:.3f}",
            }
        )
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _format_counter_summary(
    counters: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    rows = []
    for name, values in counters.items():
        rows.append(
            {
                "name": name,
                "total": f"{values['total']:.3f}",
                "avg_per_generation": f"{values['avg_per_generation']:.3f}",
                "nonzero_generation_count": str(
                    int(values["nonzero_generation_count"])
                ),
                "avg_per_nonzero_generation": f"{values['avg_per_nonzero_generation']:.3f}",
            }
        )
    rows.sort(key=lambda row: float(row["total"]), reverse=True)
    return rows
