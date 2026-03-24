"""Profiler suite analysis pipeline orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .html_report import render_html_report
from .io import discover_profile_suites
from .plots import render_comparison_charts, render_suite_charts


def run_analysis(input_dir: Path, output_dir: Path) -> None:
    suites, skipped = discover_profile_suites(input_dir)
    if not suites:
        raise SystemExit(f"No profile_suite.json files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now()
    comparison_charts = render_comparison_charts(suites)

    run_sections = []
    for suite in suites:
        charts = render_suite_charts(suite)
        dropped = ", ".join(
            f"seed {member.seed} ({member.disposition})"
            for member in suite.dropped_members
        )
        trim_note = f"6 runs total; dropped {dropped}; averaged the remaining 4 runs"
        run_sections.append(
            {
                "name": suite.name,
                "label": suite.label,
                "experiment_name": suite.experiment_name,
                "suite_name": suite.suite_name,
                "mode": "suite",
                "total_generation_count": suite.generations,
                "generation_count": len(suite.kept_members),
                "trim_note": trim_note,
                "avg_generation_ms": f"{suite.avg_generation_ms:.2f}",
                "top_event_name": _top_event_name(suite),
                "top_event_total_ms": f"{_top_event_total_ms(suite):.2f}",
                "top_event_avg_ms": f"{_top_event_avg_ms(suite):.2f}",
                "top_event_nonzero_generation_count": str(
                    int(_top_event_nonzero_count(suite))
                ),
                "path": str(suite.path),
                "run_meta": suite.raw["suite"],
                "run_notes": _suite_notes(suite),
                "summary_meta": {
                    "suite_name": suite.suite_name,
                    "cpu_generation_count": str(
                        suite.raw["aggregate"].get("cpu_generation_count_avg", 0.0)
                    ),
                    "gpu_generation_count": str(
                        suite.raw["aggregate"].get("gpu_generation_count_avg", 0.0)
                    ),
                    "path_count_note": "Averages are computed from the four kept runs.",
                },
                "summary_events": _format_event_summary(suite.aggregate_events),
                "summary_counters": _format_counter_summary(suite.aggregate_counters),
                "charts": [chart.__dict__ for chart in charts],
                "members": [
                    {
                        "seed": str(member.seed),
                        "avg_generation_ms": f"{member.avg_generation_ms:.2f}",
                        "run_total_ms": f"{member.run_total_ms:.2f}",
                        "generation_count": str(member.generation_count),
                        "disposition": member.disposition,
                    }
                    for member in suite.members
                ],
            }
        )

    # Reverse Profile Details order (most recent first) while keeping comparison charts in original order
    run_sections.reverse()

    report_name = f"profile_report_{generated_at.strftime('%Y%m%d_%H%M%S')}.html"
    report_path = output_dir / report_name
    report_html = render_html_report(
        {
            "generated_at": generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "report_name": report_name,
            "input_dir": str(input_dir),
            "run_count": len(suites),
            "skipped_count": len(skipped),
            "summary_rows": [
                {
                    "label": suite.label,
                    "total_generations": suite.generations,
                    "trimmed_generations": len(suite.kept_members),
                    "avg_generation_ms": f"{suite.avg_generation_ms:.2f}",
                    "top_event": _top_event_name(suite),
                    "top_event_avg_ms": f"{_top_event_avg_ms(suite):.2f}",
                    "top_event_nonzero_generation_count": str(
                        int(_top_event_nonzero_count(suite))
                    ),
                }
                for suite in suites
            ],
            "comparison_charts": [chart.__dict__ for chart in comparison_charts],
            "run_sections": run_sections,
            "skipped_profiles": [
                {"name": skipped_suite.path.name, "reason": skipped_suite.reason}
                for skipped_suite in skipped
            ],
        }
    )
    report_path.write_text(report_html, encoding="utf-8")

    print(f"Analysed {len(suites)} profiler suites.")
    print(f"Wrote self-contained profiler report to {report_path}")


def _suite_notes(suite) -> list[str]:
    return [
        "Profiler suites run six fixed seeds from profiler.lua.",
        "The fastest and slowest runs are dropped before aggregation.",
        "Aggregate timing and counter tables use the remaining four runs only.",
    ]


def _top_event_name(suite) -> str:
    best_name = "generation_total"
    best_value = 0.0
    for name, values in suite.aggregate_events.items():
        if name == "generation_total":
            continue
        avg = values.get("avg_ms_per_generation", 0.0)
        if avg > best_value:
            best_name = name
            best_value = avg
    return best_name


def _top_event_total_ms(suite) -> float:
    return suite.aggregate_events.get(_top_event_name(suite), {}).get("total_ms", 0.0)


def _top_event_avg_ms(suite) -> float:
    return suite.aggregate_events.get(_top_event_name(suite), {}).get(
        "avg_ms_per_generation", 0.0
    )


def _top_event_nonzero_count(suite) -> float:
    return suite.aggregate_events.get(_top_event_name(suite), {}).get(
        "nonzero_generation_count", 0.0
    )


def _format_event_summary(events: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    rows = []
    generation_total = events.get("generation_total", {}).get("total_ms", 0.0)
    for name, values in events.items():
        if values.get("total_ms", 0.0) <= 0.0:
            continue
        total_ms = values.get("total_ms", 0.0)
        percentage = (
            (total_ms / generation_total * 100) if generation_total > 0 else 0.0
        )
        rows.append(
            {
                "name": name,
                "percentage": f"{percentage:.1f}",
                "avg_ms_per_generation": f"{values.get('avg_ms_per_generation', 0.0):.3f}",
                "nonzero_generation_count": str(
                    int(values.get("nonzero_generation_count", 0.0))
                ),
                "avg_ms_per_nonzero_generation": f"{values.get('avg_ms_per_nonzero_generation', 0.0):.3f}",
                "total_ms": f"{total_ms:.3f}",
            }
        )
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _format_counter_summary(
    counters: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    rows = []
    for name, values in counters.items():
        if values.get("total", 0.0) <= 0.0:
            continue
        rows.append(
            {
                "name": name,
                "total": f"{values.get('total', 0.0):.3f}",
                "avg_per_generation": f"{values.get('avg_per_generation', 0.0):.3f}",
                "nonzero_generation_count": str(
                    int(values.get("nonzero_generation_count", 0.0))
                ),
                "avg_per_nonzero_generation": f"{values.get('avg_per_nonzero_generation', 0.0):.3f}",
            }
        )
    rows.sort(key=lambda row: float(row["total"]), reverse=True)
    return rows
