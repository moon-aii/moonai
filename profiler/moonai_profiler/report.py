"""Profiler report generation."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt

from .html_report import render_html_report
from .io import ProfileSuite, load_suites


@dataclass(frozen=True)
class Chart:
    title: str
    image_uri: str
    caption: str


def generate_report(input_dir: Path, output_dir: Path) -> None:
    """Generate HTML report from profiler suites."""
    suites = load_suites(input_dir)
    if not suites:
        raise SystemExit(f"No profile suites found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()

    # Generate comparison charts
    comparison = [
        _chart_window_comparison(suites),
        _chart_hotspots(suites),
        _chart_timelines(suites),
    ]

    # Generate per-suite sections
    sections = [_build_section(s) for s in reversed(suites)]

    # Build and write report
    report = render_html_report(
        {
            "generated_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "report_name": f"profile_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html",
            "input_dir": str(input_dir),
            "run_count": len(suites),
            "summary_rows": [
                {
                    "name": s.name,
                    "total_windows": s.windows,
                    "kept_runs": len(s.kept),
                    "avg_window_ms": f"{s.avg_window_ms:.2f}",
                    "top_event": _top_event(s)[0],
                    "top_event_avg_ms": f"{_top_event(s)[1]:.2f}",
                }
                for s in suites
            ],
            "comparison_charts": [c.__dict__ for c in comparison],
            "run_sections": sections,
        }
    )

    report_path = (
        output_dir / f"profile_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Analysed {len(suites)} profiler suites.")
    print(f"Wrote report to {report_path}")


def _build_section(suite: ProfileSuite) -> dict:
    """Build a report section for a single suite."""
    dropped_str = ", ".join(f"seed {m.seed} ({m.disposition})" for m in suite.dropped)

    return {
        "name": suite.name,
        "total_window_count": suite.windows,
        "kept_run_count": len(suite.kept),
        "trim_note": f"6 runs total; dropped {dropped_str}; averaged the remaining 4 runs"
        if dropped_str
        else "All runs kept",
        "path": str(suite.path),
        "summary_events": _format_events(suite.events),
        "charts": [c.__dict__ for c in _suite_charts(suite)],
        "members": [
            {
                "seed": str(m.seed),
                "avg_window_ms": f"{m.avg_window_ms:.2f}",
                "run_total_ms": f"{m.run_total_ms:.2f}",
                "window_count": str(m.window_count),
                "disposition": m.disposition,
            }
            for m in suite.members
        ],
    }


def _format_events(events: dict) -> list[dict]:
    """Format event stats for display."""
    window_total = events.get("window_total", {}).get("total_ms", 0.0)
    rows = []
    for name, values in events.items():
        total = values.get("total_ms", 0.0)
        if total <= 0:
            continue
        pct = (total / window_total * 100) if window_total > 0 else 0.0
        rows.append(
            {
                "name": name,
                "percentage": f"{pct:.1f}",
                "avg_ms_per_window": f"{values.get('avg_ms_per_window', 0.0):.3f}",
                "nonzero_window_count": str(int(values.get("nonzero_window_count", 0))),
                "avg_ms_per_nonzero_window": f"{values.get('avg_ms_per_nonzero_window', 0.0):.3f}",
                "total_ms": f"{total:.3f}",
            }
        )
    return sorted(rows, key=lambda r: float(r["total_ms"]), reverse=True)


def _top_event(suite: ProfileSuite) -> tuple[str, float]:
    """Find the top non-window event by average time."""
    best_name = "window_total"
    best_value = 0.0
    for name, values in suite.events.items():
        if name == "window_total":
            continue
        avg = values.get("avg_ms_per_window", 0.0)
        if avg > best_value:
            best_name = name
            best_value = avg
    return best_name, best_value


# Chart generation


def _chart_window_comparison(suites: list[ProfileSuite]) -> Chart:
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    labels = [s.name for s in suites]
    values = [s.avg_window_ms for s in suites]

    # Bar chart on primary y-axis (left)
    bars = ax1.bar(labels, values, color="#355070", label="Avg Window Time")
    ax1.set_ylabel("Average report-window time (ms)", color="#355070")
    ax1.tick_params(axis="y", labelcolor="#355070")
    ax1.set_title("Trimmed Suite Report-Window Time")
    ax1.tick_params(axis="x", rotation=35, labelsize=7)
    ax1.grid(axis="y", alpha=0.25)

    # Standard deviation line on secondary y-axis (right)
    ax2 = ax1.twinx()
    stddev_values = [s.stddev for s in suites]
    ax2.plot(
        labels,
        stddev_values,
        color="#c97b63",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Std Dev",
    )
    ax2.set_ylabel("Standard deviation (ms)", color="#c97b63")
    ax2.tick_params(axis="y", labelcolor="#c97b63")

    fig.tight_layout()
    return Chart(
        title="Average Report-Window Time",
        image_uri=_to_data_uri(fig),
        caption="Bars: Mean of middle four runs. Line: Standard deviation (lower is more stable).",
    )


def _chart_hotspots(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels = [s.name for s in suites]
    pairs = [_top_event(s) for s in suites]
    values = [p[1] for p in pairs]
    ax.bar(labels, values, color="#577590")
    ax.set_ylabel("Average event time per report window (ms)")
    ax.set_title("Top Aggregate Hotspot Per Suite")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    for i, (name, val) in enumerate(pairs):
        ax.text(i, val, name, rotation=90, va="bottom", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.88)
    return Chart(
        title="Dominant Hotspots",
        image_uri=_to_data_uri(fig),
        caption="Highest non-window-total aggregate event among kept runs.",
    )


def _chart_timelines(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    has_data = False
    for suite in suites:
        members = [m for m in suite.kept if m.window_times_ms] or [
            m for m in suite.members if m.window_times_ms
        ]
        if not members:
            continue
        count = min(len(m.window_times_ms) for m in members)
        if count == 0:
            continue
        averaged = [
            sum(m.window_times_ms[i] for m in members) / len(members)
            for i in range(count)
        ]
        has_data = True
        ax.plot(
            list(range(count)),
            averaged,
            label=suite.name,
            color="#355070",
            alpha=0.95,
            linewidth=1.8,
        )

    if has_data:
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlabel("Report Window")
    ax.set_ylabel("Report-window time (ms)")
    ax.set_title("Per-Suite Report-Window Time Lines")
    ax.set_xlim(left=0)
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Per-Suite Report-Window Lines",
        image_uri=_to_data_uri(fig),
        caption="Per-window mean across suite seeds, using kept runs when available.",
    )


def _suite_charts(suite: ProfileSuite) -> list[Chart]:
    """Generate charts for a single suite."""
    return [
        _chart_member_means(suite),
        _chart_top_events(suite),
    ]


def _chart_member_means(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [f"seed {m.seed}" for m in suite.members]
    values = [m.avg_window_ms for m in suite.members]
    colors = [
        "#355070" if m.disposition == "kept" else "#c97b63" for m in suite.members
    ]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average report-window time (ms)")
    ax.set_title(f"Suite Member Means - {suite.name}")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Member Run Means",
        image_uri=_to_data_uri(fig),
        caption="Orange bars are dropped runs. Blue bars are kept runs.",
    )


def _chart_top_events(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pairs = [
        (name, stats["avg_ms_per_window"])
        for name, stats in suite.events.items()
        if name != "window_total" and stats.get("avg_ms_per_window", 0.0) > 0.0
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:8]
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    ax.barh(labels, values, color="#6d597a")
    ax.invert_yaxis()
    ax.set_xlabel("Average time per report window (ms)")
    ax.set_title(f"Top Aggregate Events - {suite.name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Top Aggregate Events",
        image_uri=_to_data_uri(fig),
        caption="Aggregate event ranking from four kept runs.",
    )


def _to_data_uri(fig) -> str:
    """Convert matplotlib figure to base64 data URI."""
    buffer = BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return f"data:image/svg+xml;base64,{b64encode(buffer.getvalue()).decode('ascii')}"
