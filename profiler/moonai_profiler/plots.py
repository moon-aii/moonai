"""Plot helpers for profiler suite analysis."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt

from .io import ProfileSuite


@dataclass(frozen=True)
class Chart:
    title: str
    image_uri: str
    caption: str


def render_comparison_charts(suites: list[ProfileSuite]) -> list[Chart]:
    return [
        _render_suite_generation_comparison(suites),
        _render_suite_variability_comparison(suites),
        _render_suite_hotspot_comparison(suites),
        _render_cross_run_generation_lines(suites),
    ]


def render_suite_charts(suite: ProfileSuite) -> list[Chart]:
    return [
        _render_member_generation_chart(suite),
        _render_top_event_breakdown(suite),
        _render_member_run_totals(suite),
    ]


def _render_suite_generation_comparison(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [suite.label for suite in suites]
    values = [suite.avg_generation_ms for suite in suites]
    ax.bar(labels, values, color="#355070")
    ax.set_ylabel("Average generation time (ms)")
    ax.set_title("Trimmed Suite Generation Time")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Average Generation Time",
        image_uri=_figure_to_data_uri(fig),
        caption="Each bar is the mean of the middle four runs after dropping the fastest and slowest suite members.",
    )


def _render_suite_variability_comparison(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [suite.label for suite in suites]
    values = [suite.avg_generation_ms_stddev for suite in suites]
    ax.bar(labels, values, color="#b56576")
    ax.set_ylabel("Stddev of kept run means (ms)")
    ax.set_title("Run-to-Run Stability")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Suite Variability",
        image_uri=_figure_to_data_uri(fig),
        caption="Lower is better. Uses the four kept runs only.",
    )


def _render_suite_hotspot_comparison(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    labels = [suite.label for suite in suites]
    pairs = [_top_event_pair(suite) for suite in suites]
    values = [pair[1] for pair in pairs]
    ax.bar(labels, values, color="#577590")
    ax.set_ylabel("Average event time per generation (ms)")
    ax.set_title("Top Aggregate Hotspot Per Suite")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    for index, (name, value) in enumerate(pairs):
        ax.text(index, value, name, rotation=90, va="bottom", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.88)
    return Chart(
        title="Dominant Hotspots",
        image_uri=_figure_to_data_uri(fig),
        caption="Each bar shows the highest non-generation aggregate event among the kept runs.",
    )


def _render_cross_run_generation_lines(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    has_data = False
    for suite in suites:
        members = [
            member for member in suite.kept_members if member.generation_times_ms
        ]
        if not members:
            members = [member for member in suite.members if member.generation_times_ms]
        if not members:
            continue

        generation_count = min(len(member.generation_times_ms) for member in members)
        if generation_count == 0:
            continue

        averaged_generation_times = [
            sum(member.generation_times_ms[index] for member in members) / len(members)
            for index in range(generation_count)
        ]

        has_data = True
        generations = list(range(generation_count))
        ax.plot(
            generations,
            averaged_generation_times,
            label=suite.label,
            color="#355070",
            alpha=0.95,
            linewidth=1.8,
        )

    if has_data:
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Generation time (ms)")
    ax.set_title("Per-Suite Generation Time Lines")
    ax.set_xlim(left=0)
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Per-Suite Generation Lines",
        image_uri=_figure_to_data_uri(fig),
        caption="Each line is the per-generation mean across suite seeds, using kept runs when available.",
    )


def _render_member_generation_chart(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [f"seed {member.seed}" for member in suite.members]
    values = [member.avg_generation_ms for member in suite.members]
    colors = [
        "#355070" if member.disposition == "kept" else "#c97b63"
        for member in suite.members
    ]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average generation time (ms)")
    ax.set_title(f"Suite Member Means - {suite.label}")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Member Run Means",
        image_uri=_figure_to_data_uri(fig),
        caption="Orange bars are dropped runs. Blue bars are kept runs that feed the aggregate summary.",
    )


def _render_top_event_breakdown(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pairs = [
        (name, stats["avg_ms_per_generation"])
        for name, stats in suite.aggregate_events.items()
        if name != "generation_total" and stats.get("avg_ms_per_generation", 0.0) > 0.0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    pairs = pairs[:8]
    labels = [name for name, _ in pairs]
    values = [value for _, value in pairs]
    ax.barh(labels, values, color="#6d597a")
    ax.invert_yaxis()
    ax.set_xlabel("Average time per generation (ms)")
    ax.set_title(f"Top Aggregate Events - {suite.label}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Top Aggregate Events",
        image_uri=_figure_to_data_uri(fig),
        caption="Aggregate event ranking computed from the four kept runs.",
    )


def _render_member_run_totals(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [f"seed {member.seed}" for member in suite.members]
    values = [member.run_total_ms for member in suite.members]
    colors = [
        "#2d6a4f" if member.disposition == "kept" else "#c97b63"
        for member in suite.members
    ]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Run total time (ms)")
    ax.set_title(f"Suite Member Run Totals - {suite.label}")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Run Total Time",
        image_uri=_figure_to_data_uri(fig),
        caption="Whole-run wall time per member run. Useful for cross-checking process-level stability.",
    )


def _top_event_pair(suite: ProfileSuite) -> tuple[str, float]:
    pairs = [
        (name, stats["avg_ms_per_generation"])
        for name, stats in suite.aggregate_events.items()
        if name != "generation_total" and stats.get("avg_ms_per_generation", 0.0) > 0.0
    ]
    if not pairs:
        return ("none", 0.0)
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs[0]


def _figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
