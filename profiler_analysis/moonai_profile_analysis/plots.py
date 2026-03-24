"""Plot helpers for profiler analysis."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt

from .io import ProfileRun


@dataclass(frozen=True)
class Chart:
    title: str
    image_uri: str
    caption: str


def render_comparison_charts(runs: list[ProfileRun]) -> list[Chart]:
    return [
        _render_generation_comparison(runs),
        _render_hotspot_comparison(runs),
    ]


def render_run_charts(run: ProfileRun) -> list[Chart]:
    return [
        _render_generation_timeline(run),
        _render_key_event_timeline(run),
        _render_top_event_breakdown(run),
    ]


def _render_generation_comparison(runs: list[ProfileRun]) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [run.label for run in runs]
    values = [run.avg_generation_ms for run in runs]
    colors = [_mode_color(run.mode) for run in runs]

    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average generation time (ms)")
    ax.set_title("Average Generation Wall Time")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Average Generation Time",
        image_uri=_figure_to_data_uri(fig),
        caption="Lower is better. `generation_total` is measured only in headless mode.",
    )


def _render_hotspot_comparison(runs: list[ProfileRun]) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [run.label for run in runs]
    values = [run.top_event_avg_ms for run in runs]
    ax.bar(labels, values, color="#c97b63")
    ax.set_ylabel("Average hotspot time per generation (ms)")
    ax.set_title("Top Non-Generation Hotspot Per Run")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for index, run in enumerate(runs):
        ax.text(
            index,
            values[index],
            run.top_event_name,
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    return Chart(
        title="Dominant Hotspots",
        image_uri=_figure_to_data_uri(fig),
        caption="Each bar shows the largest non-`generation_total` event ranked by average time per generation.",
    )


def _render_generation_timeline(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        run.generations["generation"],
        run.generations["event::generation_total"],
        color="#2d6a4f",
        linewidth=2,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Generation wall time (ms)")
    ax.set_title(f"Generation Wall Time - {run.label}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Generation Timeline",
        image_uri=_figure_to_data_uri(fig),
        caption="Per-generation `generation_total` trend for this profile run.",
    )


def _render_key_event_timeline(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    event_names = _top_event_names(run, limit=4)
    for event_name in event_names:
        column = f"event::{event_name}"
        if column not in run.generations:
            continue
        ax.plot(
            run.generations["generation"],
            run.generations[column],
            linewidth=1.8,
            label=event_name,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Event time (ms)")
    ax.set_title(f"Key Event Timelines - {run.label}")
    ax.grid(alpha=0.25)
    if event_names:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return Chart(
        title="Key Event Timelines",
        image_uri=_figure_to_data_uri(fig),
        caption="Top summary events plotted across generations for quick hotspot drift detection.",
    )


def _render_top_event_breakdown(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    summary_events = run.raw["summary"]["events"]
    pairs = [
        (name, float(values.get("total_ms", 0.0) or 0.0))
        for name, values in summary_events.items()
        if name != "generation_total"
        and float(values.get("total_ms", 0.0) or 0.0) > 0.0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    pairs = pairs[:8]

    labels = [name for name, _ in pairs]
    values = [value for _, value in pairs]
    ax.barh(labels, values, color="#577590")
    ax.invert_yaxis()
    ax.set_xlabel("Total time (ms)")
    ax.set_title(f"Top Event Totals - {run.label}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Top Event Totals",
        image_uri=_figure_to_data_uri(fig),
        caption="Top accumulated event totals from the summary section of the profile.",
    )


def _top_event_names(run: ProfileRun, limit: int) -> list[str]:
    summary_events = run.raw["summary"]["events"]
    pairs = [
        (name, float(values.get("total_ms", 0.0) or 0.0))
        for name, values in summary_events.items()
        if name != "generation_total"
        and float(values.get("total_ms", 0.0) or 0.0) > 0.0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in pairs[:limit]]


def _mode_color(mode: str) -> str:
    if mode == "gpu":
        return "#355070"
    if mode == "mixed":
        return "#b56576"
    return "#6d597a"


def _figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
