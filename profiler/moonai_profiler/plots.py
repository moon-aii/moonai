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
        _render_generation_timeline_comparison(runs),
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
    ax.set_title("Average Generation Wall Time (IQR-trimmed)")
    ax.tick_params(axis="x", rotation=35, labelsize=6)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Average Generation Time",
        image_uri=_figure_to_data_uri(fig),
        caption="Lower is better. Averaged over IQR-trimmed generations only.",
    )


def _render_generation_timeline_comparison(runs: list[ProfileRun]) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    gen_col = "event::generation_total"
    for run in runs:
        frame = run.trimmed_generations
        if gen_col not in frame.columns:
            continue
        ax.plot(
            frame["generation"],
            frame[gen_col],
            linewidth=1.8,
            label=run.label,
            color=_mode_color(run.mode),
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Generation wall time (ms)")
    ax.set_title("Generation Wall Time Comparison (IQR-trimmed)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return Chart(
        title="Generation Timeline Comparison",
        image_uri=_figure_to_data_uri(fig),
        caption="Per-generation wall time for each run overlaid (IQR-trimmed generations only).",
    )


def _render_hotspot_comparison(runs: list[ProfileRun]) -> Chart:
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    labels = [run.label for run in runs]
    values = [run.top_event_avg_ms for run in runs]
    ax.bar(labels, values, color="#c97b63")
    ax.set_ylabel("Average hotspot time per active generation (ms)")
    ax.set_title("Top Non-Generation Hotspot Per Run (IQR-trimmed)")
    ax.tick_params(axis="x", rotation=35, labelsize=6)
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
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.26, top=0.88)
    return Chart(
        title="Dominant Hotspots",
        image_uri=_figure_to_data_uri(fig),
        caption="Each bar shows the largest non-`generation_total` event ranked by average time per active generation.",
    )


def _render_generation_timeline(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    gen_col = "event::generation_total"
    frame = run.trimmed_generations
    if gen_col in frame.columns:
        ax.plot(
            frame["generation"],
            frame[gen_col],
            color="#2d6a4f",
            linewidth=2,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "generation_total not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Generation wall time (ms)")
    ax.set_title(f"Generation Wall Time - {run.label}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Generation Timeline",
        image_uri=_figure_to_data_uri(fig),
        caption="Per-generation `generation_total` trend (IQR-trimmed generations).",
    )


def _render_key_event_timeline(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    event_names = _top_event_names(run, limit=4)
    frame = run.trimmed_generations
    for event_name in event_names:
        column = f"event::{event_name}"
        if column not in frame.columns:
            continue
        ax.plot(
            frame["generation"],
            frame[column],
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
        caption="Events are chosen by average time per active generation (IQR-trimmed).",
    )


def _render_top_event_breakdown(run: ProfileRun) -> Chart:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pairs = [
        (name, stats["avg_ms_per_nonzero_generation"])
        for name, stats in run.trimmed_summary_events.items()
        if name != "generation_total" and stats["avg_ms_per_nonzero_generation"] > 0.0
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    pairs = pairs[:8]

    labels = [name for name, _ in pairs]
    values = [value for _, value in pairs]
    ax.barh(labels, values, color="#577590")
    ax.invert_yaxis()
    ax.set_xlabel("Average time per active generation (ms)")
    ax.set_title(f"Top Event Active-Generation Averages - {run.label}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Top Event Active-Generation Averages",
        image_uri=_figure_to_data_uri(fig),
        caption="Top summary events ranked by average time per active generation (IQR-trimmed).",
    )


def _top_event_names(run: ProfileRun, limit: int) -> list[str]:
    pairs = [
        (name, stats["avg_ms_per_nonzero_generation"])
        for name, stats in run.trimmed_summary_events.items()
        if name != "generation_total" and stats["avg_ms_per_nonzero_generation"] > 0.0
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
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
