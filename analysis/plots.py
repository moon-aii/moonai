"""Plot generation helpers for MoonAI analysis."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .io import RunData, load_optional_csv

STYLE = {
    "predator_count": "#DC2626",
    "prey_count": "#16A34A",
    "predator_species": "#7C3AED",
    "prey_species": "#2563EB",
    "avg_complexity": "#F97316",
    "nodes": "#0EA5E9",
    "connections": "#EF4444",
}

COMPARISON_METRICS = [
    "predator_species",
    "prey_species",
    "avg_complexity",
    "predator_count",
    "prey_count",
]


@dataclass(frozen=True)
class ConditionAggregate:
    label: str
    runs: list[RunData]
    summary_frame: pd.DataFrame
    representative_run: RunData


@dataclass(frozen=True)
class EmbeddedChart:
    title: str
    image_uri: str
    caption: str


def build_condition_aggregate(label: str, runs: list[RunData]) -> ConditionAggregate:
    tick_frames = []
    for run in runs:
        frame = run.stats[["tick", *COMPARISON_METRICS]].copy()
        frame["run"] = run.name
        tick_frames.append(frame)

    combined = pd.concat(tick_frames, ignore_index=True)
    grouped = combined.groupby("tick")
    summary = pd.DataFrame({"tick": sorted(combined["tick"].unique())})
    for metric in COMPARISON_METRICS:
        summary[f"{metric}_mean"] = grouped[metric].mean().reindex(summary["tick"]).to_numpy()
        summary[f"{metric}_std"] = grouped[metric].std(ddof=0).fillna(0.0).reindex(summary["tick"]).to_numpy()

    representative_run = max(runs, key=lambda run: float(run.stats["avg_complexity"].iloc[-1]))
    return ConditionAggregate(
        label=label,
        runs=runs,
        summary_frame=summary,
        representative_run=representative_run,
    )


def render_condition_charts(aggregate: ConditionAggregate) -> list[EmbeddedChart]:
    return [
        render_population_chart(aggregate),
        render_species_chart(aggregate.representative_run, aggregate.label),
        render_complexity_chart(aggregate),
    ]


def render_comparison_charts(
    aggregates: list[ConditionAggregate],
) -> list[EmbeddedChart]:
    charts: list[EmbeddedChart] = []
    for metric in COMPARISON_METRICS:
        figure, axis = plt.subplots(figsize=(12, 6))
        for aggregate in aggregates:
            frame = aggregate.summary_frame
            mean = frame[f"{metric}_mean"]
            std = frame[f"{metric}_std"]
            axis.plot(frame["tick"], mean, label=aggregate.label, linewidth=1.7)
            axis.fill_between(frame["tick"], mean - std, mean + std, alpha=0.12)

        axis.set_xlabel("Tick")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(f"{metric.replace('_', ' ').title()} by Condition")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        charts.append(
            EmbeddedChart(
                title=metric.replace("_", " ").title(),
                image_uri=_figure_to_data_uri(figure),
                caption="Mean curve with one standard deviation band across runs.",
            )
        )
    return charts


def render_population_chart(aggregate: ConditionAggregate) -> EmbeddedChart:
    frame = aggregate.summary_frame
    figure, axis = plt.subplots(figsize=(12, 6))
    _plot_mean_std(axis, frame, "predator_count", "Predators", STYLE["predator_count"])
    _plot_mean_std(axis, frame, "prey_count", "Prey", STYLE["prey_count"])
    axis.set_xlabel("Tick")
    axis.set_ylabel("Population")
    axis.set_title(f"{aggregate.label} - Population (mean +/- std across {len(aggregate.runs)} runs)")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    return EmbeddedChart(
        title="Population",
        image_uri=_figure_to_data_uri(figure),
        caption=f"{aggregate.label} predator and prey population trends across {len(aggregate.runs)} runs.",
    )


def render_species_chart(run: RunData, label: str) -> EmbeddedChart:
    species_path = run.path / "species.csv"
    species_frame = load_optional_csv(species_path, required_columns=["tick", "population", "species_id", "size"])
    figure, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    if species_frame is None:
        for axis in axes:
            axis.text(0.5, 0.5, "species.csv not available", ha="center", va="center")
        for axis in axes:
            axis.set_axis_off()
    else:
        axes[0].plot(
            run.stats["tick"],
            run.stats["predator_species"],
            color=STYLE["predator_species"],
            linewidth=1.7,
            label="Predator Species",
        )
        axes[0].plot(
            run.stats["tick"],
            run.stats["prey_species"],
            color=STYLE["prey_species"],
            linewidth=1.7,
            label="Prey Species",
        )
        axes[0].set_ylabel("Species")
        axes[0].set_title(f"{label} - Species Count ({run.name})")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        predator_frame = species_frame[species_frame["population"] == "predator"]
        prey_frame = species_frame[species_frame["population"] == "prey"]

        predator_pivot = predator_frame.pivot_table(index="tick", columns="species_id", values="size", fill_value=0)
        if predator_pivot.empty:
            axes[1].text(0.5, 0.5, "No predator species data", ha="center", va="center")
            axes[1].set_axis_off()
        else:
            predator_colors = [plt.get_cmap("tab20")(index % 20) for index in range(len(predator_pivot.columns))]
            axes[1].stackplot(
                predator_pivot.index,
                predator_pivot.T.values,
                labels=[f"P{species_id}" for species_id in predator_pivot.columns],
                colors=predator_colors,
                alpha=0.85,
            )
            axes[1].set_ylabel("Predator Size")
            axes[1].set_title("Predator Species Distribution")
            axes[1].grid(True, alpha=0.3)
            if len(predator_pivot.columns) <= 15:
                axes[1].legend(loc="upper right", fontsize=8, ncol=3)

        prey_pivot = prey_frame.pivot_table(index="tick", columns="species_id", values="size", fill_value=0)
        if prey_pivot.empty:
            axes[2].text(0.5, 0.5, "No prey species data", ha="center", va="center")
            axes[2].set_axis_off()
        else:
            prey_colors = [plt.get_cmap("tab20b")(index % 20) for index in range(len(prey_pivot.columns))]
            axes[2].stackplot(
                prey_pivot.index,
                prey_pivot.T.values,
                labels=[f"Q{species_id}" for species_id in prey_pivot.columns],
                colors=prey_colors,
                alpha=0.85,
            )
            axes[2].set_xlabel("Tick")
            axes[2].set_ylabel("Prey Size")
            axes[2].set_title("Prey Species Distribution")
            axes[2].grid(True, alpha=0.3)
            if len(prey_pivot.columns) <= 15:
                axes[2].legend(loc="upper right", fontsize=8, ncol=3)

    return EmbeddedChart(
        title="Species",
        image_uri=_figure_to_data_uri(figure),
        caption=f"Representative run `{run.name}` species count and size distribution for {label}.",
    )


def render_complexity_chart(aggregate: ConditionAggregate) -> EmbeddedChart:
    frame = aggregate.summary_frame
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    _plot_mean_std(axes[0], frame, "avg_complexity", "Average Complexity", STYLE["avg_complexity"])
    axes[0].set_ylabel("Connections")
    axes[0].set_title(f"{aggregate.label} - Complexity (mean +/- std across {len(aggregate.runs)} runs)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    genome_points = _load_genome_counts(aggregate.representative_run.path / "genomes.json")
    if genome_points is None:
        axes[1].text(0.5, 0.5, "genomes.json not available", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        axes[1].plot(
            genome_points["tick"],
            genome_points["num_nodes"],
            label="Nodes",
            color=STYLE["nodes"],
            linewidth=1.7,
        )
        axes[1].plot(
            genome_points["tick"],
            genome_points["num_connections"],
            label="Connections",
            color=STYLE["connections"],
            linewidth=1.7,
        )
        axes[1].set_xlabel("Tick")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Representative Genome Structure ({aggregate.representative_run.name})")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

    return EmbeddedChart(
        title="Complexity",
        image_uri=_figure_to_data_uri(figure),
        caption=f"{aggregate.label} aggregate complexity plus representative genome structure history.",
    )


def _plot_mean_std(axis: plt.Axes, frame: pd.DataFrame, metric: str, label: str, color: str) -> None:
    mean = frame[f"{metric}_mean"]
    std = frame[f"{metric}_std"]
    axis.plot(frame["tick"], mean, label=label, color=color, linewidth=1.7)
    axis.fill_between(frame["tick"], mean - std, mean + std, color=color, alpha=0.12)


def _load_genome_counts(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        genomes = json.load(handle)
    if not genomes:
        return None
    return pd.DataFrame(
        {
            "tick": [entry["tick"] for entry in genomes],
            "num_nodes": [entry["num_nodes"] for entry in genomes],
            "num_connections": [entry["num_connections"] for entry in genomes],
        }
    )


def _figure_to_data_uri(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
