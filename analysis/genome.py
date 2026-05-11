"""Genome loading and visualization."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .io import load_json

NODE_TYPE_NAMES = {0: "Input", 1: "Hidden", 2: "Output", 3: "Bias"}
NODE_COLORS = {0: "#3B82F6", 1: "#F59E0B", 2: "#EF4444", 3: "#8B5CF6"}


def load_latest_genome(run_dir: Path) -> dict | None:
    genomes_path = run_dir / "genomes.json"
    if not genomes_path.is_file():
        return None

    genomes = load_json(genomes_path)
    if not genomes:
        return None
    return genomes[-1]


def render_genome_plot(genome: dict, title: str) -> str:
    if "genome" in genome and isinstance(genome["genome"], dict):
        genome = genome["genome"]

    graph = nx.DiGraph()
    node_types: dict[int, int] = {}

    for node in genome.get("nodes", []):
        node_id = node["id"]
        node_type = node["type"]
        node_types[node_id] = node_type
        graph.add_node(node_id, type=node_type)

    for connection in genome.get("connections", []):
        if connection.get("enabled", False):
            graph.add_edge(connection["in"], connection["out"], weight=connection["weight"])

    positions = _build_layout(node_types)
    figure, axis = plt.subplots(figsize=(14, 8))

    edges = list(graph.edges(data=True))
    if edges:
        max_weight = max(abs(data["weight"]) for _, _, data in edges) or 1.0
        for source, target, data in edges:
            weight = data["weight"]
            color = "#2563EB" if weight > 0 else "#DC2626"
            alpha = min(abs(weight) / max_weight * 0.8 + 0.2, 1.0)
            width = abs(weight) / max_weight * 2.5 + 0.3
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target)],
                edge_color=color,
                alpha=alpha,
                width=width,
                ax=axis,
                connectionstyle="arc3,rad=0.1",
                arrows=True,
                arrowsize=12,
            )

    for node_type, color in NODE_COLORS.items():
        nodes = [node_id for node_id, current_type in node_types.items() if current_type == node_type]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=nodes,
                node_color=color,
                node_size=420,
                edgecolors="white",
                linewidths=1.5,
                ax=axis,
            )

    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="white", ax=axis)

    legend = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=NODE_TYPE_NAMES[node_type],
        )
        for node_type, color in NODE_COLORS.items()
        if any(current_type == node_type for current_type in node_types.values())
    ]
    if legend:
        axis.legend(handles=legend, loc="upper right")

    axis.set_title(title)
    axis.axis("off")
    buffer = io.BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _build_layout(node_types: dict[int, int]) -> dict[int, tuple[float, float]]:
    inputs = [node_id for node_id, node_type in node_types.items() if node_type == 0]
    bias = [node_id for node_id, node_type in node_types.items() if node_type == 3]
    hidden = [node_id for node_id, node_type in node_types.items() if node_type == 1]
    outputs = [node_id for node_id, node_type in node_types.items() if node_type == 2]

    positions: dict[int, tuple[float, float]] = {}
    for index, node_id in enumerate(sorted(inputs)):
        positions[node_id] = (0.0, -float(index))
    for index, node_id in enumerate(sorted(bias)):
        positions[node_id] = (0.0, -float(len(inputs) + index))
    for index, node_id in enumerate(sorted(outputs)):
        positions[node_id] = (2.0, -(len(outputs) - 1) / 2.0 + index)
    rows = max(len(hidden), 1)
    for index, node_id in enumerate(sorted(hidden)):
        positions[node_id] = (1.0, -(rows - 1) / 2.0 + index)
    return positions
