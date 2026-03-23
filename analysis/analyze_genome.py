#!/usr/bin/env python3
"""Genome network topology visualization library for MoonAI analysis.

Entry point: analysis/cli.py genome <run_dir> [options]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


NODE_TYPE_NAMES = {0: "Input", 1: "Hidden", 2: "Output", 3: "Bias"}
NODE_COLORS = {0: "#42A5F5", 1: "#FFA726", 2: "#EF5350", 3: "#AB47BC"}


def load_genome(genomes_path: str, generation: int = -1) -> dict:
    """Load a specific genome from genomes.json. Default: last generation."""
    with open(genomes_path) as f:
        genomes = json.load(f)

    if not genomes:
        print("Error: no genomes found", file=sys.stderr)
        sys.exit(1)

    if generation < 0:
        return genomes[-1]

    for g in genomes:
        if g["generation"] == generation:
            return g

    print(f"Error: generation {generation} not found", file=sys.stderr)
    sys.exit(1)


def visualize_genome(genome: dict, output: str = None):
    """Draw the neural network topology."""
    if not HAS_NETWORKX:
        print("Error: networkx is required. Install with: pip install networkx",
              file=sys.stderr)
        sys.exit(1)

    G = nx.DiGraph()

    # Add nodes
    node_types = {}
    for node in genome["nodes"]:
        nid = node["id"]
        ntype = node["type"]
        node_types[nid] = ntype
        G.add_node(nid, type=ntype)

    # Add edges
    for conn in genome["connections"]:
        if conn["enabled"]:
            G.add_edge(conn["in"], conn["out"], weight=conn["weight"])

    # Layout: arrange by layer
    inputs = [n for n, t in node_types.items() if t == 0]
    bias = [n for n, t in node_types.items() if t == 3]
    outputs = [n for n, t in node_types.items() if t == 2]
    hidden = [n for n, t in node_types.items() if t == 1]

    pos = {}
    # Inputs on the left
    for i, n in enumerate(sorted(inputs)):
        pos[n] = (0, -i)
    # Bias below inputs
    for i, n in enumerate(bias):
        pos[n] = (0, -(len(inputs) + i))
    # Outputs on the right
    for i, n in enumerate(sorted(outputs)):
        pos[n] = (2, -(len(inputs) - len(outputs)) / 2 - i)
    # Hidden in the middle
    for i, n in enumerate(sorted(hidden)):
        rows = max(len(hidden), 1)
        pos[n] = (1, -(rows - 1) / 2 + i * (rows - 1) / max(rows - 1, 1))

    fig, ax = plt.subplots(figsize=(14, 8))

    # Draw edges with weight-based colors
    edges = G.edges(data=True)
    if edges:
        weights = [d["weight"] for _, _, d in edges]
        max_w = max(abs(w) for w in weights) if weights else 1.0

        for u, v, d in edges:
            w = d["weight"]
            color = "#2196F3" if w > 0 else "#F44336"
            alpha = min(abs(w) / max_w * 0.8 + 0.2, 1.0)
            width = abs(w) / max_w * 2.5 + 0.3
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                   edge_color=color, alpha=alpha,
                                   width=width, ax=ax,
                                   connectionstyle="arc3,rad=0.1",
                                   arrows=True, arrowsize=12)

    # Draw nodes
    for ntype, color in NODE_COLORS.items():
        nodelist = [n for n, t in node_types.items() if t == ntype]
        if nodelist:
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                                   node_color=color, node_size=400,
                                   edgecolors="white", linewidths=1.5, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=10, label=NODE_TYPE_NAMES[t])
        for t, c in NODE_COLORS.items()
        if any(nt == t for nt in node_types.values())
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    gen = genome.get("generation", "?")
    fitness = genome.get("fitness", 0)
    ax.set_title(f"MoonAI - Neural Network Topology "
                 f"(Gen {gen}, Fitness: {fitness:.3f}, "
                 f"Nodes: {genome['num_nodes']}, "
                 f"Connections: {genome['num_connections']})")
    ax.axis("off")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
    else:
        plt.show()


