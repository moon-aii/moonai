#!/usr/bin/env python3
"""Genome complexity plotting library for MoonAI analysis.

Entry point: analysis/cli.py complexity <run_dir> [options]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt


def plot(run_dir, output=None):
    """Generate network complexity plot for one run.

    Args:
        run_dir: Path to run output directory.
        output:  Save path for PNG. If None, calls plt.show().

    Returns:
        Path to saved file if output was specified, else None.
    """
    run_dir = Path(run_dir)
    stats_path = run_dir / "stats.csv"
    genomes_path = run_dir / "genomes.json"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    if stats_path.exists():
        df = pd.read_csv(stats_path, comment="#")
        ax1.plot(df["generation"], df["avg_complexity"],
                 label="Avg Connections", color="#FF7043", linewidth=1.5)
        ax1.set_ylabel("Average Connections per Genome")
        ax1.set_title("MoonAI - Network Complexity Over Generations")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    if genomes_path.exists():
        with open(genomes_path) as f:
            genomes = json.load(f)

        gens  = [g["generation"] for g in genomes]
        nodes = [g["num_nodes"] for g in genomes]
        conns = [g["num_connections"] for g in genomes]

        ax2.plot(gens, nodes, label="Nodes (best genome)",
                 color="#42A5F5", linewidth=1.5)
        ax2.plot(gens, conns, label="Connections (best genome)",
                 color="#EF5350", linewidth=1.5)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Count")
        ax2.set_title("Best Genome Structure")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
        plt.close(fig)
        return Path(output)
    else:
        plt.show()
        return None


