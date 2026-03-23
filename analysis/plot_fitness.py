#!/usr/bin/env python3
"""Fitness curve plotting library for MoonAI analysis.

Entry point: analysis/cli.py plot <run_dir> [options]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt


def plot(run_dir, output=None, smooth=1):
    """Generate fitness + complexity plot for one run directory.

    Args:
        run_dir: Path to run output directory (or stats.csv directly).
        output:  Save path for PNG. If None, calls plt.show().
        smooth:  Rolling average window size (1 = no smoothing).

    Returns:
        Path to saved file if output was specified, else None.
    """
    path = Path(run_dir)
    csv_path = path / "stats.csv" if path.is_dir() else path
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return None

    df = pd.read_csv(csv_path, comment="#")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    gen = df["generation"]
    best = df["best_fitness"]
    avg = df["avg_fitness"]

    if smooth > 1:
        best = best.rolling(smooth, min_periods=1).mean()
        avg = avg.rolling(smooth, min_periods=1).mean()

    ax.plot(gen, best, label="Best Fitness", color="#2196F3", linewidth=1.5)
    ax.plot(gen, avg, label="Avg Fitness", color="#FF9800", linewidth=1.5, alpha=0.8)
    ax.fill_between(gen, avg, best, alpha=0.1, color="#2196F3")
    ax.set_ylabel("Fitness")
    ax.set_title("MoonAI - Fitness Over Generations")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(gen, df["avg_complexity"], label="Avg Complexity",
             color="#9C27B0", linewidth=1.5)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Genome Complexity (connections)")
    ax2.set_title("Network Complexity Over Generations")
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


