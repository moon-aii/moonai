#!/usr/bin/env python3
"""Multi-run comparison plotting library for MoonAI analysis.

Entry point: analysis/cli.py compare <run_dirs...> [options]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

VALID_METRICS = [
    "best_fitness", "avg_fitness", "num_species",
    "avg_complexity", "predator_count", "prey_count",
]


def compare(run_dirs, metric="best_fitness", output=None, smooth=1, labels=None):
    """Generate a metric comparison overlay for multiple runs.

    Args:
        run_dirs: List of paths to run output directories (or stats.csv files).
        metric:   Column name to plot (default: best_fitness).
        output:   Save path for PNG. If None, calls plt.show().
        smooth:   Rolling average window size (1 = no smoothing).
        labels:   Optional list of legend labels (defaults to directory names).

    Returns:
        Path to saved file if output was specified, else None.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, run_dir in enumerate(run_dirs):
        path = Path(run_dir)
        csv_path = path / "stats.csv" if path.is_dir() else path

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path, comment="#")
        label = (labels[i] if labels and i < len(labels)
                 else (path.name if path.is_dir() else path.stem))

        values = df[metric]
        if smooth > 1:
            values = values.rolling(smooth, min_periods=1).mean()

        ax.plot(df["generation"], values, label=label, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Generation")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"MoonAI - {metric.replace('_', ' ').title()} Comparison")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output}")
        plt.close(fig)
        return Path(output)
    else:
        plt.show()
        return None


