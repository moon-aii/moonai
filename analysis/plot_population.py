#!/usr/bin/env python3
"""Population dynamics plotting library for MoonAI analysis.

Entry point: analysis/cli.py population <run_dir> [options]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt


def plot(run_dir, output=None):
    """Generate predator/prey population dynamics plot for one run.

    Args:
        run_dir: Path to run output directory (or stats.csv directly).
        output:  Save path for PNG. If None, calls plt.show().

    Returns:
        Path to saved file if output was specified, else None.
    """
    path = Path(run_dir)
    csv_path = path / "stats.csv" if path.is_dir() else path
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        return None

    df = pd.read_csv(csv_path, comment="#")

    fig, ax = plt.subplots(figsize=(12, 6))

    gen = df["generation"]
    ax.plot(gen, df["predator_count"], label="Predators (alive at gen end)",
            color="#E53935", linewidth=1.5)
    ax.plot(gen, df["prey_count"], label="Prey (alive at gen end)",
            color="#43A047", linewidth=1.5)

    ax.fill_between(gen, 0, df["predator_count"], alpha=0.15, color="#E53935")
    ax.fill_between(gen, 0, df["prey_count"], alpha=0.15, color="#43A047")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Count")
    ax.set_title("MoonAI - Population Dynamics")
    ax.legend(loc="upper right")
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


