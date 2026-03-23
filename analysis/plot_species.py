#!/usr/bin/env python3
"""Species dynamics plotting library for MoonAI analysis.

Entry point: analysis/cli.py species <run_dir> [options]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot(run_dir, output=None):
    """Generate species count + distribution plot for one run.

    Args:
        run_dir: Path to run output directory (must contain species.csv).
        output:  Save path for PNG. If None, calls plt.show().

    Returns:
        Path to saved file if output was specified, else None.
    """
    run_dir = Path(run_dir)
    species_path = run_dir / "species.csv"
    stats_path = run_dir / "stats.csv"

    if not species_path.exists():
        print(f"Error: {species_path} not found", file=sys.stderr)
        return None

    sp_df = pd.read_csv(species_path, comment="#")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path, comment="#")
        ax1.plot(stats_df["generation"], stats_df["num_species"],
                 color="#7B1FA2", linewidth=1.5)
        ax1.fill_between(stats_df["generation"], 0, stats_df["num_species"],
                         alpha=0.15, color="#7B1FA2")
    else:
        species_counts = sp_df.groupby("generation")["species_id"].nunique()
        ax1.plot(species_counts.index, species_counts.values,
                 color="#7B1FA2", linewidth=1.5)

    ax1.set_ylabel("Number of Species")
    ax1.set_title("MoonAI - Species Count Over Generations")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    pivot = sp_df.pivot_table(index="generation", columns="species_id",
                               values="size", fill_value=0)
    colors = cm.Set3(np.linspace(0, 1, len(pivot.columns)))
    ax2.stackplot(pivot.index, pivot.T.values, labels=[f"S{s}" for s in pivot.columns],
                  colors=colors, alpha=0.8)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Species Size")
    ax2.set_title("Species Size Distribution")
    if len(pivot.columns) <= 15:
        ax2.legend(loc="upper right", fontsize=8, ncol=3)
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


