#!/usr/bin/env python3
"""Report generation library for MoonAI analysis.

Entry point: analysis/cli.py report [options]
"""

import matplotlib
matplotlib.use('Agg')

import statistics
import sys
from pathlib import Path

# Scripts live flat in analysis/; Python adds the script's directory to sys.path
# automatically when run as `python3 analysis/report.py`, so these imports work.
from utils import find_runs, load_config, load_stats, condition_label, CONDITIONS
from plot_fitness import plot as plot_fitness
from plot_population import plot as plot_population
from compare_experiments import compare

METRICS = ["best_fitness", "avg_fitness", "num_species", "avg_complexity"]


def generate_plots(runs: list[Path], plots_dir: Path) -> None:
    """Generate per-condition plots and cross-condition comparisons."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by condition
    groups: dict[str, list[Path]] = {}
    for run in runs:
        try:
            cfg = load_config(run)
            label = condition_label(cfg)
        except Exception as e:
            print(f"  Warning: could not load config for {run.name}: {e}", file=sys.stderr)
            continue
        groups.setdefault(label, []).append(run)

    print(f"Grouped {len(runs)} runs into {len(groups)} conditions.")

    # Per-condition: single representative run (first seed alphabetically)
    for label, group in sorted(groups.items()):
        rep = group[0]
        for plot_fn, suffix in [(plot_fitness, "fitness"), (plot_population, "population")]:
            out = plots_dir / f"{label}_{suffix}.png"
            try:
                plot_fn(rep, output=str(out))
            except Exception as e:
                print(f"  Warning: {suffix} plot failed for {label}: {e}", file=sys.stderr)

    # Cross-condition comparisons (one overlay plot per metric, all conditions)
    for metric in METRICS:
        out = plots_dir / f"compare_{metric}.png"
        # Use one representative run per condition for cleaner comparison
        rep_runs = [groups[label][0] for label in sorted(groups)]
        rep_labels = sorted(groups.keys())
        try:
            compare(rep_runs, metric=metric, output=str(out), labels=rep_labels)
        except Exception as e:
            print(f"  Warning: comparison plot failed for {metric}: {e}", file=sys.stderr)


def build_summary(runs: list[Path], generation: int) -> dict[str, list[dict]]:
    """Load per-run stats and group by condition."""
    groups: dict[str, list[dict]] = {}
    for run in runs:
        try:
            cfg = load_config(run)
            df = load_stats(run)
        except Exception as e:
            print(f"  Skipping {run.name}: {e}", file=sys.stderr)
            continue

        label = condition_label(cfg)
        row_df = df[df["generation"] == generation]
        row = row_df.iloc[-1] if not row_df.empty else df.iloc[-1]

        entry = {m: float(row[m]) for m in METRICS if m in row.index}
        groups.setdefault(label, []).append(entry)
    return groups


def print_table(groups: dict[str, list[dict]], generation: int) -> str:
    """Print and return a markdown summary table."""
    col_w = 22
    headers = ["condition"] + METRICS
    header = "| " + " | ".join(h.ljust(col_w) for h in headers) + " |"
    sep    = "| " + " | ".join("-" * col_w for _ in headers) + " |"

    lines = [
        f"\nResults at generation {generation} (mean ± std across seeds)\n",
        header,
        sep,
    ]

    for label in sorted(groups.keys()):
        entries = groups[label]
        row_parts = [label.ljust(col_w)]
        for metric in METRICS:
            vals = [e[metric] for e in entries if metric in e]
            if not vals:
                row_parts.append("N/A".ljust(col_w))
                continue
            mean = sum(vals) / len(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cell = f"{mean:.3f} ± {std:.3f}"
            row_parts.append(cell.ljust(col_w))
        lines.append("| " + " | ".join(row_parts) + " |")

    lines.append("")
    return "\n".join(lines)


def run_report(output_dir: Path, plots_dir: Path, min_generations: int, generation: int) -> int:
    """Generate the complete report: plots + summary table. Returns exit code."""
    runs = find_runs(output_dir, min_generations=min_generations)
    if not runs:
        print(f"No qualifying runs found in {output_dir} "
              f"(min_generations={min_generations})", file=sys.stderr)
        return 1

    print(f"Found {len(runs)} qualifying runs in {output_dir}/")

    generate_plots(runs, plots_dir)

    groups = build_summary(runs, generation=generation)
    table = print_table(groups, generation=generation)
    print(table)

    summary_path = plots_dir / "summary.md"
    summary_path.write_text(table)
    print(f"Summary written to {summary_path}")

    return 0
