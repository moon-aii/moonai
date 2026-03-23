#!/usr/bin/env python3
"""MoonAI analysis CLI — single entry point for all post-run analysis.

Usage:
    uv run python analysis/cli.py <command> [options]

Commands:
    plot        Fitness and complexity curves for one run
    population  Predator/prey counts over generations
    species     Species count and size distribution
    complexity  Genome node/connection count over generations
    compare     Overlay one metric across multiple runs
    genome      Neural network topology of the best genome
    report      Generate complete report: all plots + summary table
"""

import sys
import argparse
from pathlib import Path

# Make sibling modules importable without package installation
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')

from plot_fitness        import plot as _plot_fitness
from plot_population     import plot as _plot_population
from plot_species        import plot as _plot_species
from plot_complexity     import plot as _plot_complexity
from compare_experiments import compare as _compare, VALID_METRICS
from analyze_genome      import load_genome, visualize_genome
import report as _report

_PROJECT_ROOT  = Path(__file__).parent.parent
_DEFAULT_OUT   = _PROJECT_ROOT / "output"
_DEFAULT_PLOTS = _PROJECT_ROOT / "output" / "plots"


# ── Command handlers ──────────────────────────────────────────────────────────

def cmd_plot(args):
    result = _plot_fitness(args.run_dir, output=args.output, smooth=args.smooth)
    if result is None and args.output:
        sys.exit(1)


def cmd_population(args):
    result = _plot_population(args.run_dir, output=args.output)
    if result is None and args.output:
        sys.exit(1)


def cmd_species(args):
    result = _plot_species(args.run_dir, output=args.output)
    if result is None and args.output:
        sys.exit(1)


def cmd_complexity(args):
    result = _plot_complexity(args.run_dir, output=args.output)
    if result is None and args.output:
        sys.exit(1)


def cmd_compare(args):
    _compare(args.run_dirs, metric=args.metric, output=args.output, smooth=args.smooth)


def cmd_genome(args):
    run_dir = Path(args.run_dir)
    genomes_path = run_dir / "genomes.json"
    if not genomes_path.exists():
        print(f"Error: {genomes_path} not found", file=sys.stderr)
        sys.exit(1)
    genome = load_genome(str(genomes_path), args.generation)
    print(f"Genome from generation {genome.get('generation', '?')}:")
    print(f"  Fitness:     {genome.get('fitness', 0):.3f}")
    print(f"  Nodes:       {genome['num_nodes']}")
    print(f"  Connections: {genome['num_connections']}")
    visualize_genome(genome, args.output)


def cmd_report(args):
    rc = _report.run_report(
        Path(args.output_dir),
        Path(args.plots_dir),
        args.min_generations,
        args.generation,
    )
    sys.exit(rc)


# ── Argument parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="moonai-analysis",
        description="MoonAI post-run analysis tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<command>")

    # ── plot ─────────────────────────────────────────────────────────────
    p = sub.add_parser("plot", help="Fitness and complexity curves for one run")
    p.add_argument("run_dir", help="Path to run output directory")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.add_argument("--smooth", type=int, default=1, metavar="N",
                   help="Rolling average window (default: 1 = no smoothing)")
    p.set_defaults(func=cmd_plot)

    # ── population ───────────────────────────────────────────────────────
    p = sub.add_parser("population", help="Predator/prey counts over generations")
    p.add_argument("run_dir", help="Path to run output directory")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.set_defaults(func=cmd_population)

    # ── species ──────────────────────────────────────────────────────────
    p = sub.add_parser("species", help="Species count and size distribution")
    p.add_argument("run_dir", help="Path to run output directory")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.set_defaults(func=cmd_species)

    # ── complexity ───────────────────────────────────────────────────────
    p = sub.add_parser("complexity", help="Genome node/connection count over generations")
    p.add_argument("run_dir", help="Path to run output directory")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.set_defaults(func=cmd_complexity)

    # ── compare ──────────────────────────────────────────────────────────
    p = sub.add_parser("compare", help="Overlay one metric across multiple runs")
    p.add_argument("run_dirs", nargs="+", metavar="run_dir",
                   help="Paths to run output directories")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.add_argument("--metric", default="best_fitness", choices=VALID_METRICS,
                   help="Column to plot (default: best_fitness)")
    p.add_argument("--smooth", type=int, default=1, metavar="N",
                   help="Rolling average window (default: 1 = no smoothing)")
    p.set_defaults(func=cmd_compare)

    # ── genome ───────────────────────────────────────────────────────────
    p = sub.add_parser("genome", help="Neural network topology of the best genome")
    p.add_argument("run_dir", help="Path to run output directory")
    p.add_argument("-g", "--generation", type=int, default=-1,
                   help="Generation to visualize (default: -1 = last)")
    p.add_argument("-o", "--output", help="Save path (PNG); omit to show interactively")
    p.set_defaults(func=cmd_genome)

    # ── report ───────────────────────────────────────────────────────────
    p = sub.add_parser("report", help="Generate complete report: all plots + summary table")
    p.add_argument("--output-dir", default=str(_DEFAULT_OUT), metavar="DIR",
                   help=f"Run output directory (default: {_DEFAULT_OUT})")
    p.add_argument("--plots-dir", default=str(_DEFAULT_PLOTS), metavar="DIR",
                   help=f"Directory to write plots into (default: {_DEFAULT_PLOTS})")
    p.add_argument("--min-generations", type=int, default=190, metavar="N",
                   help="Minimum generation rows to include a run (default: 190)")
    p.add_argument("--generation", type=int, default=200, metavar="N",
                   help="Generation to sample for summary table (default: 200)")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
