"""Command-line entry point for MoonAI analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "output" / "experiments"
    default_output = project_root / "output" / "analysis"

    parser = argparse.ArgumentParser(
        prog="moonai-analysis",
        description="Generate a self-contained MoonAI HTML analysis report.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input),
        metavar="DIR",
        help=f"Simulation output directory (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        metavar="DIR",
        help=f"Analysis output directory (default: {default_output})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
