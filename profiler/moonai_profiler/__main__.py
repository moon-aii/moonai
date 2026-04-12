"""Command-line entry point for MoonAI profiler analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .report import generate_report


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    default_input = project_root / "output" / "profiles"
    default_output = project_root / "profiler" / "output"

    parser = argparse.ArgumentParser(
        prog="moonai-profiler",
        description="Generate a self-contained MoonAI profiler HTML report.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input),
        metavar="DIR",
        help=f"Profiler output directory (default: {default_input})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        metavar="DIR",
        help=f"Profiler report output directory (default: {default_output})",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate_report(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
