"""Profiler report generation."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .html_report import render_html_report
from .io import AveragedNode, ProfileSuite, load_suites


@dataclass(frozen=True)
class Chart:
    title: str
    image_uri: str
    caption: str


def _sort_nodes_by_time(nodes: list[AveragedNode]) -> list[AveragedNode]:
    """Sort nodes by total time descending."""
    return sorted(nodes, key=lambda c: c.total_ms, reverse=True)


def generate_report(input_dir: Path, output_dir: Path) -> None:
    """Generate HTML report from profiler suites."""
    suites = load_suites(input_dir)
    if not suites:
        raise SystemExit(f"No profile suites found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now()

    comparison = [
        _chart_frame_comparison(suites),
        _chart_timelines(suites),
    ]

    # Generate per-suite sections
    sections = [_build_section(s) for s in reversed(suites)]

    # Build and write report
    report = render_html_report(
        {
            "generated_at": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "report_name": f"profile_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html",
            "input_dir": str(input_dir),
            "run_count": len(suites),
            "summary_rows": _build_summary_rows(suites),
            "comparison_charts": [c.__dict__ for c in comparison],
            "run_sections": sections,
        }
    )

    report_path = (
        output_dir / f"profile_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    )
    report_path.write_text(report, encoding="utf-8")

    # Also write a copy as profile.html for easy browser refresh
    stable_path = output_dir / "profile.html"
    copy2(report_path, stable_path)

    print(f"Analysed {len(suites)} profiler suites.")
    print(f"Wrote report to {report_path}")
    print(f"Also available at {stable_path} (always latest)")


def _build_summary_rows(suites: list[ProfileSuite]) -> list[dict]:
    """Build summary rows with change percentage."""
    rows = []
    prev_avg = None

    for s in suites:
        avg_frame_ms = s.avg_frame_ms

        if prev_avg is None:
            change_pct = "N/A"
            change_class = ""
        else:
            change = ((avg_frame_ms - prev_avg) / prev_avg) * 100
            change_pct = f"{change:+.1f}%"
            change_class = (
                "positive" if change > 0 else "negative" if change < 0 else ""
            )

        rows.append(
            {
                "name": s.name,
                "total_frames": s.frames,
                "kept_runs": len([m for m in s.members if m.disposition == "kept"]),
                "avg_frame_ms": f"{avg_frame_ms:.2f}",
                "change_pct": change_pct,
                "change_class": change_class,
            }
        )

        prev_avg = avg_frame_ms

    return rows


def _build_section(suite: ProfileSuite) -> dict:
    """Build a report section for a single suite."""
    # Generate charts
    member_chart = _chart_member_means(suite)
    flame_graph = _chart_flame_graph(suite) if suite.tree else None

    return {
        "name": suite.name,
        "total_frame_count": suite.frames,
        "kept_run_count": len([m for m in suite.members if m.disposition == "kept"]),
        "path": str(suite.path),
        "members": [
            {
                "seed": str(m.seed),
                "avg_frame_ms": f"{m.avg_frame_ms:.2f}",
                "disposition": m.disposition,
            }
            for m in suite.members
        ],
        "member_chart": member_chart.__dict__,
        "events": _format_tree_events(suite.tree, suite.frames) if suite.tree else [],
        "flame_graph": flame_graph.__dict__ if flame_graph else None,
        "metadata": {
            "suite_name": suite.metadata.get("suite_name", "N/A"),
            "frame_count": suite.metadata.get("frame_count", "N/A"),
            "report_interval_steps": suite.metadata.get("report_interval_steps", "N/A"),
            "gpu_allowed": suite.metadata.get("gpu_allowed", False),
            "platform": suite.metadata.get("platform", "N/A"),
            "cuda_compiled": suite.metadata.get("cuda_compiled", False),
            "openmp_compiled": suite.metadata.get("openmp_compiled", False),
            "generated_at_utc": suite.metadata.get("generated_at_utc", "N/A"),
        },
    }


def _format_tree_events(tree: AveragedNode | None, frame_count: int) -> list[dict]:
    """Format tree events with indentation to show hierarchy."""
    if tree is None:
        return []

    # Use tree.count for total frames across all kept runs (not per-run frame_count)
    total_frames = tree.count
    rows = []

    def traverse_node(node: AveragedNode, depth: int, parent_total_ms: float) -> None:
        # Calculate percentage based on parent time (root shows 100%)
        pct = (node.total_ms / parent_total_ms * 100) if parent_total_ms > 0 else 0.0

        # Calculate average per frame (consistent with percentage column)
        # This shows the average time contribution per frame
        avg_per_frame = node.total_ms / total_frames if total_frames > 0 else 0.0

        # Add 4-space indentation (rendered via CSS white-space: pre)
        indented_name = " " * (4 * depth) + node.name
        rows.append(
            {
                "name": indented_name,
                "raw_name": node.name,
                "depth": depth,
                "percentage": f"{pct:.1f}",
                "avg_ms": f"{avg_per_frame:.3f}",
                "count": str(node.count),
                "total_ms": f"{node.total_ms:.3f}",
            }
        )

        # Recursively process children (sorted by inclusive time descending)
        sorted_children = _sort_nodes_by_time(node.children)
        for child in sorted_children:
            traverse_node(child, depth + 1, node.total_ms)

    # Root node uses its own total as the baseline (shows 100%)
    traverse_node(tree, 0, tree.total_ms)
    return rows


# Chart generation


def _chart_frame_comparison(suites: list[ProfileSuite]) -> Chart:
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    labels = [s.name for s in suites]
    values = [s.avg_frame_ms for s in suites]

    # Bar chart on primary y-axis (left)
    bars = ax1.bar(labels, values, color="#355070", label="Avg Frame Time")
    ax1.set_ylabel("Average frame time (ms)", color="#355070")
    ax1.tick_params(axis="y", labelcolor="#355070")
    ax1.set_title("Trimmed Suite Frame Time")
    ax1.tick_params(axis="x", rotation=35, labelsize=7)
    ax1.grid(axis="y", alpha=0.25)

    # Standard deviation line on secondary y-axis (right)
    ax2 = ax1.twinx()
    stddev_values = [s.stddev for s in suites]
    ax2.plot(
        labels,
        stddev_values,
        color="#c97b63",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Std Dev",
    )
    ax2.set_ylabel("Standard deviation (ms)", color="#c97b63")
    ax2.tick_params(axis="y", labelcolor="#c97b63")

    fig.tight_layout()
    return Chart(
        title="Average Frame Time",
        image_uri=_to_data_uri(fig),
        caption="Bars: Mean of middle four runs. Line: Standard deviation (lower is more stable).",
    )


def _chart_timelines(suites: list[ProfileSuite]) -> Chart:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    has_data = False
    for suite in suites:
        timeline = suite.frame_timeline_ms
        if not timeline:
            continue
        has_data = True
        ax.plot(
            list(range(1, len(timeline) + 1)),
            timeline,
            label=suite.name,
            color="#355070",
            alpha=0.95,
            linewidth=1.8,
        )

    if has_data:
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame time (ms)")
    ax.set_title("Per-Suite Frame Time Lines")
    ax.set_xlim(left=0)
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Per-Suite Frame Time Lines",
        image_uri=_to_data_uri(fig),
        caption="Per-frame mean across suite seeds, using kept runs when available.",
    )


def _chart_member_means(suite: ProfileSuite) -> Chart:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"seed {m.seed}" for m in suite.members]
    values = [m.avg_frame_ms for m in suite.members]
    colors = [
        "#355070" if m.disposition == "kept" else "#c97b63" for m in suite.members
    ]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average frame time (ms)")
    ax.set_title(f"Suite Member Means")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return Chart(
        title="Member Run Means",
        image_uri=_to_data_uri(fig),
        caption="Orange bars are dropped runs. Blue bars are kept runs.",
    )


def _chart_flame_graph(suite: ProfileSuite) -> Chart | None:
    """Generate a flame graph visualization of the call tree."""
    if not suite.tree:
        return None

    # Use total frames across all kept runs (not per-run)
    total_frames = suite.tree.count

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color palette for different event types
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_map = {}
    color_idx = 0

    def get_color(name: str) -> tuple:
        """Get consistent color for an event name."""
        nonlocal color_idx
        if name not in color_map:
            color_map[name] = colors[color_idx % len(colors)]
            color_idx += 1
        return color_map[name]

    def draw_flame_node(
        node: AveragedNode, x: float, y: float, width: float, height: float
    ):
        """Recursively draw flame graph rectangles."""
        if width <= 0:
            return

        # Draw this node
        color = get_color(node.name)
        rect = mpatches.Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(rect)

        # Add text label if rectangle is wide enough
        if width > 0.05 and node.name != "frame_total":
            # Calculate per-frame average (consistent with table)
            avg_per_frame = node.total_ms / total_frames if total_frames > 0 else 0.0
            text_color = "white" if np.mean(color[:3]) < 0.5 else "black"
            ax.text(
                x + width / 2,
                y + height / 2,
                f"{node.name}\n{avg_per_frame:.2f}ms",
                ha="center",
                va="center",
                fontsize=min(10, max(6, int(width * 100))),
                color=text_color,
                weight="bold" if y > 0.5 else "normal",
            )

        # Draw children
        if node.children:
            # Sort children by inclusive time (descending)
            sorted_children = _sort_nodes_by_time(node.children)

            child_x = x

            for child in sorted_children:
                child_width = (
                    width * (child.total_ms / node.total_ms) if node.total_ms > 0 else 0
                )
                draw_flame_node(child, child_x, y + height, child_width, height)
                child_x += child_width

    # Calculate dimensions
    total_time = suite.tree.total_ms
    height_per_level = 0.15
    max_depth = _get_max_depth(suite.tree)

    # Draw the flame graph (root at bottom, children stack upward)
    draw_flame_node(suite.tree, 0, 0, 1.0, height_per_level)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, (max_depth + 1) * height_per_level)
    ax.set_xlabel("Relative time (width = inclusive time proportion)")
    ax.set_ylabel("Call depth")
    ax.set_title(f"Flame Graph - {suite.name}")
    ax.set_yticks(
        [i * height_per_level + height_per_level / 2 for i in range(max_depth + 1)]
    )
    ax.set_yticklabels([f"L{i}" for i in range(0, max_depth + 1)])
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return Chart(
        title="Flame Graph",
        image_uri=_to_data_uri(fig),
        caption="Call hierarchy visualization. Width = inclusive time proportion, height = call depth. Hover over blocks for details.",
    )


def _get_max_depth(node: AveragedNode, current_depth: int = 0) -> int:
    """Get maximum depth of the tree."""
    if not node.children:
        return current_depth
    return max(_get_max_depth(child, current_depth + 1) for child in node.children)


def _to_data_uri(fig) -> str:
    """Convert matplotlib figure to base64 data URI."""
    buffer = BytesIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    return f"data:image/svg+xml;base64,{b64encode(buffer.getvalue()).decode('ascii')}"
