#!/usr/bin/env python3
"""Overlay Part 2 datasets on one student panel and one teacher panel.

Each dataset is one color. A dotted line connects every evaluation. A solid
line with a point at each focused evaluation connects the cycles that trained
that dataset. One figure is written per run and task.

Reads the same export_csvFile.csv files as plot_focused_unfocused.py.
"""

import argparse
import importlib.util
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOADER_PATH = os.path.join(SCRIPT_DIR, "plot_focused_unfocused.py")
_spec = importlib.util.spec_from_file_location("plot_focused_unfocused", _LOADER_PATH)
curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curves)


DATASET_ORDER = [
    "CANDID-PTX",
    "RSNApneumonia",
    "ChestX-Det",
    "SIIM-ACRpneumothorax",
]
DATASET_COLOR = {
    "CANDID-PTX": "#4C78A8",
    "RSNApneumonia": "#54A24B",
    "ChestX-Det": "#F58518",
    "SIIM-ACRpneumothorax": "#B279A2",
}
RUN_TITLE = {
    "cls": "Classification only",
    "loc": "Localization only",
    "seg": "Segmentation only",
    "cls_loc": "Classification + localization",
    "cls_loc_seg": "Classification + localization + segmentation",
}
TASK_ORDER = ("classification", "localization", "segmentation")
FOCUSED_STYLE = "-"
UNFOCUSED_STYLE = (0, (1.2, 1.6))


def dataset_sort_key(name):
    if name in DATASET_ORDER:
        return DATASET_ORDER.index(name)
    return len(DATASET_ORDER)


def draw_curve(ax, points, n_tasks, tested, color):
    """Dotted line through every point. Solid line and markers on focused points only."""
    ordered = sorted(points, key=lambda row: row["epoch"])
    if not ordered:
        return
    curves.plot_curve(ax, ordered, n_tasks, color, UNFOCUSED_STYLE, None, linewidth=1.5)
    focused = [row for row in ordered if row["training"] == tested]
    curves.plot_curve(
        ax,
        focused,
        n_tasks,
        color,
        FOCUSED_STYLE,
        None,
        marker="o",
        linewidth=2.3,
    )


def style_axis(ax, title, ylabel, x_max, y_limits, show_ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("Cycle")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(1, x_max)
    ax.set_ylim(*y_limits)
    if x_max <= 16:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def y_limits(scores):
    finite = [score for score in scores if score is not None]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    span = max(high - low, 1.0)
    pad = 0.08 * span
    return low - pad, high + pad


def plot_overlay(rows, n_tasks, x_max, task_type, run_name, out_path):
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row_task, dataset = row["tested"]
        if row_task != task_type:
            continue
        by_dataset[dataset][row["model"]].append(row)
    datasets = sorted(by_dataset, key=dataset_sort_key)
    if len(datasets) < 2:
        return False

    scores = [
        row["score"]
        for dataset in datasets
        for model_rows in by_dataset[dataset].values()
        for row in model_rows
    ]
    limits = y_limits(scores)
    ylabel = curves.TASK_YLABEL[task_type]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, model, show_ylabel in zip(axes, ("student", "teacher"), (True, False)):
        for dataset in datasets:
            draw_curve(
                ax,
                by_dataset[dataset][model],
                n_tasks,
                (task_type, dataset),
                DATASET_COLOR.get(dataset, "#333333"),
            )
        style_axis(
            ax,
            model.title(),
            ylabel,
            x_max,
            limits,
            show_ylabel,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=DATASET_COLOR.get(dataset, "#333333"),
            linewidth=2.3,
            label=curves.DISPLAY_DATASET.get(dataset, dataset),
        )
        for dataset in datasets
    ]
    legend_handles.extend([
        plt.Line2D(
            [0], [0], color="black", linewidth=2.3, linestyle=FOCUSED_STYLE,
            marker="o", markersize=4.5, label="Focused",
        ),
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle=UNFOCUSED_STYLE, label="Unfocused"),
    ])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    title = RUN_TITLE.get(run_name, run_name)
    fig.suptitle("{} — {}".format(title, task_type), fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=curves.RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to <results-root>/analysis/part2_overlay.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Part 2 run name to plot. Repeat for several. Default: all five.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.results_root, "analysis", "part2_overlay")
    os.makedirs(output_dir, exist_ok=True)
    selected = set(args.run) if args.run else set(curves.PART2_RUNS)

    wrote = 0
    for part, run, run_dir in curves.runs(args.results_root):
        if part != "part_2" or run not in selected:
            continue
        csv_path = os.path.join(run_dir, "export_csvFile.csv")
        if not os.path.isfile(csv_path):
            print("Missing {}".format(csv_path))
            continue
        rows = curves.prefer_end_of_epoch(curves.read_rows(csv_path))
        if not rows:
            print("No test rows in {}".format(csv_path))
            continue
        n_tasks = curves.cycle_length(part, run, rows)
        x_max = curves.highest_cycle(rows, n_tasks) + 0.4
        present = {row["tested"][0] for row in rows}
        for task_type in TASK_ORDER:
            if task_type not in present:
                continue
            out_path = os.path.join(output_dir, "{}_{}.png".format(run, task_type))
            if plot_overlay(rows, n_tasks, x_max, task_type, run, out_path):
                wrote += 1
                print("Wrote {}".format(out_path))
    if wrote == 0:
        print("No part 2 overlay figures were written.")


if __name__ == "__main__":
    main()
