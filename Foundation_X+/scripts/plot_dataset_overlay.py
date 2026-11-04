#!/usr/bin/env python3
"""Overlay Part 2 datasets with student and teacher on one plot.

Each dataset is one color. The student is a solid line and the teacher is a
dashed line. A thinner line connects every evaluation. A thicker line of the
same style connects the cycles that trained that dataset. The highest student
and teacher score for each dataset is listed on the figure. One figure is
written per run and task.

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
STUDENT_STYLE = "-"
TEACHER_STYLE = (0, (5, 2))


def dataset_sort_key(name):
    if name in DATASET_ORDER:
        return DATASET_ORDER.index(name)
    return len(DATASET_ORDER)


def draw_curve(ax, points, n_tasks, tested, color, linestyle):
    """Thin line through every evaluation. Thicker line through focused evaluations."""
    ordered = sorted(points, key=lambda row: row["epoch"])
    if not ordered:
        return
    ax.plot(
        [curves.cycle_x(row["epoch"], n_tasks) for row in ordered],
        [row["score"] for row in ordered],
        color=color,
        linestyle=linestyle,
        linewidth=1.15,
        zorder=2,
    )
    focused = sorted(
        (row for row in ordered if row["training"] == tested),
        key=lambda row: row["epoch"],
    )
    if len(focused) < 2:
        return
    ax.plot(
        [curves.cycle_x(row["epoch"], n_tasks) for row in focused],
        [row["score"] for row in focused],
        color=color,
        linestyle=linestyle,
        linewidth=2.3,
        zorder=3,
    )


def best_point(points):
    if not points:
        return None
    return max(points, key=lambda row: (row["score"], -row["epoch"]))


def best_text(points, n_tasks):
    best = best_point(points)
    if best is None:
        return "n/a"
    return "{:.3f} at cycle {}".format(best["score"], curves.cycle_number(best["epoch"], n_tasks))


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


def datasets_in_train_order(by_dataset, task_type, order):
    """Datasets on this figure, in the order their task is trained."""
    seen = []
    for trained_type, dataset in order:
        if trained_type == task_type and dataset in by_dataset and dataset not in seen:
            seen.append(dataset)
    for dataset in sorted(by_dataset, key=dataset_sort_key):
        if dataset not in seen:
            seen.append(dataset)
    return seen


def train_position(task_type, dataset, order):
    for index, task in enumerate(order, start=1):
        if task == (task_type, dataset):
            return index
    return None


def plot_overlay(rows, n_tasks, x_max, task_type, run_name, order, out_path):
    by_dataset = defaultdict(lambda: defaultdict(list))
    for row in rows:
        row_task, dataset = row["tested"]
        if row_task != task_type:
            continue
        by_dataset[dataset][row["model"]].append(row)
    datasets = datasets_in_train_order(by_dataset, task_type, order)
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
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for dataset in datasets:
        color = DATASET_COLOR.get(dataset, "#333333")
        tested = (task_type, dataset)
        draw_curve(ax, by_dataset[dataset]["student"], n_tasks, tested, color, STUDENT_STYLE)
        draw_curve(ax, by_dataset[dataset]["teacher"], n_tasks, tested, color, TEACHER_STYLE)
    title = RUN_TITLE.get(run_name, run_name)
    style_axis(ax, "{} — {}".format(title, task_type), ylabel, x_max, limits, True)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=DATASET_COLOR.get(dataset, "#333333"),
            linewidth=2.3,
            label="{} {}".format(
                train_position(task_type, dataset, order) or "",
                curves.DISPLAY_DATASET.get(dataset, dataset),
            ).strip(),
        )
        for dataset in datasets
    ]
    legend_handles.extend([
        plt.Line2D([0], [0], color="black", linewidth=2.3, linestyle=STUDENT_STYLE, label="Student"),
        plt.Line2D([0], [0], color="black", linewidth=2.3, linestyle=TEACHER_STYLE, label="Teacher"),
        plt.Line2D([0], [0], color="black", linewidth=2.3, label="Focused"),
        plt.Line2D([0], [0], color="black", linewidth=1.15, label="All evaluations"),
    ])
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    highest_lines = []
    for dataset in datasets:
        name = curves.DISPLAY_DATASET.get(dataset, dataset)
        position = train_position(task_type, dataset, order) or ""
        highest_lines.append("{} {}    student {}    teacher {}".format(
            position,
            name,
            best_text(by_dataset[dataset]["student"], n_tasks),
            best_text(by_dataset[dataset]["teacher"], n_tasks),
        ).strip())
    fig.text(0.5, 0.01, "Highest\n" + "\n".join(highest_lines), ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.18 + 0.035 * len(datasets))
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
        order = curves.training_order(rows, n_tasks)
        x_max = curves.highest_cycle(rows, n_tasks) + 0.4
        present = {row["tested"][0] for row in rows}
        for task_type in TASK_ORDER:
            if task_type not in present:
                continue
            out_path = os.path.join(output_dir, "{}_{}.png".format(run, task_type))
            if plot_overlay(rows, n_tasks, x_max, task_type, run, order, out_path):
                wrote += 1
                print("Wrote {}".format(out_path))
    if wrote == 0:
        print("No part 2 overlay figures were written.")


if __name__ == "__main__":
    main()
