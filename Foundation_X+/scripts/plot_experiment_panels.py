#!/usr/bin/env python3
"""One figure per focused task, with a panel for each other task type.

The experiment directory is a folder such as part_2_v4_norm. Each task folder
inside it is read from export_csvFile.csv. For CANDID-PTX localization in a
run that trains all three tasks, the figure has three panels. Every panel
repeats the focused student and teacher lines. The classification panel adds
one unfocused pair per classification dataset, the localization panel adds one
pair per other localization dataset, and the segmentation panel adds one pair
per segmentation dataset.

Each dataset has its own color. The student is a solid line and the teacher is
a lighter, thinner line of that color. Focused lines stay solid blue and red.

Figures are written to <experiment>/analysis/panels/<task folder>/.
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


def _load(name):
    path = os.path.join(SCRIPT_DIR, name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curves = _load("plot_focused_unfocused.py")
pairs = _load("plot_experiment_pairs.py")

TASK_TYPES = ("classification", "localization", "segmentation")
PANEL_TITLE = {
    "classification": "After classification",
    "localization": "After localization",
    "segmentation": "After segmentation",
}
DATASET_COLOR = {
    "CANDID-PTX": "#4C78A8",
    "RSNApneumonia": "#54A24B",
    "ChestX-Det": "#F58518",
    "SIIM-ACRpneumothorax": "#B279A2",
}
FALLBACK_COLORS = ["#E45756", "#72B7B2", "#FF9DA6", "#9D755D", "#BAB0AC", "#B6992D"]
RUN_TITLE = {
    "cls": "Classification only",
    "loc": "Localization only",
    "seg": "Segmentation only",
    "cls_loc": "Classification + localization",
    "cls_loc_seg": "Classification + localization + segmentation",
    "candidptx_cls": "CANDID-PTX classification",
    "candidptx_loc": "CANDID-PTX localization",
    "candidptx_seg": "CANDID-PTX segmentation",
    "candidptx_cls_loc": "CANDID-PTX classification + localization",
    "candidptx_cls_seg": "CANDID-PTX classification + segmentation",
    "candidptx_loc_seg": "CANDID-PTX localization + segmentation",
    "candidptx_cls_loc_seg": "CANDID-PTX classification + localization + segmentation",
}


def lighten(hex_color, amount=0.55):
    text = hex_color.lstrip("#")
    channels = [int(text[index:index + 2], 16) for index in (0, 2, 4)]
    mixed = [int(round(channel + (255 - channel) * amount)) for channel in channels]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def dataset_color(dataset, used):
    if dataset in DATASET_COLOR:
        return DATASET_COLOR[dataset]
    if dataset not in used:
        used[dataset] = FALLBACK_COLORS[len(used) % len(FALLBACK_COLORS)]
    return used[dataset]


def display_dataset(dataset):
    return curves.DISPLAY_DATASET.get(dataset, dataset)


def display_focus(focus):
    task_type, dataset = focus
    return "{} {}".format(display_dataset(dataset), task_type)


def select_rows(rows, tested, training, model):
    return [
        row for row in rows
        if row["tested"] == tested and row["training"] == training and row["model"] == model
    ]


def panels_for(order, focus):
    grouped = defaultdict(list)
    for task in order:
        if task == focus:
            continue
        grouped[task[0]].append(task)
    return [(task_type, grouped[task_type]) for task_type in TASK_TYPES if grouped[task_type]]


def draw_focused(ax, rows, focus, n_tasks):
    curves.plot_curve(
        ax, select_rows(rows, focus, focus, "student"), n_tasks,
        curves.STUDENT_SOLID, "-", None, linewidth=1.15, zorder=2,
    )
    curves.plot_curve(
        ax, select_rows(rows, focus, focus, "teacher"), n_tasks,
        curves.TEACHER_SOLID, "-", None, linewidth=1.15, zorder=2,
    )


def draw_unfocused(ax, rows, focus, other, n_tasks, color):
    curves.plot_curve(
        ax, select_rows(rows, focus, other, "student"), n_tasks,
        color, "-", None, linewidth=1.7, zorder=4,
    )
    curves.plot_curve(
        ax, select_rows(rows, focus, other, "teacher"), n_tasks,
        lighten(color), (0, (4, 1.5)), None, linewidth=1.25, zorder=4,
    )


def style_panel(ax, title, ylabel, x_max, limits, show_ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Cycle")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(1, x_max)
    ax.set_ylim(*limits)
    if x_max <= 16:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_has_scores(rows, focus, others):
    for other in others:
        if select_rows(rows, focus, other, "student") or select_rows(rows, focus, other, "teacher"):
            return True
    return False


def plot_focus(rows, focus, order, n_tasks, x_max, run, out_path):
    if not select_rows(rows, focus, focus, "student") and not select_rows(rows, focus, focus, "teacher"):
        return False
    panels = [
        (task_type, others)
        for task_type, others in panels_for(order, focus)
        if panel_has_scores(rows, focus, others)
    ]
    if not panels:
        return False
    task_type = focus[0]
    scores = [
        row["score"]
        for row in rows
        if row["tested"] == focus
    ]
    limits = pairs.y_limits(scores)
    ylabel = curves.TASK_YLABEL[task_type]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.1 * len(panels), 4.6), sharey=True, squeeze=False,
    )
    color_of = {}
    seen_datasets = []
    for ax, (panel_type, others), show_ylabel in zip(axes[0], panels, [True] + [False] * (len(panels) - 1)):
        draw_focused(ax, rows, focus, n_tasks)
        for other in others:
            color = dataset_color(other[1], color_of)
            if other[1] not in seen_datasets:
                seen_datasets.append(other[1])
            draw_unfocused(ax, rows, focus, other, n_tasks, color)
        style_panel(ax, PANEL_TITLE[panel_type], ylabel, x_max, limits, show_ylabel)

    legend = [
        plt.Line2D([0], [0], color=curves.STUDENT_SOLID, linewidth=1.15, label="Focused student"),
        plt.Line2D([0], [0], color=curves.TEACHER_SOLID, linewidth=1.15, label="Focused teacher"),
    ]
    for dataset in seen_datasets:
        legend.append(plt.Line2D(
            [0], [0], color=dataset_color(dataset, color_of), linewidth=1.7,
            label=display_dataset(dataset),
        ))
    legend.extend([
        plt.Line2D([0], [0], color="black", linewidth=1.7, label="Unfocused student"),
        plt.Line2D(
            [0], [0], color="0.55", linewidth=1.2, linestyle=(0, (4, 1.5)),
            label="Unfocused teacher",
        ),
    ])
    fig.legend(
        handles=legend, loc="lower center", ncol=min(len(legend), 4),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.06),
    )
    focus_rows = {
        "student": select_rows(rows, focus, focus, "student"),
        "teacher": select_rows(rows, focus, focus, "teacher"),
    }
    fig.text(
        0.5, 0.005,
        "Highest focused student: {}    Highest focused teacher: {}".format(
            curves.best_text(focus_rows["student"], n_tasks),
            curves.best_text(focus_rows["teacher"], n_tasks),
        ),
        ha="center", va="bottom", fontsize=9,
    )
    run_title = RUN_TITLE.get(run, run)
    fig.suptitle("{} — {}".format(display_focus(focus), run_title), fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0.18, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_run(run, run_dir, output_dir):
    csv_path = os.path.join(run_dir, "export_csvFile.csv")
    rows = curves.prefer_end_of_epoch(curves.read_rows(csv_path))
    if not rows:
        print("No test rows in {}".format(csv_path))
        return 0
    part = pairs.infer_part(run)
    n_tasks = curves.cycle_length(part, run, rows)
    order = curves.training_order(rows, n_tasks)
    if len(order) < 2:
        print("Skipping {}: only one training task.".format(run))
        return 0
    x_max = curves.highest_cycle(rows, n_tasks) + 1
    wrote = 0
    for focus in order:
        filename = "{}.png".format(curves.safe_name(curves.task_label(focus)))
        out_path = os.path.join(output_dir, run, filename)
        if plot_focus(rows, focus, order, n_tasks, x_max, run, out_path):
            wrote += 1
            print("Wrote {}".format(out_path))
    return wrote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        help="One experiment folder, for example /scratch/hflechsi/Foundation_X/results/part_2_v4_norm",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to <experiment_dir>/analysis/panels.",
    )
    args = parser.parse_args()
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        parser.error("Not a directory: {}".format(experiment_dir))
    output_dir = args.output_dir or os.path.join(experiment_dir, "analysis", "panels")
    os.makedirs(output_dir, exist_ok=True)

    folders = pairs.task_folders(experiment_dir)
    if not folders:
        print("No task folders with export_csvFile.csv in {}".format(experiment_dir))
        return
    wrote = 0
    for run, run_dir in folders:
        wrote += plot_run(run, run_dir, output_dir)
    if wrote == 0:
        print("No panel figures were written.")


if __name__ == "__main__":
    main()
