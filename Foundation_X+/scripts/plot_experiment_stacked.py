#!/usr/bin/env python3
"""Student and teacher panels stacked, one figure per focused task.

The top row is the student and the bottom row is the teacher. Columns are the
other task types, as in plot_experiment_panels.py. The focused curve is a
solid line. Each other dataset is a lighter dashed line of its own color.
Those lines are this task's score on the cycles when that dataset was trained.

Figures are written to <experiment>/analysis/stacked/<task folder>/.
"""

import argparse
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PANELS_PATH = os.path.join(SCRIPT_DIR, "plot_experiment_panels.py")
_spec = importlib.util.spec_from_file_location("plot_experiment_panels", _PANELS_PATH)
panels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panels)

curves = panels.curves
pairs = panels.pairs
MODELS = ("student", "teacher")
MODEL_COLOR = {
    "student": curves.STUDENT_SOLID,
    "teacher": curves.TEACHER_SOLID,
}


def style_axis(ax, title, ylabel, x_max, limits, show_title, show_xlabel, show_ylabel):
    if show_title:
        ax.set_title(title, fontsize=12)
    if show_xlabel:
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


def draw_model(ax, rows, focus, others, n_tasks, model, color_of):
    curves.plot_curve(
        ax, panels.select_rows(rows, focus, focus, model), n_tasks,
        MODEL_COLOR[model], "-", None, linewidth=1.5, zorder=2,
    )
    for other in others:
        color = panels.dataset_color(other[1], color_of)
        curves.plot_curve(
            ax, panels.select_rows(rows, focus, other, model), n_tasks,
            panels.lighten(color), "--", None, linewidth=1.45, zorder=4,
        )


def plot_focus(rows, focus, order, n_tasks, x_max, run, out_path):
    if not any(panels.select_rows(rows, focus, focus, model) for model in MODELS):
        return False
    grouped = []
    for panel_type, others in panels.panels_for(order, focus):
        kept = panels.others_with_scores(rows, focus, others)
        if kept:
            grouped.append((panel_type, kept))
    if not grouped:
        return False

    task_type = focus[0]
    metric = curves.TASK_YLABEL[task_type]
    scores = [row["score"] for row in rows if row["tested"] == focus]
    limits = pairs.y_limits(scores)
    fig, axes = plt.subplots(
        2, len(grouped), figsize=(5.1 * len(grouped), 7.2), sharex=True, sharey=True, squeeze=False,
    )
    color_of = {}
    seen_datasets = []
    for col, (panel_type, others) in enumerate(grouped):
        for other in others:
            if other[1] not in seen_datasets:
                seen_datasets.append(other[1])
        for row_index, model in enumerate(MODELS):
            ax = axes[row_index][col]
            draw_model(ax, rows, focus, others, n_tasks, model, color_of)
            style_axis(
                ax,
                panels.PANEL_TITLE[panel_type],
                "{} {}".format(model.title(), metric),
                x_max,
                limits,
                show_title=(row_index == 0),
                show_xlabel=(row_index == len(MODELS) - 1),
                show_ylabel=(col == 0),
            )

    legend = [
        plt.Line2D([0], [0], color=curves.STUDENT_SOLID, linewidth=1.5, label="Focused student"),
        plt.Line2D([0], [0], color=curves.TEACHER_SOLID, linewidth=1.5, label="Focused teacher"),
    ]
    for dataset in seen_datasets:
        legend.append(plt.Line2D(
            [0], [0], color=panels.lighten(panels.dataset_color(dataset, color_of)),
            linewidth=1.45, linestyle="--",
            label="After {}".format(panels.display_dataset(dataset)),
        ))
    fig.legend(
        handles=legend, loc="lower center", ncol=min(len(legend), 4),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.05),
    )
    fig.text(
        0.5, 0.005,
        "Highest focused student: {}    Highest focused teacher: {}".format(
            curves.best_text(panels.select_rows(rows, focus, focus, "student"), n_tasks),
            curves.best_text(panels.select_rows(rows, focus, focus, "teacher"), n_tasks),
        ),
        ha="center", va="bottom", fontsize=9,
    )
    run_title = panels.RUN_TITLE.get(run, run)
    fig.suptitle("{} — {}".format(panels.display_focus(focus), run_title), fontsize=14, y=1.02)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
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
        help="Defaults to <experiment_dir>/analysis/stacked.",
    )
    args = parser.parse_args()
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        parser.error("Not a directory: {}".format(experiment_dir))
    output_dir = args.output_dir or os.path.join(experiment_dir, "analysis", "stacked")
    os.makedirs(output_dir, exist_ok=True)

    folders = pairs.task_folders(experiment_dir)
    if not folders:
        print("No task folders with export_csvFile.csv in {}".format(experiment_dir))
        return
    wrote = 0
    for run, run_dir in folders:
        wrote += plot_run(run, run_dir, output_dir)
    if wrote == 0:
        print("No stacked figures were written.")


if __name__ == "__main__":
    main()
