#!/usr/bin/env python3
"""Plot one experiment directory, one focused dataset against one other dataset.

The experiment directory is a folder such as part_2_v4_norm. Each task folder
inside it (cls, loc, seg, cls_loc, cls_loc_seg, and the part 1 names) is read
from its export_csvFile.csv. logs and analysis are skipped.

For every tested task, each figure is that dataset's focused student and
teacher lines, plus that same dataset's unfocused student and teacher lines
from one other training task. A separate figure is written for every other
task. Lines have no markers. The highest focused and unfocused student and
teacher scores are listed on the figure.

Figures are written to <experiment>/analysis/<task folder>/.
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

SKIP_DIRS = {"logs", "analysis"}


def infer_part(run):
    if run in curves.PART2_RUNS:
        return "part_2"
    if run in curves.PART1_RUNS:
        return "part_1"
    return "experiment"


def task_folders(experiment_dir):
    found = []
    for name in sorted(os.listdir(experiment_dir)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        path = os.path.join(experiment_dir, name)
        if os.path.isfile(os.path.join(path, "export_csvFile.csv")):
            found.append((name, path))
    return found


def y_limits(scores):
    finite = [score for score in scores if score is not None]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    span = max(high - low, 1.0)
    pad = 0.08 * span
    return low - pad, high + pad


def display_task(task):
    task_type, dataset = task
    name = curves.DISPLAY_DATASET.get(dataset, dataset)
    return "{} {}".format(name, task_type)


def plot_pair(rows, focus, other, n_tasks, x_max, out_path):
    """Focused lines are the focus dataset after its own training step.

    Unfocused lines are that same dataset after the other task was trained.
    """
    focus_rows = defaultdict(list)
    other_rows = defaultdict(list)
    for row in rows:
        if row["tested"] != focus:
            continue
        if row["training"] == focus:
            focus_rows[row["model"]].append(row)
        elif row["training"] == other:
            other_rows[row["model"]].append(row)
    if not focus_rows["student"] and not focus_rows["teacher"]:
        return False
    if not other_rows["student"] and not other_rows["teacher"]:
        return False

    task_type, dataset = focus
    focus_name = curves.DISPLAY_DATASET.get(dataset, dataset)
    other_name = display_task(other)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    curves.plot_curve(
        ax, focus_rows["student"], n_tasks, curves.STUDENT_SOLID, "-",
        "{} student focused".format(focus_name),
    )
    curves.plot_curve(
        ax, focus_rows["teacher"], n_tasks, curves.TEACHER_SOLID, "-",
        "{} teacher focused".format(focus_name),
    )
    curves.plot_curve(
        ax, other_rows["student"], n_tasks, curves.STUDENT_UNFOCUSED, "--",
        "Student after {}".format(other_name),
        linewidth=1.4,
    )
    curves.plot_curve(
        ax, other_rows["teacher"], n_tasks, curves.TEACHER_UNFOCUSED, "--",
        "Teacher after {}".format(other_name),
        linewidth=1.4,
    )

    scores = [
        row["score"]
        for pool in (focus_rows, other_rows)
        for model_rows in pool.values()
        for row in model_rows
    ]
    low, high = y_limits(scores)
    ax.set_title("{} {} focused, unfocused after {}".format(focus_name, task_type, other_name))
    ax.set_xlabel("Cycle")
    ax.set_ylabel(curves.TASK_YLABEL[task_type])
    ax.set_xlim(1, x_max)
    ax.set_ylim(low, high)
    if x_max <= 16:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    highest = "\n".join([
        "Highest focused student: {}    Highest focused teacher: {}".format(
            curves.best_text(focus_rows["student"], n_tasks),
            curves.best_text(focus_rows["teacher"], n_tasks),
        ),
        "Highest unfocused student: {}    Highest unfocused teacher: {}".format(
            curves.best_text(other_rows["student"], n_tasks),
            curves.best_text(other_rows["teacher"], n_tasks),
        ),
    ])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.01, highest, fontsize=9, va="bottom")
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
    part = infer_part(run)
    n_tasks = curves.cycle_length(part, run, rows)
    order = curves.training_order(rows, n_tasks)
    if len(order) < 2:
        print("Skipping {}: only one training task, so there is no other dataset.".format(run))
        return 0
    x_max = curves.highest_cycle(rows, n_tasks) + 1
    wrote = 0
    for focus in order:
        for other in order:
            if other == focus:
                continue
            filename = "{}__{}.png".format(
                curves.safe_name(curves.task_label(focus)),
                curves.safe_name(curves.task_label(other)),
            )
            out_path = os.path.join(output_dir, run, filename)
            if plot_pair(rows, focus, other, n_tasks, x_max, out_path):
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
        help="Defaults to <experiment_dir>/analysis.",
    )
    args = parser.parse_args()
    experiment_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(experiment_dir):
        parser.error("Not a directory: {}".format(experiment_dir))
    output_dir = args.output_dir or os.path.join(experiment_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    folders = task_folders(experiment_dir)
    if not folders:
        print("No task folders with export_csvFile.csv in {}".format(experiment_dir))
        return
    wrote = 0
    for run, run_dir in folders:
        wrote += plot_run(run, run_dir, output_dir)
    if wrote == 0:
        print("No pair figures were written.")


if __name__ == "__main__":
    main()
