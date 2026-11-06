#!/usr/bin/env python3
"""Write one CSV of the best Part 1 and Part 2 scores.

Each experiment folder under the results root is scanned. A run is included
when its folder name is one of the Part 1 or Part 2 task folders and it
contains export_csvFile.csv. Localization uses mAP40. Classification uses
AUC and segmentation uses mean Dice.

The best student and teacher scores are the highest focused scores, measured
on the epochs when that same task was trained. epoch_done is the latest epoch
written in that run.

The CSV is written to <results-root>/combined_results.csv.
"""

import argparse
import csv
import importlib.util
import os
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOADER_PATH = os.path.join(SCRIPT_DIR, "plot_focused_unfocused.py")
_spec = importlib.util.spec_from_file_location("plot_focused_unfocused", _LOADER_PATH)
curves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curves)

SCORE_COLUMN = {
    "classification": "auc",
    "localization": "map40",
    "segmentation": "dice",
}
METRIC_NAME = {
    "classification": "AUC",
    "localization": "mAP40",
    "segmentation": "Dice",
}
FIELDS = [
    "experiment",
    "part",
    "run",
    "dataset",
    "task",
    "metric",
    "epoch_done",
    "student_best",
    "student_epoch",
    "teacher_best",
    "teacher_epoch",
]
SKIP_DIRS = {"logs", "analysis", "panels", "stacked"}


def read_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if not raw or all(not cell.strip() for cell in raw):
                continue
            if raw[0].strip().lower() == "epoch":
                continue
            padded = list(raw) + [""] * (len(curves.COLUMNS) - len(raw))
            record = {name: padded[i].strip() for i, name in enumerate(curves.COLUMNS)}
            training = curves.parse_training_task(record["task_train"])
            tested = curves.parse_tested_task(record["task_test"], record["dataset"])
            if training is None or tested is None:
                continue
            task_type = tested[0]
            score = curves.parse_score(record[SCORE_COLUMN[task_type]])
            if score is None:
                continue
            try:
                epoch = int(float(record["epoch"]))
            except ValueError:
                continue
            model = record["model"].strip().lower()
            if model not in {"student", "teacher"}:
                continue
            phase = "B" if "_B_Train" in record["task_train"] else "A" if "_A_Train" in record["task_train"] else ""
            rows.append({
                "epoch": epoch,
                "model": model,
                "tested": tested,
                "training": training,
                "score": score,
                "phase": phase,
            })
    return rows


def run_dirs(experiment_dir):
    found = []
    if not os.path.isdir(experiment_dir):
        return found
    for name in sorted(os.listdir(experiment_dir)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        part = None
        if name in curves.PART1_RUNS:
            part = "part_1"
        elif name in curves.PART2_RUNS:
            part = "part_2"
        if part is None:
            continue
        path = os.path.join(experiment_dir, name)
        if os.path.isfile(os.path.join(path, "export_csvFile.csv")):
            found.append((part, name, path))
    return found


def experiments(results_root):
    """Return (name, path) for each experiment, or the root itself when it is one."""
    if run_dirs(results_root):
        return [(os.path.basename(os.path.abspath(results_root)), results_root)]
    found = []
    for name in sorted(os.listdir(results_root)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        path = os.path.join(results_root, name)
        if os.path.isdir(path) and run_dirs(path):
            found.append((name, path))
    return found


def best_focused(rows, tested, model):
    pool = [
        row for row in rows
        if row["tested"] == tested and row["model"] == model and row["training"] == tested
    ]
    return curves.best_score(pool)


def format_score(row):
    if row is None:
        return ""
    return "{:.4f}".format(row["score"])


def format_epoch(row):
    if row is None:
        return ""
    return str(row["epoch"])


def rows_for_run(experiment, part, run, rows):
    epoch_done = max(row["epoch"] for row in rows)
    by_tested = defaultdict(list)
    for row in rows:
        by_tested[row["tested"]].append(row)
    output = []
    for tested in sorted(by_tested, key=curves.task_label):
        task_type, dataset = tested
        student = best_focused(rows, tested, "student")
        teacher = best_focused(rows, tested, "teacher")
        output.append({
            "experiment": experiment,
            "part": part,
            "run": run,
            "dataset": dataset,
            "task": task_type,
            "metric": METRIC_NAME[task_type],
            "epoch_done": epoch_done,
            "student_best": format_score(student),
            "student_epoch": format_epoch(student),
            "teacher_best": format_score(teacher),
            "teacher_epoch": format_epoch(teacher),
        })
    return output


def collect(results_root):
    output = []
    for experiment, experiment_dir in experiments(results_root):
        runs = run_dirs(experiment_dir)
        order = {name: index for index, name in enumerate(curves.PART1_RUNS + curves.PART2_RUNS)}
        runs.sort(key=lambda item: order.get(item[1], len(order)))
        for part, run, run_dir in runs:
            csv_path = os.path.join(run_dir, "export_csvFile.csv")
            rows = curves.prefer_end_of_epoch(read_rows(csv_path))
            if not rows:
                print("No test rows in {}".format(csv_path))
                continue
            output.extend(rows_for_run(experiment, part, run, rows))
            print("Read {}".format(csv_path))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_root",
        nargs="?",
        default=curves.RESULTS_ROOT,
        help="Results root, or one experiment folder such as part_2_v4_norm.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="CSV path. Defaults to <results-root>/combined_results.csv.",
    )
    args = parser.parse_args()
    results_root = os.path.abspath(args.results_root)
    if not os.path.isdir(results_root):
        parser.error("Not a directory: {}".format(results_root))
    output_path = args.output or os.path.join(results_root, "combined_results.csv")
    rows = collect(results_root)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {} rows to {}".format(len(rows), output_path))


if __name__ == "__main__":
    main()
