#!/usr/bin/env python3
"""Plot focused and unfocused test curves for part 1 and part 2.

Each figure is one evaluated dataset and task. The x-axis is the cycle.
Solid blue and solid red are the student and teacher on cycles that trained
that exact task. Dashed green and dashed orange connect every cycle for the
student and teacher, so each dashed curve passes through its solid curve.

Also writes best_epoch_metrics.csv. Student and teacher best scores are
chosen separately. The primary score is classification AUC, localization
mAP50, or segmentation mean Dice.
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


RESULTS_ROOT = "/scratch/hflechsi/Foundation_X/results"

PART1_RUNS = [
    "candidptx_cls",
    "candidptx_loc",
    "candidptx_seg",
    "candidptx_cls_loc",
    "candidptx_cls_seg",
    "candidptx_loc_seg",
    "candidptx_cls_loc_seg",
]
PART2_RUNS = [
    "cls",
    "loc",
    "seg",
    "cls_loc",
    "cls_loc_seg",
]

# Tasks per cycle in the part 1 and part 2 launch scripts.
RUN_CYCLE_LENGTH = {
    ("part_1", "candidptx_cls"): 1,
    ("part_1", "candidptx_loc"): 1,
    ("part_1", "candidptx_seg"): 1,
    ("part_1", "candidptx_cls_loc"): 2,
    ("part_1", "candidptx_cls_seg"): 2,
    ("part_1", "candidptx_loc_seg"): 2,
    ("part_1", "candidptx_cls_loc_seg"): 3,
    ("part_2", "cls"): 4,
    ("part_2", "loc"): 4,
    ("part_2", "seg"): 3,
    ("part_2", "cls_loc"): 8,
    ("part_2", "cls_loc_seg"): 11,
}

STUDENT_SOLID = "#1f4bff"
TEACHER_SOLID = "#d62728"
STUDENT_DASHED = "#2ca02c"
TEACHER_DASHED = "#ff7f0e"

# Written row order. The file header swaps Dataset and Task-Train.
COLUMNS = [
    "epoch",
    "task_train",
    "dataset",
    "model",
    "task_test",
    "auc",
    "map40",
    "map50",
    "map50_95",
    "dice",
]

TASK_SCORE = {
    "classification": "auc",
    "localization": "map50",
    "segmentation": "dice",
}
TASK_YLABEL = {
    "classification": "AUC",
    "localization": "mAP50",
    "segmentation": "Mean Dice",
}

# Tokens embedded in task_todo, mapped to the dataset name written at test time.
TOKEN_TO_DATASET = {
    "CANDIDPTX": "CANDID-PTX",
    "RSNAPneumonia": "RSNApneumonia",
    "RSNAPNEUMONIA": "RSNApneumonia",
    "ChestXDet": "ChestX-Det",
    "CHESTXDET": "ChestX-Det",
    "SIIMACR": "SIIM-ACRpneumothorax",
    "CheXpert": "CheXpert",
    "NIH": "NIHChestXray14",
    "VinDRCXR": "VinDRCXR",
    "NIHShenzhen": "NIHShenzhen",
    "MIMICII": "MIMICII",
    "TBX11k": "TBX11k",
    "TBX11K": "TBX11k",
    "NODE21": "Node21",
    "Node21": "Node21",
}

# Head index saved in ckpt_E{epoch}_TH{head}.pth for foundation_x5.
TASK_HEAD = {
    ("classification", "CheXpert"): 0,
    ("classification", "NIHChestXray14"): 1,
    ("classification", "VinDRCXR"): 2,
    ("classification", "NIHShenzhen"): 3,
    ("classification", "MIMICII"): 4,
    ("classification", "TBX11k"): 5,
    ("localization", "TBX11k"): 6,
    ("classification", "Node21"): 7,
    ("localization", "Node21"): 8,
    ("classification", "CANDID-PTX"): 9,
    ("localization", "CANDID-PTX"): 10,
    ("segmentation", "CANDID-PTX"): 11,
    ("classification", "RSNApneumonia"): 12,
    ("localization", "RSNApneumonia"): 13,
    ("classification", "ChestX-Det"): 14,
    ("localization", "ChestX-Det"): 15,
    ("segmentation", "ChestX-Det"): 16,
    ("classification", "SIIM-ACRpneumothorax"): 17,
    ("localization", "SIIM-ACRpneumothorax"): 18,
    ("segmentation", "SIIM-ACRpneumothorax"): 19,
}

TRAIN_RE = re.compile(
    r"^(Classification|Localization|Segmentation)_(.+?)_(?:[ABC]_)?Train$"
)


def runs(results_root):
    found = []
    for name in PART1_RUNS:
        found.append(("part_1", name, os.path.join(results_root, "part_1", name)))
    for name in PART2_RUNS:
        found.append(("part_2", name, os.path.join(results_root, "part_2", name)))
    return found


def parse_training_task(task_train):
    """Return (task_type, dataset) from a task_todo such as Classification_CANDIDPTX_B_Train."""
    if not task_train or task_train in {"None", "-"}:
        return None
    match = TRAIN_RE.match(task_train.strip())
    if match is None:
        return None
    task_type = match.group(1).lower()
    token = match.group(2)
    dataset = TOKEN_TO_DATASET.get(token)
    if dataset is None:
        return None
    return task_type, dataset


def parse_tested_task(task_test, dataset):
    if not task_test or "_" not in task_test:
        return None
    task_type, _, rest = task_test.partition("_")
    task_type = task_type.lower()
    if task_type not in TASK_SCORE:
        return None
    tested_dataset = rest if rest else dataset
    if dataset and tested_dataset != dataset:
        tested_dataset = dataset
    return task_type, tested_dataset


def parse_score(text):
    if text is None:
        return None
    text = str(text).strip()
    if text in {"", "-", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if not raw or all(not cell.strip() for cell in raw):
                continue
            if raw[0].strip().lower() == "epoch":
                continue
            padded = list(raw) + [""] * (len(COLUMNS) - len(raw))
            record = {name: padded[i].strip() for i, name in enumerate(COLUMNS)}
            training = parse_training_task(record["task_train"])
            tested = parse_tested_task(record["task_test"], record["dataset"])
            if training is None or tested is None:
                continue
            task_type, dataset = tested
            score = parse_score(record[TASK_SCORE[task_type]])
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


def prefer_end_of_epoch(rows):
    """Keep the end-of-epoch score when both the A and B steps were logged."""
    ranked = {"B": 2, "A": 1, "": 0}
    chosen = {}
    for row in rows:
        key = (row["epoch"], row["model"], row["tested"], row["training"])
        current = chosen.get(key)
        if current is None or ranked[row["phase"]] >= ranked[current["phase"]]:
            chosen[key] = row
    return list(chosen.values())


def task_label(task):
    task_type, dataset = task
    return "{} {}".format(dataset, task_type)


def cycle_length(part, run, rows):
    scheduled = RUN_CYCLE_LENGTH.get((part, run))
    observed = len({row["training"] for row in rows})
    if scheduled is None:
        return max(observed, 1)
    return max(scheduled, observed, 1)


def cycle_x(epoch, n_tasks):
    """Position of an epoch inside the cycle axis. Cycle 1 starts at x=1."""
    return ((epoch - 1) / float(n_tasks)) + 1


def cycle_number(epoch, n_tasks):
    return ((epoch - 1) // n_tasks) + 1


def checkpoint_name(epoch, training_task):
    head = TASK_HEAD.get(training_task)
    if head is None:
        return ""
    return "ckpt_E{}_TH{}.pth".format(epoch, head)


def highest_cycle(rows, n_tasks):
    if not rows:
        return 1
    return max(cycle_number(row["epoch"], n_tasks) for row in rows)


def plot_curve(ax, points, n_tasks, color, linestyle, label):
    ordered = sorted(points, key=lambda row: row["epoch"])
    if not ordered:
        return
    xs = [cycle_x(row["epoch"], n_tasks) for row in ordered]
    ys = [row["score"] for row in ordered]
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=label,
    )


def plot_task(rows, tested, n_tasks, x_max, out_path):
    task_rows = [row for row in rows if row["tested"] == tested]
    if not task_rows:
        return
    by_model = defaultdict(list)
    focused = defaultdict(list)
    for row in task_rows:
        by_model[row["model"]].append(row)
        if row["training"] == tested:
            focused[row["model"]].append(row)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_curve(ax, by_model["student"], n_tasks, STUDENT_DASHED, "--", "Student focused and unfocused")
    plot_curve(ax, by_model["teacher"], n_tasks, TEACHER_DASHED, "--", "Teacher focused and unfocused")
    plot_curve(ax, focused["student"], n_tasks, STUDENT_SOLID, "-", "Student focused")
    plot_curve(ax, focused["teacher"], n_tasks, TEACHER_SOLID, "-", "Teacher focused")

    task_type, dataset = tested
    ax.set_title("{} {}".format(dataset, task_type))
    ax.set_xlabel("Cycle")
    ax.set_ylabel(TASK_YLABEL[task_type])
    ax.set_xlim(1, x_max)
    if x_max <= 16:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def best_score(points):
    if not points:
        return None
    return max(points, key=lambda row: (row["score"], -row["epoch"]))


def join_numbers(values):
    return ",".join(str(value) for value in values)


def best_rows_for_run(part, run, rows, n_tasks):
    by_tested = defaultdict(list)
    for row in rows:
        by_tested[row["tested"]].append(row)
    output = []
    for tested in sorted(by_tested, key=task_label):
        task_rows = by_tested[tested]
        focused_epochs = sorted({
            row["epoch"] for row in task_rows if row["training"] == tested
        })
        record = {
            "part": part,
            "run": run,
            "plotted_dataset": tested[1],
            "task": tested[0],
            "epochs_done": join_numbers(focused_epochs),
            "cycles_done": join_numbers(cycle_number(epoch, n_tasks) for epoch in focused_epochs),
        }
        for model in ("student", "teacher"):
            model_rows = [row for row in task_rows if row["model"] == model]
            focused_rows = [row for row in model_rows if row["training"] == tested]
            for kind, pool in (("focused", focused_rows), ("all", model_rows)):
                prefix = "{}_{}".format(model, kind)
                best = best_score(pool)
                if best is None:
                    record["{}_cycle".format(prefix)] = ""
                    record["{}_epoch".format(prefix)] = ""
                    record["{}_score".format(prefix)] = ""
                    record["{}_checkpoint".format(prefix)] = ""
                    continue
                record["{}_cycle".format(prefix)] = cycle_number(best["epoch"], n_tasks)
                record["{}_epoch".format(prefix)] = best["epoch"]
                record["{}_score".format(prefix)] = "{:.6f}".format(best["score"])
                record["{}_checkpoint".format(prefix)] = checkpoint_name(best["epoch"], best["training"])
        output.append(record)
    return output


BEST_FIELDS = [
    "part",
    "run",
    "plotted_dataset",
    "task",
    "epochs_done",
    "cycles_done",
    "student_focused_cycle",
    "student_focused_epoch",
    "student_focused_score",
    "student_focused_checkpoint",
    "teacher_focused_cycle",
    "teacher_focused_epoch",
    "teacher_focused_score",
    "teacher_focused_checkpoint",
    "student_all_cycle",
    "student_all_epoch",
    "student_all_score",
    "student_all_checkpoint",
    "teacher_all_cycle",
    "teacher_all_epoch",
    "teacher_all_score",
    "teacher_all_checkpoint",
]


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for figures and best_epoch_metrics.csv. Defaults to <results-root>/analysis.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or os.path.join(args.results_root, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    best = []
    wrote_plot = False
    for part, run, run_dir in runs(args.results_root):
        csv_path = os.path.join(run_dir, "export_csvFile.csv")
        if not os.path.isfile(csv_path):
            print("Missing {}".format(csv_path))
            continue
        rows = prefer_end_of_epoch(read_rows(csv_path))
        if not rows:
            print("No test rows in {}".format(csv_path))
            continue
        n_tasks = cycle_length(part, run, rows)
        x_max = highest_cycle(rows, n_tasks) + 1
        part_dir = "part1" if part == "part_1" else "part2"
        figure_dir = os.path.join(output_dir, part_dir)
        tested_tasks = sorted({row["tested"] for row in rows}, key=task_label)
        for tested in tested_tasks:
            filename = "{}_{}.png".format(run, safe_name(task_label(tested)))
            figure_path = os.path.join(figure_dir, filename)
            plot_task(rows, tested, n_tasks, x_max, figure_path)
            wrote_plot = True
            print("Wrote {}".format(figure_path))
        best.extend(best_rows_for_run(part, run, rows, n_tasks))

    best_path = os.path.join(output_dir, "best_epoch_metrics.csv")
    with open(best_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_FIELDS)
        writer.writeheader()
        writer.writerows(best)
    print("Wrote {}".format(best_path))
    if not wrote_plot:
        print("No part 1 or part 2 export_csvFile.csv files were found.")


if __name__ == "__main__":
    main()
