#!/usr/bin/env python3
"""Plot focused and unfocused test curves for part 1 and part 2.

Each figure is one evaluated dataset and task. A solid blue line is the
student on epochs that trained that exact task. A solid red line is the
teacher on those epochs. Every other training task is a dotted line, and
the student and teacher each get their own color.

Also writes best_epoch_metrics.csv. Student and teacher best epochs are
chosen separately by the highest primary score: classification AUC,
localization mAP50, or segmentation mean Dice.
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

FOCUSED_STUDENT_COLOR = "#1f4bff"
FOCUSED_TEACHER_COLOR = "#d62728"
# Distinct colors that are not the solid blue or solid red.
OTHER_COLORS = [
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#a1d99b",
    "#fd8d3c",
    "#9e9ac8",
    "#c49c94",
    "#f7b6d2",
    "#9edae5",
    "#cedb9c",
    "#636363",
    "#74c476",
    "#e6550d",
    "#6a51a3",
    "#ce6dbd",
    "#31a354",
    "#de9ed6",
    "#b5cf6b",
    "#fc8d62",
]


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


def checkpoint_name(epoch, training_task):
    head = TASK_HEAD.get(training_task)
    if head is None:
        return ""
    return "ckpt_E{}_TH{}.pth".format(epoch, head)


def series_style(model, training, tested, color_of):
    focused = training == tested
    if focused and model == "student":
        return FOCUSED_STUDENT_COLOR, "-", "Student, {}".format(task_label(training))
    if focused and model == "teacher":
        return FOCUSED_TEACHER_COLOR, "-", "Teacher, {}".format(task_label(training))
    color = color_of[(model, training)]
    who = "Student" if model == "student" else "Teacher"
    return color, ":", "{}, {}".format(who, task_label(training))


def assign_other_colors(training_tasks, tested):
    """One color per dotted line. Blue and red stay on the solid lines."""
    color_of = {}
    index = 0
    others = sorted(task for task in training_tasks if task != tested)
    for training in others:
        for model in ("student", "teacher"):
            color_of[(model, training)] = OTHER_COLORS[index % len(OTHER_COLORS)]
            index += 1
    return color_of


def plot_task(rows, tested, out_path):
    task_rows = [row for row in rows if row["tested"] == tested]
    if not task_rows:
        return
    training_tasks = {row["training"] for row in task_rows}
    color_of = assign_other_colors(training_tasks, tested)
    grouped = defaultdict(list)
    for row in task_rows:
        grouped[(row["model"], row["training"])].append(row)

    def sort_key(item):
        model, training = item
        focused_rank = 0 if training == tested else 1
        model_rank = 0 if model == "student" else 1
        return (focused_rank, task_label(training), model_rank)

    fig_width = 10
    fig_height = 6
    n_lines = len(grouped)
    if n_lines > 8:
        fig_height = 7
        fig_width = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for key in sorted(grouped, key=sort_key):
        points = sorted(grouped[key], key=lambda row: row["epoch"])
        epochs = [row["epoch"] for row in points]
        scores = [row["score"] for row in points]
        color, linestyle, label = series_style(key[0], key[1], tested, color_of)
        ax.plot(
            epochs,
            scores,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=label,
        )
    task_type, dataset = tested
    ax.set_title("{} {}".format(dataset, task_type))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(TASK_YLABEL[task_type])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def best_rows_for_run(part, run, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["tested"], row["training"], row["model"])].append(row)
    tested_training = {(tested, training) for tested, training, _model in grouped}
    output = []
    for tested, training in sorted(tested_training, key=lambda item: (task_label(item[0]), task_label(item[1]))):
        record = {
            "part": part,
            "run": run,
            "plotted_dataset": tested[1],
            "task": tested[0],
            "training_task": task_label(training),
            "focused": "yes" if training == tested else "no",
            "student_epoch": "",
            "student_score": "",
            "student_checkpoint": "",
            "teacher_epoch": "",
            "teacher_score": "",
            "teacher_checkpoint": "",
        }
        for model in ("student", "teacher"):
            points = grouped.get((tested, training, model), [])
            if not points:
                continue
            best = max(points, key=lambda row: (row["score"], -row["epoch"]))
            record["{}_epoch".format(model)] = best["epoch"]
            record["{}_score".format(model)] = "{:.6f}".format(best["score"])
            record["{}_checkpoint".format(model)] = checkpoint_name(best["epoch"], training)
        output.append(record)
    return output


BEST_FIELDS = [
    "part",
    "run",
    "plotted_dataset",
    "task",
    "training_task",
    "focused",
    "student_epoch",
    "student_score",
    "student_checkpoint",
    "teacher_epoch",
    "teacher_score",
    "teacher_checkpoint",
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
        tested_tasks = sorted({row["tested"] for row in rows}, key=task_label)
        for tested in tested_tasks:
            filename = "{}_{}_{}.png".format(part, run, safe_name(task_label(tested)))
            plot_task(rows, tested, os.path.join(output_dir, filename))
            wrote_plot = True
            print("Wrote {}".format(os.path.join(output_dir, filename)))
        best.extend(best_rows_for_run(part, run, rows))

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
