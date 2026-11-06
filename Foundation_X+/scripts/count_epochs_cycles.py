#!/usr/bin/env python3
"""Print epochs and cycles completed for each run in an experiment directory.

The directory is one experiment folder, such as part_1, part_2, or part_3.
Each task folder inside it (candidptx_cls, cls_loc, all, and the other launch
names) is one run. logs and analysis are skipped.

Epochs are the latest epoch in export_csvFile.csv or in a saved
ckpt_E*_TH*.pth. Training starts at epoch 1, so that number is how many
epochs have been run. One cycle is one pass through every task scheduled for
that run. A run on epoch 45 with 20 tasks per cycle has finished 2 cycles
and is 5 epochs into the next one.
"""

import argparse
import csv
import os
import re


SKIP_DIRS = {"logs", "analysis", "panels", "stacked"}

# Tasks per cycle in the part 1, part 2, and part 3 launch scripts.
# Used when a run has not yet repeated its first task, so the CSV cannot
# show the cycle length by itself.
TASKS_PER_CYCLE = {
    "candidptx_cls": 1,
    "candidptx_loc": 1,
    "candidptx_seg": 1,
    "candidptx_cls_loc": 2,
    "candidptx_cls_seg": 2,
    "candidptx_loc_seg": 2,
    "candidptx_cls_loc_seg": 3,
    "cls": 4,
    "loc": 4,
    "seg": 3,
    "cls_loc": 8,
    "cls_loc_seg": 11,
    "all": 20,
}

RUN_ORDER = list(TASKS_PER_CYCLE)
CKPT_RE = re.compile(r"ckpt_E(\d+)")
PHASE_RE = re.compile(r"_[ABC]_Train$")


def task_key(task_train):
    """Drop the A/B step so both logs of one epoch name the same task."""
    return PHASE_RE.sub("_Train", task_train.strip())


def csv_epochs(csv_path):
    """Return {epoch: training task} from export_csvFile.csv."""
    found = {}
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if not raw or all(not cell.strip() for cell in raw):
                continue
            if raw[0].strip().lower() == "epoch":
                continue
            try:
                epoch = int(float(raw[0].strip()))
            except ValueError:
                continue
            task = raw[1].strip() if len(raw) > 1 else ""
            if epoch not in found and task and task not in {"-", "None"}:
                found[epoch] = task_key(task)
            elif epoch not in found:
                found[epoch] = ""
    return found


def checkpoint_epoch(run_dir):
    """Latest epoch saved as ckpt_E{epoch}_TH{head}.pth, or 0."""
    latest = 0
    for name in os.listdir(run_dir):
        match = CKPT_RE.search(name)
        if match is None:
            continue
        latest = max(latest, int(match.group(1)))
    return latest


def observed_cycle_length(epoch_tasks):
    """Epochs between the first task and the next time that task is trained."""
    ordered = sorted(epoch for epoch, task in epoch_tasks.items() if task)
    if len(ordered) < 2:
        return None
    first_epoch = ordered[0]
    first_task = epoch_tasks[first_epoch]
    for epoch in ordered[1:]:
        if epoch_tasks[epoch] == first_task:
            length = epoch - first_epoch
            if length > 0:
                return length
    return None


def cycle_length(run, epoch_tasks):
    observed = observed_cycle_length(epoch_tasks)
    scheduled = TASKS_PER_CYCLE.get(run)
    if observed is None:
        return scheduled
    if scheduled is None:
        return observed
    return max(observed, scheduled)


def is_run_dir(path):
    if os.path.isfile(os.path.join(path, "export_csvFile.csv")):
        return True
    for name in os.listdir(path):
        if CKPT_RE.search(name):
            return True
    return False


def run_dirs(experiment_dir):
    found = []
    for name in os.listdir(experiment_dir):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        path = os.path.join(experiment_dir, name)
        if os.path.isdir(path) and is_run_dir(path):
            found.append((name, path))
    order = {name: index for index, name in enumerate(RUN_ORDER)}
    found.sort(key=lambda item: (order.get(item[0], len(order)), item[0]))
    return found


def experiments(directory):
    """One experiment, or each experiment inside a results root."""
    runs = run_dirs(directory)
    if runs or is_run_dir(directory):
        return [(os.path.basename(os.path.abspath(directory)), directory)]
    found = []
    for name in sorted(os.listdir(directory)):
        if name in SKIP_DIRS or name.startswith("."):
            continue
        path = os.path.join(directory, name)
        if os.path.isdir(path) and (run_dirs(path) or is_run_dir(path)):
            found.append((name, path))
    return found


def summarize_run(run, run_dir):
    csv_path = os.path.join(run_dir, "export_csvFile.csv")
    epoch_tasks = csv_epochs(csv_path) if os.path.isfile(csv_path) else {}
    csv_epoch = max(epoch_tasks) if epoch_tasks else 0
    ckpt_epoch = checkpoint_epoch(run_dir)
    epochs = max(csv_epoch, ckpt_epoch)
    length = cycle_length(run, epoch_tasks)
    return {
        "run": run,
        "epochs": epochs,
        "csv_epoch": csv_epoch,
        "ckpt_epoch": ckpt_epoch,
        "tasks_per_cycle": length,
    }


def epochs_text(row):
    epochs = row["epochs"]
    csv_epoch = row["csv_epoch"]
    ckpt_epoch = row["ckpt_epoch"]
    if csv_epoch and ckpt_epoch and csv_epoch != ckpt_epoch:
        return "{} (csv {}, ckpt {})".format(epochs, csv_epoch, ckpt_epoch)
    return str(epochs)


def cycles_text(row):
    length = row["tasks_per_cycle"]
    epochs = row["epochs"]
    if not length:
        return "?"
    done, into = divmod(epochs, length)
    if into == 0:
        return str(done)
    return "{} + {}/{}".format(done, into, length)


def print_experiment(name, experiment_dir):
    if is_run_dir(experiment_dir) and not run_dirs(experiment_dir):
        rows = [summarize_run(name, experiment_dir)]
    else:
        rows = [summarize_run(run, path) for run, path in run_dirs(experiment_dir)]
    if not rows:
        print("No runs in {}".format(experiment_dir))
        return
    labeled = []
    for row in rows:
        labeled.append((
            row["run"],
            epochs_text(row),
            cycles_text(row),
            "-" if row["tasks_per_cycle"] is None else str(row["tasks_per_cycle"]),
        ))
    headers = ("run", "epochs", "cycles", "tasks_per_cycle")
    widths = [
        max(len(headers[index]), *(len(item[index]) for item in labeled))
        for index in range(4)
    ]
    print(name)
    print("  ".join(headers[index].ljust(widths[index]) for index in range(4)))
    for item in labeled:
        print("  ".join(item[index].ljust(widths[index]) for index in range(4)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        help="One experiment folder, such as part_1, part_2, or part_3.",
    )
    args = parser.parse_args()
    directory = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(directory):
        parser.error("Not a directory: {}".format(directory))
    found = experiments(directory)
    if not found:
        print("No runs with export_csvFile.csv or checkpoints in {}".format(directory))
        return
    for index, (name, path) in enumerate(found):
        if index:
            print()
        print_experiment(name, path)


if __name__ == "__main__":
    main()
