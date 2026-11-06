#!/usr/bin/env python3
"""Draw the Part 2 synergy and recipe tables as slide images.

Part 2 trains several datasets at once. Each CSV becomes one synergy image
with a block per task: the task by itself, classification + localization,
and all three tasks. The delta is all three minus the task by itself.

The recipe images compare the original run with the later Normal and Diff
runs. One image per task. The first three score columns are the task by
itself. The next three are all three tasks together. The last columns are
the later all-three runs minus the original all-three run.

Defaults are Results/csv/part2.csv, part2_norm.csv, and part2_diff.csv.
Scores are the best focused value in the CSV. Localization is mAP40.
Images are written to Results/tables/.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATASET_ORDER = (
    "CANDID-PTX",
    "ChestX-Det",
    "RSNApneumonia",
    "SIIM-ACRpneumothorax",
)
DATASET_LABEL = {
    "CANDID-PTX": "CANDID-PTX",
    "ChestX-Det": "ChestX-Det",
    "RSNApneumonia": "RSNA",
    "SIIM-ACRpneumothorax": "SIIM-ACR",
}
TASKS = (
    ("classification", "Classification (AUC)", "cls"),
    ("localization", "Localization (mAP40)", "loc"),
    ("segmentation", "Segmentation (Dice)", "seg"),
)
RUN_EPOCH = (
    ("cls", "classification"),
    ("loc", "localization"),
    ("seg", "segmentation"),
    ("cls_loc", "cls+loc"),
    ("cls_loc_seg", "all three"),
)

INK = "#14202B"
MUTED = "#5C6B76"
HEADER_BG = "#1B3A4B"
GROUP_BG = "#2E5670"
HEADER_FG = "#FFFFFF"
ZEBRA = "#F4F7F8"
LINE = "#D5DDE3"
POSITIVE = "#0E7A43"
NEGATIVE = "#B42318"
DASH = "\u2014"


def load_csv(path):
    rows = {}
    experiment = None
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            experiment = record["experiment"].strip()
            key = (record["run"].strip(), record["dataset"].strip(), record["task"].strip())
            rows[key] = record
    if experiment is None:
        raise SystemExit("No rows in {}".format(path))
    return experiment, rows


def short_label(path, experiment):
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    if "norm" in name:
        return "Normal"
    if "diff" in name:
        return "Diff"
    if stem in {"part2", "part_2"} or experiment == "part_2":
        return "Original"
    return experiment


def cell_value(rows, run, dataset, task, model):
    record = rows.get((run, dataset, task))
    if record is None:
        return None
    raw = record["{}_best".format(model)].strip()
    if not raw:
        return None
    return float(raw)


def format_score(value):
    if value is None:
        return DASH
    return "{:.2f}".format(value)


def format_delta(value):
    if value is None:
        return DASH
    sign = "+" if value > 0 else "\u2212" if value < 0 else ""
    return "{}{:.2f}".format(sign, abs(value))


def run_epoch(rows, run):
    for record in rows.values():
        if record["run"].strip() == run:
            return record["epoch_done"].strip()
    return None


def datasets_for(rows, task):
    found = []
    for dataset in DATASET_ORDER:
        if any(key[2] == task and key[1] == dataset for key in rows):
            found.append(dataset)
    return found


def only_run(task):
    return {"classification": "cls", "localization": "loc", "segmentation": "seg"}[task]


def synergy_section(rows, task, title, model):
    headers = ["Dataset", "Task only", "Cls + Loc", "All three", "All three \u2212 task only"]
    body = []
    for dataset in datasets_for(rows, task):
        only = cell_value(rows, only_run(task), dataset, task, model)
        paired = cell_value(rows, "cls_loc", dataset, task, model)
        together = cell_value(rows, "cls_loc_seg", dataset, task, model)
        delta = None if only is None or together is None else together - only
        body.append([
            DATASET_LABEL[dataset],
            format_score(only),
            format_score(paired),
            format_score(together),
            format_delta(delta),
        ])
    return {"title": title, "headers": headers, "rows": body, "groups": None, "deltas": {4}}


def epoch_note(rows):
    bits = []
    for run, label in RUN_EPOCH:
        done = run_epoch(rows, run)
        bits.append("{} {}".format(label, done if done else DASH))
    return "Latest epoch written: {}.".format(", ".join(bits))


def recipe_section(loaded, task, title, model):
    labels = [item[0] for item in loaded]
    base_label = labels[0]
    headers = ["Dataset"] + labels + labels + ["{} \u2212 {}".format(label, base_label) for label in labels[1:]]
    groups = [
        ("Task only", 1, len(labels)),
        ("All three", len(labels) + 1, 2 * len(labels)),
    ]
    if len(labels) > 1:
        groups.append(("All three \u2212 {}".format(base_label), 2 * len(labels) + 1, len(headers) - 1))
    body = []
    run = only_run(task)
    for dataset in datasets_for(loaded[0][2], task):
        only_scores = [cell_value(item[2], run, dataset, task, model) for item in loaded]
        all_scores = [cell_value(item[2], "cls_loc_seg", dataset, task, model) for item in loaded]
        deltas = []
        for score in all_scores[1:]:
            deltas.append(None if score is None or all_scores[0] is None else score - all_scores[0])
        body.append(
            [DATASET_LABEL[dataset]]
            + [format_score(score) for score in only_scores]
            + [format_score(score) for score in all_scores]
            + [format_delta(delta) for delta in deltas]
        )
    delta_columns = set(range(2 * len(labels) + 1, len(headers)))
    return {"title": title, "headers": headers, "rows": body, "groups": groups, "deltas": delta_columns}


def column_widths(headers, rows):
    widths = []
    for index, header in enumerate(headers):
        longest = len(header)
        for row in rows:
            longest = max(longest, len(row[index]))
        if index == 0:
            widths.append(max(2.15, longest * 0.105 + 0.42))
        else:
            widths.append(max(1.28, longest * 0.105 + 0.42))
    return widths


def cell_color(text, is_delta):
    if not is_delta or text in {DASH, "0.00"}:
        return INK, "normal"
    if text.startswith("+"):
        return POSITIVE, "bold"
    if text.startswith("\u2212"):
        return NEGATIVE, "bold"
    return INK, "normal"


def draw_header_cell(ax, x, y, width, height, text, align):
    ax.add_patch(Rectangle((x, y), width, height, facecolor=HEADER_BG, edgecolor="none"))
    ax.text(
        x + width - 0.14 if align == "right" else x + 0.14,
        y + height / 2,
        text,
        fontsize=11,
        color=HEADER_FG,
        fontweight="bold",
        va="center",
        ha=align,
    )


def draw_sections(path, title, subtitle, footnote, sections):
    margin = 0.28
    row_h = 0.46
    header_h = 0.50
    group_h = 0.36
    label_h = 0.34
    gap = 0.24
    title_h = 0.40
    subtitle_h = 0.46 if subtitle else 0
    footnote_h = 0.40 if footnote else 0.12
    prepared = []
    for section in sections:
        if not section["rows"]:
            continue
        widths = column_widths(section["headers"], section["rows"])
        prepared.append((section, widths))
    if not prepared:
        raise SystemExit("No rows to draw for {}".format(path))
    show_labels = len(prepared) > 1
    fig_w = max(sum(widths) for _section, widths in prepared) + margin * 2
    fig_h = 0.20 + title_h + subtitle_h + footnote_h
    for section, _widths in prepared:
        fig_h += header_h + len(section["rows"]) * row_h + gap
        if show_labels:
            fig_h += label_h
        if section["groups"]:
            fig_h += group_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y = fig_h - 0.16
    ax.text(margin, y, title, fontsize=15, fontweight="bold", color=INK, va="top", ha="left")
    y -= title_h
    if subtitle:
        ax.text(margin, y, subtitle, fontsize=10, color=MUTED, va="top", ha="left")
        y -= subtitle_h

    for section, widths in prepared:
        if show_labels:
            y -= label_h
            ax.text(margin, y + 0.06, section["title"], fontsize=12, fontweight="bold", color=INK, va="center", ha="left")
        if section["groups"]:
            y -= group_h
            for label, start, end in section["groups"]:
                x = margin + sum(widths[:start])
                span = sum(widths[start:end + 1])
                ax.add_patch(Rectangle((x, y), span, group_h, facecolor=GROUP_BG, edgecolor="none"))
                ax.text(x + span / 2, y + group_h / 2, label, fontsize=10, color=HEADER_FG, fontweight="bold", va="center", ha="center")
        y -= header_h
        x = margin
        for index, (width, header) in enumerate(zip(widths, section["headers"])):
            draw_header_cell(ax, x, y, width, header_h, header, "left" if index == 0 else "right")
            x += width
        for row_index, row in enumerate(section["rows"]):
            y -= row_h
            x = margin
            for col_index, (width, text) in enumerate(zip(widths, row)):
                if row_index % 2 == 1:
                    ax.add_patch(Rectangle((x, y), width, row_h, facecolor=ZEBRA, edgecolor="none", zorder=0))
                color, weight = cell_color(text, col_index in section["deltas"])
                ax.text(
                    x + width - 0.14 if col_index else x + 0.14,
                    y + row_h / 2,
                    text,
                    fontsize=11.5,
                    color=color,
                    fontweight=weight,
                    va="center",
                    ha="right" if col_index else "left",
                    zorder=2,
                )
                x += width
            ax.plot([margin, margin + sum(widths)], [y, y], color=LINE, linewidth=0.6, zorder=1)
        y -= gap

    if footnote:
        ax.text(margin, 0.14, footnote, fontsize=8.5, color=MUTED, va="bottom", ha="left")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def recipe_epochs(loaded, task):
    run = only_run(task)
    bits = []
    for label, _experiment, rows in loaded:
        bits.append("{} {}/{}".format(
            label,
            run_epoch(rows, run) or DASH,
            run_epoch(rows, "cls_loc_seg") or DASH,
        ))
    return "Epochs, task only / all three: {}.".format(", ".join(bits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs",
        nargs="*",
        help="Combined result CSVs. The first is the baseline. Defaults to original, normal, and diff.",
    )
    parser.add_argument("--model", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "Results", "tables"))
    args = parser.parse_args()
    paths = args.csvs or [
        os.path.join(REPO_ROOT, "Results", "csv", "part2.csv"),
        os.path.join(REPO_ROOT, "Results", "csv", "part2_norm.csv"),
        os.path.join(REPO_ROOT, "Results", "csv", "part2_diff.csv"),
    ]
    loaded = []
    for path in paths:
        experiment, rows = load_csv(path)
        loaded.append((short_label(path, experiment), experiment, rows))

    written = []
    for label, experiment, rows in loaded:
        sections = [synergy_section(rows, task, title, args.model) for task, title, _run in TASKS]
        out_path = os.path.join(args.out, "{}_synergy_{}.png".format(experiment, args.model))
        draw_sections(
            out_path,
            "Part 2 synergy \u2014 {}".format(label),
            "{} \u00b7 best focused {} \u00b7 one row per dataset".format(experiment, args.model),
            epoch_note(rows),
            sections,
        )
        written.append(out_path)

    if len(loaded) >= 2:
        later = " and ".join("{} ({})".format(label, experiment) for label, experiment, _rows in loaded[1:])
        base_label, base_experiment, _base_rows = loaded[0]
        subtitle = "{} are the later training runs. Baseline is {} ({}).".format(
            later, base_label, base_experiment,
        )
        for task, title, _run in TASKS:
            section = recipe_section(loaded, task, title, args.model)
            out_path = os.path.join(args.out, "part2_recipe_{}_{}.png".format(task, args.model))
            draw_sections(
                out_path,
                "Part 2 recipe \u2014 {}".format(title),
                subtitle,
                recipe_epochs(loaded, task) + " Deltas use the all-three runs.",
                [section],
            )
            written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
