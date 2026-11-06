#!/usr/bin/env python3
"""Draw the Part 1 synergy and recipe tables as slide images.

Reads the combined-results CSVs (one row per run, dataset, and task).
Each CSV becomes one synergy table: CANDID-PTX trained alone, in each pair,
and as classification + localization + segmentation. The normal and updated
CSVs also produce the recipe comparison. Defaults are Results/csv/part1.csv
(original runs), part1_norm.csv, and part1_diff.csv.

Scores are the best focused value already stored in the CSV. Localization
is mAP40. Images are written to Results/tables/.
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

TASKS = (
    ("classification", "Classification (AUC)"),
    ("localization", "Localization (mAP40)"),
    ("segmentation", "Segmentation (Dice)"),
)
SETUPS = (
    ("alone", "Alone"),
    ("cls_loc", "Cls + Loc"),
    ("cls_seg", "Cls + Seg"),
    ("loc_seg", "Loc + Seg"),
    ("all", "All three"),
)
RUN_FOR = {
    ("classification", "alone"): "candidptx_cls",
    ("localization", "alone"): "candidptx_loc",
    ("segmentation", "alone"): "candidptx_seg",
    ("classification", "cls_loc"): "candidptx_cls_loc",
    ("localization", "cls_loc"): "candidptx_cls_loc",
    ("classification", "cls_seg"): "candidptx_cls_seg",
    ("segmentation", "cls_seg"): "candidptx_cls_seg",
    ("localization", "loc_seg"): "candidptx_loc_seg",
    ("segmentation", "loc_seg"): "candidptx_loc_seg",
    ("classification", "all"): "candidptx_cls_loc_seg",
    ("localization", "all"): "candidptx_cls_loc_seg",
    ("segmentation", "all"): "candidptx_cls_loc_seg",
}

INK = "#14202B"
MUTED = "#5C6B76"
HEADER_BG = "#1B3A4B"
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
            rows[(record["run"].strip(), record["task"].strip())] = record
    if experiment is None:
        raise SystemExit("No rows in {}".format(path))
    return experiment, rows


def cell_value(rows, task, setup, model):
    run = RUN_FOR.get((task, setup))
    if run is None:
        return None
    record = rows.get((run, task))
    if record is None:
        return None
    raw = record["{}_best".format(model)].strip()
    if not raw:
        return None
    return float(raw)


def epoch_done(rows, task, setup):
    run = RUN_FOR.get((task, setup))
    record = rows.get((run, task)) if run else None
    if record is None:
        return None
    return record["epoch_done"].strip()


def format_score(value):
    if value is None:
        return DASH
    return "{:.2f}".format(value)


def format_delta(value):
    if value is None:
        return DASH
    sign = "+" if value > 0 else "\u2212" if value < 0 else ""
    return "{}{:.2f}".format(sign, abs(value))


def short_label(path, experiment):
    name = os.path.basename(path).lower()
    stem = os.path.splitext(name)[0]
    if "norm" in name:
        return "Normal"
    if "diff" in name:
        return "Updated"
    if stem in {"part1", "part_1"} or experiment == "part_1":
        return "Original"
    return experiment


def synergy_rows(rows, model):
    body = []
    for task, label in TASKS:
        values = [cell_value(rows, task, setup, model) for setup, _title in SETUPS]
        alone, all_three = values[0], values[-1]
        delta = None if alone is None or all_three is None else all_three - alone
        body.append([label] + [format_score(value) for value in values] + [format_delta(delta)])
    return body


def epoch_note(rows):
    bits = []
    for task, label in TASKS:
        done = epoch_done(rows, task, "alone")
        short = label.split(" ", 1)[0].lower()
        bits.append("{} {}".format(short, done if done else DASH))
    all_done = epoch_done(rows, "classification", "all")
    return "Latest epoch written: {} alone; all three {}.".format(
        ", ".join(bits), all_done if all_done else DASH,
    )


def recipe_rows(left_rows, right_rows, model):
    body = []
    for task, label in TASKS:
        left_alone = cell_value(left_rows, task, "alone", model)
        right_alone = cell_value(right_rows, task, "alone", model)
        left_all = cell_value(left_rows, task, "all", model)
        right_all = cell_value(right_rows, task, "all", model)
        delta = None if left_all is None or right_all is None else right_all - left_all
        body.append([
            label,
            format_score(left_alone),
            format_score(right_alone),
            format_score(left_all),
            format_score(right_all),
            format_delta(delta),
        ])
    return body


def column_widths(headers, rows):
    widths = []
    for index, header in enumerate(headers):
        longest = len(header)
        for row in rows:
            longest = max(longest, len(row[index]))
        if index == 0:
            widths.append(max(2.55, longest * 0.105 + 0.42))
        else:
            widths.append(max(1.35, longest * 0.112 + 0.48))
    return widths


def draw_table(path, title, subtitle, footnote, headers, rows):
    widths = column_widths(headers, rows)
    margin = 0.28
    row_h = 0.50
    header_h = 0.56
    title_h = 0.42
    subtitle_h = 0.32 if subtitle else 0
    footnote_h = 0.36 if footnote else 0.12
    fig_w = sum(widths) + margin * 2
    fig_h = 0.22 + title_h + subtitle_h + 0.16 + header_h + len(rows) * row_h + footnote_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    y = fig_h - 0.18
    ax.text(margin, y, title, fontsize=15, fontweight="bold", color=INK, va="top", ha="left")
    y -= title_h
    if subtitle:
        ax.text(margin, y, subtitle, fontsize=10, color=MUTED, va="top", ha="left")
        y -= subtitle_h
    y -= 0.16
    y -= header_h
    x = margin
    for width, header in zip(widths, headers):
        ax.add_patch(Rectangle((x, y), width, header_h, facecolor=HEADER_BG, edgecolor="none"))
        ax.text(
            x + width - 0.16 if header != headers[0] else x + 0.16,
            y + header_h / 2,
            header,
            fontsize=11,
            color=HEADER_FG,
            fontweight="bold",
            va="center",
            ha="right" if header != headers[0] else "left",
        )
        x += width

    for row_index, row in enumerate(rows):
        y -= row_h
        x = margin
        for col_index, (width, text) in enumerate(zip(widths, row)):
            if row_index % 2 == 1:
                ax.add_patch(Rectangle((x, y), width, row_h, facecolor=ZEBRA, edgecolor="none", zorder=0))
            color = INK
            weight = "normal"
            if col_index == len(row) - 1 and text not in {DASH, "0.00"}:
                if text.startswith("+"):
                    color = POSITIVE
                    weight = "bold"
                elif text.startswith("\u2212"):
                    color = NEGATIVE
                    weight = "bold"
            ax.text(
                x + width - 0.16 if col_index else x + 0.16,
                y + row_h / 2,
                text,
                fontsize=12,
                color=color,
                fontweight=weight,
                va="center",
                ha="right" if col_index else "left",
                zorder=2,
            )
            x += width
        ax.plot([margin, margin + sum(widths)], [y, y], color=LINE, linewidth=0.6, zorder=1)

    if footnote:
        ax.text(margin, 0.16, footnote, fontsize=8.5, color=MUTED, va="bottom", ha="left")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs",
        nargs="*",
        help="Combined result CSVs. Defaults to the original, normal, and updated Part 1 files.",
    )
    parser.add_argument(
        "--model",
        choices=("teacher", "student"),
        default="teacher",
        help="Which best score to print. Default: teacher.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "Results", "tables"),
        help="Directory for the PNG files.",
    )
    args = parser.parse_args()
    paths = args.csvs or [
        os.path.join(REPO_ROOT, "Results", "csv", "part1.csv"),
        os.path.join(REPO_ROOT, "Results", "csv", "part1_norm.csv"),
        os.path.join(REPO_ROOT, "Results", "csv", "part1_diff.csv"),
    ]
    loaded = []
    for path in paths:
        experiment, rows = load_csv(path)
        loaded.append((path, experiment, short_label(path, experiment), rows))

    model_name = args.model.capitalize()
    written = []
    for _path, experiment, label, rows in loaded:
        headers = ["Task"] + [title for _setup, title in SETUPS] + ["All three \u2212 alone"]
        out_path = os.path.join(args.out, "{}_synergy_{}.png".format(experiment, args.model))
        draw_table(
            out_path,
            "Part 1 synergy \u2014 {}".format(label),
            "{} \u00b7 best focused {} \u00b7 CANDID-PTX".format(experiment, args.model),
            epoch_note(rows),
            headers,
            synergy_rows(rows, args.model),
        )
        written.append(out_path)

    by_label = {item[2]: item for item in loaded}
    pair = None
    if "Normal" in by_label and "Updated" in by_label:
        pair = (by_label["Normal"], by_label["Updated"])
    elif len(loaded) == 2:
        pair = (loaded[0], loaded[1])
    if pair is not None:
        (_left_path, left_experiment, left_label, left_rows) = pair[0]
        (_right_path, right_experiment, right_label, _right_rows) = pair[1]
        headers = [
            "Task",
            "{}, alone".format(left_label),
            "{}, alone".format(right_label),
            "{}, all three".format(left_label),
            "{}, all three".format(right_label),
            "{} \u2212 {}".format(right_label, left_label),
        ]
        out_path = os.path.join(args.out, "part1_recipe_{}.png".format(args.model))
        draw_table(
            out_path,
            "Training recipe \u2014 {}".format(model_name),
            "{} ({}) minus {} ({}) on the all-three run".format(
                right_label, right_experiment, left_label, left_experiment,
            ),
            "Best focused {} score. Classification is AUC, localization is mAP40, segmentation is mean Dice.".format(args.model),
            headers,
            recipe_rows(left_rows, _right_rows, args.model),
        )
        written.append(out_path)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
