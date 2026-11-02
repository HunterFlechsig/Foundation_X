#!/usr/bin/env python3
"""Part 1 focused vs unfocused curves for Foundation X+ pretraining logs.

Part 1 is the datasets that carry classification, localization, and segmentation
together: CANDID-PTX, ChestX-Det, and SIIM-ACR.

For each of those datasets and each evaluated task, this writes one figure per
other task. The focused line is the score on the evaluated task at the epochs
that task was trained. The unfocused line is the same score at the epochs the
other task was trained. Student and teacher are separate panels on each figure.

It also writes ``best_scores_teacher_student.csv`` with the best score of each
curve.

The trainer appends rows as
``[epoch, task_todo, eval_dataset, model, task_test, AUC, mAP40, mAP50, mAP50_95, DICE]``
under the header
``Epoch, Dataset, Task-Train, Model, Task-Test, ...``,
so the Dataset and Task-Train values are swapped relative to the header. The
loader detects that layout. A file whose columns already match the header is
accepted too.

Pick a results folder with ``--run`` or pass any directory with ``--folder``.

Examples
--------
python scripts/plot_part1_focused_unfocused.py --list-runs
python scripts/plot_part1_focused_unfocused.py --run run104
python scripts/plot_part1_focused_unfocused.py --folder /path/to/one/run
python scripts/plot_part1_focused_unfocused.py --run run101 --loc-metric mAP40
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Output directories of the three full pretraining runs. ``--folder`` accepts
# any other results directory that contains export_csvFile.csv.
RUNS = {
    "run104": {
        "path": (
            "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/"
            "IntegratedModel_FoundationX3/"
            "run104_Ark6F6_ClsLocSeg_b24_AdamW_LockReleaseAll_RCons_1LocDec"
        ),
        "description": "Swin-B lock-release run (v106)",
    },
    "run105": {
        "path": (
            "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/"
            "IntegratedModel_FoundationX3/"
            "run105_Ark6F6_ClsLocSeg_b24_AdamW_LockReleaseAll_RCons_1LocDec_SwinL768_v3"
        ),
        "description": "Swin-L 768 v3 lock-release run (v106 Swin-L)",
    },
    "run101": {
        "path": (
            "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/"
            "IntegratedModel_FoundationX3/run101_ClsLocSeg_SwinL768"
        ),
        "description": "Swin-L 768 lock-release run (v107 Swin-L)",
    },
}

PART1_DATASETS = ("CANDID-PTX", "ChestX-Det", "SIIM-ACR")
TASKS = ("CLS", "LOC", "SEG")
TASK_FROM_KIND = {
    "classification": "CLS",
    "localization": "LOC",
    "segmentation": "SEG",
}
TASK_LABEL = {
    "CLS": "classification",
    "LOC": "localization",
    "SEG": "segmentation",
}
# Keys are letters and digits only, uppercased. Eval names such as
# "SIIM-ACRpneumothorax" and train tokens such as "SIIMACR" meet here.
DATASET_FROM_KEY = {
    "CANDIDPTX": "CANDID-PTX",
    "CHESTXDET": "ChestX-Det",
    "SIIMACR": "SIIM-ACR",
    "SIIMACRPNEUMOTHORAX": "SIIM-ACR",
}
MODELS = ("Student", "Teacher")
LOC_METRICS = ("mAP40", "mAP50", "mAP50_95")
PRIMARY_METRIC = {"CLS": "AUC", "LOC": "mAP50", "SEG": "DICE"}
METRIC_COLUMN = {
    "AUC": "auc",
    "mAP40": "map40",
    "mAP50": "map50",
    "mAP50_95": "map5095",
    "DICE": "dice",
}
METRIC_AXIS = {
    "AUC": "AUC (%)",
    "mAP40": "mAP40 (%)",
    "mAP50": "mAP50 (%)",
    "mAP50_95": "mAP50:95 (%)",
    "DICE": "Mean Dice (%)",
}

_TRAIN_RE = re.compile(
    r"^(?P<kind>Classification|Localization|Segmentation)_(?P<token>.+?)(?:_[AB])?_Train$",
    re.IGNORECASE,
)
_EVAL_RE = re.compile(
    r"^(?P<kind>Classification|Localization|Segmentation)_(?P<token>.+)$",
    re.IGNORECASE,
)
_HEADER_ALIASES = {
    "epoch": "Epoch",
    "dataset": "Dataset",
    "task-train": "Task-Train",
    "task_train": "Task-Train",
    "model": "Model",
    "task-test": "Task-Test",
    "task_test": "Task-Test",
    "auc": "AUC",
    "map40": "mAP40",
    "map50": "mAP50",
    "map50_95": "mAP50_95",
    "map50-95": "mAP50_95",
    "map5095": "mAP50_95",
    "dice": "DICE",
}
_REQUIRED = ("Epoch", "Dataset", "Task-Train", "Model", "Task-Test", "AUC", "mAP40", "mAP50", "mAP50_95", "DICE")

FOCUSED_COLOR = "#0B6E4F"
UNFOCUSED_COLOR = "#C44900"


def canonical_dataset(token: object) -> str | None:
    """Map a train token or an eval dataset name onto a part-1 dataset name."""
    if token is None or (isinstance(token, float) and np.isnan(token)):
        return None
    key = re.sub(r"[^A-Za-z0-9]", "", str(token)).upper()
    return DATASET_FROM_KEY.get(key)


def _normalize_header(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        alias = _HEADER_ALIASES.get(str(column).strip().lower())
        if alias is not None:
            renamed[column] = alias
    frame = frame.rename(columns=renamed)
    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(
            "export csv is missing columns: " + ", ".join(missing)
        )
    return frame


def _columns_are_swapped(dataset_values: pd.Series) -> bool:
    """True when the Dataset column holds task_todo values such as ``..._Train``."""
    text = dataset_values.dropna().astype(str).str.strip()
    text = text[text.ne("") & text.str.lower().ne("none") & text.str.lower().ne("nan")]
    if text.empty:
        raise ValueError("export csv has a header but no result rows")
    return bool(text.str.contains(r"_Train$", case=False, regex=True).mean() >= 0.5)


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace({"-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load one run's export csv into one row per logged evaluation."""
    frame = pd.read_csv(csv_path)
    frame = _normalize_header(frame)
    epoch_text = frame["Epoch"].astype(str).str.strip()
    frame = frame[epoch_text.str.fullmatch(r"-?\d+")].copy()
    if frame.empty:
        raise ValueError(f"no numeric epochs in {csv_path}")

    swapped = _columns_are_swapped(frame["Dataset"])
    if swapped:
        train_raw = frame["Dataset"]
        eval_raw = frame["Task-Train"]
    else:
        train_raw = frame["Task-Train"]
        eval_raw = frame["Dataset"]

    train_parts = train_raw.astype(str).str.strip().str.extract(_TRAIN_RE)
    eval_parts = frame["Task-Test"].astype(str).str.strip().str.extract(_EVAL_RE)
    eval_from_task = eval_parts["token"].map(canonical_dataset)
    eval_from_column = eval_raw.map(canonical_dataset)
    eval_dataset = eval_from_task.where(eval_from_task.notna(), eval_from_column)

    tidy = pd.DataFrame(
        {
            "epoch": frame["Epoch"].astype(int).to_numpy(),
            "model": frame["Model"].astype(str).str.strip().str.title().to_numpy(),
            "train_task": train_parts["kind"].str.lower().map(TASK_FROM_KIND).to_numpy(),
            "train_dataset": train_parts["token"].map(canonical_dataset).to_numpy(),
            "eval_task": eval_parts["kind"].str.lower().map(TASK_FROM_KIND).to_numpy(),
            "eval_dataset": eval_dataset.to_numpy(),
            "auc": _numeric(frame["AUC"]).to_numpy(),
            "map40": _numeric(frame["mAP40"]).to_numpy(),
            "map50": _numeric(frame["mAP50"]).to_numpy(),
            "map5095": _numeric(frame["mAP50_95"]).to_numpy(),
            "dice": _numeric(frame["DICE"]).to_numpy(),
            "_row": np.arange(len(frame)),
        }
    )
    return tidy.dropna(subset=["train_task", "eval_task"]).reset_index(drop=True)


def _series(frame: pd.DataFrame, dataset: str, eval_task: str, train_task: str, model: str) -> pd.DataFrame:
    """One curve: this dataset's eval task, measured on epochs trained on train_task."""
    mask = (
        (frame["eval_dataset"] == dataset)
        & (frame["train_dataset"] == dataset)
        & (frame["eval_task"] == eval_task)
        & (frame["train_task"] == train_task)
        & (frame["model"] == model)
    )
    sub = frame.loc[mask].sort_values(["epoch", "_row"])
    # Lock then release can leave two rows for one epoch. Keep the later row.
    return sub.groupby("epoch", as_index=False).tail(1)


def _best(sub: pd.DataFrame, column: str) -> tuple[float, int] | None:
    values = sub[column].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    position = int(np.nanargmax(values))
    return float(values[position]), int(sub["epoch"].iloc[position])


def best_score_table(frame: pd.DataFrame, run_name: str, loc_metric: str) -> pd.DataFrame:
    """Best teacher and student score on every part-1 curve."""
    rows = []
    for dataset in PART1_DATASETS:
        for eval_task in TASKS:
            for train_task in TASKS:
                condition = "focused" if train_task == eval_task else "unfocused"
                for model in MODELS:
                    sub = _series(frame, dataset, eval_task, train_task, model)
                    if sub.empty:
                        continue
                    for metric, column in METRIC_COLUMN.items():
                        found = _best(sub, column)
                        if found is None:
                            continue
                        score, epoch = found
                        plotted_metric = loc_metric if eval_task == "LOC" else PRIMARY_METRIC[eval_task]
                        rows.append(
                            {
                                "run": run_name,
                                "dataset": dataset,
                                "eval_task": eval_task,
                                "train_task": train_task,
                                "condition": condition,
                                "model": model,
                                "metric": metric,
                                "best_score": round(score, 5),
                                "best_epoch": epoch,
                                "n_points": int(np.isfinite(sub[column].to_numpy(dtype=float)).sum()),
                                "on_plot": metric == plotted_metric,
                            }
                        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["dataset", "eval_task", "train_task", "model", "metric"]
    ).reset_index(drop=True)


def _ylim(series_list: list[np.ndarray]) -> tuple[float, float] | None:
    finite = [values[np.isfinite(values)] for values in series_list]
    finite = [values for values in finite if values.size]
    if not finite:
        return None
    joined = np.concatenate(finite)
    low = float(joined.min())
    high = float(joined.max())
    if high == low:
        pad = 1.0
    else:
        pad = 0.08 * (high - low)
    return low - pad, high + pad


def _draw_curve(axis, epochs: np.ndarray, scores: np.ndarray, *, focused: bool, label: str) -> None:
    axis.plot(
        epochs,
        scores,
        color=FOCUSED_COLOR if focused else UNFOCUSED_COLOR,
        linestyle="-" if focused else "--",
        marker="o" if focused else "s",
        markersize=4.5,
        linewidth=1.8,
        label=label,
    )


def plot_comparison(
    frame: pd.DataFrame,
    dataset: str,
    eval_task: str,
    other_task: str,
    loc_metric: str,
    out_path: Path,
    run_name: str,
) -> bool:
    """Save one focused-vs-unfocused figure. Returns False when both curves are empty."""
    metric = loc_metric if eval_task == "LOC" else PRIMARY_METRIC[eval_task]
    column = METRIC_COLUMN[metric]
    panels = []
    for model in MODELS:
        focused = _series(frame, dataset, eval_task, eval_task, model)
        unfocused = _series(frame, dataset, eval_task, other_task, model)
        panels.append((model, focused, unfocused))
    if all(focused.empty and unfocused.empty for _, focused, unfocused in panels):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    collected = []
    for axis, (model, focused, unfocused) in zip(axes, panels):
        if not focused.empty:
            scores = focused[column].to_numpy(dtype=float)
            collected.append(scores)
            _draw_curve(
                axis,
                focused["epoch"].to_numpy(),
                scores,
                focused=True,
                label=f"Focused (trained on {eval_task})",
            )
        if not unfocused.empty:
            scores = unfocused[column].to_numpy(dtype=float)
            collected.append(scores)
            _draw_curve(
                axis,
                unfocused["epoch"].to_numpy(),
                scores,
                focused=False,
                label=f"Unfocused (trained on {other_task})",
            )
        if focused.empty and unfocused.empty:
            axis.text(0.5, 0.5, "No logged points", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(model)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)

    handles = []
    labels = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    y_limits = _ylim(collected)
    if y_limits is not None:
        axes[0].set_ylim(*y_limits)
    axes[0].set_ylabel(METRIC_AXIS[metric])
    fig.suptitle(
        f"{dataset} {TASK_LABEL[eval_task]} — focused vs trained on {TASK_LABEL[other_task]}"
        + (f"  ({run_name})" if run_name else ""),
        fontsize=12,
    )
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def comparisons():
    """Each evaluated task paired with each of the other two training tasks."""
    for dataset in PART1_DATASETS:
        for eval_task in TASKS:
            for other_task in TASKS:
                if other_task == eval_task:
                    continue
                yield dataset, eval_task, other_task


def slug(dataset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", dataset.lower())


def plot_part1(csv_path: Path, out_dir: Path, run_name: str, loc_metric: str) -> dict:
    """Write the part-1 figures and the best-score csv. Returns a short summary."""
    if loc_metric not in LOC_METRICS:
        raise ValueError(f"loc_metric must be one of {LOC_METRICS}")
    frame = load_results(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    skipped = []
    for dataset, eval_task, other_task in comparisons():
        filename = f"{slug(dataset)}__{eval_task.lower()}_focused_vs_trained_on_{other_task.lower()}.png"
        path = out_dir / filename
        if plot_comparison(frame, dataset, eval_task, other_task, loc_metric, path, run_name):
            written.append(path)
        else:
            skipped.append(f"{dataset} {eval_task} vs {other_task}")

    table = best_score_table(frame, run_name, loc_metric)
    csv_out = out_dir / "best_scores_teacher_student.csv"
    table.to_csv(csv_out, index=False)
    return {
        "rows": int(len(frame)),
        "plots": written,
        "skipped": skipped,
        "best_csv": csv_out,
        "best_rows": int(len(table)),
    }


def locate_csv(folder: Path, csv_name: str) -> Path:
    if folder.is_file():
        return folder
    candidate = folder / csv_name
    if not candidate.is_file():
        raise SystemExit(f"No {csv_name} in {folder}")
    return candidate


def default_out_dir(source: Path) -> Path:
    if source.is_file():
        return source.parent / "plots" / "part1"
    return source / "plots" / "part1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot part-1 focused vs unfocused curves and write the best teacher and student scores.",
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--run",
        choices=tuple(RUNS),
        help="Which of the three pretraining result folders to read.",
    )
    choice.add_argument(
        "--folder",
        type=Path,
        help="Results directory (or the export csv itself) to analyze instead of a named run.",
    )
    choice.add_argument(
        "--list-runs",
        action="store_true",
        help="Print the three named result folders and exit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write figures and the best-score csv. Default: <folder>/plots/part1.",
    )
    parser.add_argument(
        "--csv-name",
        default="export_csvFile.csv",
        help="Csv filename inside the results folder.",
    )
    parser.add_argument(
        "--loc-metric",
        choices=LOC_METRICS,
        default="mAP50",
        help="Localization score drawn on the curves. The csv still records mAP40, mAP50, and mAP50:95.",
    )
    return parser


def print_runs() -> None:
    for name, info in RUNS.items():
        print(f"{name}: {info['description']}")
        print(f"  {info['path']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_runs:
        print_runs()
        return 0

    if args.run is not None:
        source = Path(RUNS[args.run]["path"])
        run_name = args.run
    else:
        source = args.folder.expanduser().resolve()
        run_name = source.parent.name if source.is_file() else source.name

    csv_path = locate_csv(source, args.csv_name)
    out_dir = args.out if args.out is not None else default_out_dir(source if source.is_dir() else source.parent)
    summary = plot_part1(csv_path, out_dir, run_name, args.loc_metric)

    print(f"Read {summary['rows']} evaluation rows from {csv_path}")
    print(f"Wrote {len(summary['plots'])} figures to {out_dir}")
    for path in summary["plots"]:
        print(f"  {path.name}")
    if summary["skipped"]:
        print("Skipped comparisons with no part-1 points:")
        for name in summary["skipped"]:
            print(f"  {name}")
    print(f"Best teacher and student scores: {summary['best_csv']} ({summary['best_rows']} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
