"""Checks for the part-1 focused vs unfocused plotter."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).with_name("plot_part1_focused_unfocused.py")
spec = importlib.util.spec_from_file_location("plot_part1_focused_unfocused", SCRIPT)
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)

HEADER = ["Epoch", "Dataset", "Task-Train", "Model", "Task-Test", "AUC", "mAP40", "mAP50", "mAP50_95", "DICE"]
DATASETS = (
    ("CANDID-PTX", "CANDIDPTX"),
    ("ChestX-Det", "ChestXDet"),
    ("SIIM-ACRpneumothorax", "SIIMACR"),
)
TASKS = (
    ("CLS", "Classification"),
    ("LOC", "Localization"),
    ("SEG", "Segmentation"),
)


def _metric_cells(eval_task, score):
    if eval_task == "CLS":
        return [score, "-", "-", "-", "-"]
    if eval_task == "LOC":
        # mAP40, mAP50, mAP50:95 sit a fixed offset apart so each column is checkable.
        return ["-", score - 3, score, score - 7, "-"]
    return ["-", "-", "-", "-", score]


def _score(train_ds, train_task, eval_ds, eval_task, model, cycle):
    if eval_ds != train_ds:
        return 10.0
    teacher = 1.5 if model == "Teacher" else 0.0
    if train_task == eval_task:
        return 60.0 + 5.0 * cycle + teacher
    return 40.0 + cycle + teacher


def write_swapped_log(path: Path) -> None:
    """Write a log in the trainer's column order, Dataset holding task_todo."""
    rows = [HEADER, ["Epoch", "Dataset", "Task-Train", "Model", "Task-Test", "AUC", "mAP40", "mAP50", "mAP50_95", "DICE"]]
    epoch = 1
    for cycle in (1, 2):
        for eval_name, token in DATASETS:
            for task_code, task_word in TASKS:
                task_todo = f"{task_word}_{token}_B_Train"
                for other_name, _other_token in DATASETS:
                    for eval_code, eval_word in TASKS:
                        for model in ("Student", "Teacher"):
                            # Cross-dataset evals stay in the file and must not enter the curves.
                            score = _score(eval_name, task_code, other_name, eval_code, model, cycle)
                            cells = _metric_cells(eval_code, score)
                            rows.append(
                                [epoch, task_todo, other_name, model, f"{eval_word}_{other_name}", *cells]
                            )
                epoch += 1
    # Same epoch as the first CANDID classification step, written earlier and lower.
    # The later row already in the file is the one the curve should keep.
    rows.insert(
        2,
        [1, "Classification_CANDIDPTX_A_Train", "CANDID-PTX", "Student", "Classification_CANDID-PTX", 1.0, "-", "-", "-", "-"],
    )
    # Training a different dataset must not count as CANDID "trained on localization".
    rows.append(
        [99, "Localization_NIHChestXray14_B_Train", "CANDID-PTX", "Student", "Classification_CANDID-PTX", 99.0, "-", "-", "-", "-"]
    )
    rows.append([3, "None", "CANDID-PTX", "Student", "Classification_CANDID-PTX", 99.0, "-", "-", "-", "-"])
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(path, index=False)


class PlotPart1Test(unittest.TestCase):
    def test_swapped_log_plots_and_best_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "export_csvFile.csv"
            write_swapped_log(csv_path)
            out = root / "plots"
            summary = plotter.plot_part1(csv_path, out, "run104", "mAP50")

            self.assertEqual(len(summary["plots"]), 18)
            self.assertEqual(summary["skipped"], [])
            self.assertTrue(summary["best_csv"].is_file())
            for path in summary["plots"]:
                self.assertGreater(path.stat().st_size, 1000)

            table = pd.read_csv(summary["best_csv"])
            candid_cls_student = table[
                (table["dataset"] == "CANDID-PTX")
                & (table["eval_task"] == "CLS")
                & (table["model"] == "Student")
                & (table["metric"] == "AUC")
            ]
            focused = candid_cls_student[candid_cls_student["condition"] == "focused"].iloc[0]
            self.assertEqual(focused["train_task"], "CLS")
            self.assertEqual(focused["best_score"], 70.0)
            self.assertEqual(int(focused["best_epoch"]), 10)
            self.assertEqual(int(focused["n_points"]), 2)
            self.assertTrue(bool(focused["on_plot"]))

            unfocused_loc = candid_cls_student[candid_cls_student["train_task"] == "LOC"].iloc[0]
            self.assertEqual(unfocused_loc["condition"], "unfocused")
            self.assertEqual(unfocused_loc["best_score"], 42.0)

            teacher = table[
                (table["dataset"] == "CANDID-PTX")
                & (table["eval_task"] == "CLS")
                & (table["train_task"] == "CLS")
                & (table["model"] == "Teacher")
                & (table["metric"] == "AUC")
            ].iloc[0]
            self.assertEqual(teacher["best_score"], 71.5)

            loc = table[
                (table["dataset"] == "SIIM-ACR")
                & (table["eval_task"] == "LOC")
                & (table["train_task"] == "LOC")
                & (table["model"] == "Student")
            ]
            by_metric = dict(zip(loc["metric"], loc["best_score"]))
            self.assertEqual(by_metric["mAP50"], 70.0)
            self.assertEqual(by_metric["mAP40"], 67.0)
            self.assertEqual(by_metric["mAP50_95"], 63.0)
            self.assertTrue(bool(loc.loc[loc["metric"] == "mAP50", "on_plot"].iloc[0]))
            self.assertFalse(bool(loc.loc[loc["metric"] == "mAP40", "on_plot"].iloc[0]))

            seg = table[
                (table["dataset"] == "ChestX-Det")
                & (table["eval_task"] == "SEG")
                & (table["train_task"] == "CLS")
                & (table["model"] == "Teacher")
                & (table["metric"] == "DICE")
            ].iloc[0]
            self.assertEqual(seg["condition"], "unfocused")
            self.assertEqual(seg["best_score"], 43.5)

            names = {path.name for path in summary["plots"]}
            self.assertIn("candidptx__cls_focused_vs_trained_on_loc.png", names)
            self.assertIn("candidptx__cls_focused_vs_trained_on_seg.png", names)
            self.assertIn("siimacr__loc_focused_vs_trained_on_cls.png", names)
            self.assertIn("chestxdet__seg_focused_vs_trained_on_loc.png", names)

    def test_header_aligned_csv_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export_csvFile.csv"
            pd.DataFrame(
                [
                    ["CANDID-PTX", "Classification_CANDIDPTX_B_Train", "Student", "Classification_CANDID-PTX", 80, "-", "-", "-", "-"],
                    ["SIIM-ACRpneumothorax", "Segmentation_SIIMACR_A_Train", "Teacher", "Segmentation_SIIM-ACRpneumothorax", "-", "-", "-", "-", 55],
                ],
                columns=["Dataset", "Task-Train", "Model", "Task-Test", "AUC", "mAP40", "mAP50", "mAP50_95", "DICE"],
            ).assign(Epoch=[4, 9])[
                ["Epoch", "Dataset", "Task-Train", "Model", "Task-Test", "AUC", "mAP40", "mAP50", "mAP50_95", "DICE"]
            ].to_csv(path, index=False)

            tidy = plotter.load_results(path)
            self.assertEqual(list(tidy["train_dataset"]), ["CANDID-PTX", "SIIM-ACR"])
            self.assertEqual(list(tidy["eval_dataset"]), ["CANDID-PTX", "SIIM-ACR"])
            self.assertEqual(list(tidy["train_task"]), ["CLS", "SEG"])
            self.assertEqual(list(tidy["eval_task"]), ["CLS", "SEG"])
            self.assertEqual(list(tidy["model"]), ["Student", "Teacher"])
            self.assertEqual(tidy.loc[0, "auc"], 80)
            self.assertEqual(tidy.loc[1, "dice"], 55)

    def test_cli_list_runs_and_folder(self):
        self.assertEqual(plotter.main(["--list-runs"]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "export_csvFile.csv"
            write_swapped_log(csv_path)
            out = root / "from-cli"
            code = plotter.main(["--folder", str(root), "--out", str(out), "--loc-metric", "mAP40"])
            self.assertEqual(code, 0)
            table = pd.read_csv(out / "best_scores_teacher_student.csv")
            plotted = table[(table["eval_task"] == "LOC") & (table["on_plot"] == True)]  # noqa: E712
            self.assertGreater(len(plotted), 0)
            self.assertTrue((plotted["metric"] == "mAP40").all())
            self.assertTrue((out / "candidptx__cls_focused_vs_trained_on_loc.png").is_file())


if __name__ == "__main__":
    unittest.main()
