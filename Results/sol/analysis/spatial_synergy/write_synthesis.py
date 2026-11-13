"""Combine the Grad-CAM and gradient summaries into one note.

Run after both SOL jobs. Reads their JSON summaries and does not load a model.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRADCAM_JSON = ROOT / "gradcam_synergy" / "gradcam_summary.json"
GRADIENT_JSON = ROOT / "gradient_interaction" / "gradient_summary.json"
OUTPUT = Path(__file__).resolve().parent / "SPATIAL_SYNERGY_SYNTHESIS.md"


def bucket(row):
    mean = float(row["mean"])
    positive = float(row["fraction_positive"])
    negative = float(row["fraction_negative"])
    if mean > 0.1 and positive >= 0.7:
        return "aligned"
    if mean < -0.1 and negative >= 0.7:
        return "conflicting"
    if abs(mean) <= 0.1:
        return "neutral"
    return "mixed"


def entire_rows(payload):
    rows = []
    for row in payload["rows"]:
        if row["stage"] == "entire_backbone":
            rows.append(row)
    return rows


def find_row(rows, exposure, left, right):
    for row in rows:
        if int(row["exposure"]) == exposure and row["left"] == left and row["right"] == right:
            return row
    raise SystemExit("Missing gradient row {} {} {}".format(exposure, left, right))


def rank_pair(name, value):
    if value == "aligned":
        return "SUGGESTIVE", "{} gradients are generally aligned on the shared backbone.".format(name)
    if value == "conflicting":
        return "SUGGESTIVE", "{} gradients are generally conflicting on the shared backbone.".format(name)
    if value == "neutral":
        return "SUGGESTIVE", "{} gradients are approximately neutral on the shared backbone.".format(name)
    return "NOT SUPPORTED", "{} gradients do not show one direction. The batches are mixed.".format(name)


def main():
    missing = [str(path) for path in (GRADCAM_JSON, GRADIENT_JSON) if not path.is_file()]
    if missing:
        raise SystemExit("Run both analyses first. Missing:\n" + "\n".join(missing))
    gradcam = json.loads(GRADCAM_JSON.read_text())
    gradient = json.loads(GRADIENT_JSON.read_text())
    rows = entire_rows(gradient)
    early_loc = bucket(find_row(rows, 10, "cls", "loc"))
    early_seg = bucket(find_row(rows, 10, "cls", "seg"))
    late_loc = bucket(find_row(rows, 60, "cls", "loc"))
    late_seg = bucket(find_row(rows, 60, "cls", "seg"))
    early_pair = bucket(find_row(rows, 10, "loc", "seg"))
    late_pair = bucket(find_row(rows, 60, "loc", "seg"))

    concentration = gradcam["concentration_interpretation"]
    exposure10_suggestive = "Exposure 10:" in concentration and "is SUGGESTIVE" in concentration
    # The sentence form is "Exposure 10: CLS + LOC versus CLS is SUGGESTIVE."
    gradcam_any = "is SUGGESTIVE" in concentration or "SUGGESTIVE." in gradcam["ground_truth_interpretation"]

    loc_rank, loc_text = rank_pair("Classification and localization", early_loc if early_loc == late_loc else "mixed")
    if early_loc != late_loc:
        loc_rank = "NOT SUPPORTED"
        loc_text = (
            "Classification–localization cosine is {} at exposure 10 and {} at exposure 60. "
            "The two checkpoints do not give the same description."
        ).format(early_loc, late_loc)
    seg_rank, seg_text = rank_pair("Classification and segmentation", early_seg if early_seg == late_seg else "mixed")
    if early_seg != late_seg:
        seg_rank = "NOT SUPPORTED"
        seg_text = (
            "Classification–segmentation cosine is {} at exposure 10 and {} at exposure 60. "
            "The two checkpoints do not give the same description."
        ).format(early_seg, late_seg)

    agrees = []
    if early_loc in {"neutral", "mixed"} and late_loc in {"neutral", "mixed"}:
        agrees.append(
            "SUGGESTIVE: localization gradients are not a consistent opposing direction, which agrees with the near-zero within-cycle localization effect on classification AUC."
        )
    elif early_loc == "conflicting" and late_loc == "conflicting":
        agrees.append(
            "The cosine is conflicting at both checkpoints. That does not agree with the near-zero localization effect on AUC. The AUC result stays the performance measurement."
        )
    else:
        agrees.append(
            "Localization cosine does not land in one bin at both checkpoints, so it neither confirms nor overturns the near-zero AUC effect."
        )
    if early_seg == "conflicting" and late_seg == "conflicting":
        agrees.append(
            "SUGGESTIVE: segmentation gradients oppose the classification gradient at both checkpoints. That agrees with segmentation updates lowering classification AUC. It does not show that the opposition is the cause."
        )
    elif early_seg in {"aligned", "neutral"} and late_seg in {"aligned", "neutral"}:
        agrees.append(
            "Segmentation gradients are not opposing at either checkpoint. The repeated AUC drop is then not explained by an opposite gradient direction. A large step along a similar direction, or the classification head refitting on the next classification update, remains open."
        )
    else:
        agrees.append(
            "Segmentation cosine changes bin between exposure 10 and exposure 60, so a single gradient story does not cover both the early strong interference and the later smaller interference."
        )

    gradient_clear = early_seg in {"aligned", "conflicting"} or early_loc in {"aligned", "conflicting"}
    if gradient_clear:
        second = "B. Gradient-alignment figure"
        reason = (
            "The within-cycle AUC panel is the performance result. The gradient figure is the second panel because a directional cosine is readable as one comparison and it speaks to the interference the AUC panel already shows. "
            "Grad-CAM rows are a supplement: a projected poster cannot hold six images by four models by two exposures."
        )
    elif gradcam_any or exposure10_suggestive:
        second = "A. Grad-CAM comparison, exposure 10 only"
        reason = (
            "The gradient bins are not a single direction, so a gradient panel would not add a clear claim. "
            "Exposure-10 Grad-CAM is the second panel. Exposure 60 stays out of the poster."
        )
    else:
        second = "no second panel"
        reason = (
            "Neither new analysis produced a stable directional result. The poster should keep the within-cycle AUC comparison and leave Grad-CAM and the cosine table in the supplement."
        )

    lines = [
        "# Spatial supervision and CANDID-PTX classification",
        "",
        "Part I v2, one run per configuration. Grad-CAM uses matched classification exposure. "
        "Gradient cosine uses the three-task student only. Neither analysis replaces the within-cycle AUC results in `CLASSIFICATION_SYNERGY_ANALYSIS.md`.",
        "",
        "## Answers",
        "",
        "1. Does spatial supervision visibly change classification attention?",
        "",
        gradcam["concentration_interpretation"],
        "",
        "The maps themselves are in `Results/sol/analysis/gradcam_synergy/figures/`. A numerical concentration shift is not the same thing as a visible change. Read the exposure-10 overlays before using them.",
        "",
        "2. Does localization make classification attention more spatially meaningful?",
        "",
        gradcam["ground_truth_interpretation"],
        "",
        "Rank for a poster claim that localization made the CAM more meaningful: NOT SUPPORTED unless the ground-truth paragraph above says SUGGESTIVE for CLS + LOC, and even then the rank of that paragraph is suggestive rather than supported.",
        "",
        "3. Does segmentation make classification attention more spatially meaningful?",
        "",
        "Same ground-truth paragraph. A poster claim that segmentation made the CAM more meaningful is NOT SUPPORTED unless that paragraph says SUGGESTIVE for CLS + SEG or CLS + LOC + SEG.",
        "",
        "4. Are classification and localization gradients generally aligned, neutral, or conflicting?",
        "",
        "{} {}".format(loc_rank, loc_text),
        "Exposure 10 is {}. Exposure 60 is {}. LOC↔SEG is {} at exposure 10 and {} at exposure 60.".format(
            early_loc, late_loc, early_pair, late_pair,
        ),
        "",
        "5. Are classification and segmentation gradients generally aligned, neutral, or conflicting?",
        "",
        "{} {}".format(seg_rank, seg_text),
        "Exposure 10 is {}. Exposure 60 is {}.".format(early_seg, late_seg),
        "",
        "6. Does gradient behavior agree with the observed within-cycle AUC changes?",
        "",
        " ".join(agrees),
        "",
        "7. What can we reasonably say about why classification synergy is limited?",
        "",
        "SUPPORTED, from the earlier AUC analysis and not from these two runs of maps and gradients: a segmentation update lowers classification AUC and the next classification update puts that loss back; a localization update does not move classification AUC in one direction. "
        "The new measurements are complementary. They can agree or disagree with that pattern. They do not identify the cause.",
        "",
        "## Poster",
        "",
        "Use C. Existing within-cycle AUC analysis.",
        "Second item: {}.".format(second),
        "",
        reason,
        "",
        "At most those two. Do not add a third panel.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
