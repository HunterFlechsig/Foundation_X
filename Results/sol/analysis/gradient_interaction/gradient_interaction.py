"""Task-gradient cosine similarity on the shared Swin backbone.

Uses the CLS + LOC + SEG Part I v2 student at classification exposures 10 and 60.
The optimizer is never stepped. Primary losses are the task losses only.
Teacher consistency is not included.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fx_runtime as fx


PAIRS = (("cls", "loc"), ("cls", "seg"), ("loc", "seg"))
GROUP_ORDER = ("patch_embed", "stage_1", "stage_2", "stage_3", "stage_4", "entire_backbone")


def parse_args():
    parser = argparse.ArgumentParser(description="Part I v2 shared-backbone gradient cosine")
    parser.add_argument("--results-root", type=Path, default=fx.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seeded_loader(dataset, batch_size, seed, num_workers, collate_fn):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def take_batches(loader, count):
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= count:
            break
    if len(batches) < count:
        raise SystemExit("Loader produced {} batches; requested {}".format(len(batches), count))
    return batches


def classification_loss(model, batch, criterion, device):
    images, target = batch
    images = images.float().to(device)
    target = target.float().to(device)
    logits, _features = model.backbone[0].extra_features(images, fx.CLS_HEAD)
    if target.shape != logits.shape:
        target = target.view_as(logits)
    return criterion(logits, target)


def localization_loss(model, criterion, batch, device, use_dn):
    samples, targets = batch
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    model.task_DetHead = fx.LOC_HEAD
    if use_dn:
        outputs, _features, _encoder = model(samples, targets)
    else:
        outputs, _features, _encoder = model(samples)
    loss_dict = criterion(outputs, targets)
    weights = criterion.weight_dict
    included = [key for key in loss_dict.keys() if key in weights]
    loss = sum(loss_dict[key] * weights[key] for key in included)
    return loss, included


def segmentation_boxes(model, postprocessors, images, imgsize):
    """Detector boxes as Python lists, matching the segmentation training step."""
    from util import box_ops

    model.task_DetHead = fx.SEG_HEAD
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs, _features, _encoder = model(images)
        target_sizes = torch.tensor([[imgsize, imgsize]] * images.shape[0], device=images.device)
        results = postprocessors["bbox"](outputs, target_sizes, not_to_xyxy=True)
    boxes = []
    for result in results:
        scores = result["scores"]
        converted = box_ops.box_xyxy_to_cxcywh(result["boxes"])
        selected = converted[scores > 0.3]
        valid = []
        for box in selected:
            _cx, _cy, width, height = box
            if float(width) > 0 and float(height) > 0:
                valid.append(box.detach().cpu().tolist())
        boxes.append(valid if valid else [[0.0, 0.0, 0.0, 0.0]])
    if was_training:
        model.train()
    return boxes


def segmentation_loss(model, postprocessors, batch, device, imgsize, dice_loss):
    images, masks = batch
    images = images.float().to(device)
    masks = masks.float().to(device)
    boxes = segmentation_boxes(model, postprocessors, images, imgsize)
    model.eval()
    prediction, _backbone, _features = model.backbone[0].extra_features_seg_sharedLocSeg(
        images, roi_align=boxes, head_n=fx.SEG_HEAD, model_to_train="student",
    )
    prediction = torch.sigmoid(prediction)
    return dice_loss(masks, prediction)


def capture(model):
    groups = {}
    missing = {}
    prefixes = {name: [prefix] for name, prefix in fx.SHARED_PREFIXES.items()}
    prefixes["entire_backbone"] = list(fx.SHARED_PREFIXES.values())
    for name, prefix_list in prefixes.items():
        flat, n_missing, _matched = fx.flat_grad(model, prefix_list)
        groups[name] = flat
        missing[name] = n_missing
    return groups, missing


def backward_task(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    groups, missing = capture(model)
    value = float(loss.detach().cpu())
    model.zero_grad(set_to_none=True)
    del loss
    return groups, missing, value


def write_report(output_dir, rows, meta):
    lines = []
    lines.append("## Results")
    lines.append("")
    lines.append("### SUPPORTED OBSERVATION")
    lines.append("")
    lines.append(
        "Cosine similarity was computed on {} minibatches per task at each checkpoint. "
        "Batch size {}. Seed {}. The three tasks use separate loaders, so a pair is not a same-image comparison. "
        "The model was in eval mode, so stochastic depth was off. The optimizer was not stepped.".format(
            meta["batches"], meta["batch_size"], meta["seed"],
        )
    )
    lines.append("")
    lines.append("Entire shared backbone (patch embedding and four Swin stages):")
    lines.append("")
    lines.append("| Checkpoint | Pair | Mean | Median | Std | Min | Max | Fraction positive | Fraction negative | Batches |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["stage"] != "entire_backbone":
            continue
        lines.append(
            "| E{} ({}) | {} ↔ {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {} |".format(
                row["epoch"], row["exposure"], row["left"], row["right"],
                float(row["mean"]), float(row["median"]), float(row["std"]),
                float(row["min"]), float(row["max"]),
                float(row["fraction_positive"]), float(row["fraction_negative"]),
                row["n_batches"],
            )
        )
    lines.append("")
    lines.append("Per-stage means are in `gradient_cosine_results.csv`.")
    lines.append("")
    lines.append("### INTERPRETATION / HYPOTHESIS")
    lines.append("")
    lines.append(meta["interpretation"])
    lines.append("")
    lines.append(
        "Cosine similarity does not prove why the classification AUC moves. "
        "It says whether the task gradients on these shared weights point together, against each other, or neither, on these batches."
    )
    lines.append("")
    text = "\n".join(lines)
    path = output_dir / "GRADIENT_INTERACTION_ANALYSIS.md"
    original = path.read_text()
    start = "<!-- RESULTS -->"
    end = "<!-- /RESULTS -->"
    if start not in original or end not in original:
        raise SystemExit("GRADIENT_INTERACTION_ANALYSIS.md is missing the RESULTS markers")
    path.write_text(original[:original.index(start)] + start + "\n" + text + "\n" + original[original.index(end):])


def interpret(rows):
    """Descriptive bins fixed before the run. Not a significance test."""
    messages = []
    for row in rows:
        if row["stage"] != "entire_backbone":
            continue
        mean = float(row["mean"])
        positive = float(row["fraction_positive"])
        negative = float(row["fraction_negative"])
        if mean > 0.1 and positive >= 0.7:
            bucket = "generally aligned"
        elif mean < -0.1 and negative >= 0.7:
            bucket = "generally conflicting"
        elif abs(mean) <= 0.1:
            bucket = "approximately neutral"
        else:
            bucket = "mixed, with no single direction"
        messages.append(
            "Exposure {} {}↔{} is {} (mean {:+.3f}, fraction positive {:.2f}, fraction negative {:.2f}).".format(
                row["exposure"], row["left"], row["right"], bucket, mean, positive, negative,
            )
        )
    messages.append(
        "Bins: aligned if mean > 0.1 and at least 70% of batches are positive; "
        "conflicting if mean < -0.1 and at least 70% are negative; "
        "neutral if the absolute mean is at most 0.1; otherwise mixed."
    )
    messages.append(
        "Localization updates in the published within-cycle analysis are about neutral for classification AUC. "
        "A neutral or mixed CLS↔LOC cosine agrees with that pattern. A conflicting CLS↔LOC cosine would not. "
        "Segmentation updates lower classification AUC. A conflicting CLS↔SEG cosine agrees with that direction. "
        "An aligned or neutral CLS↔SEG cosine would leave the AUC drop unexplained by opposing gradients."
    )
    return " ".join(messages)


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    resolved = fx.require_checkpoints(args.results_root, (10, 60), ("cls_loc_seg",))

    build_dir = output_dir / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    train_args, mods = fx.build_args(
        build_dir,
        "candidptxCLS_candidptxLOC_candidptxSEG",
        args.batch_size,
        args.num_workers,
        train_only=True,
        score_only=False,
    )
    heads = mods["DATASETS_HEADS"]
    if heads["CANDIDPTX_CLS"] != fx.CLS_HEAD or heads["CANDIDPTX_LOC"] != fx.LOC_HEAD or heads["CANDIDPTX_SEG"] != fx.SEG_HEAD:
        raise SystemExit("CANDID head indices changed: {}".format(heads))

    from utils_segmentation import torch_dice_coef_loss
    import util.misc as utils

    loaders = mods["dataloader_return"](train_args)
    cls_loader = loaders[20]
    loc_loader = loaders[30]
    seg_loader = loaders[50]
    if cls_loader is None or loc_loader is None or seg_loader is None:
        raise SystemExit("One of the CANDID train loaders was not built")
    if not isinstance(cls_loader.dataset, mods["CANDIDPTX"]):
        raise SystemExit("Classification loader is not CANDIDPTX")
    if not isinstance(seg_loader.dataset, mods["Candid_PTX_PXSDataset"]):
        raise SystemExit("Segmentation loader is not Candid_PTX_PXSDataset")

    cls_batches = take_batches(
        seeded_loader(cls_loader.dataset, args.batch_size, args.seed + 1, args.num_workers, None),
        args.batches,
    )
    loc_batches = take_batches(
        seeded_loader(loc_loader.dataset, args.batch_size, args.seed + 2, args.num_workers, utils.collate_fn),
        args.batches,
    )
    seg_batches = take_batches(
        seeded_loader(seg_loader.dataset, args.batch_size, args.seed + 3, args.num_workers, None),
        args.batches,
    )

    bce = torch.nn.BCEWithLogitsLoss()
    model, criterion, postprocessors = fx.build_student(train_args, mods)
    fx.shared_parameter_names(model)
    model, criterion, postprocessors = fx.move_to_device(model, criterion, postprocessors, device)
    bce.to(device)
    model.eval()
    criterion.eval()

    summary_rows = []
    batch_rows = []
    loss_keys = None
    for exposure in (10, 60):
        checkpoint, epoch = resolved[(exposure, "cls_loc_seg")]
        stored_epoch, _unused = fx.load_student(model, checkpoint)
        model.to(device)
        model.eval()
        print("Loaded", checkpoint.name, "file epoch", stored_epoch, "expected", epoch, flush=True)
        for batch_index in range(args.batches):
            cls_grad, cls_missing, cls_value = backward_task(
                model, classification_loss(model, cls_batches[batch_index], bce, device),
            )
            loc_loss, loss_keys = localization_loss(
                model, criterion, loc_batches[batch_index], device, train_args.use_dn,
            )
            loc_grad, loc_missing, loc_value = backward_task(model, loc_loss)
            seg_grad, seg_missing, seg_value = backward_task(
                model,
                segmentation_loss(
                    model, postprocessors, seg_batches[batch_index], device, train_args.imgsize, torch_dice_coef_loss,
                ),
            )
            named = {"cls": cls_grad, "loc": loc_grad, "seg": seg_grad}
            missing = {"cls": cls_missing, "loc": loc_missing, "seg": seg_missing}
            for left, right in PAIRS:
                for stage in GROUP_ORDER:
                    value = fx.cosine(named[left][stage], named[right][stage])
                    batch_rows.append({
                        "exposure": exposure,
                        "epoch": epoch,
                        "checkpoint": checkpoint.name,
                        "batch": batch_index,
                        "stage": stage,
                        "left": left,
                        "right": right,
                        "cosine": value,
                        "loss_cls": cls_value,
                        "loss_loc": loc_value,
                        "loss_seg": seg_value,
                        "missing_cls": missing["cls"][stage],
                        "missing_loc": missing["loc"][stage],
                        "missing_seg": missing["seg"][stage],
                    })
            print(
                "exposure", exposure, "batch", batch_index,
                "losses", round(cls_value, 4), round(loc_value, 4), round(seg_value, 4),
                flush=True,
            )
            del cls_grad, loc_grad, seg_grad

    grouped = {}
    for row in batch_rows:
        key = (row["exposure"], row["epoch"], row["checkpoint"], row["stage"], row["left"], row["right"])
        grouped.setdefault(key, []).append(row["cosine"])
    for key, values in grouped.items():
        exposure, epoch, checkpoint, stage, left, right = key
        stats = fx.describe(values)
        summary_rows.append({
            "exposure": exposure,
            "epoch": epoch,
            "checkpoint": checkpoint,
            "stage": stage,
            "left": left,
            "right": right,
            **stats,
        })

    summary_rows.sort(key=lambda row: (row["exposure"], GROUP_ORDER.index(row["stage"]), row["left"], row["right"]))
    csv_path = output_dir / "gradient_cosine_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    batch_path = output_dir / "gradient_cosine_batches.csv"
    with batch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(batch_rows)

    interpretation = interpret(summary_rows)
    meta = {
        "batches": args.batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "localization_loss_keys": loss_keys,
        "interpretation": interpretation,
        "primary_losses": {
            "classification": "BCEWithLogitsLoss on classification head 2",
            "localization": "weighted detection criterion, without teacher consistency and without the encoder L2 term",
            "segmentation": "dice loss after detached detector boxes; no teacher consistency",
        },
    }
    (output_dir / "gradient_summary.json").write_text(json.dumps({
        "meta": meta,
        "rows": summary_rows,
    }, indent=2))
    write_report(output_dir, summary_rows, meta)
    print("Wrote", csv_path)
    print("Localization loss terms:", loss_keys)


if __name__ == "__main__":
    main()
