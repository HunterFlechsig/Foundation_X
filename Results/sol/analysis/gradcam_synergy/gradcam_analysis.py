"""Classification Grad-CAM for the four Part I v2 CANDID-PTX students.

Run on SOL. Does not train and does not write into the checkpoint directories.
The vendored pytorch_grad_cam.base_cam.BaseCAM.forward indexes outputs[2] and
reshapes that tensor to 230400, so this script computes Grad-CAM itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fx_runtime as fx


DISPLAY_MEAN = 0.5056
DISPLAY_STD = 0.252
POSITIVE_THRESHOLD = 0.5
N_IMAGES = 6
# Pre-specified before looking at maps: a joint model is called more
# concentrated on this set only when both summaries move that way on at least
# this many of the selected images.
MIN_AGREEING_IMAGES = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Part I v2 classification Grad-CAM")
    parser.add_argument("--results-root", type=Path, default=fx.DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def target_layer(model):
    layer = model.backbone[0].layers[3].blocks[-1]
    return layer


def classification_grad_cam(model, image_bchw, layer):
    """Standard Grad-CAM on the last stage-4 Swin block.

    The block returns tokens (B, 49, 1024). Those are reshaped to (B, 1024, 7, 7)
    before the channel weights are applied. The returned map is ReLU(CAM) at 7x7,
    before display normalization.
    """
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    def save_gradient(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_hook = layer.register_forward_hook(save_activation)
    backward_hook = layer.register_full_backward_hook(save_gradient)
    model.zero_grad(set_to_none=True)
    try:
        logits, _features = model.backbone[0].extra_features(image_bchw, fx.CLS_HEAD)
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise RuntimeError("Expected one classification logit, got {}".format(tuple(logits.shape)))
        logits[0, 0].backward()
    finally:
        forward_hook.remove()
        backward_hook.remove()

    activation = captured.get("activation")
    gradient = captured.get("gradient")
    if not torch.is_tensor(activation) or activation.ndim != 3:
        raise RuntimeError("Stage-4 block did not return (B, tokens, channels); got {}".format(type(activation)))
    batch, tokens, channels = activation.shape
    side = int(tokens ** 0.5)
    if side * side != tokens or side != fx.CAM_TOKENS:
        raise RuntimeError("Expected {}x{} tokens, got {} tokens".format(fx.CAM_TOKENS, fx.CAM_TOKENS, tokens))
    activation = activation.transpose(1, 2).reshape(batch, channels, side, side)
    gradient = gradient.transpose(1, 2).reshape(batch, channels, side, side)
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1))
    return cam[0].detach()


def upsample(cam):
    return F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]


def display_image(image_chw):
    image = image_chw.detach().float().cpu()
    image = image * DISPLAY_STD + DISPLAY_MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def display_cam(cam_up):
    cam = cam_up.detach().float().cpu()
    low = float(cam.min())
    high = float(cam.max())
    if high > low:
        cam = (cam - low) / (high - low)
    else:
        cam = torch.zeros_like(cam)
    return cam.numpy(), low, high


def normalized_entropy(cam_up):
    energy = cam_up.detach().float().clamp(min=0).cpu().reshape(-1)
    total = float(energy.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    probability = energy / total
    entropy = float((-(probability * (probability + 1e-12).log()).sum() / np.log(probability.numel())))
    top_count = max(1, probability.numel() // 4)
    top_fraction = float(torch.topk(energy, top_count).values.sum() / total)
    half_area = float((energy >= 0.5 * energy.max()).float().mean())
    return entropy, top_fraction, half_area


def score_split(model, dataset, device, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    probabilities = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for images, _labels in loader:
            images = images.float().to(device)
            logits, _features = model.backbone[0].extra_features(images, fx.CLS_HEAD)
            probabilities.append(torch.sigmoid(logits.view(-1)).detach().cpu())
            cursor += images.shape[0]
    if cursor != len(dataset):
        raise RuntimeError("Scored {} images from a dataset of {}".format(cursor, len(dataset)))
    return torch.cat(probabilities).numpy()


def stem_of(path):
    return Path(str(path)).stem


def index_stems(paths, source):
    table = {}
    collisions = []
    for index, path in enumerate(paths):
        key = stem_of(path)
        if key in table:
            collisions.append(key)
        else:
            table[key] = {"index": index, "path": str(path), "source": source}
    return table, collisions


def load_loc_annotations(json_path, source):
    if not Path(json_path).is_file():
        return {}, ["missing {}".format(json_path)]
    with open(json_path) as handle:
        payload = json.load(handle)
    images = {image["id"]: image for image in payload.get("images", [])}
    boxes = {image_id: [] for image_id in images}
    for annotation in payload.get("annotations", []):
        boxes.setdefault(annotation["image_id"], []).append(annotation.get("bbox", []))
    table = {}
    collisions = []
    for image_id, image in images.items():
        key = stem_of(image.get("file_name", ""))
        record = {
            "file_name": image.get("file_name", ""),
            "width": image.get("width"),
            "height": image.get("height"),
            "boxes": boxes.get(image_id, []),
            "source": source,
        }
        if key in table:
            collisions.append(key)
        else:
            table[key] = record
    return table, collisions


def boxes_to_mask(record):
    width = record.get("width") or 0
    height = record.get("height") or 0
    if width <= 0 or height <= 0:
        return None
    mask = np.zeros((224, 224), dtype=np.float32)
    for box in record["boxes"]:
        if len(box) != 4:
            continue
        x, y, box_w, box_h = [float(value) for value in box]
        if box_w <= 0 or box_h <= 0:
            continue
        x0 = int(np.floor(x * 224.0 / width))
        y0 = int(np.floor(y * 224.0 / height))
        x1 = int(np.ceil((x + box_w) * 224.0 / width))
        y1 = int(np.ceil((y + box_h) * 224.0 / height))
        x0 = min(max(x0, 0), 223)
        y0 = min(max(y0, 0), 223)
        x1 = min(max(x1, 0), 224)
        y1 = min(max(y1, 0), 224)
        mask[y0:y1, x0:x1] = 1.0
    if float(mask.sum()) <= 0:
        return None
    return mask


def unique_quantile_indices(order, count):
    if len(order) <= count:
        return list(order)
    chosen = []
    for rank in range(count):
        position = int(round((rank + 0.5) / count * (len(order) - 1)))
        while position < len(order) and order[position] in chosen:
            position += 1
        if position >= len(order):
            position = int(round((rank + 0.5) / count * (len(order) - 1)))
            while position >= 0 and order[position] in chosen:
                position -= 1
        chosen.append(order[position])
    return chosen


def select_images(labels, probabilities):
    """Jointly correct positive test images, spread across mean confidence."""
    positive = labels == 1
    correct = np.ones(len(labels), dtype=bool)
    for scores in probabilities.values():
        correct &= scores >= POSITIVE_THRESHOLD
    eligible = np.flatnonzero(positive & correct)
    rule = (
        "positive label; sigmoid probability >= {} on all four exposure-10 students; "
        "six images at evenly spaced ranks of the four-model mean probability"
    ).format(POSITIVE_THRESHOLD)
    if len(eligible) < N_IMAGES:
        raise SystemExit(
            "Only {} jointly correct positive test images. Selection was not relaxed.".format(len(eligible))
        )
    mean_probability = np.mean(np.stack([probabilities[key][eligible] for key in fx.RUN_ORDER], axis=0), axis=0)
    order = eligible[np.argsort(mean_probability)]
    chosen = unique_quantile_indices(order.tolist(), N_IMAGES)
    return chosen, rule, int(positive.sum()), int(len(eligible))


def region_measures(cam_up, mask):
    energy = cam_up.detach().float().cpu().numpy()
    total = float(energy.sum())
    if total <= 0 or mask is None:
        return {"energy_fraction": float("nan"), "pointing_hit": "", "center_inside": ""}
    fraction = float((energy * mask).sum() / total)
    flat = energy.reshape(-1)
    peak = int(flat.argmax())
    peak_y, peak_x = divmod(peak, energy.shape[1])
    pointing = int(mask[peak_y, peak_x] > 0.5)
    yy, xx = np.mgrid[0:energy.shape[0], 0:energy.shape[1]]
    center_y = float((yy * energy).sum() / total)
    center_x = float((xx * energy).sum() / total)
    cy = min(max(int(round(center_y)), 0), mask.shape[0] - 1)
    cx = min(max(int(round(center_x)), 0), mask.shape[1] - 1)
    center = int(mask[cy, cx] > 0.5)
    return {"energy_fraction": fraction, "pointing_hit": pointing, "center_inside": center}


def save_figure(path, original, columns, title):
    count = 1 + len(columns)
    figure, axes = plt.subplots(1, count, figsize=(3.1 * count, 3.5), constrained_layout=True)
    if count == 1:
        axes = [axes]
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    for axis, (name, image, kind) in zip(axes[1:], columns):
        if kind == "overlay":
            axis.imshow(original, cmap="gray", vmin=0, vmax=1)
            axis.imshow(image, cmap="jet", vmin=0, vmax=1, alpha=0.45)
        elif kind == "raw":
            axis.imshow(image, cmap="jet", vmin=0, vmax=1)
        elif kind == "gt":
            axis.imshow(original, cmap="gray", vmin=0, vmax=1)
            if image is not None:
                axis.contour(image, levels=[0.5], colors=["#00FF66"], linewidths=1.0)
        axis.set_title(name)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def paired_direction(reference, other, want_lower):
    agree = 0
    deltas = []
    for left, right in zip(reference, other):
        if left != left or right != right:
            continue
        delta = right - left
        deltas.append(delta)
        if want_lower and delta < 0:
            agree += 1
        if (not want_lower) and delta > 0:
            agree += 1
    return agree, deltas


def write_observations(output_dir, summary):
    lines = []
    lines.append("## Observations")
    lines.append("")
    lines.append("Exposure 10 is the primary comparison. Exposure 60 uses the same six images and is a check, not a second poster grid.")
    lines.append("")
    lines.append("### SUPPORTED OBSERVATION")
    lines.append("")
    lines.append(
        "Six classification-test images were selected before any map was drawn. "
        "Each is a positive case that all four exposure-10 students score at or above {}. "
        "The test split has {} positives and {} jointly correct positives. "
        "The selected mean probabilities run from {:.3f} to {:.3f}.".format(
            POSITIVE_THRESHOLD,
            summary["n_positive"],
            summary["n_jointly_correct_positive"],
            summary["selected_mean_probability_min"],
            summary["selected_mean_probability_max"],
        )
    )
    lines.append("")
    if summary["spatial_overlap"]["selected_with_seg"] == 0 and summary["spatial_overlap"]["selected_with_loc"] == 0:
        lines.append(
            "None of the selected classification-test images matched a localization box file or a segmentation mask by filename stem. "
            "Grad-CAM stays qualitative on spatial ground truth. Concentration numbers below do not use a mask."
        )
    else:
        lines.append(
            "Filename-stem overlap with the selected images: segmentation {}, localization {}. "
            "Energy fraction, pointing-game hit, and CAM-center hit are reported only for those matches. "
            "No thresholded Dice or IoU is reported.".format(
                summary["spatial_overlap"]["selected_with_seg"],
                summary["spatial_overlap"]["selected_with_loc"],
            )
        )
    lines.append("")
    lines.append("Mean normalized CAM entropy and fraction of ReLU energy in the highest 25% of pixels, exposure 10:")
    lines.append("")
    lines.append("| Model | Entropy | Top-25% energy | Area above half-max |")
    lines.append("| --- | --- | --- | --- |")
    for run_key in fx.RUN_ORDER:
        row = summary["exposure_10_concentration"][run_key]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            fx.RUN_LABEL[run_key], row["entropy"], row["top25"], row["half_area"],
        ))
    lines.append("")
    lines.append("The same summaries at exposure 60, on the same images:")
    lines.append("")
    lines.append("| Model | Entropy | Top-25% energy | Area above half-max |")
    lines.append("| --- | --- | --- | --- |")
    for run_key in fx.RUN_ORDER:
        row = summary["exposure_60_concentration"][run_key]
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} |".format(
            fx.RUN_LABEL[run_key], row["entropy"], row["top25"], row["half_area"],
        ))
    lines.append("")
    lines.append("### INTERPRETATION / HYPOTHESIS")
    lines.append("")
    lines.append(summary["concentration_interpretation"])
    lines.append("")
    lines.append(summary["ground_truth_interpretation"])
    lines.append("")
    lines.append(
        "These maps are not evidence of what the model uses. A more peaked map is a more peaked map. "
        "It is not, by itself, better localization."
    )
    lines.append("")
    marker_start = "<!-- RESULTS -->"
    marker_end = "<!-- /RESULTS -->"
    path = output_dir / "GRADCAM_ANALYSIS.md"
    text = path.read_text()
    if marker_start not in text or marker_end not in text:
        raise SystemExit("GRADCAM_ANALYSIS.md is missing the RESULTS markers")
    body = "\n".join(lines)
    replaced = text[:text.index(marker_start)] + marker_start + "\n" + body + "\n" + text[text.index(marker_end):]
    path.write_text(replaced)


def ground_truth_text(rows, overlap, gt_hits):
    if gt_hits["seg"] < 4 and gt_hits["loc"] < 4:
        return (
            "Spatial ground truth does not cover enough of the selected classification-test images for a localization summary. "
            "Cls-test stems overlapping seg-test: {}. Cls-test stems overlapping loc-val: {}. "
            "The attention comparison stays on the raw maps and the concentration summaries."
        ).format(overlap["cls_and_seg_test"], overlap["cls_and_loc_val"])
    messages = [
        "Energy inside the annotation is the fraction of ReLU CAM mass, before display min-max. "
        "Pointing and center hits are in the CSV. No Dice or IoU is computed."
    ]
    for kind, label in (("seg", "segmentation mask"), ("loc", "localization boxes")):
        if gt_hits[kind] < 4:
            continue
        for exposure in (10, 60):
            for run_key in ("cls_loc", "cls_seg", "cls_loc_seg"):
                hits = 0
                deltas = []
                for row in rows:
                    if not row["{}_gt_source".format(kind)]:
                        continue
                    left = row["{}_energy_cls_e{}".format(kind, exposure)]
                    right = row["{}_energy_{}_e{}".format(kind, run_key, exposure)]
                    if left == "" or right == "":
                        continue
                    delta = float(right) - float(left)
                    deltas.append(delta)
                    if delta > 0:
                        hits += 1
                fired = hits >= MIN_AGREEING_IMAGES and len(deltas) >= MIN_AGREEING_IMAGES
                status = "SUGGESTIVE" if fired else "NOT SUPPORTED on this image set"
                mean_delta = float(np.mean(deltas)) if deltas else float("nan")
                messages.append(
                    "Exposure {} {} versus CLS, {}: {}. Energy fraction was higher on {}/{} matched images (mean paired change {:+.3f}).".format(
                        exposure, fx.RUN_LABEL[run_key], label, status, hits, len(deltas), mean_delta,
                    )
                )
    messages.append(
        "A higher fraction means more of the CAM mass lies inside the annotation on these images. It does not show that the classifier uses that region."
    )
    return " ".join(messages)


def concentration_interpretation(per_image):
    """per_image[exposure][run_key] = list of (entropy, top25) aligned to selected images."""
    messages = []
    for exposure in (10, 60):
        reference_entropy = [item[0] for item in per_image[exposure]["cls"]]
        reference_top = [item[1] for item in per_image[exposure]["cls"]]
        for run_key in ("cls_loc", "cls_seg", "cls_loc_seg"):
            entropy_hits, entropy_deltas = paired_direction(
                reference_entropy, [item[0] for item in per_image[exposure][run_key]], want_lower=True,
            )
            top_hits, top_deltas = paired_direction(
                reference_top, [item[1] for item in per_image[exposure][run_key]], want_lower=False,
            )
            fired = entropy_hits >= MIN_AGREEING_IMAGES and top_hits >= MIN_AGREEING_IMAGES
            status = "SUGGESTIVE" if fired else "NOT SUPPORTED on this image set"
            messages.append(
                "Exposure {}: {} versus CLS is {}. Entropy was lower on {}/{} images (mean paired change {:+.3f}); "
                "top-25% energy was higher on {}/{} images (mean paired change {:+.3f}).".format(
                    exposure,
                    fx.RUN_LABEL[run_key],
                    status,
                    entropy_hits,
                    len(entropy_deltas),
                    float(np.mean(entropy_deltas)) if entropy_deltas else float("nan"),
                    top_hits,
                    len(top_deltas),
                    float(np.mean(top_deltas)) if top_deltas else float("nan"),
                )
            )
    messages.append(
        "The rule was fixed before the maps were drawn: both entropy down and top-25% energy up on at least {} of the {} images. "
        "A result that meets the rule is still suggestive, because the set has six images chosen for joint correctness and a confidence range.".format(
            MIN_AGREEING_IMAGES, N_IMAGES,
        )
    )
    return " ".join(messages)


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    resolved = fx.require_checkpoints(args.results_root, (10, 60), fx.RUN_ORDER)

    build_dir = output_dir / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    train_args, mods = fx.build_args(
        build_dir, "candidptxCLS", args.batch_size, args.num_workers, train_only=False, score_only=True,
    )
    if mods["DATASETS_HEADS"]["CANDIDPTX_CLS"] != fx.CLS_HEAD:
        raise SystemExit("CANDID classification head index changed; update fx.CLS_HEAD")

    loaders = mods["dataloader_return"](train_args)
    test_loader = loaders[21]
    if test_loader is None or not isinstance(test_loader.dataset, mods["CANDIDPTX"]):
        raise SystemExit("CANDID-PTX classification test loader was not built")
    dataset = test_loader.dataset
    labels = np.array([int(dataset.img_label[index][0]) for index in range(len(dataset))], dtype=np.int64)

    config = mods["DATASETS_CONFIG"]
    seg_root = config["seg_candidptx_root"]
    seg_sets = {}
    seg_collisions = []
    for source, list_name in (("seg_test", "seg_candidptx_testList"), ("seg_train", "seg_candidptx_trainList")):
        seg_sets[source] = mods["Candid_PTX_PXSDataset"](
            [(seg_root, config[list_name])], image_size=(224, 224), mode="test",
        )
        _table, collisions = index_stems(seg_sets[source].img_list, source)
        seg_collisions.extend(collisions)
    seg_index = {}
    for source in ("seg_test", "seg_train"):
        table, _collisions = index_stems(seg_sets[source].img_list, source)
        for key, record in table.items():
            seg_index.setdefault(key, record)

    loc_val, loc_val_collisions = load_loc_annotations(config["loc_candidptx_valList"], "loc_val")
    loc_test, loc_test_collisions = load_loc_annotations(config["loc_candidptx_testList"], "loc_test")
    loc_index = dict(loc_test)
    loc_index.update(loc_val)  # evaluation split wins when a stem is in both

    cls_stems = [stem_of(path) for path in dataset.img_list]
    cls_stem_set = set(cls_stems)
    overlap = {
        "n_cls_test": len(dataset),
        "n_seg_test_stems": len(index_stems(seg_sets["seg_test"].img_list, "seg_test")[0]),
        "n_seg_train_stems": len(index_stems(seg_sets["seg_train"].img_list, "seg_train")[0]),
        "n_loc_val_stems": len(loc_val),
        "n_loc_test_stems": len(loc_test),
        "cls_and_seg_test": len(cls_stem_set & set(index_stems(seg_sets["seg_test"].img_list, "seg_test")[0])),
        "cls_and_seg_train": len(cls_stem_set & set(index_stems(seg_sets["seg_train"].img_list, "seg_train")[0])),
        "cls_and_loc_val": len(cls_stem_set & set(loc_val)),
        "cls_and_loc_test": len(cls_stem_set & set(loc_test)),
        "stem_collisions": {
            "seg": seg_collisions[:20],
            "loc_val": loc_val_collisions[:20],
            "loc_test": loc_test_collisions[:20],
        },
    }

    probabilities = {10: {}, 60: {key: np.full(len(dataset), np.nan) for key in fx.RUN_ORDER}}
    model, criterion, postprocessors = fx.build_student(train_args, mods)
    for run_key in fx.RUN_ORDER:
        path, _epoch = resolved[(10, run_key)]
        print("Scoring", fx.RUN_LABEL[run_key], "exposure 10", path.name, flush=True)
        fx.load_student(model, path)
        model.to(device)
        model.eval()
        probabilities[10][run_key] = score_split(model, dataset, device, args.batch_size, args.num_workers)
    del model, criterion, postprocessors
    torch.cuda.empty_cache()

    chosen, rule, n_positive, n_joint = select_images(labels, probabilities[10])
    selected_rows = []
    per_image_metrics = {10: {key: [] for key in fx.RUN_ORDER}, 60: {key: [] for key in fx.RUN_ORDER}}
    stored_cams = {10: {key: {} for key in fx.RUN_ORDER}, 60: {key: {} for key in fx.RUN_ORDER}}
    gt_hits = {"seg": 0, "loc": 0}

    model, criterion, postprocessors = fx.build_student(train_args, mods)
    for exposure in (10, 60):
        for run_key in fx.RUN_ORDER:
            checkpoint, _epoch = resolved[(exposure, run_key)]
            print("Grad-CAM", fx.RUN_LABEL[run_key], "exposure", exposure, checkpoint.name, flush=True)
            fx.load_student(model, checkpoint)
            model.to(device)
            model.eval()
            layer = target_layer(model)
            for image_index in chosen:
                image_tensor, _label = dataset[image_index]
                batch = image_tensor.unsqueeze(0).float().to(device)
                with torch.no_grad():
                    logits, _features = model.backbone[0].extra_features(batch, fx.CLS_HEAD)
                probabilities[exposure][run_key][image_index] = float(torch.sigmoid(logits).view(-1)[0].cpu())
                cam = classification_grad_cam(model, batch, layer)
                cam_up = upsample(cam).cpu()
                entropy, top25, half_area = normalized_entropy(cam_up)
                per_image_metrics[exposure][run_key].append((entropy, top25, half_area))
                stored_cams[exposure][run_key][image_index] = cam_up
            torch.cuda.empty_cache()
    del model, criterion, postprocessors
    torch.cuda.empty_cache()

    for image_index in chosen:
        path = dataset.img_list[image_index]
        key = stem_of(path)
        image_tensor, _label = dataset[image_index]
        original = display_image(image_tensor)
        seg_record = seg_index.get(key)
        seg_mask = None
        seg_source = ""
        if seg_record is not None:
            seg_source = seg_record["source"]
            _image, mask = seg_sets[seg_source][seg_record["index"]]
            seg_mask = (mask.squeeze(0).numpy() > 0.5).astype(np.float32)
            if float(seg_mask.sum()) <= 0:
                seg_mask = None
            else:
                gt_hits["seg"] += 1
        loc_record = loc_index.get(key)
        loc_mask = boxes_to_mask(loc_record) if loc_record else None
        loc_source = loc_record["source"] if loc_record and loc_mask is not None else ""
        if loc_mask is not None:
            gt_hits["loc"] += 1
        gt_mask = seg_mask if seg_mask is not None else loc_mask

        row = {
            "image_index": image_index,
            "image_path": path,
            "stem": key,
            "label": int(labels[image_index]),
            "selection_reason": rule,
            "seg_gt_source": seg_source if seg_mask is not None else "",
            "loc_gt_source": loc_source,
        }
        for exposure in (10, 60):
            for run_key in fx.RUN_ORDER:
                row["p_{}_e{}".format(run_key, exposure)] = "{:.6f}".format(probabilities[exposure][run_key][image_index])

        position = chosen.index(image_index)
        for exposure in (10, 60):
            overlay_columns = []
            raw_columns = []
            for run_key in fx.RUN_ORDER:
                cam_up = stored_cams[exposure][run_key][image_index]
                entropy, top25, half_area = per_image_metrics[exposure][run_key][position]
                shown, _low, _high = display_cam(cam_up)
                overlay_columns.append((fx.RUN_LABEL[run_key], shown, "overlay"))
                raw_columns.append((fx.RUN_LABEL[run_key], shown, "raw"))
                seg_stats = region_measures(cam_up, seg_mask)
                loc_stats = region_measures(cam_up, loc_mask)
                prefix = "{}_e{}".format(run_key, exposure)
                row["entropy_" + prefix] = "{:.6f}".format(entropy)
                row["top25_" + prefix] = "{:.6f}".format(top25)
                row["half_area_" + prefix] = "{:.6f}".format(half_area)
                row["seg_energy_" + prefix] = "" if seg_stats["energy_fraction"] != seg_stats["energy_fraction"] else "{:.6f}".format(seg_stats["energy_fraction"])
                row["seg_pointing_" + prefix] = seg_stats["pointing_hit"]
                row["seg_center_" + prefix] = seg_stats["center_inside"]
                row["loc_energy_" + prefix] = "" if loc_stats["energy_fraction"] != loc_stats["energy_fraction"] else "{:.6f}".format(loc_stats["energy_fraction"])
                row["loc_pointing_" + prefix] = loc_stats["pointing_hit"]
                row["loc_center_" + prefix] = loc_stats["center_inside"]
            if gt_mask is not None:
                overlay_columns.append(("Ground truth", gt_mask, "gt"))
                raw_columns.append(("Ground truth", gt_mask, "gt"))
            title = "{}  exposure {}".format(key, exposure)
            figure_dir = output_dir / "figures" / "exposure_{}".format(exposure)
            save_figure(figure_dir / "{}_overlay.png".format(key), original, overlay_columns, title)
            save_figure(figure_dir / "{}_raw.png".format(key), original, raw_columns, title + " raw")
        selected_rows.append(row)
        print("Wrote maps for", key, flush=True)

    csv_path = output_dir / "gradcam_selected_images.csv"
    fieldnames = list(selected_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    def mean_metric(exposure, run_key, slot):
        values = [item[slot] for item in per_image_metrics[exposure][run_key]]
        return float(np.mean(values))

    summary = {
        "n_positive": n_positive,
        "n_jointly_correct_positive": n_joint,
        "selected_indices": [int(index) for index in chosen],
        "selected_mean_probability_min": float(min(
            np.mean([probabilities[10][key][index] for key in fx.RUN_ORDER]) for index in chosen
        )),
        "selected_mean_probability_max": float(max(
            np.mean([probabilities[10][key][index] for key in fx.RUN_ORDER]) for index in chosen
        )),
        "spatial_overlap": {
            **overlap,
            "selected_with_seg": gt_hits["seg"],
            "selected_with_loc": gt_hits["loc"],
        },
        "exposure_10_concentration": {
            run_key: {
                "entropy": mean_metric(10, run_key, 0),
                "top25": mean_metric(10, run_key, 1),
                "half_area": mean_metric(10, run_key, 2),
            }
            for run_key in fx.RUN_ORDER
        },
        "exposure_60_concentration": {
            run_key: {
                "entropy": mean_metric(60, run_key, 0),
                "top25": mean_metric(60, run_key, 1),
                "half_area": mean_metric(60, run_key, 2),
            }
            for run_key in fx.RUN_ORDER
        },
    }
    compact_metrics = {
        exposure: {
            run_key: [(item[0], item[1]) for item in per_image_metrics[exposure][run_key]]
            for run_key in fx.RUN_ORDER
        }
        for exposure in (10, 60)
    }
    summary["concentration_interpretation"] = concentration_interpretation(compact_metrics)
    summary["ground_truth_interpretation"] = ground_truth_text(selected_rows, overlap, gt_hits)
    (output_dir / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    write_observations(output_dir, summary)
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()
