"""Build a Part I v2 student and load one saved checkpoint.

The training entry point is not modified. This module repeats the model
construction used by the CANDID-PTX Part I v2 launch scripts, then loads the
student state_dict. It does not step an optimizer and it does not write into
the training result directories.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
FX_PLUS = REPO_ROOT / "Foundation_X+"
CONFIG_FILE = FX_PLUS / "config" / "DINO" / "DINO_4scale_swinBASE.py"
DEFAULT_RESULTS_ROOT = Path("/scratch/hflechsi/Foundation_X/results/part_1_v2")

# Module indices inside the Swin student. These are not the TH numbers in the
# checkpoint filename. TH9 means the epoch trained CANDID-PTX classification.
CLS_HEAD = 2
LOC_HEAD = 2
SEG_HEAD = 2

# Classification exposure -> epoch -> checkpoint, verified from the Part I v2
# student classification rows. Exposure k is the k-th classification epoch.
EXPOSURES = {
    10: {
        "cls": ("candidptx_cls", 10, "ckpt_E10_TH9.pth"),
        "cls_loc": ("candidptx_cls_loc", 19, "ckpt_E19_TH9.pth"),
        "cls_seg": ("candidptx_cls_seg", 19, "ckpt_E19_TH9.pth"),
        "cls_loc_seg": ("candidptx_cls_loc_seg", 28, "ckpt_E28_TH9.pth"),
    },
    60: {
        "cls": ("candidptx_cls", 60, "ckpt_E60_TH9.pth"),
        "cls_loc": ("candidptx_cls_loc", 119, "ckpt_E119_TH9.pth"),
        "cls_seg": ("candidptx_cls_seg", 119, "ckpt_E119_TH9.pth"),
        "cls_loc_seg": ("candidptx_cls_loc_seg", 178, "ckpt_E178_TH9.pth"),
    },
}

RUN_ORDER = ("cls", "cls_loc", "cls_seg", "cls_loc_seg")
RUN_LABEL = {
    "cls": "CLS",
    "cls_loc": "CLS + LOC",
    "cls_seg": "CLS + SEG",
    "cls_loc_seg": "CLS + LOC + SEG",
}

# Last Swin-B stage at 224 input, patch size 4: 7 x 7 tokens, 1024 channels.
CAM_TOKENS = 7

SHARED_PREFIXES = {
    "patch_embed": "backbone.0.patch_embed.",
    "stage_1": "backbone.0.layers.0.",
    "stage_2": "backbone.0.layers.1.",
    "stage_3": "backbone.0.layers.2.",
    "stage_4": "backbone.0.layers.3.",
}


def _import_training_modules():
    os.chdir(FX_PLUS)
    if str(FX_PLUS) not in sys.path:
        sys.path.insert(0, str(FX_PLUS))
    from main_Consolidated import build_model_main, get_args_parser
    from util.slconfig import SLConfig
    from datasets_medical import CANDIDPTX, Candid_PTX_PXSDataset, dataloader_return
    from config.config_datasets import DATASETS_CONFIG, DATASETS_HEADS

    return {
        "build_model_main": build_model_main,
        "get_args_parser": get_args_parser,
        "SLConfig": SLConfig,
        "CANDIDPTX": CANDIDPTX,
        "Candid_PTX_PXSDataset": Candid_PTX_PXSDataset,
        "dataloader_return": dataloader_return,
        "DATASETS_CONFIG": DATASETS_CONFIG,
        "DATASETS_HEADS": DATASETS_HEADS,
    }


def build_args(output_dir, cyclictask, batch_size, num_workers, train_only, score_only):
    mods = _import_training_modules()
    parser = argparse.ArgumentParser("fx-analysis", parents=[mods["get_args_parser"]()])
    argv = [
        "--taskcomponent", "foundation_x5_pretraining",
        "--cyclictask", cyclictask,
        "--dataset_file", "foundation6Ark6_datasets",
        "--classification_dataset", "foundation6Ark6_datasets",
        "--serverC", "SOL",
        "--imgsize", "224",
        "--backbonemodel", "Swin-B",
        "--numClasses", "1",
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--output_dir", str(output_dir),
        "--coco_path", "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/",
        "-c", str(CONFIG_FILE),
        "--opt", "adamw",
        "--weight-decay", "0.0001",
        "--options",
        "dn_scalar=100",
        "embed_init_tgt=TRUE",
        "dn_label_coef=1.0",
        "dn_bbox_coef=1.0",
        "use_ema=False",
        "dn_box_noise_scale=1.0",
    ]
    if train_only:
        argv.append("--train-only")
    if score_only:
        argv.append("--score-only")
    args = parser.parse_args(argv)
    if args.local_rank is None:
        args.local_rank = 0

    cfg = mods["SLConfig"].fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for key, value in cfg_dict.items():
        if key not in args_vars:
            setattr(args, key, value)
        else:
            raise ValueError("Config key {} collides with an argparse field".format(key))

    # Same overrides main() applies for foundation_x5 after the config merge.
    args.num_classes = 14
    args.dn_labelbook_size = 15
    # Checkpointing changes the backward tape, not the weights. Grad-CAM and
    # the gradient comparison both need the ordinary backward tape.
    args.use_checkpoint = False
    args.debug = False
    if not hasattr(args, "amp"):
        args.amp = False
    return args, mods


def build_student(args, mods):
    model, criterion, postprocessors = mods["build_model_main"](args)
    model.eval()
    criterion.eval()
    return model, criterion, postprocessors


def load_student(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise SystemExit("Missing checkpoint: {}".format(path))
    saved = torch.load(str(path), map_location="cpu")
    if "model" not in saved:
        raise SystemExit("Checkpoint has no student 'model' state_dict: {}".format(path))
    state = saved["model"]
    stored_epoch = saved.get("epoch")
    del saved
    if any(str(key).startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items() if str(key).startswith("module.")}
    incompatible = model.load_state_dict(state, strict=True)
    del state
    return stored_epoch, incompatible


def checkpoint_path(results_root, exposure, run_key):
    run_name, epoch, filename = EXPOSURES[exposure][run_key]
    return Path(results_root) / run_name / filename, epoch


def require_checkpoints(results_root, exposures, run_keys):
    missing = []
    resolved = {}
    for exposure in exposures:
        for run_key in run_keys:
            path, epoch = checkpoint_path(results_root, exposure, run_key)
            resolved[(exposure, run_key)] = (path, epoch)
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise SystemExit("Missing checkpoints:\n" + "\n".join(missing))
    return resolved


def move_to_device(model, criterion, postprocessors, device):
    model.to(device)
    criterion.to(device)
    for value in postprocessors.values():
        if isinstance(value, torch.nn.Module):
            value.to(device)
    return model, criterion, postprocessors


def shared_parameter_names(model):
    grouped = {name: [] for name in SHARED_PREFIXES}
    for parameter_name, _ in model.named_parameters():
        for group, prefix in SHARED_PREFIXES.items():
            if parameter_name.startswith(prefix):
                grouped[group].append(parameter_name)
                break
    empty = [name for name, names in grouped.items() if not names]
    if empty:
        sample = [name for name, _ in model.named_parameters() if "backbone.0" in name][:8]
        raise SystemExit(
            "Shared Swin groups matched no parameters: {}. Sample backbone names: {}".format(empty, sample)
        )
    return grouped


def flat_grad(model, prefixes):
    chunks = []
    missing = 0
    matched = 0
    for name, parameter in model.named_parameters():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        matched += 1
        if parameter.grad is None:
            missing += 1
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(parameter.grad.detach().float().cpu().reshape(-1))
    if matched == 0:
        raise RuntimeError("No parameters matched {}".format(prefixes))
    return torch.cat(chunks), missing, matched


def cosine(left, right):
    left64 = left.double()
    right64 = right.double()
    denom = torch.linalg.vector_norm(left64) * torch.linalg.vector_norm(right64)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(left64, right64) / denom)


def describe(values):
    import statistics
    finite = [value for value in values if value == value]
    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "fraction_positive": float("nan"),
            "fraction_negative": float("nan"),
            "n_batches": 0,
            "n_undefined": len(values),
        }
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "fraction_positive": sum(value > 0 for value in finite) / len(finite),
        "fraction_negative": sum(value < 0 for value in finite) / len(finite),
        "n_batches": len(finite),
        "n_undefined": len(values) - len(finite),
    }
