# Classification Grad-CAM, CANDID-PTX Part I v2

Student models only. No retraining. The numbers and the interpretation below the marker are written by `gradcam_analysis.py` after it runs on SOL.

## Scientific question

Does localization or segmentation supervision change where the CANDID-PTX classification student attends?

The comparison is classification Grad-CAM from four Part I v2 runs:

- CLS
- CLS + LOC
- CLS + SEG
- CLS + LOC + SEG

The hypothesis is not that the spatially supervised maps must look better. The question is whether those runs produce visibly different or more spatially concentrated classification attention. A peaked map is not, by itself, better localization.

## Checkpoint selection

Matched classification exposure, not each run's best checkpoint. A best-checkpoint comparison would mix different amounts of classification training with the task mix. Exposure 10 is the primary comparison. Exposure 60 uses the same images as a check.

Exposure *k* is the *k*-th classification epoch in that run, from the Part I v2 student rows.

| Exposure | CLS | CLS + LOC | CLS + SEG | CLS + LOC + SEG |
| --- | --- | --- | --- | --- |
| 10 | `ckpt_E10_TH9.pth` | `ckpt_E19_TH9.pth` | `ckpt_E19_TH9.pth` | `ckpt_E28_TH9.pth` |
| 60 | `ckpt_E60_TH9.pth` | `ckpt_E119_TH9.pth` | `ckpt_E119_TH9.pth` | `ckpt_E178_TH9.pth` |

TH9 is the cyclic task index for a CANDID-PTX classification epoch. The classification module inside the student is head 2, a linear layer with one output.

Exposure 10 is primary because all four runs are still in the high classification band, before the joint runs fall behind classification-only around exposure 20. It is also the early three-task checkpoint used for the gradient analysis (`ckpt_E28_TH9.pth`). Exposure 60 is the last exposure shared by all four runs and the late gradient checkpoint. Exposure 20 is not plotted: it is one point inside a short deficit, and a third exposure would turn the poster into a grid.

Checkpoints are read from `/scratch/hflechsi/Foundation_X/results/part_1_v2/`. The script stops if one is missing.

## Target layer

`backbone[0].layers[3].blocks[-1]`, the last block of Swin-B stage 4.

Classification (`extra_features`) runs patch embedding, then the four Swin stages, then a classification-only LayerNorm, average pool, and head 2. The last stage-4 block is the last spatial map that all three tasks actually compute. The classification LayerNorm and the pool are not used by localization or segmentation, so they are not the target.

Swin-B at 224 input, patch size 4, has a 7×7 token grid and 1024 channels at stage 4. The block returns `(batch, 49, 1024)`. Grad-CAM reshapes that to `(batch, 1024, 7, 7)` before the channel weights. The script stops if the token count is not 49.

The repo copy of `pytorch_grad_cam` is not used. Its `BaseCAM.forward` indexes `outputs[2]` and reshapes that tensor to 230400, which is not this classification head.

Activation checkpointing is turned off for the analysis forward. That flag does not change the loaded weights. It only keeps the backward tape that Grad-CAM needs. The student is in eval mode, so stochastic depth is off.

## Image-selection rule

The pool is the CANDID-PTX classification test list, `CANDIDPTX_cls_test.txt`, with the test2 transform used at evaluation: resize to 224×224, ToTensor, chest X-ray normalization (mean 0.5056, std 0.252). The same transformed tensor is given to every model.

An image is eligible when its label is positive and every exposure-10 student gives sigmoid probability at least 0.5. Six images are taken at evenly spaced ranks of the four-model mean probability. The rule is applied before any map is drawn. The script does not relax it if fewer than six images qualify.

Exposure 60 is drawn on those same six images. Selection does not use exposure-60 correctness.

## Spatial ground truth

Classification images are DICOMs under `CANDID-PTX/dataset`. Segmentation lists are DICOM path plus RLE, also under that directory. Localization boxes are COCO JSON over PNGs in `CANDID-PTX/png`. The evaluation localization split is the val JSON (`candidptx_pneumothorax_val`), not the test JSON. The script matches filename stems and records how many classification-test stems hit seg-test, seg-train, loc-val, and loc-test. A stem that appears in more than one file is marked, and the evaluation split wins.

If a selected image has a segmentation mask, the figure's ground-truth column is that mask. Otherwise it is the localization boxes, scaled from the JSON width and height onto the 224 canvas. Energy fraction, pointing-game hit, and CAM-center hit are computed only for those matches. The CAM is not thresholded into a mask, and Dice and IoU are not reported.

## Normalization

Each map is ReLU Grad-CAM, upsampled bilinearly to 224×224. Display divides by that map's own min and max, so every panel uses the same function and the same jet range of 0 to 1. Absolute magnitudes are not on one shared color scale: a shared scale would hide the shape of a weaker map. Concentration and energy fractions use the ReLU map before that display scaling. Overlays use the same denormalized X-ray and the same alpha.

## Selected images

The script writes `gradcam_selected_images.csv` with the path, stem, label, selection reason, probability under each student, and the per-image summaries.

## Limitations

Six images. One run per configuration. Eval-mode maps, not the stochastic training forward. Classification, localization, and segmentation do not use the same file format, so a missing stem match is a real limit on quantitative overlap. Grad-CAM is not a measurement of feature use.

<!-- RESULTS -->

Not computed yet. On SOL:

```bash
bash Results/sol/analysis/gradcam_synergy/run_gradcam_sol.sh
```

<!-- /RESULTS -->
