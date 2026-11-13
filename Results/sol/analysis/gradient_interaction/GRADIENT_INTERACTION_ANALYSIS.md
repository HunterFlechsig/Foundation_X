# Gradient interaction, CANDID-PTX Part I v2

Three-task student only. No retraining and no optimizer step. The table below the marker is written by `gradient_interaction.py` after it runs on SOL.

## Scientific question

Do classification, localization, and segmentation produce compatible or conflicting gradients on the shared Foundation X+ backbone?

This is aimed at the within-cycle AUC result, not a replacement for it:

- Localization updates are about neutral for classification.
- Segmentation updates often lower classification AUC, and the next classification update recovers it.

The cosine says whether those task gradients point together or apart. It does not prove why the AUC moves.

## Checkpoints

Both files are from `candidptx_cls_loc_seg`. The epoch-to-exposure mapping was checked against the Part I v2 student classification rows.

| Role | Exposure | Epoch | File |
| --- | --- | --- | --- |
| Early, where the first 20 segmentation updates all lowered classification AUC | 10 | 28 | `ckpt_E28_TH9.pth` |
| Late, the last classification checkpoint, where the segmentation effect is still negative and smaller | 60 | 178 | `ckpt_E178_TH9.pth` |

The loaded weights are the student `model` entry. The teacher and the optimizer state are not loaded.

## Primary losses

Each task is back-propagated alone, after `zero_grad`, and the gradients are read before the next task. Nothing is stepped.

- Classification: `BCEWithLogitsLoss` on head 2. The training step mixes this with feature consistency at coefficient 0.5. That mixture is not in the primary gradient.
- Localization: the weighted detection criterion used by `train_one_epoch`, including the auxiliary decoder terms that are in `weight_dict`. With `use_dn` true, the forward receives the targets, as in training. Teacher consistency is not added. The encoder L2 term is not added; it is on `transformer.encoder`, not on the shared stages, so it would not have entered these cosine vectors anyway.
- Segmentation: dice loss from `extra_features_seg_sharedLocSeg`, head 2. Detector boxes are produced in eval mode under `no_grad`, threshold 0.3, and passed in as Python lists, which is how the training step stops the segmentation loss from flowing through the detector. Teacher consistency is not added.

## Shared parameters

Included:

- `backbone.0.patch_embed`
- `backbone.0.layers.0` through `backbone.0.layers.3` (Swin stages 1–4, including patch merging inside the stages)

Excluded: classification LayerNorm and pooling, classification heads, `Locnorm`, `Segnorm`, the segmentation pyramid and heads, and the detection transformer. Localization does apply `Locnorm` on the stages it returns, and that gradient still flows back into the shared stages. The norm parameters themselves are not in the cosine.

The script checks that each prefix matches at least one parameter and stops if a prefix is empty.

## Batches

Default 20 minibatches per task per checkpoint, batch size 4, seeds 1, 2, and 3 for the three loaders. Training used batch size 24. Batch size 4 is a memory choice for a detection backward with activation checkpointing turned off. The cosine is an average over batches because one batch is not the task gradient.

The loaders are the training loaders, each with its own seeded sampler. Classification, localization, and segmentation batches are different images. This is not a same-image gradient comparison. Preprocessing is the preprocessing each loader used in training.

The student is in eval mode, so stochastic depth is off and the three backwards see the same subnetwork. That is the task-loss direction at these weights, not a noisy training-step sample.

## Statistics

For each checkpoint, stage, and pair the CSV reports mean, median, standard deviation, minimum, maximum, fraction positive, fraction negative, and the number of finite batches. No significance test.

Descriptive bins, fixed before the run: aligned if the mean is above 0.1 and at least 70% of batches are positive; conflicting if the mean is below −0.1 and at least 70% are negative; neutral if the absolute mean is at most 0.1; otherwise mixed.

## Limitations

One run. Eval mode. Batch size 4 rather than 24. Separate images per task. Cosine on a flattened vector can hide a conflict that is large in one stage and small in another; the per-stage rows are there for that. A conflicting cosine is compatible with the AUC drop and is not a demonstration that the drop is caused by that conflict.

<!-- RESULTS -->

Not computed yet. On SOL:

```bash
bash Results/sol/analysis/gradient_interaction/run_gradient_sol.sh
```

Then:

```bash
python Results/sol/analysis/spatial_synergy/write_synthesis.py
```

<!-- /RESULTS -->
