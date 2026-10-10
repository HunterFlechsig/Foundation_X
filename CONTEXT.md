# Foundation X+ pretraining

Cyclic pretraining of one chest X-ray model across many labeled datasets, switching task heads as training proceeds.

## Language

**Dataset**:
A named chest X-ray collection, such as CANDID-PTX or NIH ChestX-ray14.
_Avoid_: server, cluster

**Server**:
The filesystem layout the dataloaders read. SOL is the ASU SOL scratch layout under `/scratch/jliang12/data/`.
_Avoid_: HTC, ASU

**Task**:
One dataset paired with one annotation type: classification, localization, or segmentation. A dataset can have more than one task. CANDID-PTX has all three; CheXpert has classification only.
_Avoid_: head, job

**Task token**:
The script name of a task, formed as the dataset key plus `CLS`, `LOC`, or `SEG`. Example: `candidptxCLS`.
_Avoid_: flag, mode

**Cyclic step**:
One pass that trains a single task token, then may evaluate. A full pretraining run is a sequence of cyclic steps, not one pass over every dataset at once.
_Avoid_: epoch (when more than one task token is selected)

**Checkpoint**:
The weights written after a cyclic step that actually trained. Several checkpoints from one smoke run live in one output folder.
_Avoid_: best model

**Smoke run**:
A pretraining launch that visits each selected task token once under the existing debug batch caps and writes its checkpoints into one results folder.
_Avoid_: full pretraining
