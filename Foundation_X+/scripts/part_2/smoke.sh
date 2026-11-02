#!/bin/bash
# Interactive check that every part_2 dataset opens.
# Jobs (h) through (k) are subsets of (l), so this one run covers all five.
# From an existing SOL GPU session, in Foundation_X+/:
#   bash scripts/part_2/smoke.sh
# --debug stops each pass after a few batches. One cycle visits each task once.

module load mamba/latest
module load cuda-11.6.2-gcc-12.1.0
source activate foundation_x_sol

cd /home/hflechsi/Foundation_X/Foundation_X+/

CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=/scratch/hflechsi/Foundation_X/results/part_2/smoke
backbone_dir=/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar

BACKBONEMODEL=Swin-B
IMGSIZE=224
coco_path=/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
DATASETFILE=foundation6Ark6_datasets

lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4

BATCHSIZE=24
num_workers=12
INIT=ark
opt=adamw
SERVER=SOL
EMAMODE=True_Epoch

cyclictask=candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG
CYCLES=1
IFS='_' read -ra TASKS <<< "$cyclictask"
total_epochs=$(( ${#TASKS[@]} * CYCLES + 1 ))

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29520
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOGFILE"

~/.conda/envs/foundation_x_sol/bin/python main_Consolidated.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --serverC $SERVER --modelEMA $EMAMODE --lockrelease --saveAllModel --debug \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
