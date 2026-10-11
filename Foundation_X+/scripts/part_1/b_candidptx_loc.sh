#!/bin/bash
#SBATCH --job-name=fx_candid_loc
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -p public
#SBATCH -t 1-00:00:00
#SBATCH -q public
#SBATCH -A grp_jliang12
#SBATCH -o /scratch/hflechsi/Foundation_X/results/part_1/logs/%x_%j.out
#SBATCH -e /scratch/hflechsi/Foundation_X/results/part_1/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hflechsi@asu.edu

# CANDID-PTX only: candidptxLOC
# 200 cycles means each selected task is trained 200 times.
# The trainer starts at epoch 1 and stops before total_epochs.

module load mamba/latest
module load cuda-11.6.2-gcc-12.1.0
source activate foundation_x_sol

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=/scratch/hflechsi/Foundation_X/results/part_1/candidptx_loc
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

cyclictask=candidptxLOC
CYCLES=200
IFS='_' read -ra TASKS <<< "$cyclictask"
total_epochs=$(( ${#TASKS[@]} * CYCLES + 1 ))

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOGFILE"

cd /home/hflechsi/Foundation_X/Foundation_X+/

~/.conda/envs/foundation_x_sol/bin/python main_Consolidated.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --serverC $SERVER --modelEMA $EMAMODE --lockrelease --saveAllModel \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
