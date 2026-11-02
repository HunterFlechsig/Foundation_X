#!/usr/bin/env bash
# Submit one Part 1 or Part 2 job with a results-folder name.
# Slurm reads log paths at submission, so this script sets them.
#
#   bash scripts/submit_job.sh scripts/part_1/a_candidptx_cls.sh part_1_v4 once_per_epoch cosine
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: submit_job.sh <sbatch_script> <experiment> [every_pass|once_per_epoch] [fixed|cosine]" >&2
    exit 2
fi

SCRIPT=$(readlink -f "$1")
EXPERIMENT="$2"
TEACHER_UPDATE="${3:-every_pass}"
EMA_MOMENTUM="${4:-fixed}"

case "$TEACHER_UPDATE" in
    every_pass|once_per_epoch) ;;
    *) echo "teacher_update must be every_pass or once_per_epoch: $TEACHER_UPDATE" >&2; exit 2 ;;
esac
case "$EMA_MOMENTUM" in
    fixed|cosine) ;;
    *) echo "ema_momentum must be fixed or cosine: $EMA_MOMENTUM" >&2; exit 2 ;;
esac

RESULTS_ROOT=/scratch/hflechsi/Foundation_X/results
LOG_DIR="${RESULTS_ROOT}/${EXPERIMENT}/logs"
mkdir -p "$LOG_DIR"

echo "Submitting $(basename "$SCRIPT")"
echo "Experiment     : ${EXPERIMENT}"
echo "Teacher update : ${TEACHER_UPDATE}"
echo "EMA momentum   : ${EMA_MOMENTUM}"
echo "Logs           : ${LOG_DIR}"

sbatch \
    --export=ALL,EXPERIMENT="${EXPERIMENT}",TEACHER_UPDATE="${TEACHER_UPDATE}",EMA_MOMENTUM="${EMA_MOMENTUM}" \
    -o "${LOG_DIR}/%x_%j.out" \
    -e "${LOG_DIR}/%x_%j.err" \
    "$SCRIPT"
