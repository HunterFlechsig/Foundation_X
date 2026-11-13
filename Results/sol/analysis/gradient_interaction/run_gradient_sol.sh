#!/usr/bin/env bash
# Shared-backbone gradient cosine for the Part I v2 three-task student.
# Run on SOL. Does not train and does not modify checkpoints.
#
#   bash Results/sol/analysis/gradient_interaction/run_gradient_sol.sh
#   sbatch Results/sol/analysis/gradient_interaction/run_gradient_sol.sh

#SBATCH --job-name=fx_grad_p1v2
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -p public
#SBATCH -t 0-08:00:00
#SBATCH -q public
#SBATCH -A grp_jliang12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hflechsi@asu.edu

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
FX_PLUS="$REPO_ROOT/Foundation_X+"

module load mamba/latest
module load cuda-11.6.2-gcc-12.1.0
source activate foundation_x_sol

cd "$FX_PLUS"
export PYTHONPATH="$REPO_ROOT:$FX_PLUS:${PYTHONPATH:-}"

python "$SCRIPT_DIR/gradient_interaction.py" \
    --results-root /scratch/hflechsi/Foundation_X/results/part_1_v2 \
    --output-dir "$SCRIPT_DIR" \
    --batch-size 4 \
    --batches 20 \
    --num-workers 4 \
    --seed 0
