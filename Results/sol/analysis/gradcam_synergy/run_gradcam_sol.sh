#!/usr/bin/env bash
# Classification Grad-CAM for the four Part I v2 students.
# Run on SOL from a login node, or submit with sbatch.
# Does not train and does not modify checkpoints.
#
#   bash Results/sol/analysis/gradcam_synergy/run_gradcam_sol.sh
#   sbatch Results/sol/analysis/gradcam_synergy/run_gradcam_sol.sh

#SBATCH --job-name=fx_gradcam_p1v2
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

python "$SCRIPT_DIR/gradcam_analysis.py" \
    --results-root /scratch/hflechsi/Foundation_X/results/part_1_v2 \
    --output-dir "$SCRIPT_DIR" \
    --batch-size 16 \
    --num-workers 4
