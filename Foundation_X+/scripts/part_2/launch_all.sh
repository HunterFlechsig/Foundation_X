#!/bin/bash
set -euo pipefail
# Submit the five part_2 jobs.
#   bash scripts/part_2/launch_all.sh <experiment> [every_pass|once_per_epoch] [fixed|cosine]
#   bash scripts/part_2/launch_all.sh part_2_v2 every_pass cosine
# Run scripts/part_2/smoke.sh once before this.

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/part_2/launch_all.sh <experiment> [every_pass|once_per_epoch] [fixed|cosine]" >&2
    exit 2
fi
EXPERIMENT="$1"
TEACHER_UPDATE="${2:-every_pass}"
EMA_MOMENTUM="${3:-fixed}"

for script in \
    h_cls.sh \
    i_loc.sh \
    j_seg.sh \
    k_cls_loc.sh \
    l_cls_loc_seg.sh
do
    "$SCRIPT_DIR/../submit_job.sh" "$SCRIPT_DIR/$script" "$EXPERIMENT" "$TEACHER_UPDATE" "$EMA_MOMENTUM"
done
