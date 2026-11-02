#!/bin/bash
set -euo pipefail
# Submit all seven CANDID-PTX part_1 jobs.
#   bash scripts/part_1/launch_all.sh <experiment> [every_pass|once_per_epoch] [fixed|cosine]
#   bash scripts/part_1/launch_all.sh part_1_v4 once_per_epoch cosine

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/part_1/launch_all.sh <experiment> [every_pass|once_per_epoch] [fixed|cosine]" >&2
    exit 2
fi
EXPERIMENT="$1"
TEACHER_UPDATE="${2:-every_pass}"
EMA_MOMENTUM="${3:-fixed}"

for script in \
    a_candidptx_cls.sh \
    b_candidptx_loc.sh \
    c_candidptx_seg.sh \
    d_candidptx_cls_loc.sh \
    e_candidptx_cls_seg.sh \
    f_candidptx_loc_seg.sh \
    g_candidptx_cls_loc_seg.sh
do
    "$SCRIPT_DIR/../submit_job.sh" "$SCRIPT_DIR/$script" "$EXPERIMENT" "$TEACHER_UPDATE" "$EMA_MOMENTUM"
done
