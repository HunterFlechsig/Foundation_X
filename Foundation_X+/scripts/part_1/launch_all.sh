#!/bin/bash
# Submit all seven CANDID-PTX part_1 jobs. Run from anywhere:
#   bash Foundation_X+/part_1/launch_all.sh

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
mkdir -p /scratch/hflechsi/Foundation_X/results/part_1/logs

for script in \
    a_candidptx_cls.sh \
    b_candidptx_loc.sh \
    c_candidptx_seg.sh \
    d_candidptx_cls_loc.sh \
    e_candidptx_cls_seg.sh \
    f_candidptx_loc_seg.sh \
    g_candidptx_cls_loc_seg.sh
do
    sbatch "$script"
done
