#!/bin/bash
# Submit the five part_2 jobs. From Foundation_X+/:
#   bash scripts/part_2/launch_all.sh
# Run scripts/part_2/smoke.sh once before this.

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
mkdir -p /scratch/hflechsi/Foundation_X/results/part_2/logs

for script in \
    h_cls.sh \
    i_loc.sh \
    j_seg.sh \
    k_cls_loc.sh \
    l_cls_loc_seg.sh
do
    sbatch "$script"
done
