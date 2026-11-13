# Spatial synergy synthesis

This file is replaced by `write_synthesis.py` after both SOL jobs have written their JSON summaries.

The ranking rules are fixed before those numbers exist:

- A Grad-CAM concentration or overlap pattern is at most SUGGESTIVE. Six selected images do not support a claim about feature use.
- Gradient cosine is at most SUGGESTIVE. Aligned, conflicting, and neutral are descriptive bins (mean above 0.1 with at least 70% of batches positive; mean below −0.1 with at least 70% negative; absolute mean at most 0.1). They are not a test, and they do not prove why AUC moves.
- The within-cycle AUC results stay the SUPPORTED performance findings.
- The poster gets the within-cycle AUC panel, plus at most one of Grad-CAM (exposure 10) or the gradient figure. A directional cosine takes the second slot over Grad-CAM, because a projected poster cannot hold the image rows. If neither new result is directional, the poster keeps only the AUC panel.

```bash
python Results/sol/analysis/spatial_synergy/write_synthesis.py
```
