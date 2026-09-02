# SISA seed/protocol comparison snapshot

The new seed bundle has a small positive mean AUC shift but weaker logloss consistency. Because GPU topology, per-GPU batch, blocked-loader drop boundaries, dataloader seed, and SISA initialization seed also changed, these are descriptive repeatability deltas, not a pure seed effect or significance estimate.

- Aligned labels: **68** across **16** cells
- Label-weighted mean ΔAUC: **+0.000833**
- Cell-macro mean ΔAUC: **+0.001619**
- AUC improved: **32/68** labels and **9/16** cells
- Label-weighted mean Δlogloss: **+0.000812** (negative is better)
- Logloss improved: **23/68** labels and **4/16** cells
- AUC changes within ±0.005: **48/68**
- Same top model for each dataset-label task: **11/17**

## Model-level cell-macro deltas

| Model | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |
|---|---:|---:|---:|---:|---:|---:|
| HiFormer | 4 | 17 | +0.003345 | 3/4 | +0.000034 | 2/4 |
| HyFormer | 4 | 17 | +0.001215 | 2/4 | +0.000782 | 1/4 |
| RankMixer | 4 | 17 | -0.000418 | 2/4 | +0.001805 | 1/4 |
| Zenith | 4 | 17 | +0.002334 | 2/4 | +0.001383 | 0/4 |

## Dataset-level cell-macro deltas

| Dataset | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |
|---|---:|---:|---:|---:|---:|---:|
| QK_Video_Action | 4 | 16 | -0.000392 | 1/4 | +0.000449 | 0/4 |
| KuaiRand_Video_Action | 4 | 24 | -0.001028 | 2/4 | +0.001014 | 2/4 |
| TencentGR_10M_Action | 4 | 8 | +0.005861 | 3/4 | +0.002259 | 1/4 |
| MerRec_Action | 4 | 20 | +0.002035 | 3/4 | +0.000281 | 1/4 |
