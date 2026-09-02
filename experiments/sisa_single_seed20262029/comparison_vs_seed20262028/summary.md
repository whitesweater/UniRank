# SISA seed/protocol comparison snapshot

Candidate seed 20262029 is compared with reference seed 20262028 under the same 2×H100, world-size 2, per-GPU batch 16384, accumulation 1 protocol. The dataloader and SISA initialization random streams intentionally advance with the experiment seed, so these are descriptive two-point repeatability deltas, not a variance or significance estimate.

- Aligned labels: **68** across **16** cells
- Label-weighted mean ΔAUC: **-0.001123**
- Cell-macro mean ΔAUC: **-0.001516**
- AUC improved: **33/68** labels and **5/16** cells
- Label-weighted mean Δlogloss: **+0.000284** (negative is better)
- Logloss improved: **27/68** labels and **9/16** cells
- AUC changes within ±0.005: **54/68**
- Same top model for each dataset-label task: **10/17**

## Model-level cell-macro deltas

| Model | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |
|---|---:|---:|---:|---:|---:|---:|
| HiFormer | 4 | 17 | -0.001975 | 0/4 | +0.001274 | 1/4 |
| HyFormer | 4 | 17 | -0.000441 | 1/4 | +0.000050 | 3/4 |
| RankMixer | 4 | 17 | +0.000010 | 3/4 | -0.000461 | 4/4 |
| Zenith | 4 | 17 | -0.003659 | 1/4 | +0.001569 | 1/4 |

## Dataset-level cell-macro deltas

| Dataset | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |
|---|---:|---:|---:|---:|---:|---:|
| QK_Video_Action | 4 | 16 | -0.000145 | 1/4 | -0.000076 | 4/4 |
| KuaiRand_Video_Action | 4 | 24 | +0.000405 | 2/4 | -0.000235 | 2/4 |
| TencentGR_10M_Action | 4 | 8 | -0.003563 | 1/4 | +0.002391 | 1/4 |
| MerRec_Action | 4 | 20 | -0.002762 | 1/4 | +0.000352 | 2/4 |
