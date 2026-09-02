# UniRank SISA expansion — HPC3/ACD final report

Generated: 2026-08-25 (Asia/Shanghai)

## Completion and protocol

- Final valid training elements: **38/38**.
- Every selected element finished with Slurm state `COMPLETED`, exit code `0:0`, finite test metrics, and both experiment completion markers.
- Every element used **4 × NVIDIA H100 80GB HBM3** on the `acd_u` partition.
- Fixed protocol: per-GPU batch 8192, global batch 32768, accumulation 1, one epoch, seed 20262027, BF16, blocked DDP.
- Final records contain 142 task-level label metrics and 71 baseline/SISA label pairs. GPU-pair and metric-pair validation errors are both zero.

Final Slurm mapping:

- Calibration array: `543255`; task 0 uses corrected retry `543887_0`.
- Main array: `543257`.
- Official-config retry: tasks 12/13 use `544786_12` and `544786_13`.
- UltraHSTU SISA recovery: tasks 29/35/37 use `545515_29`, `545515_35`, and `545515_37`.

## Agreement with the UniRank benchmark

- Baselines audited: **19/19**.
- Labels audited: **71/71**.
- Acceptance rule: absolute AUC difference from the checked-in official benchmark must be at most 0.01.
- Result: **all labels passed**.
- Largest absolute difference: **0.009366**, UltraHSTU–KuaiRand `is_click` (official 0.774418, local 0.765052).

## SISA paired results

- Mean label-level ΔAUC: **+0.003802**.
- AUC improved on **48/71** labels and decreased on **23/71**.
- Largest AUC improvement: **+0.027940**, UltraHSTU–KuaiRand `is_click`.
- Largest AUC decrease: **−0.026251**, HyFormer–Taobao `cart`.
- Mean Δlogloss: **−0.002318**; logloss improved on **45/71** labels.

These are single-seed, one-epoch paired results. They are point estimates and do not establish statistical significance.

### Label-weighted results by model

| Model | Labels | Mean ΔAUC | AUC improved | Mean Δlogloss | Logloss improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| HiFormer | 2 | +0.011561 | 2/2 | −0.004235 | 2/2 |
| HyFormer | 21 | +0.003194 | 18/21 | −0.002253 | 16/21 |
| OneTrans | 2 | +0.005123 | 2/2 | −0.000211 | 1/2 |
| RankMixer | 2 | +0.010262 | 2/2 | −0.001743 | 2/2 |
| UltraHSTU | 21 | +0.006353 | 15/21 | −0.004737 | 12/21 |
| UniMixer | 21 | −0.000171 | 7/21 | +0.000025 | 10/21 |
| Zenith | 2 | +0.009569 | 2/2 | −0.002956 | 2/2 |

### Label-weighted results by dataset

| Dataset | Labels | Mean ΔAUC | AUC improved | Mean Δlogloss | Logloss improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| KuaiRand | 18 | +0.005902 | 15/18 | −0.006481 | 14/18 |
| MerRec | 15 | +0.008241 | 11/15 | −0.001670 | 9/15 |
| QK-Video | 12 | −0.000022 | 5/12 | −0.000178 | 7/12 |
| Taobao | 12 | −0.004062 | 7/12 | +0.000048 | 5/12 |
| TencentGR | 14 | +0.006364 | 10/14 | −0.001519 | 10/14 |

The strongest model–dataset means were UltraHSTU–MerRec (+0.012726), HyFormer–MerRec (+0.012394), and HiFormer–TencentGR (+0.011561). The main regression was HyFormer–Taobao (−0.015343 mean ΔAUC across four labels).

## Recovered failures

- Task 0 originally exposed a missing default sequence pooling encoder for preprocessed TencentGR features. The implementation was aligned with the official `MaskedAveragePooling` configuration and task 0 was rerun successfully.
- Tasks 12/13 initially used drifted UniMixer TencentGR parameters. They were rerun with the official benchmark settings (`group_id=user_index`, `token_dim=8`, `num_tokens=128`); the corrected baseline passed the AUC gate with maximum absolute difference 0.006236.
- UltraHSTU SISA tasks 29/35/37 increased the FlexAttention Q/K head dimension from 256 to 272. PyTorch 2.8's compiled Triton kernel exceeded the H100 per-kernel resource limit. The final task-local recovery disabled outer TorchDynamo compilation while preserving the model, sparse mask, SISA calculation, data, batch, precision, seed, and DDP protocol. All three recovery runs crossed the first batch and completed normally with clean final logs.

## Recorded artifacts

- [`results/runs.csv`](results/runs.csv): one row per final selected training element, including Slurm mapping, hardware, protocol validation, completion state, and metric JSON.
- [`results/metrics.csv`](results/metrics.csv): all 142 task-level label metrics.
- [`results/paired_summary.csv`](results/paired_summary.csv): 71 baseline/SISA label comparisons.
- [`results/baseline_audit.csv`](results/baseline_audit.csv): 71 official-paper baseline AUC comparisons and tolerance decisions.
- [`results/unit_summary.csv`](results/unit_summary.csv): 19 model–dataset units derived from the paired metrics and baseline audit for the HTML report.
- [`results/summary.md`](results/summary.md): compact machine-generated completion snapshot.
