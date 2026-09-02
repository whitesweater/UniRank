# SISA single-seed 20262029 protocol audit

## Outcome

- Experiment scope: one new experiment seed, `20262029`, across 16 model×dataset tasks.
- Internal RNG substreams: dataloader `2028` and SISA parameters `20260823`; these are part of the same experiment and are not additional experiment seeds.
- Primary Slurm array: `549595`; task-0 same-protocol retry: `549797_0`.
- Execution window: `2026-08-26 01:44:31` to `2026-08-26 06:08:40` (wall clock `04:24:09`).
- Successful formal tasks: `16/16`, all selected runs `COMPLETED / 0:0`.
- Formal protocol: `2 × H100 80GB`, world size `2`, per-GPU batch `16384`, accumulation `1`, global batch `32768`, one epoch, BF16.
- Four-GPU fallback: not used. No formal run encountered OOM.
- Final metrics: `68` finite AUC/logloss pairs; see `summary.md` and `metrics.csv`.
- Formal-run error audit: no CUDA OOM, NCCL error/timeout, traceback, killed process, or NaN signal.

## Task-0 retry and file isolation

Primary task `549595_0` finished training, final test evaluation, and checkpoint preservation, but the original runner then exited `2:0` at line 193 with `unexpected EOF while looking for matching '"'`. This was a post-training shell syntax failure, not an OOM or model failure.

- Attempt 1 checkpoint/log were retained independently: `9,081,174,367` bytes / `10,595` bytes.
- After repairing the runner, `549797_0` reran the same `2 × H100`, batch `16384`, accumulation `1` protocol as `attempt2` and completed `0:0` in `00:27:16`.
- Attempt 2 checkpoint/log are the formal task-0 result: `9,081,174,367` bytes / `10,548` bytes.
- The collector override is `0=549797:ws2_bs16384_acc1:2`; attempt 1 remains audit evidence and does not enter `runs.csv` or `metrics.csv`.

## Resource usage

Peak GPU memory is the maximum sampled per-device allocation from the 30-second telemetry files. Every formal task used two `NVIDIA H100 80GB HBM3` devices, and every task reached a sampled utilization of 100% on at least one device.

| Task | Model | Dataset | Formal job | Elapsed | Peak GPU memory (MiB) |
|---:|---|---|---|---:|---:|
| 0 | Zenith | MerRec_Action | 549797_0 | 00:27:16 | 46,372 |
| 1 | HyFormer | MerRec_Action | 549595_1 | 00:36:05 | 70,425 |
| 2 | HiFormer | MerRec_Action | 549595_2 | 00:27:58 | 35,175 |
| 3 | RankMixer | MerRec_Action | 549595_3 | 00:21:45 | 21,539 |
| 4 | Zenith | KuaiRand_Video_Action | 549595_4 | 00:44:27 | 56,901 |
| 5 | HyFormer | KuaiRand_Video_Action | 549595_5 | 01:00:13 | 71,842 |
| 6 | HiFormer | KuaiRand_Video_Action | 549595_6 | 00:48:23 | 46,101 |
| 7 | RankMixer | KuaiRand_Video_Action | 549595_7 | 00:33:03 | 20,227 |
| 8 | Zenith | QK_Video_Action | 549595_8 | 00:35:25 | 19,481 |
| 9 | HyFormer | QK_Video_Action | 549595_9 | 01:10:19 | 64,893 |
| 10 | HiFormer | QK_Video_Action | 549595_10 | 00:36:32 | 19,379 |
| 11 | RankMixer | QK_Video_Action | 549595_11 | 00:25:46 | 13,400 |
| 12 | Zenith | TencentGR_10M_Action | 549595_12 | 01:59:59 | 39,831 |
| 13 | HyFormer | TencentGR_10M_Action | 549595_13 | 02:45:56 | 71,717 |
| 14 | HiFormer | TencentGR_10M_Action | 549595_14 | 02:12:21 | 43,365 |
| 15 | RankMixer | TencentGR_10M_Action | 549595_15 | 01:40:19 | 19,389 |

The maximum sampled allocation was `71,842 MiB` on task 5, below the H100 telemetry capacity of `81,559 MiB`.

## Training sample coverage and optimizer steps

The blocked loader applies `drop_last` independently at each parquet block. Processed rows equal the sum of full block batches multiplied by `16384`. With accumulation `1`, each rank's full-batch count is also its optimizer-step count. The batch signature was identical across all four models for each dataset.

| Dataset | Assigned rows | Rank 0 / rank 1 optimizer steps | Full batches across ranks | Processed rows | Dropped rows | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| MerRec_Action | 144,108,235 | 4,395 / 4,397 | 8,792 | 144,048,128 | 60,107 | 99.958290% |
| KuaiRand_Video_Action | 256,110,233 | 7,804 / 7,811 | 15,615 | 255,836,160 | 274,073 | 99.892986% |
| QK_Video_Action | 394,647,318 | 12,038 / 12,036 | 24,074 | 394,428,416 | 218,902 | 99.944532% |
| TencentGR_10M_Action | 586,117,494 | 17,881 / 17,883 | 35,764 | 585,957,376 | 160,118 | 99.972682% |

Reference seed `20262028` used the same two-GPU protocol and has the same train/validation/test block boundaries and processed-row counts. Therefore the `20262029` versus `20262028` comparison is not confounded by a batch-size or `drop_last` coverage change; only the experiment seed and its internal random streams differ.

## Verification

- Result collector: `complete=16/16`, `metrics=68`; task 0 is explicitly bound to `549797_0` attempt 2.
- Strict CSV audit: 16 unique task IDs, 68 aligned finite metric rows, one protocol, one experiment seed, and expected per-dataset label counts.
- Checkpoints: 16 formal non-empty `.model` files plus 16 non-empty `.log` files; formal model bytes total `46,836,605,593`.
- Same-protocol comparison: 68 aligned labels across 16 cells; outputs are under `comparison_vs_seed20262028/`.
- Portable report delivery: artifact validation, packaging, browser verification, source dialog, desktop `1440` px, and mobile `390` px checks passed; see `comparison_vs_seed20262028/report/delivery_receipt.json`.
- CPU regression suite: `78/78` passed.
- Slurm/shell syntax, Python compilation, and `git diff --check`: passed.
- Reference result protection: all four seed-`20262028` audit hashes remain unchanged from their pre-run values.
