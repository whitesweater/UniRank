# SISA single-seed 20262028 protocol audit

## Outcome

- Slurm array: `548166`
- Execution window: `2026-08-25 21:01:04` to `2026-08-26 00:48:19` (wall clock `03:47:15`)
- Successful tasks: `16/16`, all `COMPLETED / 0:0`
- Protocol: `2 x H100 80GB`, per-GPU batch `16384`, accumulation `1`, global batch `32768`
- Seed bundle: base `20262028`, dataloader `2027`, SISA parameters `20260822`
- Attempts: all task runs used `attempt1`; no retry and no four-GPU OOM fallback
- Final metrics: `68` finite AUC/logloss pairs; see `summary.md` and `metrics.csv`
- Error audit: no CUDA OOM, NCCL error/timeout, traceback, killed process, or NaN signal

## Resource usage

Peak GPU memory is the maximum sampled per-device allocation from the 30-second telemetry files.

| Task | Model | Dataset | Elapsed | Peak GPU memory (MiB) |
|---:|---|---|---:|---:|
| 0 | Zenith | MerRec_Action | 00:27:54 | 46,370 |
| 1 | HyFormer | MerRec_Action | 00:38:27 | 70,425 |
| 2 | HiFormer | MerRec_Action | 00:28:40 | 35,175 |
| 3 | RankMixer | MerRec_Action | 00:22:46 | 21,536 |
| 4 | Zenith | KuaiRand_Video_Action | 00:46:01 | 56,901 |
| 5 | HyFormer | KuaiRand_Video_Action | 01:00:39 | 71,847 |
| 6 | HiFormer | KuaiRand_Video_Action | 00:49:35 | 46,098 |
| 7 | RankMixer | KuaiRand_Video_Action | 00:34:13 | 20,227 |
| 8 | Zenith | QK_Video_Action | 00:34:25 | 19,481 |
| 9 | HyFormer | QK_Video_Action | 01:11:22 | 64,893 |
| 10 | HiFormer | QK_Video_Action | 00:32:44 | 19,379 |
| 11 | RankMixer | QK_Video_Action | 00:28:35 | 13,405 |
| 12 | Zenith | TencentGR_10M_Action | 01:58:38 | 39,831 |
| 13 | HyFormer | TencentGR_10M_Action | 02:43:41 | 71,717 |
| 14 | HiFormer | TencentGR_10M_Action | 02:16:45 | 43,365 |
| 15 | RankMixer | TencentGR_10M_Action | 01:40:01 | 19,389 |

The maximum sampled allocation was `71,847 MiB` on task 5, below the H100 telemetry capacity of `81,559 MiB`.

## Training sample coverage

Coverage is derived from the rank-local blocked-loader allocation lines in one representative run per dataset. Raw rows are the sum assigned to ranks 0 and 1. Processed rows equal the sum of rank-local full batches multiplied by `16384`; the difference is the `drop_last` remainder. The allocation is dataset- and seed-specific and is shared by all four models.

| Dataset | Assigned rows | Full batches across ranks | Processed rows | Dropped rows | Coverage |
|---|---:|---:|---:|---:|---:|
| MerRec_Action | 144,108,235 | 8,792 | 144,048,128 | 60,107 | 99.958290% |
| KuaiRand_Video_Action | 256,110,233 | 15,615 | 255,836,160 | 274,073 | 99.892986% |
| QK_Video_Action | 394,647,318 | 24,074 | 394,428,416 | 218,902 | 99.944532% |
| TencentGR_10M_Action | 586,117,494 | 35,764 | 585,957,376 | 160,118 | 99.972682% |

## Previous/new blocked-loader coverage comparison

The assigned row counts are identical between the previous and new protocols for all 12 dataset splits, but the new `2 × 16384` protocol processes fewer rows because blocked-loader `drop_last` boundaries changed. For each row below, `processed rows = sum(rank-local full batches) × per-GPU batch`; coverage is processed rows divided by assigned rows. The allocation signature was identical across all four models within each dataset/protocol, so one representative model log per dataset was sufficient after the four-model consistency check.

| Dataset | Split | Assigned rows | Previous processed | Previous coverage | New processed | New coverage | Added dropped rows | Coverage Δ (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MerRec_Action | train | 144,108,235 | 144,072,704 | 99.975344% | 144,048,128 | 99.958290% | +24,576 | -0.017054 |
| MerRec_Action | valid | 16,515,570 | 16,482,304 | 99.798578% | 16,449,536 | 99.600171% | +32,768 | -0.198407 |
| MerRec_Action | test | 11,681,154 | 11,640,832 | 99.654812% | 11,599,872 | 99.304161% | +40,960 | -0.350650 |
| KuaiRand_Video_Action | train | 256,110,233 | 255,959,040 | 99.940966% | 255,836,160 | 99.892986% | +122,880 | -0.047979 |
| KuaiRand_Video_Action | valid | 34,334,869 | 34,308,096 | 99.922024% | 34,258,944 | 99.778869% | +49,152 | -0.143155 |
| KuaiRand_Video_Action | test | 33,019,342 | 32,980,992 | 99.883856% | 32,964,608 | 99.834237% | +16,384 | -0.049619 |
| QK_Video_Action | train | 394,647,318 | 394,526,720 | 99.969442% | 394,428,416 | 99.944532% | +98,304 | -0.024909 |
| QK_Video_Action | valid | 50,079,027 | 50,053,120 | 99.948268% | 50,036,736 | 99.915551% | +16,384 | -0.032716 |
| QK_Video_Action | test | 48,579,958 | 48,562,176 | 99.963396% | 48,529,408 | 99.895945% | +32,768 | -0.067452 |
| TencentGR_10M_Action | train | 586,117,494 | 586,047,488 | 99.988056% | 585,957,376 | 99.972682% | +90,112 | -0.015374 |
| TencentGR_10M_Action | valid | 17,846,140 | 17,809,408 | 99.794174% | 17,793,024 | 99.702367% | +16,384 | -0.091807 |
| TencentGR_10M_Action | test | 153,243,512 | 153,206,784 | 99.976033% | 153,157,632 | 99.943958% | +49,152 | -0.032074 |

Representative previous/new logs were `14810_15`/`548166_2` for MerRec, `14850_11`/`548166_6` for KuaiRand, `14811_9`/`548166_10` for QK-Video, and `543257_3`/`548166_14` for TencentGR. Consequently, the old and new final test metrics are not evaluated on exactly the same `drop_last` sample subsets; the largest coverage difference is MerRec test at `-0.350650` percentage points.

## Verification

- Result collector: `complete=16/16`, `metrics=68`
- CPU regression suite: `65/65` passed
- Modified shell syntax, Python compilation, and `git diff --check`: passed
- Independent per-task audit confirmed one completion marker, finite final metrics, non-empty log/model checkpoints, two-device telemetry, correct H100/protocol/seed evidence, and zero retry artifacts for every task
