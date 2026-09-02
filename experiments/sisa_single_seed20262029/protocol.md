# SISA single-seed 20262029 protocol

## Scope

- Matrix: HiFormer, HyFormer, RankMixer, Zenith × QK-Video, KuaiRand, TencentGR, MerRec
- Logical tasks: 16 SISA runs
- Experiment seed: `20262029` (this study adds one new experiment seed only)
- Internal dataloader RNG substream: `2028`
- Internal SISA-parameter RNG substream: `20260823`
- The two internal RNG values above belong to the same `20262029` experiment and are not additional experiment seeds.
- Primary protocol: `2 × H100 80GB`, per-GPU batch `16384`, accumulation `1`, global batch `32768`
- Epochs: `1`; max sequence length: `100`; BF16 enabled
- Array concurrency: `%8`; CPUs: `16` per task; memory: `480G` per task
- OOM recovery rule: use `4 × H100`, per-GPU batch `8192`, only after a confirmed OOM. No formal task in this run required that fallback.

## Isolation and retention

- Slurm logs: `logs/unirank-sisa-seed20262029-ws2-<job>_<task>.out`
- Telemetry: `artifacts/sisa_single_seed20262029/telemetry/`
- Result collector output: `experiments/sisa_single_seed20262029/results/`
- Checkpoint/log model ID suffix: `sisa_seed20262029_ws2_bs16384_acc1_attempt1`
- Best `.model` checkpoints are preserved after test evaluation.
- Intermediate checkpoints are retained below `checkpoints/<dataset>/archive/<model_id>/<session_id>/`.
- The runner refuses to start if the target seed-specific `.log` or `.model` path already exists.

## Submission

```bash
sbatch \
  --job-name=unirank-sisa-seed20262029-ws2 \
  --export=ALL,SISA_SEED=20262029,SISA_DATALOADER_SEED=2028,SISA_PARAMETER_SEED=20260823,SISA_STUDY=sisa_single_seed20262029 \
  scripts/submit_sisa_single_seed_acd.sbatch
```

Array job ID: `549595`.

## Actual outcome and retry

- All 16 formal results used `2 × H100`, per-GPU batch `16384`, accumulation `1`.
- Primary task 0 (`549595_0`) completed training, test metrics, and checkpoint preservation, then exited `2:0` because the original runner ended with an unmatched quote at line 193.
- The failure was not OOM, NCCL, or numerical. After repairing the runner, task 0 was rerun with the same two-GPU protocol as `549797_0`, `attempt2`, and completed `0:0`.
- Formal collection selects `549797_0` for task 0. The independent `attempt1` files remain preserved as audit evidence and were not overwritten.
