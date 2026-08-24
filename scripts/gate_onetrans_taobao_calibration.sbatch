#!/bin/bash
#SBATCH --job-name=unirank-strict-gate
#SBATCH --account=lthpc
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${CALIBRATION_JOB_ID:?CALIBRATION_JOB_ID must be exported by sbatch}"
calibration_task_id="${CALIBRATION_TASK_ID:-4}"
slurm_log="logs/unirank-sisa-strict-${CALIBRATION_JOB_ID}_${calibration_task_id}.out"
checkpoint_log="checkpoints/Taobao_Action/SISA_strict_OneTrans_Taobao_Action_baseline_seed20262027.log"

.venv/bin/python scripts/audit_onetrans_taobao_calibration.py \
  --slurm-log "$slurm_log" \
  --checkpoint-log "$checkpoint_log" \
  --tolerance 0.01

echo "STRICT_CALIBRATION_GATE_COMPLETE calibration_job=${CALIBRATION_JOB_ID}_${calibration_task_id}"
