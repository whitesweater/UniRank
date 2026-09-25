#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=rise-targeted-controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-2%2
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if [[ "$task_id" != 0 && "$task_id" != 1 && "$task_id" != 2 ]]; then
  echo "Invalid targeted task: $task_id" >&2
  exit 2
fi
exec .venv/bin/python experiments/rise_sample_analysis/scripts/run_targeted_controls.py \
  --task-id "$task_id" --attempt "${RISE_TARGETED_ATTEMPT:-1}"
