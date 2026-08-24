#!/bin/bash
#SBATCH --job-name=unirank-sisa-smoke
#SBATCH --account=lthpc
#SBATCH --partition=rtx4090
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=00:30:00
#SBATCH --array=0-13%2
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONUNBUFFERED=1

experiments=(
  OneTrans_Taobao_Action
  HiFormer_Taobao_Action
  RankMixer_Taobao_Action
  Zenith_Taobao_Action
  UniMixer_Taobao_Action
  HyFormer_Taobao_Action
  UltraHSTU_Taobao_Action
)

model_index=$((SLURM_ARRAY_TASK_ID / 2))
setting_index=$((SLURM_ARRAY_TASK_ID % 2))
experiment="${experiments[$model_index]}"

arguments=(--expid "$experiment" --batch-size 16 --max-len 32)
setting=baseline
if (( setting_index == 1 )); then
  arguments+=(--sisa-enabled)
  setting=sisa
fi

echo "host=$(hostname) job_id=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} experiment=${experiment} setting=${setting}"
.venv/bin/python scripts/unirank_gpu_smoke.py "${arguments[@]}"
