#!/usr/bin/env bash
#SBATCH --partition=acd_u
#SBATCH --job-name=unirank-sisa-expansion-acd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --array=0-37%4
#SBATCH --output=logs/%x-%A_%a.out

set -uo pipefail

repo_root="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$repo_root"

for required_dir in \
  "$repo_root/logs" \
  "$repo_root/checkpoints" \
  "$repo_root/artifacts/sisa_expansion_acd/failed_attempts"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "ACD_EXPANSION_ERROR missing pre-created directory: $required_dir" >&2
    exit 2
  fi
done

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ACD_EXPANSION_ERROR SLURM_ARRAY_TASK_ID is unset" >&2
  exit 2
fi

mapfile -t gpu_names < <(
  nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/[[:space:]]*$//'
)
if (( ${#gpu_names[@]} != 4 )); then
  echo "ACD_EXPANSION_ERROR expected 4 visible GPUs, got ${#gpu_names[@]}" >&2
  exit 2
fi
first_gpu="${gpu_names[0]}"
for gpu_name in "${gpu_names[@]}"; do
  if [[ "$gpu_name" != "$first_gpu" ]]; then
    echo "ACD_EXPANSION_ERROR mixed GPU models: ${gpu_names[*]}" >&2
    exit 2
  fi
done

echo "ACD_EXPANSION_HARDWARE gpu_count=4 gpu_name=${first_gpu} host=$(hostname)"
echo "ACD_EXPANSION_ALLOCATION cpus=${SLURM_CPUS_PER_TASK:-unset} memory_per_node=${SLURM_MEM_PER_NODE:-unset}"

max_attempts="${SISA_MAX_ATTEMPTS:-2}"
if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACD_EXPANSION_ERROR SISA_MAX_ATTEMPTS must be positive" >&2
  exit 2
fi

for attempt in $(seq 1 "$max_attempts"); do
  echo "ACD_EXPANSION_ATTEMPT_START task_id=${SLURM_ARRAY_TASK_ID} attempt=${attempt}"
  bash "$repo_root/scripts/submit_sisa_expansion.sbatch"
  exit_code=$?
  if (( exit_code == 0 )); then
    echo "SISA_ACD_EXPANSION_COMPLETE task_id=${SLURM_ARRAY_TASK_ID} attempt=${attempt}"
    exit 0
  fi

  checkpoint_glob="$repo_root/checkpoints"/*/SISA_expansion_*_seed20262027.log
  failure_root="$repo_root/artifacts/sisa_expansion_acd/failed_attempts"
  for checkpoint_log in $checkpoint_glob; do
    if [[ -f "$checkpoint_log" ]]; then
      checkpoint_name="$(basename "$checkpoint_log")"
      cp -p "$checkpoint_log" \
        "$failure_root/task_${SLURM_ARRAY_TASK_ID}_attempt_${attempt}_${checkpoint_name}"
    fi
  done
  echo "ACD_EXPANSION_ATTEMPT_FAILED task_id=${SLURM_ARRAY_TASK_ID} attempt=${attempt} exit_code=${exit_code}" >&2
done

echo "ACD_EXPANSION_FAILED task_id=${SLURM_ARRAY_TASK_ID} attempts=${max_attempts}" >&2
exit 1
