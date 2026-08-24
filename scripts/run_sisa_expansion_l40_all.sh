#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "RUNNER_ERROR: this script must run inside a Slurm allocation" >&2
  exit 2
fi

max_attempts="${SISA_MAX_ATTEMPTS:-2}"
if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNNER_ERROR: SISA_MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

state_root="$repo_root/artifacts/sisa_expansion_l40"
allocation_root="$state_root/allocations/$SLURM_JOB_ID"
completed_root="$state_root/completed"
failed_root="$state_root/failed"
status_file="$allocation_root/status.tsv"
current_file="$state_root/current_task.tsv"
mkdir -p "$allocation_root" "$completed_root" "$failed_root" "$repo_root/logs"
printf '%s\n' "$SLURM_JOB_ID" > "$state_root/current_allocation"
if [[ ! -s "$status_file" ]]; then
  printf 'timestamp\ttask_id\tattempt\tstate\texit_code\tlog\n' > "$status_file"
fi

record_status() {
  local task_id="$1"
  local attempt="$2"
  local state="$3"
  local exit_code="$4"
  local log_file="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%F %T %Z')" "$task_id" "$attempt" "$state" \
    "$exit_code" "$log_file" >> "$status_file"
}

handle_signal() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date '+%F %T %Z')" "${current_task_id:-none}" \
    "${current_attempt:-0}" "allocation_signal" "143" "-" \
    >> "$status_file"
  exit 143
}
trap handle_signal TERM INT

echo "SISA_L40_RUNNER_START allocation=$SLURM_JOB_ID max_attempts=$max_attempts"

preflight_log="$allocation_root/gpu_preflight.log"
if ! srun --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --gres=gpu:l40s:4 --overlap \
  bash -lc '
    set -euo pipefail
    printf "host=%s cuda_visible_devices=%s\n" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  ' > "$preflight_log" 2>&1; then
  echo "RUNNER_ERROR: GPU preflight failed; see $preflight_log" >&2
  exit 3
fi
gpu_count="$(rg -c '^NVIDIA L40S,' "$preflight_log" 2>/dev/null || true)"
if [[ "$gpu_count" != 4 ]]; then
  echo "RUNNER_ERROR: expected four L40S GPUs, observed ${gpu_count:-0}" >&2
  sed -n '1,80p' "$preflight_log" >&2
  exit 3
fi

failed_tasks=()
for current_task_id in $(seq 0 37); do
  completion_file="$completed_root/task_${current_task_id}.ok"
  if [[ -s "$completion_file" ]]; then
    record_status "$current_task_id" 0 skipped 0 "$(sed -n '1p' "$completion_file")"
    echo "SISA_TASK_SKIP task_id=$current_task_id reason=completion_marker"
    continue
  fi

  task_succeeded=false
  for current_attempt in $(seq 1 "$max_attempts"); do
    task_log="$repo_root/logs/unirank-sisa-l40-${SLURM_JOB_ID}_${current_task_id}_attempt${current_attempt}.out"
    printf '%s\t%s\t%s\t%s\n' \
      "$(date '+%F %T %Z')" "$SLURM_JOB_ID" "$current_task_id" \
      "$current_attempt" > "$current_file"
    record_status "$current_task_id" "$current_attempt" running - "$task_log"
    echo "SISA_TASK_START task_id=$current_task_id attempt=$current_attempt log=$task_log"

    srun \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" \
      --gres=gpu:l40s:4 \
      --kill-on-bad-exit=1 \
      --export="ALL,SLURM_ARRAY_TASK_ID=$current_task_id,SLURM_ARRAY_JOB_ID=$SLURM_JOB_ID" \
      bash "$repo_root/scripts/submit_sisa_expansion.sbatch" \
      > "$task_log" 2>&1
    task_exit_code=$?

    if [[ "$task_exit_code" == 0 ]] \
      && rg -q '^SISA_EXPANSION_COMPLETE ' "$task_log"; then
      completion_tmp="$completion_file.tmp.$$"
      printf '%s\n%s\n%s\n' \
        "$task_log" "$SLURM_JOB_ID" "$(date '+%F %T %Z')" \
        > "$completion_tmp"
      mv -f "$completion_tmp" "$completion_file"
      rm -f "$failed_root/task_${current_task_id}.failed"
      record_status "$current_task_id" "$current_attempt" complete 0 "$task_log"
      echo "SISA_TASK_COMPLETE task_id=$current_task_id attempt=$current_attempt"
      task_succeeded=true
      break
    fi

    record_status "$current_task_id" "$current_attempt" failed \
      "$task_exit_code" "$task_log"
    echo "SISA_TASK_FAILED task_id=$current_task_id attempt=$current_attempt exit_code=$task_exit_code"
  done

  if [[ "$task_succeeded" != true ]]; then
    printf '%s\n%s\n%s\n' \
      "$task_log" "$task_exit_code" "$(date '+%F %T %Z')" \
      > "$failed_root/task_${current_task_id}.failed"
    failed_tasks+=("$current_task_id")
  fi
done

rm -f "$current_file"
complete_count="$(find "$completed_root" -maxdepth 1 -type f -name 'task_*.ok' | wc -l)"
if (( ${#failed_tasks[@]} > 0 )); then
  printf 'SISA_L40_RUNNER_INCOMPLETE allocation=%s complete=%s failed=%s\n' \
    "$SLURM_JOB_ID" "$complete_count" "${failed_tasks[*]}" >&2
  exit 1
fi
if [[ "$complete_count" != 38 ]]; then
  echo "SISA_L40_RUNNER_INCOMPLETE allocation=$SLURM_JOB_ID complete=$complete_count expected=38" >&2
  exit 1
fi

printf 'SISA_L40_RUNNER_COMPLETE allocation=%s complete=38\n' "$SLURM_JOB_ID"
