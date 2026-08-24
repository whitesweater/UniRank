#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

interval_seconds="${MONITOR_INTERVAL_SECONDS:-1800}"
job_name="${SISA_ALLOCATION_JOB_NAME:-unirank-sisa-l40-all}"
state_root="$repo_root/artifacts/sisa_expansion_l40"
monitor_log="$repo_root/logs/sisa-expansion-l40-monitor.log"
error_pattern='Traceback|CUDA out of memory|OutOfMemory|oom-kill|NCCL|Killed|Segmentation fault|(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)'

if ! [[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "MONITOR_ERROR: MONITOR_INTERVAL_SECONDS must be positive" >&2
  exit 2
fi
mkdir -p "$state_root" "$repo_root/logs"

snapshot() {
  local allocation_id=""
  local completed_count=0
  local failed_count=0
  local latest_log=""
  local errors=0
  local progress=""

  echo "===== $(date '+%F %T %Z') ====="
  echo "[allocation queue]"
  squeue -u "$USER" -n "$job_name" \
    -o '%i|%P|%q|%T|%M|%S|%R' 2>&1

  echo "[node allocation]"
  scontrol show node lthpc -o 2>&1 \
    | sed -n 's/.*\(Gres=[^ ]*\).*\(AllocTRES=[^ ]*\).*/\1 \2/p'

  completed_count="$(find "$state_root/completed" -maxdepth 1 -type f -name 'task_*.ok' 2>/dev/null | wc -l)"
  failed_count="$(find "$state_root/failed" -maxdepth 1 -type f -name 'task_*.failed' 2>/dev/null | wc -l)"
  printf '[task state]\ncompleted=%s/38 failed=%s\n' "$completed_count" "$failed_count"
  if [[ -s "$state_root/current_task.tsv" ]]; then
    printf 'current='
    tail -n 1 "$state_root/current_task.tsv"
  else
    echo 'current=none'
  fi

  latest_log="$(find "$repo_root/logs" -maxdepth 1 -type f \
    -name 'unirank-sisa-l40-*_attempt*.out' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
  echo "[latest training log]"
  if [[ -n "$latest_log" && -f "$latest_log" ]]; then
    errors="$(rg -i -c "$error_pattern" "$latest_log" 2>/dev/null || true)"
    progress="$(tr '\r' '\n' < "$latest_log" \
      | rg 'Rank 0 \| Epoch|Train loss:|\[Metrics\]|SISA_EXPANSION_COMPLETE|Traceback|CUDA out of memory' \
      | tail -n 8 || true)"
    printf 'file=%s errors=%s\n%s\n' "$latest_log" "${errors:-0}" "$progress"
  else
    echo 'file=none'
  fi

  allocation_id="$(squeue -h -u "$USER" -n "$job_name" -t RUNNING -o '%A' | head -n 1)"
  echo "[gpu telemetry]"
  if [[ -n "$allocation_id" ]]; then
    timeout 20s srun --overlap --jobid="$allocation_id" --nodes=1 --ntasks=1 \
      --gres=gpu:l40s:4 \
      nvidia-smi \
      --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits </dev/null 2>&1 || true
  else
    echo 'allocation_not_running'
  fi

  echo "[storage]"
  df -h "$repo_root" | tail -n 1
  echo
}

while true; do
  snapshot >> "$monitor_log" 2>&1
  completed_count="$(find "$state_root/completed" -maxdepth 1 -type f -name 'task_*.ok' 2>/dev/null | wc -l)"
  if [[ "$completed_count" == 38 ]]; then
    printf 'MONITOR_COMPLETE timestamp=%s complete=38/38\n' \
      "$(date '+%F %T %Z')" >> "$monitor_log"
    break
  fi
  if [[ -f "$state_root/STOP_MONITOR" ]]; then
    printf 'MONITOR_STOP_REQUEST timestamp=%s\n' \
      "$(date '+%F %T %Z')" >> "$monitor_log"
    break
  fi
  sleep "$interval_seconds"
done
