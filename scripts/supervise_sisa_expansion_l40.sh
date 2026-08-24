#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

job_name="${SISA_ALLOCATION_JOB_NAME:-unirank-sisa-l40-all}"
poll_seconds="${SISA_SUPERVISOR_POLL_SECONDS:-1800}"
retry_seconds="${SISA_SUPERVISOR_RETRY_SECONDS:-60}"
state_root="$repo_root/artifacts/sisa_expansion_l40"
completed_root="$state_root/completed"
supervisor_log="$state_root/supervisor.log"
lock_file="$state_root/supervisor.lock"
request_script="$repo_root/scripts/request_sisa_expansion_l40_all.sh"

if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUPERVISOR_ERROR: SISA_SUPERVISOR_POLL_SECONDS must be positive" >&2
  exit 2
fi
if ! [[ "$retry_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "SUPERVISOR_ERROR: SISA_SUPERVISOR_RETRY_SECONDS must be positive" >&2
  exit 2
fi

mkdir -p "$completed_root" "$repo_root/logs"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "SUPERVISOR_ERROR: another supervisor holds $lock_file" >&2
  exit 2
fi

log_event() {
  printf '%s\t%s\n' "$(date '+%F %T %Z')" "$1" | tee -a "$supervisor_log"
}

completed_count() {
  find "$completed_root" -maxdepth 1 -type f -name 'task_*.ok' \
    2>/dev/null | wc -l
}

handle_signal() {
  log_event "supervisor_signal"
  exit 143
}
trap handle_signal TERM INT

log_event "supervisor_start job_name=$job_name poll_seconds=$poll_seconds retry_seconds=$retry_seconds"
while true; do
  complete="$(completed_count)"
  if [[ "$complete" == 38 ]]; then
    log_event "supervisor_complete complete=38/38"
    exit 0
  fi

  if ! queue_state="$(squeue -h -u "$USER" -n "$job_name" -o '%A|%T|%R')"; then
    log_event "queue_query_failed complete=$complete/38"
    sleep "$retry_seconds"
    continue
  fi
  if [[ -n "$queue_state" ]]; then
    queue_state="$(printf '%s' "$queue_state" | tr '\n' ';')"
    log_event "allocation_observed jobs=$queue_state complete=$complete/38"
    sleep "$poll_seconds"
    continue
  fi

  log_event "allocation_request_start complete=$complete/38 time=${SISA_ALLOCATION_TIME:-7-00:00:00}"
  "$request_script"
  request_exit_code=$?
  complete="$(completed_count)"
  log_event "allocation_request_end exit_code=$request_exit_code complete=$complete/38"

  if [[ "$complete" == 38 ]]; then
    log_event "supervisor_complete complete=38/38"
    exit 0
  fi
  sleep "$retry_seconds"
done
