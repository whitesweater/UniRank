#!/usr/bin/env bash
# Audit the fixed three-task study every ten minutes; preserve all snapshots.
set -u
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
monitor_dir=artifacts/sisa_score_dim/monitor
mkdir -p "$monitor_dir"
exec 9>"$monitor_dir/monitor.lock"
flock -n 9 || { echo "A dimension-study monitor is already running."; exit 1; }

while true; do
  snapshot=$(date +%Y%m%d_%H%M%S)
  snapshot_dir="$monitor_dir/$snapshot"
  mkdir -p "$snapshot_dir"
  date --iso-8601=seconds
  if .venv/bin/python scripts/collect_sisa_score_dim_results.py \
       >"$snapshot_dir/summary.json" 2>"$snapshot_dir/collector.stderr"; then
    cat "$snapshot_dir/summary.json"
    cp experiments/ablations/sisa_score_dim/results/evidence.json "$snapshot_dir/evidence.json"
    .venv/bin/python - "$snapshot_dir/evidence.json" <<'PY'
import json
from pathlib import Path
import sys

evidence = json.loads(Path(sys.argv[1]).read_text())
terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
if evidence["complete"]:
    print("SISA_SCORE_DIM_MONITOR_COMPLETE all four dimensions audited")
    raise SystemExit(0)
new_runs = [row for row in evidence["runs"] if row["score_dim"] != 16]
if all(row["state"].split()[0] in terminal for row in new_runs):
    for row in evidence["runs"]:
        if not row["eligible"]:
            print(json.dumps({key: row.get(key) for key in
                              ("score_dim", "job_task_id", "state", "issues")}))
    print("SISA_SCORE_DIM_MONITOR_NEEDS_ATTENTION")
    raise SystemExit(2)
raise SystemExit(3)
PY
    monitor_status=$?
    if [[ "$monitor_status" == 0 || "$monitor_status" == 2 ]]; then
      exit "$monitor_status"
    fi
  else
    echo "Collection failed; preserving diagnostic output and retrying at the next interval."
    cat "$snapshot_dir/collector.stderr"
  fi
  study_arrays=$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path("experiments/ablations/sisa_score_dim/notes/submission.json").read_text())
print(",".join(sorted({row["array_id"] for row in receipt["tasks"]})))
PY
  )
  squeue -r -j "$study_arrays" -o '%.22i %.32j %.10T %.10M %R' \
    >"$snapshot_dir/queue.txt" 2>"$snapshot_dir/queue.stderr"
  cat "$snapshot_dir/queue.txt"
  sleep 600
done
