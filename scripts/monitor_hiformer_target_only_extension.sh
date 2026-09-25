#!/usr/bin/env bash
# Persist queue and evidence snapshots for the three authorized extension runs.
set -u
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
monitor_dir=artifacts/rise_hiformer_target_only_extension/monitor
mkdir -p "$monitor_dir"
exec 9>"$monitor_dir/monitor.lock"
flock -n 9 || { echo "The HiFormer extension monitor is already running."; exit 1; }

while true; do
  snapshot=$(date +%Y%m%d_%H%M%S)
  snapshot_dir="$monitor_dir/$snapshot"
  mkdir -p "$snapshot_dir"
  date --iso-8601=seconds
  if .venv/bin/python scripts/collect_hiformer_target_only_extension.py \
      >"$snapshot_dir/summary.json" 2>"$snapshot_dir/collector.stderr"; then
    cat "$snapshot_dir/summary.json"
    cp experiments/hiformer_target_only_extension/results/evidence.json "$snapshot_dir/evidence.json"
    .venv/bin/python - "$snapshot_dir/evidence.json" <<'PY'
import json
from pathlib import Path
import sys

evidence = json.loads(Path(sys.argv[1]).read_text())
if evidence["complete"]:
    print("RISE_HIFORMER_EXTENSION_COMPLETE all three results audited")
    raise SystemExit(0)
if evidence["all_terminal"]:
    print("RISE_HIFORMER_EXTENSION_NEEDS_ATTENTION")
    raise SystemExit(2)
raise SystemExit(3)
PY
    monitor_status=$?
    if [[ "$monitor_status" == 0 || "$monitor_status" == 2 ]]; then
      exit "$monitor_status"
    fi
  else
    echo "Collection failed; keeping diagnostics for the next snapshot."
    cat "$snapshot_dir/collector.stderr"
  fi
  .venv/bin/python - <<'PY' >"$snapshot_dir/queue.txt" 2>"$snapshot_dir/queue.stderr"
import json
from pathlib import Path
import subprocess

receipt = json.loads(Path("experiments/hiformer_target_only_extension/notes/submission.json").read_text())
ids = sorted({str(row["job_task_id"]).split("_")[0] for row in receipt["tasks"]})
subprocess.run(["squeue", "-r", "-j", ",".join(ids), "-o", "%.22i %.36j %.10T %.10M %R"], check=False)
PY
  cat "$snapshot_dir/queue.txt"
  sleep 600
done
