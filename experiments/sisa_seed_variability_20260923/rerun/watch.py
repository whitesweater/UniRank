"""Lightweight local watcher for this submitted batch; no extra training jobs."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--batch-dir", type=Path, default=Path(__file__).parent)
args = parser.parse_args()
HERE = args.batch_dir.resolve()
runtime_name = "retry_watch" if HERE == Path(__file__).parent else HERE.name + "_watch"
RUNTIME = ROOT / "artifacts/sisa_seed_variability_20260923" / runtime_name
RUNTIME.mkdir(parents=True, exist_ok=True)
terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
previous = {}
started = {}


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


while True:
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(Path(__file__).with_name("collect.py")),
         "--batch-dir", str(HERE)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    heartbeat = dict(at=timestamp(), pid=os.getpid(), collector_returncode=result.returncode)
    if result.returncode:
        heartbeat["error"] = result.stderr[-4000:]
    else:
        status = json.loads((HERE / "results/status.json").read_text())
        heartbeat["complete"] = status["complete_count"]
        for run in status["runs"]:
            job = run["job_task_id"]
            state = run["state"]
            if previous.get(job) != state:
                with (RUNTIME / "state_changes.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(at=timestamp(), job=job, state=state, node=run["node"])) + "\n")
                previous[job] = state
            progress = run["train_progress"]
            if progress and run["parameter_match"]:
                event = started.setdefault(job, dict(first_observed_at=timestamp(), first_progress=progress, node=run["node"]))
                event.update(last_observed_at=timestamp(), last_progress=progress,
                             progress_advanced=event["first_progress"] != progress)
        (HERE / "startup.json").write_text(json.dumps(started, indent=2) + "\n")
        if all(r["state"].split()[0] in terminal for r in status["runs"]):
            heartbeat["finished"] = True
            (RUNTIME / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2) + "\n")
            print(json.dumps(heartbeat), flush=True)
            break
    (RUNTIME / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2) + "\n")
    time.sleep(60)
