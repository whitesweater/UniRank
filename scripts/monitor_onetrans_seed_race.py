#!/usr/bin/env python3
"""Keep the first-started version of each explicitly registered OneTrans pair."""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STUDY = "onetrans_seed_completion_ws2"
PLAN = ROOT / "experiments" / STUDY / "notes/race_plan.json"
RUNTIME = ROOT / "artifacts" / STUDY
ALLOWED_TASKS = (0, 1, 2, 3, 4, 5, 8, 9)
STARTED_STATES = {"RUNNING", "COMPLETING", "COMPLETED", "FAILED", "TIMEOUT",
                  "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "SUSPENDED"}
TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                   "PREEMPTED", "CANCELLED", "BOOT_FAIL", "DEADLINE"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def select_winner(jobs):
    candidates = []
    for protocol, job in jobs.items():
        start = job.get("start", "")
        if (job.get("state") in STARTED_STATES
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", start)):
            candidates.append((start, 0 if protocol == "ws4" else 1, protocol))
    return min(candidates)[2] if candidates else None


def atomic_json(path, value):
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_plan():
    plan = json.loads(PLAN.read_text())
    if plan.get("policy") != "first_started_wins":
        raise ValueError("First-start cancellation policy is not authorized in the plan")
    if tuple(plan.get("task_ids", [])) != ALLOWED_TASKS:
        raise ValueError("Race task scope must equal the eight retained non-Taobao tasks")
    arrays = plan.get("arrays", {})
    if set(arrays) != {"ws4", "ws2"} or len(set(arrays.values())) != 2:
        raise ValueError("Expected two distinct registered arrays")
    if not all(str(array).isdigit() and int(array) > 0 for array in arrays.values()):
        raise ValueError("Unsafe array identity")
    return plan


def observations(plan):
    arrays = ",".join(str(value) for value in plan["arrays"].values())
    commands = [
        ["squeue", "-h", "-r", "-j", arrays, "-o", "%i|%j|%T|%S"],
        ["sacct", "--array", "-n", "-P", "-j", arrays,
         "--format=JobID,JobName%64,State%32,Start"],
    ]
    records, succeeded = {}, 0
    allowed = {f"{array}_{task_id}" for array in plan["arrays"].values()
               for task_id in ALLOWED_TASKS}
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, timeout=45)
        if result.returncode:
            continue
        succeeded += 1
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4 or parts[0] not in allowed:
                continue
            job_id, name, state, start = parts[:4]
            protocol = next(key for key, array in plan["arrays"].items()
                            if job_id.startswith(str(array) + "_"))
            if name != plan["job_names"][protocol]:
                raise ValueError(f"Refusing to act on job-name mismatch for {job_id}")
            record = dict(job_id=job_id, name=name, state=state.split()[0].rstrip("+"), start=start)
            # A current queue RUNNING observation takes precedence over stale accounting PENDING.
            if job_id not in records or records[job_id]["state"] == "PENDING":
                records[job_id] = record
    if not succeeded:
        raise RuntimeError("Both scheduler queries failed; no cancellation attempted")
    return records


def poll(plan, state):
    records = observations(plan)
    state["updated_at"] = now()
    state["observations"] = records
    decisions = state.setdefault("decisions", {})
    for task_id in ALLOWED_TASKS:
        key = str(task_id)
        jobs = {protocol: records.get(f"{array}_{task_id}", {})
                for protocol, array in plan["arrays"].items()}
        if key not in decisions:
            winner = select_winner(jobs)
            if winner is None:
                continue
            loser = "ws2" if winner == "ws4" else "ws4"
            decisions[key] = dict(selected_at=now(), winner_protocol=winner,
                                  winner_job_id=f"{plan['arrays'][winner]}_{task_id}",
                                  winner_start=jobs[winner]["start"], loser_protocol=loser,
                                  loser_job_id=f"{plan['arrays'][loser]}_{task_id}",
                                  cancellation_done=False)
            atomic_json(RUNTIME / "race_state.json", state)
            print("SELECTED " + json.dumps(decisions[key]), flush=True)
        decision = decisions[key]
        if decision.get("cancellation_done"):
            continue
        loser = records.get(decision["loser_job_id"], {})
        if loser.get("state") in TERMINAL_STATES:
            decision.update(cancellation_done=True, loser_terminal_state=loser["state"],
                            confirmed_at=now())
            continue
        if not loser:
            # Do not assume a missing accounting row means a job was cancelled.
            continue
        result = subprocess.run(["scancel", decision["loser_job_id"]], text=True,
                                capture_output=True, timeout=30)
        decision["last_cancel_attempt"] = now()
        decision["cancel_returncode"] = result.returncode
        decision["cancel_stderr"] = result.stderr.strip()
        if result.returncode == 0:
            decision["cancellation_done"] = True
            print(f"CANCELLED {decision['loser_job_id']}", flush=True)
    complete = len(decisions) == len(ALLOWED_TASKS) and all(
        decision.get("cancellation_done") for decision in decisions.values())
    state["status"] = "all_pairs_selected" if complete else "watching"
    atomic_json(RUNTIME / "race_state.json", state)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-hours", type=float, default=168)
    args = parser.parse_args()
    if args.interval < 5 or args.max_hours <= 0:
        parser.error("interval must be >=5 seconds and max-hours must be positive")
    plan = load_plan()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "race_monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = RUNTIME / "race_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "started_at": now(), "arrays": plan["arrays"], "decisions": {}}
    if state["arrays"] != plan["arrays"]:
        raise ValueError("Existing race state refers to different arrays")
    (RUNTIME / "race_monitor.pid").write_text(str(os.getpid()) + "\n")
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        try:
            if poll(plan, state) or args.once:
                return 0
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
            state.update(status="query_error", updated_at=now(), last_error=str(error))
            atomic_json(state_file, state)
            print(f"MONITOR_ERROR {error}", flush=True)
            if args.once:
                return 1
        time.sleep(args.interval)
    state.update(status="monitor_expired", updated_at=now())
    atomic_json(state_file, state)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
