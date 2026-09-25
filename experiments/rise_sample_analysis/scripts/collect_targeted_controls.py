"""Audit targeted controls without touching frozen training/export sources.

All attempts remain visible. Only completed, mutually consistent scheduler,
configuration, model-construction, checkpoint, Test and coverage evidence is
eligible. Native A is freshly audited; historical E uses its selected archive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import collect_new_experiments as BASE

ROOT = BASE.ROOT
TASKS = (
    {"model": "HyFormer", "dataset": "MerRec_Action", "mode": "fixed_cumulative",
     "dense": 7804797, "embedding": 802271456, "stored": 810103901},
    {"model": "HiFormer", "dataset": "QK_Video_Action", "mode": "target_only",
     "dense": 27826102, "embedding": 140001120, "stored": 167827222},
    {"model": "HiFormer", "dataset": "QK_Video_Action", "mode": "field_only",
     "dense": 27887120, "embedding": 140001120, "stored": 167888240},
)
RUNTIME_ERRORS = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|torch\.OutOfMemoryError|"
                            r"CUDA error:[^\r\n]*|device-side assert[^\r\n]*|illegal memory access|"
                            r"NCCL[^\r\n]*(?:error|timeout|timed out)|DistBackendError|FloatingPointError", re.IGNORECASE)


def discover(root, array="644098"):
    specs = []
    for index, task in enumerate(TASKS):
        prefix = f"RISE_targeted_{task['model']}_{task['dataset']}_{task['mode']}_seed20262028_attempt"
        artifact_parent = Path("artifacts/rise_targeted_controls")
        checkpoint_parent = Path("checkpoints") / task["dataset"]
        attempts = {1}
        for parent in (artifact_parent, checkpoint_parent):
            for path in (Path(root) / parent).glob(prefix + "*"):
                match = re.fullmatch(re.escape(prefix) + r"(\d+)(?:\.(?:log|model))?", path.name)
                if match:
                    attempts.add(int(match[1]))
        for attempt in sorted(attempts):
            run_id = prefix + str(attempt)
            specs.append({**task, "kind": "targeted_control", "setting": task["mode"], "task_index": index,
                          "logical_task": f"targeted_{index}", "attempt": attempt, "run_id": run_id,
                          "expected_job_task": f"{array}_{index}", "artifact_dir": str(artifact_parent / run_id),
                          "manifest": str(artifact_parent / run_id / "frozen_manifest.json"),
                          "checkpoint_log": str(checkpoint_parent / (run_id + ".log"))})
    return specs


def embedded_json(text, marker):
    """Decode each marker even if concurrent print calls share a line."""
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(re.escape(marker) + r"\s*", text):
        value, _ = decoder.raw_decode(text[match.end():].lstrip())
        values.append(value)
    return values


def progress_snapshot(text):
    snapshots = {}
    pattern = r"Rank (\d+) \| Epoch (\d+):[^\r\n]*?\|\s*(\d+)/(\d+)\s*\["
    for rank, epoch, step, total in re.findall(pattern, text):
        rank, epoch, step, total = map(int, (rank, epoch, step, total))
        key = str(rank)
        previous = snapshots.get(key)
        if previous is None or epoch > previous["epoch"]:
            snapshots[key] = {"epoch": epoch, "first_step": step, "latest_step": step,
                              "total_steps": total, "logged_updates": 1, "advancing_within_snapshot": False}
        elif epoch == previous["epoch"]:
            previous.update(latest_step=step, total_steps=total, logged_updates=previous["logged_updates"] + 1,
                            advancing_within_snapshot=step > previous["first_step"])
    return snapshots


def expected_model_audit(spec, rank):
    mode, run_id = spec["mode"], spec["run_id"]
    rows = []
    if mode == "fixed_cumulative":
        for group, reference, offset in (("seq_layers", "sequence_end", 0), ("decode_layers", "query", 100)):
            for layer in range(3):
                rows.append({"site": f"{group}.{layer}", "reference": reference, "enabled": True,
                             "class_name": "FixedCumulativeScoreBias", "parameter_seed": 20260822 + offset + layer,
                             "hidden_size": 256, "heads": 2, "score_dim": 16, "stored": 21012, "active": 16404})
    else:
        target = {"site": "attention_layers", "reference": "query", "enabled": mode == "target_only"}
        if target["enabled"]:
            target.update(class_name="SISAScoreBias", parameter_seed=20260822, hidden_size=48,
                          heads=1, score_dim=16, stored=1970, active=1970)
        rows.append(target)
        for layer in range(3):
            field = {"site": f"mixer_layers.{layer}", "reference": "sequence_end", "enabled": mode == "field_only"}
            if field["enabled"]:
                field.update(class_name="SISAScoreBias", parameter_seed=20260922 + layer, hidden_size=256,
                             heads=2, score_dim=16, stored=20996, active=20996)
            rows.append(field)
    return {"mode": mode, "sites": rows, "enabled_site_count": sum(row["enabled"] for row in rows),
            "stored_parameters": spec["stored"], "active_parameters": spec["dense"] + spec["embedding"],
            "active_dense_parameters": spec["dense"], "active_embedding_parameters": spec["embedding"],
            "rank": rank, "model_id": run_id}


def model_audit_issues(spec, actual, rank):
    expected = expected_model_audit(spec, rank)
    return [f"rank{rank} model audit mismatch: {key}" for key in sorted(expected.keys() | actual.keys())
            if actual.get(key) != expected.get(key)]


def audit_attempt(spec, sources, accounts, reference):
    artifact = Path(spec["artifact_dir"])
    manifest = sources.json(spec["manifest"]) or {}
    started = sources.json(artifact / "started.json") or {}
    outcome = sources.json(artifact / "outcome.json") or {}
    expected_job = spec["expected_job_task"]
    if started.get("array_job_id") and started.get("array_task_id") is not None:
        expected_job = f"{started['array_job_id']}_{started['array_task_id']}"
    account = BASE.select_account(accounts, expected_job, started.get("job_id"))
    slurm_path = f"logs/rise-targeted-controls-{account['job_task_id']}.out"
    log = sources.read(spec["checkpoint_log"]) or ""
    slurm = sources.read(slurm_path) or ""
    completed = account["state"] == "COMPLETED" and account["exit_code"] == "0:0"
    record = {**spec, **account, "seed": 20262028, "slurm_log": slurm_path,
              "eligible": False, "selected": False, "issues": [], "pending_checks": [],
              "metrics": {}, "macro_auc": None, "macro_logloss": None,
              "model_audits": {}, "protocol_fields_checked": 0, "source_hash_checks": [],
              "parameter_counts": BASE.parameter_counts(log),
              "coverage": BASE.HELPER.parse_coverage(slurm, 16384),
              "progress": progress_snapshot(slurm), "runtime_error_signals": RUNTIME_ERRORS.findall(slurm + "\n" + log)}
    issues, pending = record["issues"], record["pending_checks"]
    if not completed:
        pending.append(f"Scheduler not yet successful: {account['state']}/{account['exit_code']}")
    if outcome.get("returncode") != 0:
        pending.append("Successful process outcome unavailable")
    if not started.get("job_id"):
        pending.append("Started allocation receipt unavailable")
    elif str(started["job_id"]) != str(account["raw_job_id"]):
        issues.append("Started allocation/scheduler mismatch")
    if outcome.get("job_id") is not None and str(outcome["job_id"]) != str(started.get("job_id")):
        issues.append("Outcome allocation mismatch")
    names = started.get("gpu_names", [])
    record["gpu_names"] = names
    if len(names) != 2 or not all("H100" in name for name in names):
        pending.append("Two-H100 allocation proof unavailable")
    if not manifest:
        pending.append("Frozen manifest unavailable")
    else:
        identity = {"run_id": spec["run_id"], "task_id": spec["task_index"], "attempt": spec["attempt"],
                    "mode": spec["mode"], "model": spec["model"], "dataset": spec["dataset"],
                    "reference_job_task": reference["job_task_id"], "allowed_parameter_differences": ["model_id"]}
        for key, value in identity.items():
            if manifest.get(key) != value:
                issues.append(f"Frozen manifest identity mismatch: {key}")
        if manifest.get("reference_params") != reference["params"]:
            issues.append("Frozen reference Params differ from historical E")
        expected_reference_hash = sources.records.get(reference["checkpoint_log"], {}).get("sha256")
        if manifest.get("reference_log_sha256") != expected_reference_hash:
            issues.append("Frozen reference log SHA mismatch")
        frozen_counts = manifest.get("parameter_expectations", {})
        for key, value in {"active_dense": spec["dense"], "active_embedding": spec["embedding"],
                           "active_total": spec["dense"] + spec["embedding"], "stored_total": spec["stored"]}.items():
            if frozen_counts.get(key) != value:
                issues.append(f"Frozen parameter expectation mismatch: {key}")
        source_issues, record["source_hash_checks"] = BASE.source_manifest_check(manifest, sources)
        issues.extend(source_issues)
        for name, text in (("checkpoint", log), ("Slurm", slurm)):
            if "INFO Params:" not in text:
                pending.append(f"{name} runtime Params unavailable")
                continue
            try:
                actual = BASE.parsed_params(text)
                issues.extend(f"{name}: {issue}" for issue in BASE.protocol_issues(
                    spec, actual, reference["params"], manifest.get("effective_params", {})))
                if name == "checkpoint":
                    record["protocol_fields_checked"] = len(actual)
            except (ValueError, TypeError) as error:
                (issues if completed else pending).append(f"{name} parameter parse: {error}")
    for rank in (0, 1):
        model_audit = sources.json(artifact / f"model_audit_rank{rank}.json")
        if model_audit is None:
            pending.append(f"rank{rank} construction audit unavailable")
        else:
            record["model_audits"][str(rank)] = model_audit
            issues.extend(model_audit_issues(spec, model_audit, rank))
    try:
        workers = embedded_json(slurm, "RISE_TARGETED_WORKER")
        expected_workers = [{"task_id": spec["task_index"], "mode": spec["mode"], "rank": rank,
                             "run_id": spec["run_id"]} for rank in (0, 1)]
        if sorted(workers, key=lambda row: row["rank"]) != expected_workers:
            pending.append("Exact two worker identity markers unavailable")
        embedded_audits = embedded_json(slurm, "RISE_TARGETED_MODEL_AUDIT")
        if len(embedded_audits) != 2:
            pending.append("Two embedded construction audits unavailable")
        else:
            for rank, audit in enumerate(sorted(embedded_audits, key=lambda row: row["rank"])):
                issues.extend(model_audit_issues(spec, audit, rank))
                if audit != record["model_audits"].get(str(rank)):
                    issues.append(f"rank{rank} construction audit differs between artifact and log")
    except (ValueError, KeyError) as error:
        (issues if completed else pending).append(f"Worker/construction marker parse: {error}")
    counts = record["parameter_counts"]
    if any(value is None for value in counts.values()):
        pending.append("Active parameter-count log unavailable")
    else:
        expected_counts = {"total_parameters": spec["dense"] + spec["embedding"],
                           "dense_parameters": spec["dense"], "embedding_parameters": spec["embedding"]}
        if counts != expected_counts or counts != BASE.parameter_counts(slurm):
            issues.append("Logged active parameter counts differ from frozen expectation or dual log")
    issues.extend(BASE.HELPER.coverage_issues(record["coverage"], reference["coverage"], partial=not completed))
    record["checkpoint"] = BASE.inspect_checkpoint((sources.root / spec["checkpoint_log"]).with_suffix(".model"))
    if not record["checkpoint"]["exists"] or not record["checkpoint"]["zip_metadata_readable"]:
        pending.append("Readable model checkpoint unavailable")
    if record["runtime_error_signals"]:
        issues.append("Runtime CUDA/NCCL/Python error signals present")
    marker = f"RISE_TARGETED_COMPLETE run_id={spec['run_id']}"
    if slurm.splitlines().count(marker) != 1:
        pending.append("Unique targeted completion marker unavailable")
    metrics = None
    if "Test evaluation" in log:
        try:
            metrics = BASE.HELPER.parse_test_metrics(log, BASE.LABELS[spec["dataset"]])
            if metrics != BASE.HELPER.parse_test_metrics(slurm, BASE.LABELS[spec["dataset"]]):
                issues.append("Final Test metrics disagree between logs")
            preserved = re.findall(r"Preserved best model checkpoint: (.+)", log)
            if len(preserved) != 1 or Path(preserved[0]).name != spec["run_id"] + ".model":
                pending.append("Unique best-checkpoint preservation receipt unavailable")
        except (ValueError, TypeError) as error:
            (issues if completed else pending).append(f"Final Test parse: {error}")
    else:
        pending.append("Final Test evaluation unavailable")
    if not reference["eligible"]:
        issues.append("Historical E failed its independent audit")
    if completed and not issues and not pending and metrics is not None:
        record.update(eligible=True, metrics=metrics,
                      macro_auc=statistics.mean(value["AUC"] for value in metrics.values()),
                      macro_logloss=statistics.mean(value["logloss"] for value in metrics.values()))
    return record


def select_attempts(runs):
    for logical in {run["logical_task"] for run in runs}:
        eligible = [run for run in runs if run["logical_task"] == logical and run["eligible"]]
        if eligible:
            max(eligible, key=lambda run: run["attempt"])["selected"] = True


def collect(root, targeted_specs, native_specs, accounts):
    sources = BASE.Sources(root)
    references = {(spec["model"], spec["dataset"]): None for spec in targeted_specs}
    for key in references:
        references[key] = BASE.reference_audit(*key, sources)
    native = [BASE.audit_attempt(spec, sources, accounts, references[(spec["model"], spec["dataset"])]) for spec in native_specs]
    select_attempts(native)
    runs = [audit_attempt(spec, sources, accounts, references[(spec["model"], spec["dataset"])]) for spec in targeted_specs]
    select_attempts(runs)
    comparisons = []
    for run in runs:
        if not run["selected"]:
            continue
        key = (run["model"], run["dataset"])
        candidates = [row for row in native if (row["model"], row["dataset"]) == key and row["selected"]]
        if len(candidates) != 1:
            continue
        baseline = candidates[0]
        reference = references[key]
        for label in [*BASE.LABELS[run["dataset"]], "__macro__"]:
            values = {}
            for name, metrics in (("Baseline", baseline["metrics"]), ("Full_RISE", reference["metrics"]), ("Targeted", run["metrics"])):
                item = metrics[label] if label != "__macro__" else {
                    metric: statistics.mean(point[metric] for point in metrics.values()) for metric in ("AUC", "logloss")}
                values.update({f"{name}_{metric}": item[metric] for metric in ("AUC", "logloss")})
            comparisons.append({"model": run["model"], "dataset": run["dataset"], "mode": run["mode"],
                                "attempt": run["attempt"], "targeted_job": run["job_task_id"],
                                "baseline_job": baseline["job_task_id"], "full_RISE_job": reference["job_task_id"],
                                "label": label, **values,
                                "delta_AUC_vs_Baseline": values["Targeted_AUC"] - values["Baseline_AUC"],
                                "delta_AUC_vs_Full_RISE": values["Targeted_AUC"] - values["Full_RISE_AUC"],
                                "delta_logloss_vs_Baseline": values["Targeted_logloss"] - values["Baseline_logloss"],
                                "delta_logloss_vs_Full_RISE": values["Targeted_logloss"] - values["Full_RISE_logloss"]})
    return runs, native, list(references.values()), comparisons, list(sources.records.values())


def accounting(root, specs, target_array, baseline_array):
    ids = {target_array, baseline_array}
    for spec in specs:
        path = Path(root) / spec["artifact_dir"] / "started.json"
        if path.exists():
            raw = json.loads(path.read_text()).get("job_id")
            if raw:
                ids.add(str(raw))
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(sorted(ids)),
               "--format=JobID%64,JobIDRaw%64,State%32,ExitCode,Elapsed,NodeList%64"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return BASE.parse_accounting(result.stdout), {"command": command, "stdout": result.stdout, "stderr": result.stderr,
                                                "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def write_outputs(output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "targeted_training_snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    run_rows, label_rows = [], []
    for run in evidence["runs"]:
        row = {key: run[key] for key in ("model", "dataset", "mode", "task_index", "attempt", "run_id", "job_task_id",
                                         "raw_job_id", "state", "exit_code", "elapsed", "node", "eligible", "selected", "macro_auc", "macro_logloss")}
        row.update(run["parameter_counts"], checkpoint_exists=run["checkpoint"]["exists"],
                   issues="; ".join(run["issues"]), pending_checks="; ".join(run["pending_checks"]),
                   progress_json=json.dumps(run["progress"], sort_keys=True), runtime_error_count=len(run["runtime_error_signals"]))
        run_rows.append(row)
        for label, metrics in run["metrics"].items():
            label_rows.append({"model": run["model"], "dataset": run["dataset"], "mode": run["mode"],
                               "attempt": run["attempt"], "job_task_id": run["job_task_id"], "selected": run["selected"],
                               "label": label, **metrics})
    files = {
        "targeted_training_evidence.json": json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "targeted_training_runs.csv": BASE.csv_text(run_rows, ["model", "dataset", "mode", "state", "eligible"]),
        "targeted_training_label_metrics.csv": BASE.csv_text(label_rows, ["model", "dataset", "mode", "label", "AUC", "logloss"]),
        "targeted_training_comparisons.csv": BASE.csv_text(evidence["comparisons"], ["model", "dataset", "mode", "label", "Baseline_AUC", "Full_RISE_AUC", "Targeted_AUC"]),
        "targeted_training_sources.csv": BASE.csv_text(evidence["sources"], ["path", "bytes", "sha256"]),
        "targeted_training_accounting.json": json.dumps(evidence["accounting"], indent=2) + "\n",
    }
    for name, text in files.items():
        (snapshot / name).write_text(text)
        temporary = output / f"{name}.tmp-{os.getpid()}"
        temporary.write_text(text)
        os.replace(temporary, output / name)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-array", default="644098")
    parser.add_argument("--baseline-array", default="643809")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/rise_sample_analysis/results")
    args = parser.parse_args()
    specs = discover(ROOT, args.target_array)
    wanted = {(task["model"], task["dataset"]) for task in TASKS}
    native_specs = [spec for spec in BASE.discover(ROOT, args.baseline_array)
                    if spec["kind"] == "native_baseline" and (spec["model"], spec["dataset"]) in wanted]
    accounts, receipt = accounting(ROOT, [*specs, *native_specs], args.target_array, args.baseline_array)
    runs, native, references, comparisons, sources = collect(ROOT, specs, native_specs, accounts)
    evidence = {"generated_utc": datetime.now(timezone.utc).isoformat(), "runs": runs, "native_baseline_audits": native,
                "historical_E_audits": references, "comparisons": comparisons, "sources": sources, "accounting": receipt,
                "collector_sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in (Path(__file__).resolve(), Path(BASE.__file__).resolve(), BASE.HELPER_PATH)},
                "scope": "Three targeted seed20262028 controls, same-protocol native A and historical E; no seed20262027 mixing",
                "construction_gate": "Exact two-rank site identity, class, reference, seed offset, active/stored capacity; artifact and embedded log agree",
                "incomplete_policy": "Pending/running/failed attempts retain status and progress only, without eligible metrics",
                "attempt_policy": "Every attempt retained; highest eligible attempt selected without reference to score",
                "coverage_scope": "Per-rank split counts; not proof of byte-identical historical sample contents",
                "capacity_scope": "Fixed cumulative and site removals are not matched in active parameter count",
                "progress_scope": "Advancement is established from successive tqdm updates inside the captured log snapshot; no long-running polling"}
    snapshot = write_outputs(args.output_dir, evidence)
    print(json.dumps({"attempts": len(runs), "eligible": sum(run["eligible"] for run in runs), "snapshot": str(snapshot)}))
    for run in runs:
        print(json.dumps({key: run[key] for key in ("job_task_id", "mode", "state", "node", "eligible", "progress",
                                                   "runtime_error_signals", "issues", "pending_checks")}))


if __name__ == "__main__":
    main()
