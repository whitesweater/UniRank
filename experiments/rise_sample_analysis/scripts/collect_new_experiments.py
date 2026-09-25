"""Audit new matched native baselines and RankMixer mechanism controls.

No jobs are launched. Every attempt remains a status record; metrics are exposed
only after scheduler, outcome, protocol, source, checkpoint and coverage gates.
Repeated collection keeps immutable timestamped snapshots and refreshes only
this study's new_training_* outputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = ROOT / "scripts/collect_sisa_component_results.py"
SPEC = importlib.util.spec_from_file_location("rise_existing_component_collector", HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
LABELS = {"MerRec_Action": ("Like", "Cart", "Offer", "Checkout", "Purchase"),
          "QK_Video_Action": ("click", "follow", "like", "share"),
          "TencentGR_10M_Action": ("is_click", "is_conversion"),
          "KuaiRand_Video_Action": ("is_click", "is_follow", "is_like", "is_comment", "is_forward", "long_view")}
BASELINES = (("HyFormer", "MerRec_Action"), ("HiFormer", "QK_Video_Action"),
             ("HiFormer", "TencentGR_10M_Action"))
ERROR_PATTERN = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|torch\.OutOfMemoryError|"
                           r"NCCL (?:error|timeout)|DistBackendError|FloatingPointError", re.IGNORECASE)


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.records = {}

    def read(self, relative):
        name = str(relative)
        if name not in self.cache:
            path = self.root / relative
            if not path.exists():
                self.cache[name] = None
                return None
            data = path.read_bytes()
            self.cache[name] = data.decode("utf-8")
            self.records[name] = {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return self.cache[name]

    def json(self, relative):
        text = self.read(relative)
        return json.loads(text) if text is not None else None


def parsed_params(text):
    return {key: HELPER.parsed_value(value) for key, value in HELPER.parse_params(text).items()}


def parameter_counts(text):
    result = HELPER.parameter_counts(text)
    matches = re.findall(r"Number of embedding parameters: (\d+)\.", text)
    result["embedding_parameters"] = int(matches[0]) if len(matches) == 1 else None
    return result


def parse_accounting(text):
    records = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 6 or "." in fields[0] or not re.fullmatch(r"\d+(?:_\d+)?", fields[0]):
            continue
        records.append(dict(job_task_id=fields[0], raw_job_id=fields[1], state=fields[2].split()[0],
                            exit_code=fields[3], elapsed=fields[4], node=fields[5]))
    return records


def select_account(records, expected_task, started_raw=None):
    """Prefer the actual allocation, not sacct's duplicate array placeholder."""
    if started_raw is not None:
        candidates = [record for record in records if record["raw_job_id"] == str(started_raw)]
    else:
        candidates = [record for record in records if record["job_task_id"] == expected_task]
    if not candidates:
        return {"job_task_id": expected_task, "raw_job_id": started_raw, "state": "UNKNOWN",
                "exit_code": None, "elapsed": None, "node": None}
    actual = [record for record in candidates
              if record["raw_job_id"] != record["job_task_id"].split("_")[0] or record["state"] != "PENDING"]
    selected = actual or candidates
    unique = {tuple(sorted(record.items())): record for record in selected}
    if len(unique) != 1:
        raise ValueError(f"Ambiguous scheduler accounting for {expected_task}: {selected}")
    return next(iter(unique.values()))


def discover(root, baseline_array="643809", control_array="643822"):
    attempts = []
    for index, (model, dataset) in enumerate(BASELINES):
        prefix = f"RISE_matched_{model}_{dataset}_baseline_seed20262028_attempt"
        artifact_parent = Path("artifacts/rise_sample_analysis/matched_baselines")
        checkpoint_parent = Path("artifacts/rise_sample_analysis/training_checkpoints") / dataset
        numbers = {1}
        for parent in (artifact_parent, checkpoint_parent):
            for path in (Path(root) / parent).glob(prefix + "*"):
                match = re.fullmatch(re.escape(prefix) + r"(\d+)(?:\.(?:log|model))?", path.name)
                if match:
                    numbers.add(int(match[1]))
        for attempt in sorted(numbers):
            run_id = prefix + str(attempt)
            attempts.append(dict(kind="native_baseline", model=model, dataset=dataset, setting="A",
                                 logical_task=f"baseline_{index}", task_index=index, attempt=attempt, run_id=run_id,
                                 expected_job_task=f"{baseline_array}_{index}",
                                 artifact_dir=str(artifact_parent / run_id),
                                 manifest=str(artifact_parent / run_id / "frozen_manifest.json"),
                                 checkpoint_log=str(checkpoint_parent / (run_id + ".log")),
                                 slurm_prefix="unirank-rise-matched-baseline"))
    for index, control in enumerate(("local", "fixed")):
        model, dataset = "RankMixer", "KuaiRand_Video_Action"
        prefix = f"RISE_controls_{model}_{dataset}_{control}_seed20262028_attempt"
        artifact_parent = Path("artifacts/rise_rankmixer_controls")
        checkpoint_parent = Path("checkpoints") / dataset
        numbers = {1}
        for parent in (artifact_parent, checkpoint_parent):
            for path in (Path(root) / parent).glob(prefix + "*"):
                match = re.fullmatch(re.escape(prefix) + r"(\d+)(?:\.(?:log|model))?", path.name)
                if match:
                    numbers.add(int(match[1]))
        for attempt in sorted(numbers):
            run_id = prefix + str(attempt)
            attempts.append(dict(kind="rankmixer_control", model=model, dataset=dataset, setting=control,
                                 logical_task=f"control_{control}", task_index=index, attempt=attempt, run_id=run_id,
                                 expected_job_task=f"{control_array}_{index}",
                                 artifact_dir=str(artifact_parent / run_id),
                                 manifest=str(artifact_parent / run_id / "manifest.json"),
                                 checkpoint_log=str(checkpoint_parent / (run_id + ".log")),
                                 slurm_prefix="rise-rankmixer-controls"))
    return attempts


def protocol_issues(spec, actual, reference, frozen):
    allowed = {"model_id", "model_root", "sisa_enabled"} if spec["kind"] == "native_baseline" else {"model_id"}
    issues = []
    for key in sorted(actual.keys() | frozen.keys()):
        if key not in actual or key not in frozen or actual[key] != frozen[key]:
            issues.append(f"Runtime/frozen configuration mismatch: {key}")
    for key in sorted(actual.keys() | reference.keys()):
        if key not in allowed and (key not in actual or key not in reference or actual[key] != reference[key]):
            issues.append(f"Historical E protocol mismatch: {key}")
    if actual.get("model_id") != spec["run_id"]:
        issues.append("Runtime identifier mismatch")
    enabled = spec["kind"] != "native_baseline"
    if actual.get("sisa_enabled", False) is not enabled:
        issues.append("Native/control SISA switch mismatch")
    required = {"model": spec["model"], "dataset_id": spec["dataset"], "seed": 20262028,
                "world_size": 2, "batch_size": 16384, "epochs": 1, "max_len": 100,
                "dataloader_seed": 2027, "sisa_parameter_seed": 20260822,
                "enable_bf16": True, "enable_torch_compile": True}
    for key, value in required.items():
        if actual.get(key) != value:
            issues.append(f"Required protocol mismatch: {key}")
    return issues


def source_manifest_check(manifest, sources):
    mapping = manifest.get("source_sha256", {})
    if not mapping:
        return ["Missing frozen source hashes"], []
    issues, entries = [], []
    for relative, expected in sorted(mapping.items()):
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not re.fullmatch(r"[0-9a-f]{64}", expected):
            issues.append(f"Malformed frozen source identity: {relative}")
            continue
        text = sources.read(relative)
        current = sources.records.get(relative, {}).get("sha256")
        matches = text is not None and current == expected
        entries.append({"path": relative, "frozen_sha256": expected, "current_sha256": current, "matches": matches})
        if not matches:
            issues.append(f"Current source differs from frozen execution manifest: {relative}")
    return issues, entries


def inspect_checkpoint(path):
    if not path.is_file():
        return {"exists": False, "bytes": 0, "zip_metadata_readable": False}
    stat = path.stat()
    result = {"exists": True, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "zip_metadata_readable": False, "full_content_hash_computed": False}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.endswith("/data.pkl")]
            if len(metadata) != 1:
                raise ValueError("Expected one serialized state dictionary")
            result.update(zip_entries=len(names), state_metadata_bytes=len(archive.read(metadata[0])),
                          zip_metadata_readable=True)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        result["error"] = str(error)
    return result


def reference_audit(model, dataset, sources):
    text = sources.read("experiments/sisa_single_seed20262028/results/runs.csv")
    records = list(csv.DictReader(io.StringIO(text or "")))
    selected = [record for record in records if record["model"] == model and record["dataset"] == dataset]
    if len(selected) != 1:
        raise ValueError("Historical reference is not unique")
    archive = selected[0]
    log = sources.read(archive["checkpoint_log"]) or ""
    slurm = sources.read(archive["slurm_log"]) or ""
    params = parsed_params(log)
    metrics = HELPER.parse_test_metrics(log, LABELS[dataset])
    issues = []
    if archive.get("complete") != "True" or archive.get("state") != "COMPLETED" or archive.get("exit_code") != "0:0":
        issues.append("Historical E lacks accepted completion receipt")
    if metrics != HELPER.parse_test_metrics(slurm, LABELS[dataset]):
        issues.append("Historical E final Test logs disagree")
    if metrics != json.loads(archive["metrics_json"]):
        issues.append("Historical E differs from selected metric archive")
    if parsed_params(slurm) != params:
        issues.append("Historical E parameter logs disagree")
    coverage = HELPER.parse_coverage(slurm, 16384)
    if {(entry["split"], entry["rank"]) for entry in coverage} != {(split, rank) for split in ("train", "valid", "test") for rank in (0, 1)}:
        issues.append("Historical E lacks complete split/rank coverage")
    counts = parameter_counts(log)
    if counts != parameter_counts(slurm) or any(value is None for value in counts.values()):
        issues.append("Historical E parameter-count records incomplete or inconsistent")
    return {"model": model, "dataset": dataset, "job_task_id": archive["job_task_id"],
            "checkpoint_log": archive["checkpoint_log"], "slurm_log": archive["slurm_log"],
            "state_source": "frozen historical selected-run archive", "params": params,
            "metrics": metrics, "coverage": coverage, "parameter_counts": counts,
            "eligible": not issues, "issues": issues,
            "historical_training_source_revision_verified": False}


def audit_attempt(spec, sources, accounts, reference):
    manifest = sources.json(spec["manifest"]) or {}
    artifact = Path(spec["artifact_dir"])
    started = sources.json(artifact / "started.json") or {}
    outcome = sources.json(artifact / "outcome.json") or {}
    raw = started.get("job_id") if spec["kind"] == "native_baseline" else manifest.get("job_id")
    expected = spec["expected_job_task"]
    if spec["kind"] == "rankmixer_control" and manifest.get("array_job_id") and manifest.get("array_task_id") is not None:
        expected = f"{manifest['array_job_id']}_{manifest['array_task_id']}"
    account = select_account(accounts, expected, raw)
    slurm_path = f"logs/{spec['slurm_prefix']}-{account['job_task_id']}.out"
    log = sources.read(spec["checkpoint_log"]) or ""
    slurm = sources.read(slurm_path) or ""
    record = dict(spec, **{key: account[key] for key in ("job_task_id", "raw_job_id", "state", "exit_code", "elapsed", "node")},
                  seed=20262028, slurm_log=slurm_path, eligible=False, selected=False,
                  metrics={}, macro_auc=None, macro_logloss=None, issues=[], pending_checks=[],
                  protocol_fields_checked=0, source_hash_checks=[], outcome=outcome,
                  parameter_counts=parameter_counts(log), coverage=HELPER.parse_coverage(slurm, 16384))
    issues, pending = record["issues"], record["pending_checks"]
    completed = account["state"] == "COMPLETED" and account["exit_code"] == "0:0"
    if not completed:
        pending.append(f"Scheduler not yet successful: {account['state']}/{account['exit_code']}")
    if not manifest:
        pending.append("Execution manifest not available")
    else:
        if manifest.get("run_id") != spec["run_id"] or manifest.get("attempt") != spec["attempt"]:
            issues.append("Artifact attempt identity mismatch")
        frozen = manifest.get("effective_params", {})
        if not frozen:
            issues.append("No frozen effective configuration")
        source_issues, record["source_hash_checks"] = source_manifest_check(manifest, sources)
        issues.extend(source_issues)
        if spec["kind"] == "native_baseline":
            if manifest.get("reference_params") != reference["params"]:
                issues.append("Frozen baseline reference parameters differ from E")
            ref_hash = sources.records.get(reference["checkpoint_log"], {}).get("sha256")
            if manifest.get("reference_log_sha256") != ref_hash:
                issues.append("Frozen reference log hash mismatch")
            if manifest.get("task_id") != spec["task_index"]:
                issues.append("Baseline logical-task identity mismatch")
        elif manifest.get("control") != spec["setting"] or manifest.get("reference_job_task") != reference["job_task_id"]:
            issues.append("Control/reference identity mismatch")
        for label, text in (("checkpoint", log), ("Slurm", slurm)):
            if "INFO Params:" not in text:
                pending.append(f"{label} runtime parameters not yet available")
                continue
            try:
                actual = parsed_params(text)
                issues.extend(f"{label}: {item}" for item in protocol_issues(spec, actual, reference["params"], frozen))
                if label == "checkpoint":
                    record["protocol_fields_checked"] = len(actual)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                issues.append(f"{label} parameter parse: {error}")
    if not raw:
        pending.append("Started allocation identity unavailable")
    elif str(account["raw_job_id"]) != str(raw):
        issues.append("Scheduler allocation does not match started manifest")
    if outcome.get("returncode") != 0:
        pending.append("Successful process outcome unavailable")
    if outcome.get("job_id") is not None and str(outcome["job_id"]) != str(raw):
        issues.append("Outcome allocation mismatch")
    gpus = started.get("gpus", []) if spec["kind"] == "native_baseline" else manifest.get("gpu_names", [])
    record["gpu_names"] = gpus
    if len(gpus) != 2 or not all("H100" in name for name in gpus):
        pending.append("Two-H100 execution receipt unavailable")
    issues.extend(HELPER.coverage_issues(record["coverage"], reference["coverage"], partial=not completed))
    counts = record["parameter_counts"]
    if any(value is None for value in counts.values()):
        pending.append("Complete active parameter counts not yet available")
    else:
        if counts != parameter_counts(slurm):
            issues.append("Active parameter counts disagree between logs")
        if counts["total_parameters"] != counts["embedding_parameters"] + counts["dense_parameters"]:
            issues.append("Active parameter-count arithmetic mismatch")
        if counts["embedding_parameters"] != reference["parameter_counts"]["embedding_parameters"]:
            issues.append("Embedding parameter count differs from historical E")
        if spec["setting"] == "local" and counts != reference["parameter_counts"]:
            issues.append("Local control must retain E active parameter count")
    checkpoint_path = sources.root / Path(spec["checkpoint_log"]).with_suffix(".model")
    record["checkpoint"] = inspect_checkpoint(checkpoint_path)
    if not record["checkpoint"]["exists"] or not record["checkpoint"]["zip_metadata_readable"]:
        pending.append("Readable model checkpoint not yet available")
    errors = ERROR_PATTERN.findall(log + "\n" + slurm)
    if errors:
        issues.append(f"Execution error signals: {sorted(set(errors))}")
    if spec["kind"] == "rankmixer_control" and manifest:
        workers = re.findall(r"RISE_CONTROL_WORKER (\{[^\r\n]+\})", slurm)
        try:
            records = [json.loads(item) for item in workers]
            expected_workers = [{"control": spec["setting"], "rank": rank, "score_dim": 16} for rank in (0, 1)]
            if sorted(records, key=lambda item: item["rank"]) != expected_workers:
                pending.append("Exact two-rank control-construction receipts unavailable")
        except (ValueError, KeyError):
            issues.append("Malformed control worker-construction receipt")
        marker = f"RISE_CONTROL_COMPLETE run_id={spec['run_id']}"
        if slurm.splitlines().count(marker) != 1:
            pending.append("Control completion marker unavailable or repeated")
    metrics = None
    if "Test evaluation" in log:
        try:
            metrics = HELPER.parse_test_metrics(log, LABELS[spec["dataset"]])
            if metrics != HELPER.parse_test_metrics(slurm, LABELS[spec["dataset"]]):
                issues.append("Final Test metrics disagree between logs")
            preserved = re.findall(r"Preserved best model checkpoint: (.+)", log)
            if len(preserved) != 1 or Path(preserved[0]).name != spec["run_id"] + ".model":
                pending.append("Best-checkpoint preservation receipt unavailable or ambiguous")
        except (ValueError, TypeError) as error:
            (issues if completed else pending).append(f"Final Test metric audit: {error}")
    else:
        pending.append("Final Test evaluation not yet available")
    if not reference["eligible"]:
        issues.append("Historical E reference did not pass its audit")
    if completed and not issues and not pending and metrics is not None:
        record.update(eligible=True, metrics=metrics,
                      macro_auc=statistics.mean(value["AUC"] for value in metrics.values()),
                      macro_logloss=statistics.mean(value["logloss"] for value in metrics.values()))
    return record


def collect(root, specs, accounts):
    sources = Sources(root)
    references = {}
    runs = []
    for spec in specs:
        key = (spec["model"], spec["dataset"])
        if key not in references:
            references[key] = reference_audit(*key, sources)
        runs.append(audit_attempt(spec, sources, accounts, references[key]))
    # A later successful attempt replaces an earlier one by attempt number only;
    # failed/incomplete attempts remain visible and are never metric-selected.
    for logical in {run["logical_task"] for run in runs}:
        eligible = [run for run in runs if run["logical_task"] == logical and run["eligible"]]
        if eligible:
            max(eligible, key=lambda run: run["attempt"])["selected"] = True
    comparisons = []
    for run in runs:
        if not run["selected"]:
            continue
        reference = references[(run["model"], run["dataset"])]
        for label in [*LABELS[run["dataset"]], "__macro__"]:
            if label == "__macro__":
                current = {"AUC": run["macro_auc"], "logloss": run["macro_logloss"]}
                full = {metric: statistics.mean(row[metric] for row in reference["metrics"].values()) for metric in ("AUC", "logloss")}
            else:
                current, full = run["metrics"][label], reference["metrics"][label]
            comparisons.append({"model": run["model"], "dataset": run["dataset"], "setting": run["setting"],
                                "attempt": run["attempt"], "job_task_id": run["job_task_id"], "label": label,
                                "comparison": f"E-{run['setting']}", "new_run_auc": current["AUC"], "full_E_auc": full["AUC"],
                                "delta_auc_E_minus_new": full["AUC"] - current["AUC"],
                                "new_run_logloss": current["logloss"], "full_E_logloss": full["logloss"],
                                "delta_logloss_E_minus_new": full["logloss"] - current["logloss"]})
    return runs, list(references.values()), comparisons, list(sources.records.values())


def accounting(root, specs, baseline_array, control_array):
    ids = {baseline_array, control_array}
    for spec in specs:
        path = Path(root) / spec["artifact_dir"] / ("started.json" if spec["kind"] == "native_baseline" else "manifest.json")
        if path.exists():
            raw = json.loads(path.read_text()).get("job_id")
            if raw:
                ids.add(str(raw))
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(sorted(ids)),
               "--format=JobID%64,JobIDRaw%64,State%32,ExitCode,Elapsed,NodeList%64"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return parse_accounting(result.stdout), {"command": command, "stdout": result.stdout,
                                            "stderr": result.stderr, "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def csv_text(rows, empty_fields):
    stream = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else empty_fields
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_outputs(output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot = output / "new_training_snapshots" / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    run_rows = []
    label_rows = []
    for run in evidence["runs"]:
        row = {key: run[key] for key in ("kind", "model", "dataset", "setting", "logical_task", "attempt", "run_id",
                                         "job_task_id", "raw_job_id", "state", "exit_code", "elapsed", "node", "eligible",
                                         "selected", "macro_auc", "macro_logloss", "protocol_fields_checked")}
        row.update(run["parameter_counts"], checkpoint_exists=run["checkpoint"]["exists"],
                   checkpoint_bytes=run["checkpoint"]["bytes"],
                   issues="; ".join(run["issues"]), pending_checks="; ".join(run["pending_checks"]))
        run_rows.append(row)
        for label, values in run["metrics"].items():
            label_rows.append({"model": run["model"], "dataset": run["dataset"], "setting": run["setting"],
                               "attempt": run["attempt"], "job_task_id": run["job_task_id"], "selected": run["selected"],
                               "label": label, **values})
    files = {
        "new_training_evidence.json": json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "new_training_runs.csv": csv_text(run_rows, ["kind", "model", "dataset", "state", "eligible"]),
        "new_training_label_metrics.csv": csv_text(label_rows, ["model", "dataset", "setting", "attempt", "job_task_id", "label", "AUC", "logloss"]),
        "new_training_comparisons.csv": csv_text(evidence["comparisons"], ["model", "dataset", "setting", "comparison", "label", "delta_auc_E_minus_new"]),
        "new_training_sources.csv": csv_text(evidence["sources"], ["path", "bytes", "sha256"]),
        "new_training_accounting.json": json.dumps(evidence["accounting"], indent=2) + "\n",
    }
    for name, content in files.items():
        (snapshot / name).write_text(content)
        temporary = output / (name + f".tmp-{os.getpid()}")
        temporary.write_text(content)
        os.replace(temporary, output / name)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-array", default="643809")
    parser.add_argument("--control-array", default="643822")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/rise_sample_analysis/results")
    args = parser.parse_args()
    specs = discover(ROOT, args.baseline_array, args.control_array)
    accounts, accounting_receipt = accounting(ROOT, specs, args.baseline_array, args.control_array)
    runs, references, comparisons, sources = collect(ROOT, specs, accounts)
    evidence = {"generated_utc": datetime.now(timezone.utc).isoformat(), "runs": runs, "historical_references": references,
                "comparisons": comparisons, "sources": sources, "accounting": accounting_receipt,
                "collector_sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__).resolve(), HELPER_PATH)},
                "scope": "Three new native baselines and two RankMixer controls matched to historical seed20262028 E; never paired with seed20262027.",
                "eligibility": "Completed scheduler 0:0 and outcome0, frozen/runtime/reference protocol, complete dual-log final Test, parameter counts, readable checkpoint, source hashes and split/rank coverage.",
                "parameter_definition": "Logged active trainable total/embedding/dense counts; not all stored tensors. Fixed controls are not capacity-matched.",
                "coverage_scope": "Per-rank blocks, allocated rows, batches and retained rows for train/valid/test; not historical byte-identical sample proof.",
                "checkpoint_scope": "Presence, positive size and readable PyTorch ZIP state metadata; no GPU loading, forward or complete weight-file hash.",
                "historical_source_scope": "Historical E completion is archive-backed; exact historical source/environment identity remains unverified.",
                "selection": "Highest eligible attempt per logical task, never selected by metric value; every failed/incomplete attempt retained."}
    snapshot = write_outputs(args.output_dir, evidence)
    print(json.dumps({"attempts": len(runs), "eligible": sum(run["eligible"] for run in runs),
                      "selected": sum(run["selected"] for run in runs), "snapshot": str(snapshot)}))
    for run in runs:
        print(json.dumps({key: run[key] for key in ("job_task_id", "attempt", "setting", "state", "eligible", "issues", "pending_checks")}))


if __name__ == "__main__":
    main()
