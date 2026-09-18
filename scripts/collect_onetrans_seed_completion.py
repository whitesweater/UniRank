#!/usr/bin/env python3
"""Audit the ten OneTrans SISA repetitions and their frozen historical references."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_sisa_component_results import (
    Sources, coverage_issues, parameter_counts, parse_coverage, parse_params,
    parse_test_metrics, parsed_value, write_csv,
)
from scripts.run_onetrans_seed_completion import (
    DATASETS, STUDY, STUDY_DIR, expected_params, study_task,
)


def portable_params(params, dataset):
    """Normalize only the ten documented historical dataset path fields."""
    result = {key: parsed_value(value) for key, value in params.items()}
    if str(result.get("data_root", "")).endswith("/datasets"):
        result["data_root"] = "./datasets"
    for split in ("train", "valid", "test"):
        for suffix in ("data", "item_info", "user_info"):
            key = f"{split}_{suffix}"
            ending = f"/datasets/{dataset}/{split}/{suffix}"
            if str(result.get(key, "")).endswith(ending):
                result[key] = "." + ending
    return result


def config_issues(actual, expected):
    return [f"Config mismatch: {key}" for key in sorted(actual.keys() | expected.keys())
            if key not in actual or key not in expected
            or parsed_value(actual[key]) != parsed_value(expected[key])]


def check_logs(record, checkpoint, slurm, expected, expected_coverage, historical=False):
    issues = record["issues"]
    for name, text in (("checkpoint", checkpoint), ("Slurm", slurm)):
        try:
            actual = parse_params(text)
            if historical:
                actual = portable_params(actual, record["dataset"])
            issues.extend(f"{name}: {issue}" for issue in config_issues(actual, expected))
        except (ValueError, TypeError) as error:
            issues.append(f"{name}: {error}")
    record["coverage"] = parse_coverage(slurm, 8192)
    issues.extend(coverage_issues(record["coverage"], expected_coverage,
                                 partial=record["state"] not in {"COMPLETED", "COMPLETED_MARKER"}))
    record.update(parameter_counts(checkpoint))
    labels = [entry["name"] for entry in parsed_value(expected["label_col"])]
    try:
        metrics = parse_test_metrics(checkpoint, labels)
        if metrics != parse_test_metrics(slurm, labels):
            issues.append("Final Test metrics differ between checkpoint and Slurm")
        return metrics
    except (ValueError, TypeError) as error:
        issues.append(str(error))
        return {}


def finish(record, metrics):
    record.update(eligible=False, metrics={}, macro_auc=None, macro_logloss=None)
    if not record["issues"] and metrics:
        record.update(eligible=True, metrics=metrics,
                      macro_auc=statistics.mean(value["AUC"] for value in metrics.values()),
                      macro_logloss=statistics.mean(value["logloss"] for value in metrics.values()))
    return record


def audit_new_run(task, expected, expected_coverage, checkpoint, slurm, manifest, outcome,
                  account, array_job):
    issues = []
    record = dict(model=task.model, dataset=task.dataset, setting=task.setting, seed=task.seed,
                  attempt=task.attempt, run_id=task.run_id, job_task_id=f"{array_job}_{task.task_id}",
                  state=account.get("state", "NOT_SUBMITTED" if not array_job else "UNKNOWN"),
                  exit_code=account.get("exit_code"), issues=issues, historical=False,
                  source_git_revision=manifest.get("git_revision"), gpu_names=manifest.get("gpu_names", []))
    if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
        issues.append("sacct does not show COMPLETED/0:0")
    if outcome.get("returncode") != 0:
        issues.append("Missing or unsuccessful artifact outcome")
    if slurm.splitlines().count(f"ONETRANS_SEED_COMPLETE run_id={task.run_id}") != 1:
        issues.append("Missing or duplicate exact completion marker")
    expected_manifest = dict(model=task.model, dataset=task.dataset, setting="sisa", task_id=task.task_id,
                             seed=task.seed, run_id=task.run_id, attempt=task.attempt,
                             world_size=4, per_gpu_batch=8192, global_batch=32768,
                             dataloader_seed=task.dataloader_seed,
                             sisa_parameter_seed=task.sisa_parameter_seed,
                             array_job_id=str(array_job), job_id=account.get("raw_job_id"))
    for key, value in expected_manifest.items():
        if key not in manifest or manifest[key] != value:
            issues.append(f"Manifest mismatch: {key}")
    gpus = manifest.get("gpu_names", [])
    if len(gpus) != 4 or any("H100" not in name for name in gpus):
        issues.append("Manifest must record four H100 GPUs")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest.get("git_revision", "")):
        issues.append("Missing valid source Git revision")
    issues.extend(f"Manifest: {issue}" for issue in
                  config_issues(manifest.get("effective_params", {}), expected))
    embedded = re.findall(r"^ONETRANS_SEED_PROTOCOL (.+)$", slurm, re.MULTILINE)
    try:
        if len(embedded) != 1 or json.loads(embedded[0]) != manifest:
            issues.append("Embedded Slurm protocol differs from manifest")
    except ValueError:
        issues.append("Malformed embedded Slurm protocol")
    metrics = check_logs(record, checkpoint, slurm, expected, expected_coverage)
    return finish(record, metrics)


def audit_reference(row, expected, expected_coverage, checkpoint, slurm):
    issues = []
    run_id = Path(row["checkpoint_log"]).stem
    record = dict(model=row["model"], dataset=row["dataset"], setting=row["setting"], seed=int(row["seed"]),
                  attempt=None, run_id=run_id, job_task_id=row["job_task_id"], state=row["state"],
                  exit_code=row["exit_code"], issues=issues, historical=True, source_git_revision=None,
                  status_evidence=row.get("status_evidence"), gpu_names=[row["gpu_name"]] * int(row["gpu_count"]))
    if (record["state"] not in {"COMPLETED", "COMPLETED_MARKER"}
            or not record["exit_code"].startswith("0:0") or row.get("complete") != "True"):
        issues.append("Historical reference lacks accepted successful completion evidence")
    if record["model"] != "OneTrans" or record["seed"] != 20262027:
        issues.append("Historical reference identity mismatch")
    for key, value in (("world_size", "4"), ("per_gpu_batch", "8192"), ("global_batch", "32768")):
        if row.get(key) != value:
            issues.append(f"Historical protocol mismatch: {key}")
    markers = [line for line in slurm.splitlines()
               if line in {f"SISA_NATIVE_STRICT_COMPLETE run_id={run_id}",
                           f"SISA_EXPANSION_COMPLETE run_id={run_id}"}]
    if len(markers) != 1:
        issues.append("Missing or duplicate historical completion marker")
    metrics = check_logs(record, checkpoint, slurm, expected, expected_coverage, historical=True)
    if metrics != json.loads(row["metrics_json"]):
        issues.append("Historical metrics differ from frozen reference table")
    return finish(record, metrics)


def three_seed_summary(runs):
    rows = []
    for dataset in dict.fromkeys(run["dataset"] for run in runs):
        sisas = [run for run in runs if run["dataset"] == dataset and run["setting"] == "sisa"]
        if len(sisas) != 3 or {run["seed"] for run in sisas} != {20262027, 20262028, 20262029}:
            continue
        if not all(run["eligible"] for run in sisas):
            continue
        sisas.sort(key=lambda run: run["seed"])
        if any(set(run["metrics"]) != set(sisas[0]["metrics"]) for run in sisas):
            continue
        baselines = [run for run in runs if run["dataset"] == dataset and run["setting"] == "baseline"
                     and run["seed"] == 20262027 and run["eligible"]]
        baseline = baselines[0] if len(baselines) == 1 else None
        for label in ["__macro__", *sisas[0]["metrics"]]:
            row = dict(model="OneTrans", dataset=dataset, scope="macro" if label == "__macro__" else "label",
                       label=label, sisa_seed_count=3, baseline_seed=20262027 if baseline else None)
            for metric, suffix, macro in (("AUC", "auc", "macro_auc"), ("logloss", "logloss", "macro_logloss")):
                values = [run[macro] if label == "__macro__" else run["metrics"][label][metric] for run in sisas]
                for run, value in zip(sisas, values):
                    row[f"sisa_{run['seed']}_{suffix}"] = value
                row[f"sisa_{suffix}_mean"] = statistics.mean(values)
                row[f"sisa_{suffix}_sample_sd"] = statistics.stdev(values)
                row[f"baseline_20262027_{suffix}"] = (
                    baseline[macro] if label == "__macro__" else baseline["metrics"][label][metric]
                ) if baseline else None
            rows.append(row)
    return rows


def parse_overrides(values, converter):
    overrides = {}
    for value in values:
        task, replacement = value.split("=", 1)
        task, replacement = int(task), converter(replacement)
        if task not in range(10) or task in overrides or not str(replacement).isdigit() or int(replacement) < 1:
            raise ValueError(f"Invalid or duplicate task override: {value}")
        overrides[task] = replacement
    return overrides


def query_accounting(arrays):
    if not arrays:
        return {}, {"command": [], "stdout": "", "sha256": hashlib.sha256(b"").hexdigest()}
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(sorted(arrays)), "--format=JobID,JobIDRaw,State,ExitCode"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and "." not in fields[0]:
            records[fields[0]] = dict(raw_job_id=fields[1], state=fields[2], exit_code=fields[3])
    return records, dict(command=command, stdout=result.stdout, stderr=result.stderr,
                         sha256=hashlib.sha256(result.stdout.encode()).hexdigest())


def resolve_account(job_task_id, accounts, cancellation):
    if job_task_id in accounts:
        return accounts[job_task_id]
    if (cancellation.get("returncode") == 0
            and job_task_id in cancellation.get("cancelled_job_task_ids", [])):
        # Slurm may omit pending array elements cancelled before allocation.
        # This label explicitly identifies the receipt, not a sacct outcome.
        return dict(state="CANCELLED_BY_USER", exit_code=None,
                    status_evidence="successful_user_cancellation_receipt")
    return {}


def collect(array_job=None, attempt=1, task_attempts=None, task_arrays=None):
    sources = Sources(ROOT)
    config = Path("experiments") / STUDY / "configs"
    references = sources.csv(config / "reference_runs.csv")
    params = sources.json(config / "reference_params.json")
    coverage = sources.csv(config / "data_coverage.csv")
    task_rows = sources.csv(config / "tasks.csv")
    submission = sources.json(Path("experiments") / STUDY / "notes/submission.json") or {}
    cancellation = sources.json(Path("experiments") / STUDY / "notes/cancellation.json") or {}
    array_job = str(array_job or submission.get("array_job_id") or "")
    arrays = {index: str((task_arrays or {}).get(index, array_job)) for index in range(10)}
    accounts, snapshot = query_accounting({value for value in arrays.values() if value})
    if len(references) != 10 or {(row["dataset"], row["setting"]) for row in references} != {
        (dataset, setting) for dataset in DATASETS for setting in ("baseline", "sisa")
    }:
        raise ValueError("Expected exactly ten unique historical baseline/SISA references")
    if len(task_rows) != 10 or {int(row["task_id"]) for row in task_rows} != set(range(10)):
        raise ValueError("Expected exactly ten frozen SISA repetition tasks")
    for row in coverage:
        for key in ("rank", "blocks", "assigned_rows", "batches", "processed_rows"):
            row[key] = int(row[key])

    def expected_coverage(dataset, setting):
        subset = [row for row in coverage if row["dataset"] == dataset and row["setting"] == setting]
        if len(subset) != 12 or {(row["split"], row["rank"]) for row in subset} != {
            (split, rank) for split in ("train", "valid", "test") for rank in range(4)
        }:
            raise ValueError(f"Incomplete frozen four-rank coverage: {dataset}/{setting}")
        return subset

    runs = []
    for row in references:
        checkpoint, slurm = row["checkpoint_log"], row["slurm_log"]
        record = audit_reference(row, params[f"OneTrans_{row['dataset']}_{row['setting']}"],
                                 expected_coverage(row["dataset"], row["setting"]),
                                 sources.read(Path(checkpoint)) or "", sources.read(Path(slurm)) or "")
        record.update(checkpoint_log=checkpoint, slurm_log=slurm)
        runs.append(record)
    for row in sorted(task_rows, key=lambda row: int(row["task_id"])):
        index = int(row["task_id"])
        original = study_task(index, int(row["attempt"]))
        for key in ("model", "dataset", "setting", "seed", "dataloader_seed", "sisa_parameter_seed", "run_id"):
            if row[key] != str(getattr(original, key)):
                raise ValueError(f"Frozen task identity differs: {index}/{key}")
        task = study_task(index, (task_attempts or {}).get(index, attempt))
        array = arrays[index]
        checkpoint = f"checkpoints/{task.dataset}/{task.run_id}.log"
        slurm = f"logs/unirank-onetrans-seeds-{array}_{index}.out" if array else ""
        artifact = Path("artifacts") / STUDY / task.run_id
        record = audit_new_run(task, expected_params(task), expected_coverage(task.dataset, "sisa"),
                               sources.read(Path(checkpoint)) or "", (sources.read(Path(slurm)) or "") if slurm else "",
                               sources.json(artifact / "manifest.json") or {},
                               sources.json(artifact / "outcome.json") or {},
                               resolve_account(f"{array}_{index}", accounts, cancellation), array)
        record.update(checkpoint_log=checkpoint, slurm_log=slurm)
        runs.append(record)
    return dict(generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                study=STUDY, runs=runs, three_seed_summary=three_seed_summary(runs),
                sources=list(sources.records.values()), accounting_snapshot=snapshot,
                evidence_scope="Separate OneTrans diagnostic study; excluded from the accepted 4x4 SISA paper matrix.",
                summary_definition="Three full-SISA seeds only; arithmetic mean and sample SD (ddof=1). Historical baseline has only seed 20262027, so these are not three paired baseline/SISA seeds.",
                protocol_caveats="Four-rank, batch-8192 coverage is checked by counts, not sample IDs. Historical GPU models and software/source provenance differ; historical internal seed values were not explicitly logged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-job")
    parser.add_argument("--output", type=Path, default=STUDY_DIR / "results")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--task-attempt", action="append", default=[], metavar="ID=N")
    parser.add_argument("--task-array", action="append", default=[], metavar="ID=ARRAY")
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("attempt must be positive")
    try:
        evidence = collect(args.array_job, args.attempt, parse_overrides(args.task_attempt, int),
                           parse_overrides(args.task_array, str))
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["model", "dataset", "setting", "seed", "attempt", "job_task_id", "state", "exit_code", "historical",
              "eligible", "macro_auc", "macro_logloss", "total_parameters", "dense_parameters",
              "source_git_revision", "checkpoint_log", "slurm_log", "issues"]
    write_csv(args.output / "runs.csv", [{key: "; ".join(run[key]) if key == "issues" else run.get(key)
                                        for key in fields} for run in evidence["runs"]], fields)
    label_fields = ["model", "dataset", "setting", "seed", "job_task_id", "label", "AUC", "logloss"]
    labels = [dict({key: run[key] for key in label_fields[:5]}, label=label, **metrics)
              for run in evidence["runs"] for label, metrics in run["metrics"].items()]
    write_csv(args.output / "label_metrics.csv", labels, label_fields)
    summary_fields = ["model", "dataset", "scope", "label", "sisa_seed_count", "baseline_seed"]
    for suffix in ("auc", "logloss"):
        summary_fields.extend([f"sisa_{seed}_{suffix}" for seed in (20262027, 20262028, 20262029)])
        summary_fields.extend([f"sisa_{suffix}_mean", f"sisa_{suffix}_sample_sd", f"baseline_20262027_{suffix}"])
    write_csv(args.output / "three_seed_summary.csv", evidence["three_seed_summary"], summary_fields)
    print(json.dumps(dict(generated_at=evidence["generated_at"], runs=len(evidence["runs"]),
                          eligible=sum(run["eligible"] for run in evidence["runs"]),
                          complete_datasets=sum(row["scope"] == "macro" for row in evidence["three_seed_summary"]))))


if __name__ == "__main__":
    main()
