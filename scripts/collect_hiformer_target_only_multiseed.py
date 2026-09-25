#!/usr/bin/env python3
"""Audit twelve HiFormer target-only runs while keeping prior studies read-only.

Eight new attempts supplement the four seed20262028 runs. Only fully audited
final Test metrics enter comparisons. Three-seed means use sample SD (ddof=1)
and describe the historical 4-GPU/2-GPU layouts, not a pure seed intervention.
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
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "scripts/collect_hiformer_target_only_extension.py"
_SPEC = importlib.util.spec_from_file_location("hiformer_multiseed_legacy", LEGACY_PATH)
LEGACY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LEGACY)
BASE = LEGACY.BASE
STUDY = Path("experiments/hiformer_target_only_multiseed")
DATASETS = LEGACY.REPORT_DATASETS
SEEDS = (20262027, 20262028, 20262029)
DISPLAY = LEGACY.DISPLAY
REFERENCE_AUDIT = STUDY / "notes/reference_audit.json"


def normalize_submission(submission):
    tasks = submission.get("tasks", [])
    if len(tasks) != 8 or {task.get("task_index") for task in tasks} != set(range(8)):
        raise ValueError("Submission must contain exactly eight distinct new tasks")
    result = []
    for task in tasks:
        index = task["task_index"]
        seed, dataset = (20262027, 20262029)[index // 4], DATASETS[index % 4]
        attempt = task.get("attempt", 1)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("Attempt must be a positive integer")
        array = str(task.get("array_id", ""))
        if not array.isdigit():
            raise ValueError("Each task requires its own actual array identity")
        world, batch = (4, 8192) if seed == 20262027 else (2, 16384)
        run_id = f"RISE_hiformer_target_only_{dataset}_seed{seed}_ws{world}_bs{batch}_attempt{attempt}"
        expected = {"seed": seed, "dataset": dataset, "run_id": run_id,
                    "job_task_id": f"{array}_{index}",
                    "manifest": f"artifacts/rise_hiformer_target_only_multiseed/{run_id}/frozen_manifest.json",
                    "checkpoint_log": f"checkpoints/{dataset}/{run_id}.log"}
        for key, value in expected.items():
            if task.get(key) != value:
                raise ValueError(f"Submission identity mismatch: task {index}/{key}")
        path = Path(task["slurm_log"])
        if path.is_absolute() or ".." in path.parts or path.parent != Path("logs"):
            raise ValueError("Slurm log must be its actual repository-relative logs path")
        result.append({**task, "array_id": array, "model": "HiFormer", "mode": "target_only",
                       "kind": "hiformer_target_only_multiseed", "world_size": 4 if seed == 20262027 else 2,
                       "batch_size": 8192 if seed == 20262027 else 16384,
                       "artifact_dir": str(Path(task["manifest"]).parent)})
    return sorted(result, key=lambda row: row["task_index"])


def reference_audits(sources):
    """Independently reparse raw references and verify their frozen identities."""
    frozen = sources.json(REFERENCE_AUDIT)
    if not frozen:
        raise ValueError("Frozen reference audit unavailable")
    registry = frozen["reference_registry"]
    registry_text = sources.read(registry["path"])
    registry_matches = sources.records.get(registry["path"], {}).get("sha256") == registry["sha256"]
    registry_rows = list(csv.DictReader(io.StringIO(registry_text or "")))
    rows, seen = [], set()
    for original in frozen["full_rise_references"]:
        row = dict(original, seed=int(original["seed"]), model="HiFormer", issues=[])
        key = row["dataset"], row["seed"]
        if key in seen or key[0] not in DATASETS or key[1] not in SEEDS:
            raise ValueError("Duplicate or out-of-scope Full RISE reference")
        seen.add(key)
        issues = row["issues"]
        log = sources.read(row["checkpoint_log"]) or ""
        slurm = sources.read(row["slurm_log"]) or ""
        if not registry_matches or not any(record["job_task_id"] == row["job_task_id"] for record in registry_rows):
            issues.append("Frozen Full RISE registry differs or omits task")
        for name in ("checkpoint_log", "slurm_log"):
            if sources.records.get(row[name], {}).get("sha256") != row[name + "_sha256"]:
                issues.append(f"Frozen Full RISE source SHA mismatch: {name}")
        if row["archived_state"] not in {"COMPLETED", "COMPLETED_MARKER"}:
            issues.append("Historical Full RISE completion receipt unavailable")
        try:
            params = BASE.parsed_params(log)
            if params != row["params"] or BASE.parsed_params(slurm) != params:
                issues.append("Historical Full RISE parameters differ from frozen/dual log")
            normalized = dict(params)
            additions = row["historical_default_additions"]
            expected_additions = {"dataloader_seed": 2026, "sisa_parameter_seed": 20260821} if row["seed"] == 20262027 else {}
            if additions != expected_additions or any(name in normalized for name in additions):
                issues.append("Historical default reconstruction differs from approved seed27 contract")
            normalized.update(additions)
            for name, relocation in row["path_relocation_map"].items():
                if (normalized.get(name) != relocation["original"]
                        or relocation["normalized"] != relocation["original"].replace(
                            "/data_nvme/user/ywhao/proj/UniRank/datasets", "./datasets", 1)
                        or row["seed"] != 20262027 or row["dataset"] not in {"QK_Video_Action", "MerRec_Action"}):
                    issues.append(f"Unapproved reference path relocation: {name}")
                normalized[name] = relocation["normalized"]
            if normalized != row["normalized_reference_params"]:
                issues.append("Normalized Full RISE reference differs from frozen audit")
            row["raw_params"], row["params"] = params, normalized
            required = {"model": "HiFormer", "dataset_id": row["dataset"], "seed": row["seed"],
                        "world_size": 4 if row["seed"] == 20262027 else 2,
                        "batch_size": 8192 if row["seed"] == 20262027 else 16384,
                        "epochs": 1, "max_len": 100, "sisa_enabled": True,
                        "sisa_score_dim": 16, "dataloader_seed": row["seed"] - 20260001,
                        "sisa_parameter_seed": row["seed"] - 1206,
                        "enable_bf16": True, "enable_torch_compile": True}
            for name, value in required.items():
                if normalized.get(name) != value:
                    issues.append(f"Historical Full RISE required protocol mismatch: {name}")
            metrics = BASE.HELPER.parse_test_metrics(log, BASE.LABELS[row["dataset"]])
            if metrics != row["metrics"] or BASE.HELPER.parse_test_metrics(slurm, BASE.LABELS[row["dataset"]]) != metrics:
                issues.append("Historical Full RISE Test differs from frozen/dual log")
            coverage = BASE.HELPER.parse_coverage(slurm, normalized["batch_size"])
            if coverage != row["coverage"]:
                issues.append("Historical Full RISE coverage differs from frozen audit")
            expected_keys = {(split, rank) for split in ("train", "valid", "test") for rank in range(normalized["world_size"])}
            if len(coverage) != len(expected_keys) or {(entry["split"], entry["rank"]) for entry in coverage} != expected_keys:
                issues.append("Historical Full RISE lacks complete per-rank split coverage")
            row["parameter_counts"] = BASE.parameter_counts(log)
            if row["parameter_counts"] != BASE.parameter_counts(slurm) or any(value is None for value in row["parameter_counts"].values()):
                issues.append("Historical Full RISE parameter counts disagree")
        except (KeyError, ValueError, TypeError) as error:
            issues.append(f"Historical Full RISE parse: {error}")
        row["eligible"] = not issues
        rows.append(row)
    if seen != {(dataset, seed) for dataset in DATASETS for seed in SEEDS}:
        raise ValueError("Full RISE reference scope must contain all twelve cells")
    return rows


def protocol_issues(spec, actual, reference, frozen):
    issues = []
    expected = dict(reference, model_id=spec["run_id"])
    for key in sorted(actual.keys() | expected.keys() | frozen.keys()):
        if key not in actual or actual.get(key) != expected.get(key) or actual.get(key) != frozen.get(key):
            issues.append(f"Runtime/frozen/reference configuration mismatch: {key}")
    return issues


def compare_run(run, reference):
    if not run["eligible"] or not reference["eligible"]:
        return []
    rows = []
    for label in [*BASE.LABELS[run["dataset"]], "MACRO"]:
        before, after = reference["metrics"].get(label), run["metrics"].get(label)
        if label == "MACRO":
            before = {name: statistics.fmean(value[name] for value in reference["metrics"].values()) for name in ("AUC", "logloss")}
            after = {"AUC": run["macro_auc"], "logloss": run["macro_logloss"]}
        rows.append({"model": "HiFormer", "dataset": run["dataset"], "seed": run["seed"], "label": label,
                     "full_rise_job": reference["job_task_id"], "target_only_job": run["job_task_id"],
                     "full_rise_auc": before["AUC"], "target_only_auc": after["AUC"],
                     "delta_auc": after["AUC"] - before["AUC"],
                     "full_rise_logloss": before["logloss"], "target_only_logloss": after["logloss"],
                     "delta_logloss": after["logloss"] - before["logloss"]})
    return rows


def summarize(runs, references, baselines):
    seen = set()
    for run in runs:
        key = run["dataset"], run["seed"]
        if key in seen:
            raise ValueError("Duplicate dataset/seed cannot form a three-seed mean")
        seen.add(key)
    reference_map = {(row["dataset"], row["seed"]): row for row in references}
    baseline_map = {row["dataset"]: row for row in baselines if row["kind"] == "paper_seed27"}
    rows = []
    for dataset in DATASETS:
        successful = [run for run in runs if run["dataset"] == dataset and run["eligible"]]
        row = {"dataset": dataset, "completed_seeds": len(successful), "required_seeds": 3,
               "baseline_seed": 20262027, "baseline_auc": None, "baseline_logloss": None,
               "baseline_job": None, "summary_kind": "cross_layout_descriptive_sample_sd"}
        baseline = baseline_map.get(dataset)
        if baseline:
            row.update(baseline_auc=float(baseline["baseline_auc"]), baseline_logloss=float(baseline["baseline_logloss"]),
                       baseline_job=baseline["baseline_job"])
        for metric in ("auc", "logloss"):
            for prefix in ("target_only", "full_rise", "delta"):
                row[f"{prefix}_{metric}_mean"] = None
                row[f"{prefix}_{metric}_sd"] = None
        if {run["seed"] for run in successful} == set(SEEDS):
            refs = [reference_map[(dataset, run["seed"])] for run in successful]
            if all(reference["eligible"] for reference in refs):
                for metric, field in (("auc", "AUC"), ("logloss", "logloss")):
                    after = [run[f"macro_{metric}"] for run in successful]
                    before = [statistics.fmean(values[field] for values in reference["metrics"].values()) for reference in refs]
                    for prefix, values in (("target_only", after), ("full_rise", before), ("delta", [a - b for a, b in zip(after, before)])):
                        row[f"{prefix}_{metric}_mean"] = statistics.fmean(values)
                        row[f"{prefix}_{metric}_sd"] = statistics.stdev(values)
        rows.append(row)
    return rows


def completion_flags(runs):
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    complete = len(runs) == 12 and {(run["dataset"], run["seed"]) for run in runs} == expected and all(run["eligible"] for run in runs)
    new = [run for run in runs if run["seed"] != 20262028]
    terminal = len(new) == 8 and all(run["state"].split("+")[0] in LEGACY.TERMINAL_STATES for run in new)
    old_failed = any(not run["eligible"] for run in runs if run["seed"] == 20262028)
    return {"complete": complete, "all_terminal": terminal, "needs_attention": old_failed or (terminal and not complete)}



def seed_bundle(spec):
    return {"seed": spec["seed"], "dataloader_seed": spec["seed"] - 20260001,
            "sisa_parameter_seed": spec["seed"] - 1206}


def expected_worker(spec, rank):
    return {"task_id": spec["task_index"], "mode": "target_only", "rank": rank,
            "world_size": spec["world_size"], "run_id": spec["run_id"], "seed_bundle": seed_bundle(spec)}


def expected_model_audit(spec, reference, rank):
    return {**LEGACY.expected_model_audit(spec, reference, rank), "world_size": spec["world_size"],
            "run_id": spec["run_id"], "seed_bundle": seed_bundle(spec)}


def model_audit_issues(spec, reference, actual, rank):
    expected = expected_model_audit(spec, reference, rank)
    return [f"rank{rank} construction mismatch: {key}" for key in sorted(actual.keys() | expected.keys())
            if actual.get(key) != expected.get(key)]


def rng_audit_issues(spec, actual, rank):
    issues = []
    for key, value in {"rank": rank, "world_size": spec["world_size"], "run_id": spec["run_id"],
                       "seed_bundle": seed_bundle(spec), "audit_preserved_rng": True}.items():
        if actual.get(key) != value:
            issues.append(f"rank{rank} RNG identity mismatch: {key}")
    before, after = actual.get("before_audit", {}), actual.get("after_audit", {})
    required = {"torch_cpu", "python", "numpy", "torch_cuda_current"}
    if set(before) != required or any(not isinstance(value, str) or not LEGACY.re.fullmatch(r"[0-9a-f]{64}", value) for value in before.values()):
        issues.append(f"rank{rank} RNG fingerprints incomplete")
    if before != after:
        issues.append(f"rank{rank} construction audit changed RNG")
    return issues

def audit_attempt(spec, sources, accounts, reference):
    artifact = Path(spec["artifact_dir"])
    manifest = sources.json(spec["manifest"]) or {}
    started = sources.json(artifact / "started.json") or {}
    outcome = sources.json(artifact / "outcome.json") or {}
    account = BASE.select_account(accounts, spec["job_task_id"], started.get("job_id"))
    log = sources.read(spec["checkpoint_log"]) or ""
    slurm = sources.read(spec["slurm_log"]) or ""
    completed = account["state"] == "COMPLETED" and account["exit_code"] == "0:0"
    record = {**spec, **account, "seed": spec["seed"], "eligible": False,
              "issues": [], "pending_checks": [], "metrics": {}, "macro_auc": None,
              "macro_logloss": None, "protocol_fields_checked": 0, "model_audits": {},
              "source_hash_checks": [], "outcome": outcome,
              "parameter_counts": BASE.parameter_counts(log),
              "coverage": BASE.HELPER.parse_coverage(slurm, spec["batch_size"]),
              "progress": LEGACY.progress_snapshot(slurm), "gpu_names": started.get("gpu_names", [])}
    issues, pending = record["issues"], record["pending_checks"]
    if not completed:
        pending.append(f"Scheduler not successful: {account['state']}/{account['exit_code']}")
    if account["job_task_id"] != spec["job_task_id"]:
        issues.append("Scheduler task identity differs from submission")
    if not started.get("job_id"):
        pending.append("Started allocation receipt unavailable")
    else:
        if str(started["job_id"]) != str(account["raw_job_id"]):
            issues.append("Started allocation differs from scheduler")
        if f"{started.get('array_job_id')}_{started.get('array_task_id')}" != spec["job_task_id"]:
            issues.append("Started array identity differs from submission")
    if outcome.get("returncode") != 0:
        pending.append("Successful process outcome unavailable")
    if outcome and str(outcome.get("job_id")) != str(started.get("job_id")):
        issues.append("Outcome allocation differs from started receipt")
    if len(record["gpu_names"]) != spec["world_size"]:
        pending.append("Expected GPU allocation count receipt unavailable")
    counts = LEGACY.parameter_expectations(reference)
    if not manifest:
        pending.append("Frozen manifest unavailable")
    else:
        identity = {"study": "rise_hiformer_target_only_multiseed", "run_id": spec["run_id"],
                    "task_id": spec["task_index"], "attempt": spec["attempt"],
                    "model": "HiFormer", "dataset": spec["dataset"], "mode": "target_only",
                    "reference_job_task": reference["job_task_id"],
                    "reference_log": reference["checkpoint_log"],
                    "allowed_parameter_differences": sorted(["model_id", *reference["historical_default_additions"], *reference["path_relocation_map"]]),
                    "process_local_treatment": "target_only", "seed_bundle": seed_bundle(spec),
                    "world_size": spec["world_size"], "batch_size": spec["batch_size"], "global_batch": 32768}
        for key, value in identity.items():
            if manifest.get(key) != value:
                issues.append(f"Frozen manifest mismatch: {key}")
        if manifest.get("reference_params") != reference["raw_params"]:
            issues.append("Frozen raw reference parameters differ from full RISE")
        for field, expected in (("normalized_reference_params", reference["params"]),
                                ("historical_default_additions", reference["historical_default_additions"]),
                                ("path_relocation_map", {name: {"historical": change["original"], "current": change["normalized"]} for name, change in reference["path_relocation_map"].items()})):
            if manifest.get(field) != expected:
                issues.append(f"Frozen reference reconstruction mismatch: {field}")
        expected_hash = sources.records.get(reference["checkpoint_log"], {}).get("sha256")
        if manifest.get("reference_log_sha256") != expected_hash:
            issues.append("Frozen reference log SHA mismatch")
        for key in ("active_dense", "active_embedding", "active_total", "stored_total"):
            if manifest.get("parameter_expectations", {}).get(key) != counts[key]:
                issues.append(f"Frozen parameter expectation mismatch: {key}")
        required_sources = {
            "scripts/run_hiformer_target_only_multiseed.py",
            "scripts/submit_hiformer_target_only_multiseed.sbatch",
            "experiments/rise_sample_analysis/scripts/run_targeted_controls.py",
            "experiments/rise_sample_analysis/scripts/targeted_controls.py",
            "model_zoo/HiFormer.py", "unirank/pytorch/layers/attentions/sisa.py",
            "run_expid.py", "pyproject.toml", "uv.lock",
        }
        if not required_sources.issubset(manifest.get("source_sha256", {})):
            issues.append("Frozen manifest omits a required execution source")
        source_issues, record["source_hash_checks"] = BASE.source_manifest_check(manifest, sources)
        issues.extend(source_issues)
        for relative, digest in manifest.get("source_sha256", {}).items():
            snapshot_path = artifact / "source" / relative
            sources.read(snapshot_path)
            if sources.records.get(str(snapshot_path), {}).get("sha256") != digest:
                issues.append(f"Frozen source snapshot mismatch: {relative}")
        expected_config = {"model_config.yaml": {"Base": {}, spec["run_id"]: manifest.get("effective_params", {})},
                           "dataset_config.yaml": {spec["dataset"]: {}}}
        for filename, expected in expected_config.items():
            if sources.json(artifact / "config" / filename) != expected:
                issues.append(f"Frozen runtime configuration mismatch: {filename}")
        for name, text in (("checkpoint", log), ("Slurm", slurm)):
            if "INFO Params:" not in text:
                pending.append(f"{name} runtime parameters unavailable")
                continue
            try:
                actual = BASE.parsed_params(text)
                issues.extend(f"{name}: {issue}" for issue in protocol_issues(
                    spec, actual, reference["params"], manifest.get("effective_params", {})))
                if name == "checkpoint":
                    record["protocol_fields_checked"] = len(actual)
            except (ValueError, TypeError) as error:
                (issues if completed else pending).append(f"{name} parameter parse: {error}")
    record["worker_receipts"], record["rng_audits"] = {}, {}
    if started and (started.get("world_size") != spec["world_size"] or started.get("seed_bundle") != seed_bundle(spec)):
        issues.append("Started receipt seed bundle or world size differs")
    for rank in range(spec["world_size"]):
        worker = sources.json(artifact / f"worker_rank{rank}.json")
        if worker is None:
            pending.append(f"rank{rank} worker artifact unavailable")
        else:
            record["worker_receipts"][str(rank)] = worker
            if worker != expected_worker(spec, rank):
                issues.append(f"rank{rank} worker artifact differs from expected identity")
        rng = sources.json(artifact / f"rng_audit_rank{rank}.json")
        if rng is None:
            pending.append(f"rank{rank} RNG audit artifact unavailable")
        else:
            record["rng_audits"][str(rank)] = rng
            issues.extend(rng_audit_issues(spec, rng, rank))
        audit = sources.json(artifact / f"model_audit_rank{rank}.json")
        if audit is None:
            pending.append(f"rank{rank} construction artifact unavailable")
        else:
            record["model_audits"][str(rank)] = audit
            issues.extend(model_audit_issues(spec, reference, audit, rank))
    try:
        workers = LEGACY.embedded_json(slurm, "RISE_TARGETED_WORKER")
        expected_workers = [expected_worker(spec, rank) for rank in range(spec["world_size"])]
        if sorted(workers, key=lambda row: row["rank"]) != expected_workers:
            pending.append("Exact per-rank worker identity markers unavailable")
        rng_markers = LEGACY.embedded_json(slurm, "RISE_TARGETED_RNG_AUDIT")
        if len(rng_markers) != spec["world_size"]:
            pending.append("Exact per-rank RNG audit markers unavailable")
        else:
            for rank, rng in enumerate(sorted(rng_markers, key=lambda row: row["rank"])):
                issues.extend(rng_audit_issues(spec, rng, rank))
                if rng != record["rng_audits"].get(str(rank)):
                    issues.append(f"rank{rank} RNG receipt differs between artifact and log")
        audits = LEGACY.embedded_json(slurm, "RISE_TARGETED_MODEL_AUDIT")
        if len(audits) != spec["world_size"]:
            pending.append("Exact per-rank embedded construction audits unavailable")
        else:
            for rank, audit in enumerate(sorted(audits, key=lambda row: row["rank"])):
                issues.extend(model_audit_issues(spec, reference, audit, rank))
                if audit != record["model_audits"].get(str(rank)):
                    issues.append(f"rank{rank} construction differs between artifact and log")
    except (ValueError, KeyError, TypeError) as error:
        (issues if completed else pending).append(f"Construction marker parse: {error}")
    expected_counts = {"total_parameters": counts["active_total"],
                       "dense_parameters": counts["active_dense"],
                       "embedding_parameters": counts["active_embedding"]}
    if any(value is None for value in record["parameter_counts"].values()):
        pending.append("Active parameter counts unavailable")
    elif record["parameter_counts"] != expected_counts or BASE.parameter_counts(slurm) != expected_counts:
        issues.append("Dual-log active parameter counts differ from target-only expectation")
    issues.extend(BASE.HELPER.coverage_issues(record["coverage"], reference["coverage"], partial=not completed))
    record["checkpoint"] = BASE.inspect_checkpoint((sources.root / spec["checkpoint_log"]).with_suffix(".model"))
    if not record["checkpoint"]["exists"] or not record["checkpoint"]["zip_metadata_readable"]:
        pending.append("Readable saved model unavailable")
    record["runtime_error_signals"] = LEGACY.RUNTIME_ERRORS.findall(slurm + "\n" + log)
    if record["runtime_error_signals"]:
        issues.append("Runtime CUDA/NCCL/Python error signal present")
    marker = f"RISE_TARGETED_COMPLETE run_id={spec['run_id']}"
    if slurm.splitlines().count(marker) != 1:
        pending.append("Unique completion marker unavailable")
    metrics = None
    if "Test evaluation" not in log:
        pending.append("Final Test evaluation unavailable")
    else:
        try:
            metrics = BASE.HELPER.parse_test_metrics(log, BASE.LABELS[spec["dataset"]])
            if metrics != BASE.HELPER.parse_test_metrics(slurm, BASE.LABELS[spec["dataset"]]):
                issues.append("Final Test metrics differ between logs")
            preserved = LEGACY.re.findall(r"Preserved best model checkpoint: (.+)", log)
            if len(preserved) != 1 or Path(preserved[0]).name != spec["run_id"] + ".model":
                pending.append("Unique best-checkpoint preservation receipt unavailable")
        except (ValueError, TypeError) as error:
            (issues if completed else pending).append(f"Final Test parse: {error}")
    if not reference["eligible"]:
        issues.append("Historical full RISE failed its independent audit")
    if completed and not issues and not pending and metrics is not None:
        record.update(eligible=True, metrics=metrics,
                      macro_auc=statistics.fmean(value["AUC"] for value in metrics.values()),
                      macro_logloss=statistics.fmean(value["logloss"] for value in metrics.values()))
    return record



def collect(root, submission, accounts):
    sources = BASE.Sources(root)
    sources.read(STUDY / "notes/submission.json")
    references = reference_audits(sources)
    reference_map = {(row["dataset"], row["seed"]): row for row in references}
    old_submission = sources.json(LEGACY.STUDY / "notes/submission.json")
    # Pure read-only collection: do not invoke the old CLI/write_outputs.
    old = LEGACY.collect(root, old_submission, accounts)
    runs = []
    for run in old["runs"]:
        run = dict(run, seed=20262028, world_size=2, batch_size=16384, evidence_origin="reused_seed28")
        if not reference_map[(run["dataset"], 20262028)]["eligible"]:
            run["issues"].append("Frozen multiseed Full RISE reference audit failed")
            run.update(eligible=False, metrics={}, macro_auc=None, macro_logloss=None)
        runs.append(run)
    for spec in normalize_submission(submission):
        reference = reference_map[(spec["dataset"], spec["seed"])]
        run = audit_attempt(spec, sources, accounts, reference)
        run["evidence_origin"] = "new_multiseed"
        runs.append(run)
    runs.sort(key=lambda row: (DATASETS.index(row["dataset"]), row["seed"]))
    comparisons = [row for run in runs for row in compare_run(run, reference_map[(run["dataset"], run["seed"])])]
    baseline_path = LEGACY.STUDY / "results/native_baselines.csv"
    baselines = list(csv.DictReader(io.StringIO(sources.read(baseline_path) or "")))
    sources.read(LEGACY.STUDY / "results/native_baseline_evidence.json")
    source_map = {row["path"]: row for row in old["sources"]}
    source_map.update(sources.records)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), **completion_flags(runs),
            "scope": "Exactly HiFormer target_only, four datasets and seeds 20262027/28/29; reuse four seed28 runs and submit eight new runs.",
            "runs": runs, "historical_references": references, "comparisons": comparisons,
            "summary": summarize(runs, references, baselines), "native_baselines": baselines,
            "sources": list(source_map.values()),
            "limitations": [
                "Means and sample SD describe three seed/layout tuples, not strictly matched seed replication or statistical significance.",
                "Seed27 uses four GPUs/batch8192; seed28/29 use two GPUs/batch16384. Global batch is 32768 but drop_last sample coverage differs.",
                "Seed27 dataloader2026 and RISE parameter20260821 are reconstructed historical defaults, not explicit historical runtime Params.",
                "Compare placement within each seed against its corresponding Full RISE and verify per-rank sample coverage; target_only removes field parameters.",
                "Historical source/environment and dataset bytes are not fully traceable; seed27 local-path relocation preserves split suffixes.",
                "Native Baseline values are existing single-run references, not three-seed baseline means. No native Baseline is newly trained."]}


def accounting(root, specs):
    identifiers = {"648217", "644098_1", *(str(spec["array_id"]) for spec in specs)}
    old_submission = json.loads((Path(root) / LEGACY.STUDY / "notes/submission.json").read_text())
    all_specs = [*specs, LEGACY.existing_qk_spec(root), *LEGACY.normalize_submission(old_submission)]
    for spec in all_specs:
        started = Path(root) / spec["artifact_dir"] / "started.json"
        if started.exists():
            raw = json.loads(started.read_text()).get("job_id")
            if raw:
                identifiers.add(str(raw))
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(sorted(identifiers)),
               "--format=JobID%64,JobIDRaw%64,State%32,ExitCode,Elapsed,NodeList%64"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    receipt = {"command": command, "returncode": result.returncode, "stdout": result.stdout,
               "stderr": result.stderr, "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}
    return BASE.parse_accounting(result.stdout) if result.returncode == 0 else [], receipt


def render_report(evidence):
    successful = sum(run["eligible"] for run in evidence["runs"])
    lines = ["# HiFormer 仅历史汇聚接入 RISE：四数据集三种子实验", "",
             f"已完成最终测试并通过审计：{successful}/12。共复用 4 项 seed20262028 结果，新增 8 项 seed20262027/20262029 训练。", "",
             "修改版只在历史汇聚处保留 1 个 RISE 接入点，关闭 3 个字段交互层的 RISE。",
             "主要比较同一数据集、同一训练种子对应的完整 RISE。未完成或审计未通过的训练不展示测试指标。", "",
             "## 四数据集汇总", "",
             "| 数据集 | 已通过 | Baseline AUC（单次） | 完整 RISE AUC（均值 ± SD） | 修改版 AUC（均值 ± SD） | ΔAUC 均值 | Baseline LogLoss（单次） | 完整 RISE LogLoss（均值 ± SD） | 修改版 LogLoss（均值 ± SD） | ΔLogLoss 均值 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in evidence["summary"]:
        def number(key):
            return "—" if row.get(key) is None else f"{row[key]:.8f}"

        def mean_sd(prefix):
            return "—" if row.get(prefix + "_mean") is None else f"{row[prefix + '_mean']:.8f} ± {row[prefix + '_sd']:.8f}"

        lines.append(f"| {DISPLAY[row['dataset']]} | {row['completed_seeds']}/3 | {number('baseline_auc')} | "
                     f"{mean_sd('full_rise_auc')} | {mean_sd('target_only_auc')} | {number('delta_auc_mean')} | "
                     f"{number('baseline_logloss')} | {mean_sd('full_rise_logloss')} | {mean_sd('target_only_logloss')} | {number('delta_logloss_mean')} |")
    lines += ["", "三个种子均通过审计后才填写均值与样本标准差（ddof=1）；各次指标为该数据集标签等权宏平均。",
              "Baseline 为已有主表原生 HiFormer seed20262027 的单次结果，不是三种子 Baseline 均值；与修改版三均仅作描述性比较。",
              "seed20262027 使用 4 卡、单卡 batch=8192；seed20262028/29 使用 2 卡、单卡 batch=16384。全局 batch=32768、1 epoch、max_len=100、BF16/torch.compile 保持。",
              "均值 ± SD 是跨布局的描述统计，不能将 world size、每卡 batch、drop_last 覆盖变化归为纯训练种子效应，也不据此声明统计显著性。", "",
              "## 全部 12 项逐种子对照", "",
              "| 数据集 | 训练种子 | 完整 RISE 任务 | 修改版任务 | 调度/审计状态 | 完整 RISE AUC | 修改版 AUC | ΔAUC | 完整 RISE LogLoss | 修改版 LogLoss | ΔLogLoss |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    macros = {(row["dataset"], row["seed"]): row for row in evidence["comparisons"] if row["label"] == "MACRO"}
    references = {(row["dataset"], row["seed"]): row for row in evidence["historical_references"]}
    for run in evidence["runs"]:
        key = run["dataset"], run["seed"]
        row = macros.get(key)
        values = ([f"{row[name]:+.8f}" if name.startswith("delta") else f"{row[name]:.8f}"
                   for name in ("full_rise_auc", "target_only_auc", "delta_auc", "full_rise_logloss", "target_only_logloss", "delta_logloss")]
                  if row and run["eligible"] else ["—"] * 6)
        state = run["state"] + (" / 通过" if run["eligible"] else " / 待完成或待核查")
        lines.append(f"| {DISPLAY[run['dataset']]} | {run['seed']} | {references[key]['job_task_id']} | {run['job_task_id']} | {state} | " + " | ".join(values) + " |")
    lines += ["", "Δ = 修改版 − 完整 RISE；AUC 越大越好，LogLoss 越小越好。逐种子对照审查了相应 Full RISE 的配置、逐 rank 样本覆盖与最终双日志 Test。",
              "seed20262027 的 dataloader_seed=2026、RISE 参数种子=20260821 来自历史源码缺省重建，原始 Params 未显式记录；seed20262028/29 分别为 2027/20260822 与 2028/20260823。",
              "seed20262027 的 QK-Video/MerRec 仅将旧机器数据根目录重定位到当前 datasets，保留数据集和 train/valid/test 后缀。历史代码环境与数据字节完全一致尚未被追溯证明。", "",
              "## 已审计逐标签结果", "",
              "| 数据集 | 种子 | 标签 | 完整 RISE AUC | 修改版 AUC | ΔAUC | 完整 RISE LogLoss | 修改版 LogLoss | ΔLogLoss |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in evidence["comparisons"]:
        if row["label"] == "MACRO":
            continue
        lines.append(f"| {DISPLAY[row['dataset']]} | {row['seed']} | {row['label']} | {row['full_rise_auc']:.8f} | "
                     f"{row['target_only_auc']:.8f} | {row['delta_auc']:+.8f} | {row['full_rise_logloss']:.8f} | "
                     f"{row['target_only_logloss']:.8f} | {row['delta_logloss']:+.8f} |")
    lines += ["", "## 审计与来源", "",
              "results/runs.csv 保留所有任务状态；results/label_metrics.csv、comparisons.csv、summary.csv 保存逐标签、逐种子差值与三种子汇总。",
              "results/evidence.json 记录配置、构造接入点与参数量、各 rank 覆盖、模型文件、进程结果、原始路径和 SHA256；results/snapshots 保存每次采集快照。",
              "notes/reference_audit.json 固定 12 项完整 RISE 参考。seed20262028 通过原有审计函数只读复用，不改旧实验报告或结果。"]
    for run in evidence["runs"]:
        messages = run["issues"] + run["pending_checks"]
        if messages:
            lines += ["", f"{DISPLAY[run['dataset']]} / seed{run['seed']}：", *[f"- {message}" for message in messages]]
    lines.append("")
    return "\n".join(lines)


def write_outputs(output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    run_rows, label_rows = [], []
    for run in evidence["runs"]:
        row = {name: run.get(name) for name in (
            "model", "dataset", "mode", "seed", "world_size", "batch_size", "attempt", "run_id", "job_task_id",
            "raw_job_id", "evidence_origin", "state", "exit_code", "elapsed", "node", "eligible", "macro_auc",
            "macro_logloss", "checkpoint_log", "slurm_log", "protocol_fields_checked")}
        row.update(run.get("parameter_counts", {}), issues="; ".join(run["issues"]), pending_checks="; ".join(run["pending_checks"]))
        run_rows.append(row)
        if run["eligible"]:
            for label, metrics in run["metrics"].items():
                label_rows.append({"model": "HiFormer", "dataset": run["dataset"], "seed": run["seed"],
                                   "job_task_id": run["job_task_id"], "label": label, **metrics})
    files = {"runs.csv": BASE.csv_text(run_rows, ["dataset", "seed", "eligible"]),
             "label_metrics.csv": BASE.csv_text(label_rows, ["dataset", "seed", "label", "AUC", "logloss"]),
             "comparisons.csv": BASE.csv_text(evidence["comparisons"], ["dataset", "seed", "label", "delta_auc", "delta_logloss"]),
             "summary.csv": BASE.csv_text(evidence["summary"], ["dataset", "completed_seeds"]),
             "evidence.json": json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot = output / "snapshots" / stamp
    snapshot.mkdir(parents=True)
    report_text = render_report(evidence)
    (snapshot / "report.md").write_text(report_text)
    for name, content in files.items():
        (snapshot / name).write_text(content)
        temporary = output / (name + f".tmp-{os.getpid()}")
        temporary.write_text(content)
        os.replace(temporary, output / name)
    report = output.parent / "report.md"
    temporary = report.with_name(report.name + f".tmp-{os.getpid()}")
    temporary.write_text(report_text)
    os.replace(temporary, report)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / STUDY / "notes/submission.json")
    parser.add_argument("--output", type=Path, default=ROOT / STUDY / "results")
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text())
    specs = normalize_submission(submission)
    accounts, receipt = accounting(ROOT, specs)
    evidence = collect(ROOT, submission, accounts)
    evidence.update(accounting=receipt, submission=submission,
                    collector_source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in (Path(__file__).resolve(), LEGACY_PATH, LEGACY.BASE_PATH,
                                                          BASE.HELPER_PATH, LEGACY.TARGETED_PATH)})
    snapshot = write_outputs(args.output, evidence)
    print(json.dumps({**{name: evidence[name] for name in ("complete", "all_terminal", "needs_attention")},
                      "eligible": sum(run["eligible"] for run in evidence["runs"]),
                      "states": {run["job_task_id"]: run["state"] for run in evidence["runs"]},
                      "snapshot": str(snapshot)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
