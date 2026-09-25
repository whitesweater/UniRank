#!/usr/bin/env python3
"""Collect all four HiFormer target-only results without launching training.

Only successful, fully audited attempts expose metrics. The historical full
RISE reference is the same dataset's seed20262028 run, never a three-run mean.
The QK-Video result is reused; the submission still contains only three new runs.
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


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments/rise_sample_analysis/scripts/collect_new_experiments.py"
BASE_SPEC = importlib.util.spec_from_file_location("hiformer_extension_audit_helpers", BASE_PATH)
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)
TARGETED_PATH = ROOT / "experiments/rise_sample_analysis/scripts/collect_targeted_controls.py"
TARGETED_SPEC = importlib.util.spec_from_file_location("hiformer_existing_qk_audit_helpers", TARGETED_PATH)
TARGETED = importlib.util.module_from_spec(TARGETED_SPEC)
TARGETED_SPEC.loader.exec_module(TARGETED)
STUDY = Path("experiments/hiformer_target_only_extension")
DATASETS = ("MerRec_Action", "KuaiRand_Video_Action", "TencentGR_10M_Action")
REPORT_DATASETS = ("QK_Video_Action", *DATASETS)
REFERENCE_JOBS = dict(zip(DATASETS, ("548166_2", "548166_6", "548166_14")))
REFERENCE_JOBS["QK_Video_Action"] = "548166_10"
DISPLAY = dict(zip(DATASETS, ("MerRec", "KuaiRand", "TencentGR")))
DISPLAY["QK_Video_Action"] = "QK-Video"
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
RUNTIME_ERRORS = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|torch\.OutOfMemoryError|"
    r"CUDA error:[^\r\n]*|device-side assert|illegal memory access|"
    r"NCCL[^\r\n]*(?:error|timeout|timed out)|DistBackendError|FloatingPointError",
    re.IGNORECASE,
)


def normalize_submission(submission):
    tasks = submission.get("tasks", [])
    if len(tasks) != 3 or {task.get("dataset") for task in tasks} != set(DATASETS):
        raise ValueError("Submission must contain exactly the three approved datasets")
    result = []
    for task in tasks:
        dataset = task["dataset"]
        index = DATASETS.index(dataset)
        attempt = task.get("attempt", 1)
        run_id = f"RISE_hiformer_target_only_{dataset}_seed20262028_attempt{attempt}"
        if task.get("task_index") != index or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("Unexpected task index or attempt")
        expected = {
            "job_task_id": f"{submission['array_id']}_{index}",
            "run_id": run_id,
            "manifest": f"artifacts/rise_hiformer_target_only_extension/{run_id}/frozen_manifest.json",
            "checkpoint_log": f"checkpoints/{dataset}/{run_id}.log",
        }
        for key, value in expected.items():
            if task.get(key) != value:
                raise ValueError(f"Submission identity mismatch: {dataset}/{key}")
        slurm_path = Path(task["slurm_log"])
        if slurm_path.is_absolute() or ".." in slurm_path.parts or slurm_path.parent != Path("logs"):
            raise ValueError("Slurm evidence must use its actual repository-relative logs path")
        result.append({**task, "model": "HiFormer", "mode": "target_only",
                       "kind": "hiformer_target_only_extension",
                       "artifact_dir": str(Path(task["manifest"]).parent)})
    return sorted(result, key=lambda task: task["task_index"])


def parameter_expectations(reference):
    params = reference["params"]
    item_width = sum(
        int(column.get("embedding_dim", params["embedding_dim"]))
        * (len(column["name"]) if isinstance(column["name"], list) else 1)
        for column in params["feature_cols"]
        if column.get("active", True) and column.get("type") != "meta"
        and column.get("source") in {"item", "action"}
    )
    dimension = params["sisa_score_dim"]
    field_parameters = params["num_layers"] * (
        params["num_heads"] * params["token_dim"] * (2 * dimension + 1 + dimension // 2)
        + 2 * params["num_heads"]
    )
    counts = reference["parameter_counts"]
    dense = counts["dense_parameters"] - field_parameters
    embedding = counts["embedding_parameters"]
    return {"active_dense": dense, "active_embedding": embedding,
            "active_total": dense + embedding, "stored_total": dense + embedding,
            "target_hidden_size": item_width, "target_site_parameters": item_width * (2 * dimension + 1 + dimension // 2) + 2,
            "removed_field_parameters": field_parameters}


def expected_model_audit(spec, reference, rank):
    params = reference["params"]
    counts = parameter_expectations(reference)
    sites = [{"site": "attention_layers", "reference": "query", "enabled": True,
              "class_name": "SISAScoreBias", "parameter_seed": params["sisa_parameter_seed"],
              "hidden_size": counts["target_hidden_size"], "heads": 1,
              "score_dim": params["sisa_score_dim"], "stored": counts["target_site_parameters"],
              "active": counts["target_site_parameters"]}]
    sites.extend({"site": f"mixer_layers.{layer}", "reference": "sequence_end", "enabled": False}
                 for layer in range(params["num_layers"]))
    return {"mode": "target_only", "sites": sites, "enabled_site_count": 1,
            "stored_parameters": counts["stored_total"], "active_parameters": counts["active_total"],
            "active_dense_parameters": counts["active_dense"],
            "active_embedding_parameters": counts["active_embedding"],
            "rank": rank, "model_id": spec["run_id"]}


def model_audit_issues(spec, reference, actual, rank):
    expected = expected_model_audit(spec, reference, rank)
    return [f"rank{rank} construction mismatch: {key}"
            for key in sorted(expected.keys() | actual.keys()) if actual.get(key) != expected.get(key)]


def embedded_json(text, marker):
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(text[match.end():].lstrip())[0]
            for match in re.finditer(re.escape(marker) + r"\s*", text)]


def progress_snapshot(text):
    result = {}
    for rank, epoch, step, total in re.findall(
            r"Rank (\d+) \| Epoch (\d+):[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[", text):
        result[rank] = {"epoch": int(epoch), "step": int(step), "total_steps": int(total)}
    return result


def audit_attempt(spec, sources, accounts, reference):
    artifact = Path(spec["artifact_dir"])
    manifest = sources.json(spec["manifest"]) or {}
    started = sources.json(artifact / "started.json") or {}
    outcome = sources.json(artifact / "outcome.json") or {}
    account = BASE.select_account(accounts, spec["job_task_id"], started.get("job_id"))
    log = sources.read(spec["checkpoint_log"]) or ""
    slurm = sources.read(spec["slurm_log"]) or ""
    completed = account["state"] == "COMPLETED" and account["exit_code"] == "0:0"
    record = {**spec, **account, "seed": 20262028, "eligible": False,
              "issues": [], "pending_checks": [], "metrics": {}, "macro_auc": None,
              "macro_logloss": None, "protocol_fields_checked": 0, "model_audits": {},
              "source_hash_checks": [], "outcome": outcome,
              "parameter_counts": BASE.parameter_counts(log),
              "coverage": BASE.HELPER.parse_coverage(slurm, 16384),
              "progress": progress_snapshot(slurm), "gpu_names": started.get("gpu_names", [])}
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
    if len(record["gpu_names"]) != 2 or not all("H100" in name for name in record["gpu_names"]):
        pending.append("Two-H100 allocation receipt unavailable")
    counts = parameter_expectations(reference)
    if not manifest:
        pending.append("Frozen manifest unavailable")
    else:
        identity = {"study": "rise_hiformer_target_only_extension", "run_id": spec["run_id"],
                    "task_id": spec["task_index"], "attempt": spec["attempt"],
                    "model": "HiFormer", "dataset": spec["dataset"], "mode": "target_only",
                    "reference_job_task": reference["job_task_id"],
                    "reference_log": reference["checkpoint_log"],
                    "allowed_parameter_differences": ["model_id"],
                    "process_local_treatment": "target_only"}
        for key, value in identity.items():
            if manifest.get(key) != value:
                issues.append(f"Frozen manifest mismatch: {key}")
        if manifest.get("reference_params") != reference["params"]:
            issues.append("Frozen reference parameters differ from full RISE")
        expected_hash = sources.records.get(reference["checkpoint_log"], {}).get("sha256")
        if manifest.get("reference_log_sha256") != expected_hash:
            issues.append("Frozen reference log SHA mismatch")
        for key in ("active_dense", "active_embedding", "active_total", "stored_total"):
            if manifest.get("parameter_expectations", {}).get(key) != counts[key]:
                issues.append(f"Frozen parameter expectation mismatch: {key}")
        required_sources = {
            "scripts/run_hiformer_target_only_extension.py",
            "scripts/submit_hiformer_target_only_extension.sbatch",
            "experiments/rise_sample_analysis/scripts/run_targeted_controls.py",
            "experiments/rise_sample_analysis/scripts/targeted_controls.py",
            "model_zoo/HiFormer.py", "unirank/pytorch/layers/attentions/sisa.py",
            "run_expid.py", "pyproject.toml", "uv.lock",
        }
        if not required_sources.issubset(manifest.get("source_sha256", {})):
            issues.append("Frozen manifest omits a required execution source")
        source_issues, record["source_hash_checks"] = BASE.source_manifest_check(manifest, sources)
        issues.extend(source_issues)
        for name, text in (("checkpoint", log), ("Slurm", slurm)):
            if "INFO Params:" not in text:
                pending.append(f"{name} runtime parameters unavailable")
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
        audit = sources.json(artifact / f"model_audit_rank{rank}.json")
        if audit is None:
            pending.append(f"rank{rank} construction artifact unavailable")
        else:
            record["model_audits"][str(rank)] = audit
            issues.extend(model_audit_issues(spec, reference, audit, rank))
    try:
        workers = embedded_json(slurm, "RISE_TARGETED_WORKER")
        expected_workers = [{"task_id": spec["task_index"], "mode": "target_only", "rank": rank,
                             "run_id": spec["run_id"]} for rank in (0, 1)]
        if sorted(workers, key=lambda row: row["rank"]) != expected_workers:
            pending.append("Exact two worker identity markers unavailable")
        audits = embedded_json(slurm, "RISE_TARGETED_MODEL_AUDIT")
        if len(audits) != 2:
            pending.append("Exact two embedded construction audits unavailable")
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
    record["runtime_error_signals"] = RUNTIME_ERRORS.findall(slurm + "\n" + log)
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
            preserved = re.findall(r"Preserved best model checkpoint: (.+)", log)
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


def existing_qk_spec(root):
    candidates = [spec for spec in TARGETED.discover(root, array="644098")
                  if spec["model"] == "HiFormer" and spec["dataset"] == "QK_Video_Action"
                  and spec["mode"] == "target_only" and spec["attempt"] == 1]
    if len(candidates) != 1:
        raise ValueError("Expected exactly the existing QK-Video target-only attempt 644098_1")
    return candidates[0]


def native_baseline_comparisons(sources, runs):
    """Join optional audited native references without mixing their seed cohorts."""
    baseline_csv = sources.read(STUDY / "results/native_baselines.csv")
    if baseline_csv is None:
        return []
    sources.read(STUDY / "results/native_baseline_evidence.json")
    eligible = {run["dataset"]: run for run in runs if run["eligible"]}
    comparisons, seen = [], set()
    for row in csv.DictReader(io.StringIO(baseline_csv)):
        kind, dataset = row["kind"], row["dataset"]
        seed = int(row["baseline_seed"])
        expected_seed = {"paper_seed27": 20262027, "matched_seed28": 20262028}.get(kind)
        if seed != expected_seed or dataset not in REPORT_DATASETS:
            raise ValueError("Unexpected native baseline cohort, seed, or dataset")
        if (kind, dataset) in seen:
            raise ValueError("Duplicate native baseline within a seed cohort")
        seen.add((kind, dataset))
        run = eligible.get(dataset)
        if run is None:
            continue
        baseline_auc, baseline_loss = float(row["baseline_auc"]), float(row["baseline_logloss"])
        comparisons.append({
            "kind": kind, "model": "HiFormer", "dataset": dataset,
            "baseline_seed": seed, "target_only_seed": 20262028,
            "baseline_job": row["baseline_job"], "target_only_job": run["job_task_id"],
            "baseline_auc": baseline_auc, "target_only_auc": run["macro_auc"],
            "delta_auc": run["macro_auc"] - baseline_auc,
            "baseline_logloss": baseline_loss, "target_only_logloss": run["macro_logloss"],
            "delta_logloss": run["macro_logloss"] - baseline_loss,
        })
    return comparisons


def collect(root, submission, accounts):
    specs = [existing_qk_spec(root), *normalize_submission(submission)]
    sources = BASE.Sources(root)
    references, runs, comparisons = [], [], []
    frozen_manifest = sources.read("experiments/sisa_paper_report/archive/results/source_manifest.csv")
    archived_hashes = {row["path"]: row["sha256"]
                       for row in csv.DictReader(io.StringIO(frozen_manifest or ""))}
    for spec in specs:
        reference = BASE.reference_audit("HiFormer", spec["dataset"], sources)
        if reference["job_task_id"] != REFERENCE_JOBS[spec["dataset"]]:
            reference["issues"].append("Historical task differs from selected seed20262028 reference")
        for key in ("checkpoint_log", "slurm_log"):
            path = reference[key]
            if sources.records[path]["sha256"] != archived_hashes.get(path):
                reference["issues"].append(f"Historical source hash differs from frozen archive: {path}")
        reference["eligible"] = not reference["issues"]
        references.append(reference)
        if spec["dataset"] == "QK_Video_Action":
            run = TARGETED.audit_attempt(spec, sources, accounts, reference)
            run["evidence_origin"] = "existing_qk_control"
            if run["job_task_id"] != "644098_1":
                run["issues"].append("Existing QK-Video task differs from selected 644098_1")
                run.update(eligible=False, metrics={}, macro_auc=None, macro_logloss=None)
        else:
            run = audit_attempt(spec, sources, accounts, reference)
            run["evidence_origin"] = "new_extension"
        runs.append(run)
        if not run["eligible"]:
            continue
        for label in [*BASE.LABELS[spec["dataset"]], "MACRO"]:
            before = reference["metrics"].get(label)
            after = run["metrics"].get(label)
            if label == "MACRO":
                before = {metric: statistics.fmean(value[metric] for value in reference["metrics"].values())
                          for metric in ("AUC", "logloss")}
                after = {"AUC": run["macro_auc"], "logloss": run["macro_logloss"]}
            comparisons.append({"model": "HiFormer", "dataset": spec["dataset"], "label": label,
                                "seed": 20262028, "full_rise_job": reference["job_task_id"],
                                "target_only_job": run["job_task_id"],
                                "full_rise_auc": before["AUC"], "target_only_auc": after["AUC"],
                                "delta_auc": after["AUC"] - before["AUC"],
                                "full_rise_logloss": before["logloss"], "target_only_logloss": after["logloss"],
                                "delta_logloss": after["logloss"] - before["logloss"]})
    native_comparisons = native_baseline_comparisons(sources, runs)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "complete": len(runs) == 4 and all(run["eligible"] for run in runs),
            "all_terminal": bool(runs) and all(run["state"].split("+")[0] in TERMINAL_STATES for run in runs),
            "scope": "Exactly four target_only HiFormer dataset results: the existing QK-Video 644098_1 and three newly submitted extension runs; same-dataset full RISE seed20262028 references. No QK rerun or new native baseline.",
            "submitted_extension_count": 3, "reported_dataset_count": 4,
            "runs": runs, "historical_references": references, "comparisons": comparisons,
            "native_baseline_comparisons": native_comparisons,
            "sources": list(sources.records.values()),
            "limitations": [
                "Single-seed exploratory placement comparisons; no training-seed significance claim.",
                "Target-only removes field parameters and therefore is not a capacity-matched intervention.",
                "Recorded configuration and per-rank sample coverage are matched; historical code/environment and dataset-byte identity are not retrospectively proved.",
                "The main report's MerRec seed20262029 retry does not replace the seed20262028 reference used here.",
                "Historical native baseline comparisons are reported separately from the full-RISE paired differences; no three-run means are used.",
            ]}


def accounting(root, submission, specs):
    identifiers = {str(submission["array_id"]), "644098_1"}
    for spec in [existing_qk_spec(root), *specs]:
        path = Path(root) / spec["artifact_dir"] / "started.json"
        if path.exists():
            job = json.loads(path.read_text()).get("job_id")
            if job:
                identifiers.add(str(job))
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(sorted(identifiers)),
               "--format=JobID%64,JobIDRaw%64,State%32,ExitCode,Elapsed,NodeList%64"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    receipt = {"command": command, "returncode": result.returncode, "stdout": result.stdout,
               "stderr": result.stderr, "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}
    return BASE.parse_accounting(result.stdout) if result.returncode == 0 else [], receipt


def render_report(evidence):
    successful = sum(run["eligible"] for run in evidence["runs"])
    summary = f"已完成并通过审计：{successful}/4；全部通过审计后才能形成四数据集结论。"
    if evidence["complete"]:
        macro_rows = [row for row in evidence["comparisons"] if row["label"] == "MACRO"]
        improved_auc = sum(row["delta_auc"] > 0 for row in macro_rows)
        improved_loss = sum(row["delta_logloss"] < 0 for row in macro_rows)
        summary = (f"4/4 项实验均已完成最终测试并通过审计。相对同种子完整 RISE，target_only 在 "
                   f"{improved_auc}/4 个数据集上提高 Macro-AUC，在 {improved_loss}/4 个数据集上降低 Macro-LogLoss。")
    lines = ["# HiFormer 仅历史汇聚接入 RISE：四个数据集完整结果", "",
             summary, "",
             "比较对象为同一数据集 seed20262028 的完整 RISE 与 target_only（仅历史 target pooling）训练。",
             "报告共四项：QK-Video 沿用已有任务 644098_1，并重新审计原始证据；本轮新增三个训练任务，分别为 MerRec、KuaiRand、TencentGR。",
             "本轮没有重复训练 QK-Video，也没有新增原生 Baseline。", "",
             "| 数据集 | 完整 RISE 任务 | target_only 任务 | 调度状态 | 审计状态 | 完整 RISE AUC | target_only AUC | ΔAUC | 完整 RISE LogLoss | target_only LogLoss | ΔLogLoss |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    macros = {row["dataset"]: row for row in evidence["comparisons"] if row["label"] == "MACRO"}
    ordered_runs = sorted(evidence["runs"], key=lambda run: REPORT_DATASETS.index(run["dataset"]))
    for run in ordered_runs:
        if run["eligible"]:
            row = macros[run["dataset"]]
            values = [f"{row[key]:.8f}" for key in ("full_rise_auc", "target_only_auc")]
            values += [f"{row['delta_auc']:+.8f}"]
            values += [f"{row[key]:.8f}" for key in ("full_rise_logloss", "target_only_logloss")]
            values += [f"{row['delta_logloss']:+.8f}"]
        else:
            values = ["—"] * 6
        lines.append(f"| {DISPLAY[run['dataset']]} | {REFERENCE_JOBS[run['dataset']]} | {run['job_task_id']} | {run['state']} | "
                     f"{'通过' if run['eligible'] else '待完成/待核查'} | " + " | ".join(values) + " |")
    lines += ["", "Δ = target_only − 完整 RISE；AUC 越大越好，LogLoss 越小越好。",
              "所有值均为各标签等权宏平均，不能与三次运行均值混用。", "",
              "固定协议：2 张 H100、单卡 batch=16384、全局 batch=32768、1 epoch、序列长度100、",
              "seed=20262028、dataloader_seed=2027、RISE 参数种子20260822、BF16 和 torch.compile 开启。",
              "只保留1个 target 位置的 RISE，3个字段交互位置均关闭；训练参数与完整 RISE 仅 model_id 不同。", "",
              "这是单种子、参数量不同的位置对照，不据此声明训练种子统计显著性。历史代码版本与数据字节完全一致未被追溯证明。",
              "MerRec 主表更新的是 seed20262029，本实验对照仍为未替换的 seed20262028（548166_2）。"]
    if evidence["comparisons"]:
        lines += ["", "## 逐标签结果", "",
                  "| 数据集 | 标签 | 完整 RISE AUC | target_only AUC | ΔAUC | 完整 RISE LogLoss | target_only LogLoss | ΔLogLoss |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(evidence["comparisons"], key=lambda row: REPORT_DATASETS.index(row["dataset"])):
            if row["label"] == "MACRO":
                continue
            lines.append(f"| {DISPLAY[row['dataset']]} | {row['label']} | {row['full_rise_auc']:.8f} | "
                         f"{row['target_only_auc']:.8f} | {row['delta_auc']:+.8f} | {row['full_rise_logloss']:.8f} | "
                         f"{row['target_only_logloss']:.8f} | {row['delta_logloss']:+.8f} |")
    for run in evidence["runs"]:
        if run["issues"] or run["pending_checks"]:
            lines += ["", f"{DISPLAY[run['dataset']]}（{run['job_task_id']}）：",
                      *[f"- {message}" for message in run["issues"] + run["pending_checks"]]]
    native_comparisons = evidence.get("native_baseline_comparisons", [])
    for kind, title, caveat in (
            ("paper_seed27", "相对主表原生 Baseline（seed20262027）",
             "Baseline 为主表原生 HiFormer seed20262027（4卡、单卡 batch=8192），修改版为 target_only seed20262028（2卡、单卡 batch=16384）；全局 batch 均为32768。训练种子、world size 和样本覆盖不同，只能作描述性比较，不能视为严格配对实验。"),
            ("matched_seed28", "相对同种子原生 Baseline（seed20262028）",
             "仅列已有同种子原生对照：Baseline 与修改版均为 seed20262028，训练预算及逐 rank 样本覆盖匹配；这是单次运行对照，不代表统计显著性。")):
        rows = [row for row in native_comparisons if row["kind"] == kind]
        if not rows:
            continue
        lines += ["", f"## {title}", "", caveat, "",
                  "| 数据集 | Baseline 任务 | target_only 任务 | Baseline AUC | target_only AUC | ΔAUC | Baseline LogLoss | target_only LogLoss | ΔLogLoss |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(rows, key=lambda row: REPORT_DATASETS.index(row["dataset"])):
            lines.append(f"| {DISPLAY[row['dataset']]} | {row['baseline_job']} | {row['target_only_job']} | "
                         f"{row['baseline_auc']:.8f} | {row['target_only_auc']:.8f} | {row['delta_auc']:+.8f} | "
                         f"{row['baseline_logloss']:.8f} | {row['target_only_logloss']:.8f} | {row['delta_logloss']:+.8f} |")
        lines += ["", "Δ = target_only − 原生 Baseline；各标签等权宏平均，AUC 越大越好，LogLoss 越小越好。",
                  "原生参考：results/native_baselines.csv；双日志核查：results/native_baseline_evidence.json；来源指纹及动态差值：results/evidence.json。"]
    lines += ["", "审计材料：results/evidence.json；逐标签及对照：results/label_metrics.csv、results/comparisons.csv。", ""]
    return "\n".join(lines)


def write_outputs(output, evidence):
    output.mkdir(parents=True, exist_ok=True)
    run_rows, label_rows = [], []
    for run in evidence["runs"]:
        row = {key: run.get(key) for key in (
            "model", "dataset", "mode", "attempt", "run_id", "job_task_id", "raw_job_id", "seed", "evidence_origin",
            "state", "exit_code", "elapsed", "node", "eligible", "macro_auc", "macro_logloss",
            "checkpoint_log", "slurm_log", "protocol_fields_checked")}
        row.update(run["parameter_counts"], issues="; ".join(run["issues"]),
                   pending_checks="; ".join(run["pending_checks"]))
        run_rows.append(row)
        for label, metrics in run["metrics"].items():
            label_rows.append({"model": "HiFormer", "dataset": run["dataset"], "setting": "target_only",
                               "seed": 20262028, "job_task_id": run["job_task_id"], "label": label, **metrics})
    files = {
        "runs.csv": BASE.csv_text(run_rows, ["dataset", "job_task_id", "state", "eligible"]),
        "label_metrics.csv": BASE.csv_text(label_rows, ["model", "dataset", "setting", "seed", "job_task_id", "label", "AUC", "logloss"]),
        "comparisons.csv": BASE.csv_text(evidence["comparisons"], ["model", "dataset", "label", "delta_auc", "delta_logloss"]),
        "evidence.json": json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot = output / "snapshots" / stamp
    snapshot.mkdir(parents=True)
    for name, content in files.items():
        (snapshot / name).write_text(content)
        temporary = output / (name + f".tmp-{os.getpid()}")
        temporary.write_text(content)
        os.replace(temporary, output / name)
    report = output.parent / "report.md"
    temporary = report.with_name(report.name + f".tmp-{os.getpid()}")
    temporary.write_text(render_report(evidence))
    os.replace(temporary, report)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / STUDY / "notes/submission.json")
    parser.add_argument("--output", type=Path, default=ROOT / STUDY / "results")
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text())
    specs = normalize_submission(submission)
    accounts, receipt = accounting(ROOT, submission, specs)
    evidence = collect(ROOT, submission, accounts)
    evidence["accounting"] = receipt
    evidence["submission"] = submission
    evidence["collector_source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), BASE_PATH, BASE.HELPER_PATH, TARGETED_PATH)
    }
    snapshot = write_outputs(args.output, evidence)
    print(json.dumps({"complete": evidence["complete"], "all_terminal": evidence["all_terminal"],
                      "eligible": sum(run["eligible"] for run in evidence["runs"]),
                      "states": {run["job_task_id"]: run["state"] for run in evidence["runs"]},
                      "snapshot": str(snapshot)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
