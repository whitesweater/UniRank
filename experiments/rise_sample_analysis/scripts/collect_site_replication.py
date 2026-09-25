"""Audit six cross-model placement runs and preserve all source receipts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

import collect_new_experiments as BASE
import collect_targeted_controls as TARGET

ROOT = BASE.ROOT
DATASET = "QK_Video_Action"
TASKS = (("Zenith", "baseline", 11164164, 419988768),
         ("HyFormer", "baseline", 7505412, 140001120),
         ("Zenith", "history_only", 11167446, 419988768),
         ("HyFormer", "sequence_only", 7568400, 140001120),
         ("Zenith", "token_only", 11227152, 419988768),
         ("HyFormer", "query_only", 7568400, 140001120))
REFERENCES = {"history_only": "query", "token_only": "sequence_end",
              "sequence_only": "sequence_end", "query_only": "query"}


def sites(model):
    if model == "Zenith":
        return [("attention_layers", "query", 0, 80, 1)] + [
            (f"unified_interaction_layers.mixer_layers.{i}", "sequence_end", 100+i, 256, 2) for i in range(3)]
    return [(f"unified_interaction_layers.seq_layers.{i}", "sequence_end", i, 256, 2) for i in range(3)] + [
        (f"unified_interaction_layers.decode_layers.{i}", "query", 100+i, 256, 2) for i in range(3)]


def discover():
    result = []
    for index, (model, mode, dense, embedding) in enumerate(TASKS):
        prefix = f"RISE_cross_sites_{model}_{DATASET}_{mode}_seed20262028_attempt"
        for path in sorted((ROOT / "artifacts/rise_site_replication").glob(prefix + "*")):
            match = re.fullmatch(re.escape(prefix) + r"(\d+)", path.name)
            if not match:
                continue
            result.append(dict(model=model, dataset=DATASET, mode=mode, dense=dense, embedding=embedding,
                               task_index=index, attempt=int(match[1]), run_id=path.name,
                               artifact_dir=str(path.relative_to(ROOT)), logical_task=str(index),
                               kind="native_baseline" if mode == "baseline" else "site_control",
                               checkpoint_log=f"checkpoints/{DATASET}/{path.name}.log"))
    return result


def expected_audit(spec, rank):
    entries = []
    for path, reference, offset, width, heads in sites(spec["model"]):
        enabled = spec["mode"] != "baseline" and reference == REFERENCES[spec["mode"]]
        row = dict(path=path, reference=reference, enabled=enabled)
        if enabled:
            row.update(seed=20260822+offset, hidden_size=width, heads=heads, score_dim=16)
        entries.append(row)
    return dict(model=spec["model"], mode=spec["mode"], sites=entries,
                active_sites=sum(r["enabled"] for r in entries), dense=spec["dense"],
                rank=rank, run_id=spec["run_id"])


def audit(spec, sources, accounts, reference):
    artifact = Path(spec["artifact_dir"])
    manifest = sources.json(artifact / "frozen_manifest.json") or {}
    started = sources.json(artifact / "started.json") or {}
    outcome = sources.json(artifact / "outcome.json") or {}
    expected_job = f"{started.get('array_job_id', '644632')}_{started.get('array_task_id', spec['task_index'])}"
    account = BASE.select_account(accounts, expected_job, started.get("job_id"))
    slurm_path = f"logs/unirank-rise-site-replication-{expected_job}.out"
    log = sources.read(spec["checkpoint_log"]) or ""
    slurm = sources.read(slurm_path) or ""
    row = dict(spec, **account, issues=[], eligible=False, selected=False, metrics={},
               macro_auc=None, macro_logloss=None, slurm_log=slurm_path,
               parameter_counts=BASE.parameter_counts(log),
               coverage=BASE.HELPER.parse_coverage(slurm, 16384),
               checkpoint=BASE.inspect_checkpoint((sources.root / spec["checkpoint_log"]).with_suffix(".model")),
               gpu_names=started.get("gpu_names", []))
    issues = row["issues"]
    if account["state"] != "COMPLETED" or account["exit_code"] != "0:0":
        issues.append("Scheduler not completed successfully")
    if outcome.get("returncode") != 0 or str(outcome.get("job_id")) != str(started.get("job_id")):
        issues.append("Process outcome missing or mismatched")
    if str(started.get("job_id")) != str(account["raw_job_id"]):
        issues.append("Allocation identity mismatch")
    if len(row["gpu_names"]) != 2 or any("H100" not in n for n in row["gpu_names"]):
        issues.append("Expected two H100 GPUs")
    identity = {k: spec[k] for k in ("attempt", "model", "dataset", "mode", "run_id")}
    identity.update(task=spec["task_index"], reference_params=reference["params"],
                    reference_log=reference["checkpoint_log"],
                    reference_log_sha256=sources.records[reference["checkpoint_log"]]["sha256"],
                    allowed_logged_differences=["model_id", "sisa_enabled"] if spec["mode"] == "baseline" else ["model_id"],
                    expected_counts=dict(dense=spec["dense"], embedding=spec["embedding"], total=spec["dense"]+spec["embedding"]),
                    expected_sites=[list(s) for s in sites(spec["model"])])
    issues.extend(f"Manifest mismatch: {key}" for key, value in identity.items() if manifest.get(key) != value)
    source_issues, row["source_hash_checks"] = BASE.source_manifest_check(manifest, sources)
    issues.extend(source_issues)
    for name, contents in (("checkpoint", log), ("Slurm", slurm)):
        try:
            actual = BASE.parsed_params(contents)
            issues.extend(f"{name}: {issue}" for issue in BASE.protocol_issues(spec, actual, reference["params"], manifest.get("effective_params", {})))
            row["protocol_fields_checked"] = len(actual)
        except (ValueError, TypeError) as error:
            issues.append(f"{name} Params parse failure: {error}")
    for rank in (0, 1):
        if sources.json(artifact / f"model_audit_rank{rank}.json") != expected_audit(spec, rank):
            issues.append(f"Rank {rank} site audit mismatch")
    try:
        observed = TARGET.embedded_json(slurm, "RISE_SITE_REPLICATION_AUDIT")
        if sorted(observed, key=lambda item: item["rank"]) != [expected_audit(spec, r) for r in (0, 1)]:
            issues.append("Embedded two-rank site audit mismatch")
    except (ValueError, KeyError) as error:
        issues.append(f"Embedded site audit parse failure: {error}")
    counts = dict(total_parameters=spec["dense"]+spec["embedding"], dense_parameters=spec["dense"], embedding_parameters=spec["embedding"])
    if row["parameter_counts"] != counts or BASE.parameter_counts(slurm) != counts:
        issues.append("Active parameter counts mismatch")
    issues.extend(BASE.HELPER.coverage_issues(row["coverage"], reference["coverage"], partial=False))
    if not row["checkpoint"]["exists"] or not row["checkpoint"]["zip_metadata_readable"]:
        issues.append("Readable model checkpoint absent")
    row["runtime_errors"] = TARGET.RUNTIME_ERRORS.findall(log + "\n" + slurm)
    if row["runtime_errors"]:
        issues.append("Runtime error signals found")
    if slurm.splitlines().count("RISE_SITE_REPLICATION_COMPLETE " + spec["run_id"]) != 1:
        issues.append("Unique completion marker absent")
    metrics = None
    try:
        metrics = BASE.HELPER.parse_test_metrics(log, BASE.LABELS[DATASET])
        if metrics != BASE.HELPER.parse_test_metrics(slurm, BASE.LABELS[DATASET]):
            issues.append("Final Test metrics disagree")
        saved = re.findall(r"Preserved best model checkpoint: (.+)", log)
        if len(saved) != 1 or Path(saved[0]).name != spec["run_id"] + ".model":
            issues.append("Best checkpoint receipt mismatch")
    except (ValueError, TypeError) as error:
        issues.append(f"Final Test parse failure: {error}")
    if not reference["eligible"]:
        issues.append("Historical E audit failed")
    if not issues and metrics:
        row.update(eligible=True, metrics=metrics,
                   macro_auc=statistics.mean(v["AUC"] for v in metrics.values()),
                   macro_logloss=statistics.mean(v["logloss"] for v in metrics.values()))
    return row


def collect():
    specs = discover()
    accounts, receipt = TARGET.accounting(ROOT, specs, "644632", "644632")
    sources = BASE.Sources(ROOT)
    refs = {model: BASE.reference_audit(model, DATASET, sources) for model in ("Zenith", "HyFormer")}
    runs = [audit(s, sources, accounts, refs[s["model"]]) for s in specs]
    TARGET.select_attempts(runs)
    comparisons, labels = [], []
    for run in runs:
        for label, values in run["metrics"].items():
            labels.append({k: run[k] for k in ("model", "dataset", "mode", "job_task_id", "attempt", "selected")} | dict(label=label, **values))
        if not run["selected"] or run["mode"] == "baseline":
            continue
        native = [r for r in runs if r["selected"] and r["model"] == run["model"] and r["mode"] == "baseline"]
        if len(native) != 1:
            continue
        for label in [*BASE.LABELS[DATASET], "__macro__"]:
            row = {k: run[k] for k in ("model", "dataset", "mode", "job_task_id")}
            row.update(label=label, baseline_job=native[0]["job_task_id"], full_RISE_job=refs[run["model"]]["job_task_id"])
            for name, metrics in (("Baseline", native[0]["metrics"]), ("Full_RISE", refs[run["model"]]["metrics"]), ("Placement", run["metrics"])):
                for metric in ("AUC", "logloss"):
                    row[f"{name}_{metric}"] = metrics[label][metric] if label != "__macro__" else statistics.mean(v[metric] for v in metrics.values())
            row.update(delta_AUC_vs_Baseline=row["Placement_AUC"]-row["Baseline_AUC"], delta_AUC_vs_Full_RISE=row["Placement_AUC"]-row["Full_RISE_AUC"])
            comparisons.append(row)
    evidence = dict(generated_utc=datetime.now(timezone.utc).isoformat(), runs=runs, historical_E_audits=list(refs.values()),
                    accounting=receipt, sources=list(sources.records.values()), comparisons=comparisons,
                    scope="Exploratory QK placement replication, fixed seed20262028; no seed20262027 substitution.",
                    limitations=["Single training seed; no significance claim for placement differences.",
                                 "Active capacity differs by placement; retained site identity and initialization are checked.",
                                 "Historical E source revision is not verified; full logged protocol and per-rank coverage are checked."])
    compact = [{k: r[k] for k in ("model", "dataset", "mode", "task_index", "attempt", "run_id", "job_task_id", "raw_job_id", "state", "exit_code", "elapsed", "node", "eligible", "selected", "macro_auc", "macro_logloss")} |
               r["parameter_counts"] | dict(issues="; ".join(r["issues"])) for r in runs]
    return evidence, compact, labels


def main():
    evidence, compact, labels = collect()
    output = ROOT / "experiments/rise_sample_analysis/results"
    snapshot = output / "site_replication_snapshots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    snapshot.mkdir(parents=True)
    files = {"site_replication_evidence.json": json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
             "site_replication_runs.csv": BASE.csv_text(compact, ["model", "mode", "eligible"]),
             "site_replication_label_metrics.csv": BASE.csv_text(labels, ["model", "mode", "label", "AUC", "logloss"]),
             "site_replication_comparisons.csv": BASE.csv_text(evidence["comparisons"], ["model", "mode", "label", "Baseline_AUC", "Full_RISE_AUC", "Placement_AUC"]),
             "site_replication_sources.csv": BASE.csv_text(evidence["sources"], ["path", "bytes", "sha256"])}
    for name, contents in files.items():
        (snapshot / name).write_text(contents)
        temporary = output / (name+f".tmp-{os.getpid()}")
        temporary.write_text(contents); os.replace(temporary, output/name)
    print(json.dumps({"runs": len(compact), "eligible": sum(r["eligible"] for r in compact), "snapshot": str(snapshot)}))
    for row in compact:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
