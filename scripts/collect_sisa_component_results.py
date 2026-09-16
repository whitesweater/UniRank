#!/usr/bin/env python3
"""Collect audited final Test evidence for the three frozen SISA component studies.

No training is launched. Incomplete or inconsistent runs are retained as status
records without metrics. Comparisons require eligible B, C, D and historical E.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
STUDIES = (
    ("sisa_components_bcd", "RankMixer", "KuaiRand_Video_Action", "626644", "7", "components"),
    ("sisa_components_hyformer_merrec_bcd", "HyFormer", "MerRec_Action", "626760", "1", "hyformer-merrec"),
    ("sisa_components_hiformer_taac_bcd", "HiFormer", "TencentGR_10M_Action", "626779", "14", "hiformer-taac"),
)
VARIANTS = {"B": "content_only", "C": "decay_only", "D": "rotation_only", "E": "full"}
DIFFERENCES = {
    "C-B": {"C": 1, "B": -1}, "D-B": {"D": 1, "B": -1},
    "E-B": {"E": 1, "B": -1}, "E-C": {"E": 1, "C": -1},
    "E-D": {"E": 1, "D": -1}, "E-C-D+B": {"E": 1, "C": -1, "D": -1, "B": 1},
}


def parsed_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_params(text):
    matches = list(re.finditer(r"INFO Params:\s*", text))
    if len(matches) != 1:
        raise ValueError(f"Expected one Params session, got {len(matches)}")
    params, _ = json.JSONDecoder().raw_decode(text[matches[0].end():])
    if parsed_value(params.get("rank")) != 0 or parsed_value(params.get("local_rank", 0)) != 0:
        raise ValueError("Params session is not rank zero")
    return params


def protocol_issues(actual, reference, run_id, variant):
    expected = dict(reference, model_id=run_id)
    if variant != "full" or "sisa_variant" in actual:
        expected["sisa_variant"] = variant
    return [f"Frozen protocol mismatch: {key}" for key in sorted(actual.keys() | expected.keys())
            if key not in actual or key not in expected
            or parsed_value(actual[key]) != parsed_value(expected[key])]


def parse_test_metrics(text, labels):
    sections = re.split(r"\*+ Test evaluation \*+", text)
    if len(sections) != 2:
        raise ValueError(f"Expected exactly one Test evaluation section, got {len(sections) - 1}")
    records = re.findall(r"\[Task:\s*([^\]]+)\]\[Metrics\]\s*([^\r\n]+)", sections[1])
    metrics = {}
    for label, values in records:
        if label in metrics:
            raise ValueError(f"Duplicate Test label: {label}")
        match = re.fullmatch(r"logloss:\s*(\S+)\s+-\s+AUC:\s*(\S+)\s*", values)
        if not match:
            raise ValueError(f"Malformed Test metrics: {label}")
        loss, auc = map(float, match.groups())
        if not (math.isfinite(loss) and loss >= 0 and math.isfinite(auc) and 0 <= auc <= 1):
            raise ValueError(f"Nonfinite or out-of-range Test metrics: {label}")
        metrics[label] = {"AUC": auc, "logloss": loss}
    if set(metrics) != set(labels) or len(metrics) != len(labels):
        raise ValueError(f"Test label coverage mismatch: expected {labels}, got {list(metrics)}")
    return metrics


def completion_issues(account, outcome, text, run_id):
    issues = []
    if account.get("state") != "COMPLETED" or account.get("exit_code") != "0:0":
        issues.append(f"sacct not successful: {account.get('state', 'UNKNOWN')}/{account.get('exit_code', 'unknown')}")
    if outcome.get("returncode") != 0:
        issues.append("Missing or unsuccessful artifact outcome")
    marker = f"SISA_COMPONENT_COMPLETE run_id={run_id}"
    if text.splitlines().count(marker) != 1:
        issues.append("Missing or duplicate exact component completion marker")
    return issues


def parse_coverage(text, batch_size):
    by_rank = {}
    for match in re.finditer(
        r"\[Rank (\d+)\] blocked load balanced: blocks=(\d+), rows=(\d+), batches=(\d+)", text
    ):
        rank, blocks, rows, batches = map(int, match.groups())
        entries = by_rank.setdefault(rank, [])
        index = len(entries)
        entries.append({"split": ("train", "valid", "test")[index] if index < 3 else f"unexpected_{index}",
                        "rank": rank, "blocks": blocks, "assigned_rows": rows, "batches": batches,
                        "processed_rows": batches * batch_size})
    return [entry for rank in sorted(by_rank) for entry in by_rank[rank]]


def coverage_issues(actual, expected, partial=False):
    keys = ("blocks", "assigned_rows", "batches", "processed_rows")
    expected_map = {(entry["split"], entry["rank"]): entry for entry in expected}
    actual_map = {(entry["split"], entry["rank"]): entry for entry in actual}
    issues = []
    if not partial and actual_map.keys() != expected_map.keys():
        issues.append("Coverage split/rank set differs from frozen reference")
    for key, value in actual_map.items():
        if key not in expected_map or any(value[field] != expected_map[key][field] for field in keys):
            issues.append(f"Coverage mismatch: split={key[0]} rank={key[1]}")
    return issues


class Sources:
    def __init__(self, root):
        self.root = root
        self.records = {}

    def read(self, relative):
        path = self.root / relative
        if not path.exists():
            return None
        # Hash the exact byte snapshot used for parsing, including running logs.
        digest = hashlib.sha256()
        chunks = []
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                chunks.append(chunk)
        data = b"".join(chunks)
        self.records[str(relative)] = {"path": str(relative), "sha256": digest.hexdigest(), "bytes": len(data)}
        return data.decode("utf-8")

    def json(self, relative):
        text = self.read(relative)
        return json.loads(text) if text is not None else None

    def csv(self, relative):
        text = self.read(relative)
        return list(csv.DictReader(text.splitlines())) if text is not None else []


def accounting():
    command = ["sacct", "-n", "-P", "-j", "626644,626760,626779",
               "--format=JobID,JobIDRaw,State,ExitCode"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 4 or "." in fields[0]:
            continue
        records[fields[0]] = dict(raw_job_id=fields[1], state=fields[2], exit_code=fields[3])
    return records, {"command": command, "stdout": result.stdout, "stderr": result.stderr,
                     "sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def parameter_counts(text):
    values = {}
    for key, title in (("total_parameters", "Total number of parameters"),
                       ("dense_parameters", "Number of dense parameters")):
        matches = re.findall(re.escape(title) + r": (\d+)\.", text)
        values[key] = int(matches[0]) if len(matches) == 1 else None
    return values


def manifest_issues(manifest, study, model, dataset, setting, task_id, run_id, array, account):
    expected = dict(experiment=f"{model}_{dataset}", setting=setting, variant=VARIANTS[setting],
                    task_id=task_id, run_id=run_id, seed=20262028, world_size=2,
                    per_gpu_batch=16384, global_batch=32768, dataloader_seed=2027,
                    sisa_parameter_seed=20260822, array_job_id=array,
                    job_id=account.get("raw_job_id"), attempt=1)
    issues = [f"Manifest mismatch: {key}" for key, value in expected.items()
              if manifest.get(key) != value]
    # The first launcher predates the optional explicit study field.
    if "study" in manifest and manifest["study"] != study:
        issues.append("Manifest mismatch: study")
    if manifest.get("gpu_names") != ["NVIDIA H100 80GB HBM3"] * 2:
        issues.append("Manifest GPU names differ from H100 reference")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_revision", "")):
        issues.append("Manifest has no valid source Git revision")
    return issues


def collect(root, accounts):
    sources = Sources(root)
    archival_runs = sources.csv(Path("experiments/sisa_paper_report/archive/results/run_evidence.csv"))
    archival_labels = sources.csv(Path("experiments/sisa_paper_report/archive/results/label_results.csv"))
    runs = []
    for study, model, dataset, array, reference_task, log_prefix in STUDIES:
        config_dir = Path("experiments/ablations") / study / "configs"
        reference = sources.json(config_dir / "reference_params.json")
        expected_coverage = sources.csv(config_dir / "data_coverage.csv")
        for row in expected_coverage:
            for key in ("rank", "blocks", "assigned_rows", "batches", "processed_rows"):
                row[key] = int(row[key])
        labels = [entry["name"] for entry in parsed_value(reference["label_col"])]
        for index, setting in enumerate("BCDE"):
            historical = setting == "E"
            run_id = (parsed_value(reference["model_id"]) if historical else
                      f"SISA_components_{model}_{dataset}_{setting}_seed20262028_attempt1")
            job_id = f"548166_{reference_task}" if historical else f"{array}_{index}"
            slurm_path = (f"logs/unirank-sisa-seed20262028-ws2-{job_id}.out" if historical else
                          f"logs/unirank-sisa-{log_prefix}-bcd-{job_id}.out")
            checkpoint_path = f"checkpoints/{dataset}/{run_id}.log"
            slurm = sources.read(Path(slurm_path)) or ""
            checkpoint = sources.read(Path(checkpoint_path)) or ""
            issues = []
            coverage = parse_coverage(slurm, 16384)
            record = dict(study=study, model=model, dataset=dataset, setting=setting,
                          variant=VARIANTS[setting], seed=20262028, run_id=run_id,
                          job_task_id=job_id, state="UNKNOWN", eligible=False, issues=issues,
                          metrics={}, macro_auc=None, macro_logloss=None,
                          checkpoint_log=checkpoint_path, slurm_log=slurm_path, coverage=coverage,
                          gpu_names=[], source_git_revision=None, protocol_fields_checked=0,
                          **parameter_counts(checkpoint))
            if historical:
                matches = [row for row in archival_runs if row["job_task_id"] == job_id]
                archive = matches[0] if len(matches) == 1 else {}
                record["state"] = archive.get("state", "UNKNOWN")
                record["exit_code"] = archive.get("exit_code")
                if archive.get("state") != "COMPLETED" or archive.get("exit_code") != "0:0":
                    issues.append("Historical run lacks archived COMPLETED/0:0 evidence")
                for key, expected in dict(model=model, dataset=dataset, seed="20262028", setting="sisa",
                                           world_size="2", per_gpu_batch="16384", global_batch="32768",
                                           checkpoint_log=checkpoint_path, slurm_log=slurm_path).items():
                    if archive.get(key) != expected:
                        issues.append(f"Historical archive mismatch: {key}")
                marker = (f"SISA_SINGLE_SEED_COMPLETE task_id={reference_task} run_id={run_id} "
                          "protocol=ws2_bs16384_acc1")
                if slurm.splitlines().count(marker) != 1:
                    issues.append("Missing or duplicate historical completion marker")
                hardware = re.findall(r"SISA_SINGLE_SEED_HARDWARE gpu_count=(\d+) gpu_name=(.*?) host=", slurm)
                if hardware != [("2", "NVIDIA H100 80GB HBM3")]:
                    issues.append("Historical hardware does not match 2 H100 GPUs")
                else:
                    record["gpu_names"] = [hardware[0][1]] * 2
                for key in ("total_parameters", "dense_parameters"):
                    if str(record[key]) != archive.get(key):
                        issues.append(f"Historical parameter count mismatch: {key}")
            else:
                account = accounts.get(job_id, {})
                record["state"] = account.get("state", "UNKNOWN")
                record["exit_code"] = account.get("exit_code")
                record["raw_job_id"] = account.get("raw_job_id")
                artifact_dir = Path("artifacts") / study / run_id
                manifest = sources.json(artifact_dir / "manifest.json") or {}
                outcome = sources.json(artifact_dir / "outcome.json") or {}
                issues.extend(completion_issues(account, outcome, slurm, run_id))
                if manifest:
                    issues.extend(manifest_issues(manifest, study, model, dataset, setting,
                                                  index, run_id, array, account))
                    record["gpu_names"] = manifest.get("gpu_names", [])
                    record["source_git_revision"] = manifest.get("git_revision")
                    protocol_lines = re.findall(r"^SISA_COMPONENT_PROTOCOL (.+)$", slurm, re.MULTILINE)
                    if len(protocol_lines) != 1 or json.loads(protocol_lines[0]) != manifest:
                        issues.append("Embedded Slurm protocol differs from artifact manifest")
                else:
                    issues.append("Missing artifact manifest")
            # Running jobs may only have the first one or two dataset splits.
            issues.extend(coverage_issues(coverage, expected_coverage,
                                          partial=record["state"] != "COMPLETED"))
            for name, text in (("checkpoint", checkpoint), ("Slurm", slurm)):
                try:
                    actual = parse_params(text)
                    issues.extend(f"{name}: {issue}" for issue in
                                  protocol_issues(actual, reference, run_id, VARIANTS[setting]))
                    if name == "checkpoint":
                        record["protocol_fields_checked"] = len(actual)
                except (ValueError, TypeError) as error:
                    issues.append(f"{name}: {error}")
            try:
                metrics = parse_test_metrics(checkpoint, labels)
                if metrics != parse_test_metrics(slurm, labels):
                    issues.append("Final Test metrics differ between checkpoint and Slurm logs")
                if historical:
                    expected_labels = [row for row in archival_labels
                                       if row["model"] == model and row["dataset"] == dataset]
                    expected_metrics = {row["label"]: {"AUC": float(row["sisa_20262028_auc"]),
                                                       "logloss": float(row["sisa_20262028_logloss"])}
                                        for row in expected_labels}
                    if metrics != expected_metrics:
                        issues.append("Historical Test metrics differ from archived label table")
                if not issues:
                    record.update(eligible=True, metrics=metrics,
                                  macro_auc=statistics.mean(row["AUC"] for row in metrics.values()),
                                  macro_logloss=statistics.mean(row["logloss"] for row in metrics.values()))
            except (ValueError, TypeError) as error:
                issues.append(str(error))
            runs.append(record)
    return runs, list(sources.records.values())


def build_comparisons(runs):
    comparisons = []
    for study in dict.fromkeys(run["study"] for run in runs):
        cells = {run["setting"]: run for run in runs if run["study"] == study}
        if set(cells) != set("BCDE") or not all(run["eligible"] for run in cells.values()):
            continue
        labels = list(cells["E"]["metrics"])
        result = dict(study=study, model=cells["E"]["model"], dataset=cells["E"]["dataset"],
                      macro={}, labels={label: {} for label in labels})
        for name, weights in DIFFERENCES.items():
            result["macro"][name] = {
                metric: sum(weight * cells[setting][key] for setting, weight in weights.items())
                for metric, key in (("AUC", "macro_auc"), ("logloss", "macro_logloss"))}
            for label in labels:
                result["labels"][label][name] = {
                    metric: sum(weight * cells[setting]["metrics"][label][metric]
                                for setting, weight in weights.items())
                    for metric in ("AUC", "logloss")}
        comparisons.append(result)
    return comparisons


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/ablations/sisa_component_report/archive/results")
    args = parser.parse_args()
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    accounts, account_source = accounting()
    runs, sources = collect(ROOT, accounts)
    comparisons = build_comparisons(runs)
    evidence = dict(generated_at=generated_at, runs=runs, comparisons=comparisons,
                    sources=sources, accounting_snapshot=account_source,
                    metric_definition="Unweighted mean across all dataset task labels; Test only.",
                    parameter_definition="total_parameters and dense_parameters are trainable parameter counts logged by count_parameters (requires_grad=True), not all allocated tensors.",
                    comparison_definition="Raw metric differences; larger AUC and smaller logloss are better.",
                    coverage_definition="Per-rank allocated blocks/rows/batches from Slurm; processed_rows=batches*16384, as in frozen drop_last protocol. This checks counts, not sample IDs or dataset content hashes.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    run_fields = ["study", "model", "dataset", "setting", "variant", "seed", "job_task_id", "state",
                  "eligible", "macro_auc", "macro_logloss", "total_parameters", "dense_parameters",
                  "protocol_fields_checked", "source_git_revision", "checkpoint_log", "slurm_log", "issues"]
    run_rows = [{key: ("; ".join(run[key]) if key == "issues" else run.get(key))
                 for key in run_fields} for run in runs]
    write_csv(args.output / "runs.csv", run_rows, run_fields)
    label_fields = ["study", "model", "dataset", "setting", "seed", "job_task_id", "label", "AUC", "logloss"]
    label_rows = [dict({key: run[key] for key in label_fields[:6]}, label=label, **values)
                  for run in runs for label, values in run["metrics"].items()]
    write_csv(args.output / "label_metrics.csv", label_rows, label_fields)
    comparison_fields = ["study", "model", "dataset", "scope", "label", "comparison", "delta_auc", "delta_logloss"]
    comparison_rows = []
    for group in comparisons:
        for label, values in [("__macro__", group["macro"]), *group["labels"].items()]:
            for name, deltas in values.items():
                comparison_rows.append(dict(study=group["study"], model=group["model"], dataset=group["dataset"],
                                            scope="macro" if label == "__macro__" else "label", label=label,
                                            comparison=name, delta_auc=deltas["AUC"], delta_logloss=deltas["logloss"]))
    write_csv(args.output / "comparisons.csv", comparison_rows, comparison_fields)
    print(json.dumps({"generated_at": generated_at, "runs": len(runs),
                      "eligible_runs": sum(run["eligible"] for run in runs),
                      "complete_comparisons": len(comparisons),
                      "output": str(args.output.relative_to(ROOT))}, ensure_ascii=False))
    for run in runs:
        print(json.dumps({key: run[key] for key in ("job_task_id", "setting", "state", "eligible", "issues")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
