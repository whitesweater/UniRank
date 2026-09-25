#!/usr/bin/env python3
"""Collect only completed, protocol-matched HyFormer-MerRec dimension evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
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
from scripts import run_sisa_score_dim as runner

REFERENCE_SEEDS = (20262027, 20262028, 20262029)
REFERENCE_JOBS = {20262027: "543257_27", 20262028: "548166_1", 20262029: "549595_1"}


class RuntimeSources(Sources):
    """Preserve raw bytes even when concurrent Slurm progress splits UTF-8."""

    def read(self, relative):
        path = self.root / relative
        if not path.exists():
            return None
        raw = path.read_bytes()
        record = dict(path=str(relative), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        self.records[str(relative)] = record
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            if path.suffix != ".out":
                raise
            text = raw.decode("utf-8", errors="replace")
            record["utf8_replacement_characters"] = text.count("\ufffd")
            return text


def expected_parameter_counts(score_dim: int) -> dict:
    frozen = json.loads((runner.DIRECTORY / "configs/reference_evidence.json").read_text())
    # Six sites, hidden width 256, two heads; only B/C/phase widths change.
    delta = 7680 * (score_dim - 16)
    return {key: frozen["reference"][key] + delta
            for key in ("total_parameters", "dense_parameters")}


def audit_run(*, task, array_id, account, manifest, outcome, checkpoint,
              slurm, expected_coverage, expected_sources=None) -> dict:
    issues = []
    record = dict(task_id=task.task_id, score_dim=task.score_dim, run_id=task.run_id,
                  seed=runner.SEED,
                  job_task_id=f"{array_id}_{task.task_id}", attempt=task.attempt,
                  state=account.get("state", "UNKNOWN"), exit_code=account.get("exit_code"),
                  eligible=False, issues=issues, metrics={}, macro_auc=None, macro_logloss=None,
                  gpu_names=manifest.get("gpu_names", []), **parameter_counts(checkpoint))
    record["gradient_checkpointing"] = runner.uses_activation_checkpointing(task)
    record.update(world_size=runner.task_world_size(task), per_gpu_batch=runner.task_batch_size(task),
                  global_batch=32768)
    record["comparison_scope"] = ("dimension_with_activation_checkpointing"
                                  if record["gradient_checkpointing"] else "dimension_only")
    if record["world_size"] == 4:
        record["comparison_scope"] = "dimension_with_four_gpu_layout"
    if account.get("state") != "COMPLETED" or account.get("exit_code") != "0:0":
        issues.append("No COMPLETED/0:0 accounting record")
    if outcome.get("returncode") != 0:
        issues.append("Missing or unsuccessful outcome")
    if slurm.splitlines().count(f"SISA_SCORE_DIM_COMPLETE run_id={task.run_id}") != 1:
        issues.append("Missing or duplicate dimension completion marker")
    expected = runner.task_manifest(task, manifest.get("gpu_names", []),
                                    manifest.get("git_revision", ""),
                                    {"SLURM_JOB_ID": account.get("raw_job_id"),
                                     "SLURM_ARRAY_JOB_ID": array_id})
    for key, value in expected.items():
        if manifest.get(key) != value:
            issues.append(f"Manifest mismatch: {key}")
    if len(manifest.get("gpu_names", [])) != record["world_size"]:
        issues.append(f"Manifest must identify {record['world_size']} GPUs")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_revision", "")):
        issues.append("Missing valid source Git revision")
    if expected_sources is not None and manifest.get("source_sha256") != expected_sources:
        issues.append("Source hashes differ from the pre-submission snapshot")
    protocol_lines = re.findall(r"^SISA_SCORE_DIM_PROTOCOL (.+)$", slurm, re.MULTILINE)
    try:
        if len(protocol_lines) != 1 or json.loads(protocol_lines[0]) != manifest:
            issues.append("Embedded protocol differs from artifact manifest")
    except (ValueError, TypeError):
        issues.append("Malformed embedded protocol")
    actual_coverage = parse_coverage(slurm, runner.task_batch_size(task))
    issues.extend(coverage_issues(actual_coverage, expected_coverage))
    record["coverage"] = actual_coverage
    for name, text in (("checkpoint", checkpoint), ("Slurm", slurm)):
        try:
            issues.extend(f"{name}: {issue}" for issue in
                          runner.protocol_issues(parse_params(text), task))
        except (ValueError, TypeError):
            issues.append(f"{name}: missing or malformed rank-zero Params")
        if parameter_counts(text) != expected_parameter_counts(task.score_dim):
            issues.append(f"{name}: parameter counts differ from the dimension-only expectation")
        if re.search(r"Traceback \(most recent call last\)|OutOfMemoryError|CUDA out of memory|"
                     r"ChildFailedError|NCCL.*(?:[Ee]rror|[Tt]imed out)", text):
            issues.append(f"{name}: runtime failure recorded")
    try:
        labels = [entry["name"] for entry in runner.reference_params()["label_col"]]
        metrics = parse_test_metrics(checkpoint, labels)
        if metrics != parse_test_metrics(slurm, labels):
            issues.append("Final Test metrics differ between checkpoint and Slurm logs")
        if not issues:
            record.update(eligible=True, metrics=metrics,
                          macro_auc=statistics.mean(value["AUC"] for value in metrics.values()),
                          macro_logloss=statistics.mean(value["logloss"] for value in metrics.values()))
    except (ValueError, TypeError):
        issues.append("Missing or invalid final Test metrics")
    return record


def historical_reference(sources, seed: int) -> dict:
    archive = Path("experiments/sisa_paper_report/archive/results")
    matches = [row for row in sources.csv(archive / "run_evidence.csv")
               if row["job_task_id"] == REFERENCE_JOBS[seed]
               and row["model"] == "HyFormer" and row["dataset"] == "MerRec_Action"
               and row["setting"] == "sisa" and int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate historical r16 run for seed {seed}")
    archive_row = matches[0]
    checkpoint = sources.read(Path(archive_row["checkpoint_log"])) or ""
    slurm = sources.read(Path(archive_row["slurm_log"])) or ""
    world_size = int(archive_row["world_size"])
    batch_size = int(archive_row["per_gpu_batch"])
    run_id = Path(archive_row["checkpoint_log"]).stem
    labels = [entry["name"] for entry in runner.reference_params()["label_col"]]
    issues = []
    record = dict(score_dim=16, seed=seed, task_id=None, attempt=1, historical=True,
                  run_id=run_id, job_task_id=archive_row["job_task_id"],
                  state=archive_row["state"], exit_code=archive_row["exit_code"],
                  eligible=False, issues=issues, metrics={}, macro_auc=None, macro_logloss=None,
                  world_size=world_size, per_gpu_batch=batch_size, global_batch=32768,
                  gradient_checkpointing=False, comparison_scope="historical_reference",
                  checkpoint_log=archive_row["checkpoint_log"], slurm_log=archive_row["slurm_log"],
                  gpu_names=[archive_row["gpu"]] * world_size, **parameter_counts(checkpoint))
    if (record["state"], record["exit_code"], archive_row["original_metrics_match"],
            archive_row["metric_count"], archive_row["logged_sessions"]) != ("COMPLETED", "0:0", "True", "5", "1"):
        issues.append("Historical completion or original-evidence acceptance failed")
    manifest = {row["path"]: row for row in sources.csv(archive / "source_manifest.csv")}
    for field in ("checkpoint_log", "slurm_log"):
        path = archive_row[field]
        if sources.records.get(path, {}).get("sha256") != manifest.get(path, {}).get("sha256"):
            issues.append(f"Historical source hash mismatch: {path}")
    expected_params = dict(runner.reference_params(), model_id=run_id, seed=seed,
                           batch_size=batch_size, world_size=world_size)
    if seed == 20262027:
        # The original run did not explicitly log these override fields.
        expected_params.pop("dataloader_seed", None)
        expected_params.pop("sisa_parameter_seed", None)
    else:
        expected_params.update(dataloader_seed=2028, sisa_parameter_seed=20260823)
    for name, text in (("checkpoint", checkpoint), ("Slurm", slurm)):
        try:
            actual = {key: parsed_value(value) for key, value in parse_params(text).items()}
            if actual != expected_params:
                changed = [key for key in actual.keys() | expected_params.keys()
                           if actual.get(key) != expected_params.get(key)]
                issues.append(f"{name}: historical parameters differ: {sorted(changed)}")
            if name == "checkpoint":
                record["logged_params"] = actual
        except (ValueError, TypeError) as error:
            issues.append(f"{name}: {error}")
        expected_counts = {key: int(archive_row[key]) for key in ("total_parameters", "dense_parameters")}
        if parameter_counts(text) != expected_counts:
            issues.append(f"{name}: historical parameter counts disagree")
    markers = ([f"SISA_EXPANSION_COMPLETE run_id={run_id}", "SISA_ACD_EXPANSION_COMPLETE task_id=27 attempt=1"]
               if seed == 20262027 else
               [f"SISA_SINGLE_SEED_COMPLETE task_id=1 run_id={run_id} protocol=ws2_bs16384_acc1"])
    if any(slurm.splitlines().count(marker) != 1 for marker in markers):
        issues.append("Historical completion marker missing or duplicated")
    coverage_file = "data_coverage_ws4.csv" if world_size == 4 else "data_coverage.csv"
    expected_coverage = sources.csv(runner.DIRECTORY.relative_to(ROOT) / "configs" / coverage_file)
    for row in expected_coverage:
        for key in ("rank", "blocks", "assigned_rows", "batches", "processed_rows"):
            row[key] = int(row[key])
    record["coverage"] = parse_coverage(slurm, batch_size)
    issues.extend(coverage_issues(record["coverage"], expected_coverage))
    try:
        metrics = parse_test_metrics(checkpoint, labels)
        if metrics != parse_test_metrics(slurm, labels):
            issues.append("Historical checkpoint/Slurm Test metrics differ")
        label_rows = [row for row in sources.csv(archive / "label_results.csv")
                      if row["model"] == "HyFormer" and row["dataset"] == "MerRec_Action"]
        expected_metrics = {row["label"]: {"AUC": float(row[f"sisa_{seed}_auc"]),
                                           "logloss": float(row[f"sisa_{seed}_logloss"])} for row in label_rows}
        if metrics != expected_metrics:
            issues.append("Historical final Test differs from the accepted label archive")
        if not issues:
            record.update(eligible=True, metrics=metrics,
                          macro_auc=float(sum(Decimal(str(row["AUC"])) for row in metrics.values()) / 5),
                          macro_logloss=float(sum(Decimal(str(row["logloss"])) for row in metrics.values()) / 5))
    except (ValueError, TypeError) as error:
        issues.append(str(error))
    return record


def summarize_dimensions(runs: list[dict]) -> list[dict]:
    def mean(values):
        return float(sum(Decimal(str(value)) for value in values) / Decimal(len(values)))

    summaries = []
    for dim in (8, 16, 32, 64):
        group = sorted((row for row in runs if row["score_dim"] == dim), key=lambda row: row["seed"])
        expected_seeds = list(REFERENCE_SEEDS) if dim == 16 else [runner.SEED]
        seeds = [row["seed"] for row in group]
        valid = (seeds == expected_seeds and all(row["eligible"] for row in group)
                 and len({row["job_task_id"] for row in group}) == len(group))
        summary = dict(score_dim=dim, n_runs=len(group), seeds=seeds,
                       job_task_ids=[row["job_task_id"] for row in group], eligible=valid,
                       statistic="three_run_mean" if dim == 16 else "single_run",
                       comparison_scope="mixed_protocol_three_run_reference" if dim == 16
                       else "single_run_vs_three_run_reference",
                       macro_auc=None, macro_logloss=None, metrics={})
        if valid:
            labels = list(group[0]["metrics"])
            if not all(set(row["metrics"]) == set(labels) for row in group):
                summary["eligible"] = False
            else:
                summary.update(macro_auc=mean([row["macro_auc"] for row in group]),
                               macro_logloss=mean([row["macro_logloss"] for row in group]),
                               metrics={label: {metric: mean([row["metrics"][label][metric] for row in group])
                                                for metric in ("AUC", "logloss")} for label in labels},
                               total_parameters=group[0]["total_parameters"],
                               dense_parameters=group[0]["dense_parameters"])
        summaries.append(summary)
    return summaries


def comparisons(runs: list[dict]) -> list[dict]:
    references = [run for run in runs if run["score_dim"] == 16 and run["eligible"]]
    if len(references) != 1:
        return []
    reference = references[0]
    rows = []
    for run in runs:
        if not run["eligible"] or run["score_dim"] == 16:
            continue
        rows.append(dict(score_dim=run["score_dim"], reference_dim=16, label="__macro__",
                         comparison_scope=run.get("comparison_scope", "dimension_only"),
                         delta_auc=run["macro_auc"] - reference["macro_auc"],
                         delta_logloss=run["macro_logloss"] - reference["macro_logloss"]))
        for label, values in run["metrics"].items():
            rows.append(dict(score_dim=run["score_dim"], reference_dim=16, label=label,
                             comparison_scope=run.get("comparison_scope", "dimension_only"),
                             delta_auc=values["AUC"] - reference["metrics"][label]["AUC"],
                             delta_logloss=values["logloss"] - reference["metrics"][label]["logloss"]))
    return rows


def parse_accounting(text: str) -> dict:
    records = {}
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and "." not in fields[0]:
            job_id, raw_id, state, exit_code = fields[:4]
            previous = records.get(job_id)
            array_id = job_id.split("_", 1)[0]
            # --array can expand a stale pending range over an already
            # materialized task. The concrete JobIDRaw owns that task's state.
            if (previous and raw_id == array_id and state == "PENDING"
                    and previous["raw_job_id"] != array_id):
                continue
            records[job_id] = dict(raw_job_id=raw_id, state=state, exit_code=exit_code)
    return records


def accounting(array_ids: list[str]) -> tuple[dict, dict]:
    if not array_ids:
        return {}, {"command": [], "stdout": "", "stderr": ""}
    command = ["sacct", "--array", "-n", "-P", "-j", ",".join(array_ids),
               "--format=JobID,JobIDRaw,State,ExitCode"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return parse_accounting(result.stdout), dict(command=command, stdout=result.stdout, stderr=result.stderr)


def collect() -> dict:
    sources = RuntimeSources(ROOT)
    directory = runner.DIRECTORY.relative_to(ROOT)
    frozen = sources.json(directory / "configs/reference_evidence.json")
    reference = dict(frozen["reference"], score_dim=16, task_id=None, attempt=1,
                     historical=True, gradient_checkpointing=False, comparison_scope="dimension_only",
                     world_size=2, per_gpu_batch=16384, global_batch=32768)
    # Freeze the verified historical record, then require its original sources
    # to remain byte-identical whenever it is reused for a new comparison.
    issues = list(reference["issues"])
    for source in frozen["sources"]:
        sources.read(Path(source["path"]))
        actual = sources.records.get(source["path"], {})
        if actual.get("sha256") != source["sha256"]:
            issues.append(f"Historical source missing or changed: {source['path']}")
    reference["issues"] = issues
    if issues:
        reference.update(eligible=False, metrics={}, macro_auc=None, macro_logloss=None)
    submission = sources.json(directory / "notes/submission.json") or {"tasks": []}
    accounts, receipt = accounting(sorted({row["array_id"] for row in submission["tasks"]}))
    runs = [reference, historical_reference(sources, 20262027), historical_reference(sources, 20262029)]
    for index in range(3):
        matching = [row for row in submission["tasks"] if row["task_id"] == index]
        if len(matching) > 1:
            raise ValueError(f"Ambiguous selected attempt for task {index}")
        selected = matching[0] if matching else {}
        source_snapshot = selected.get("source_snapshot", str(directory / "notes/source_sha256.json"))
        expected_sources = sources.json(Path(source_snapshot)) or {}
        task = runner.study_task(index, selected.get("attempt", 1))
        coverage_name = "data_coverage_ws4.csv" if runner.task_world_size(task) == 4 else "data_coverage.csv"
        expected_coverage = sources.csv(directory / "configs" / coverage_name)
        for row in expected_coverage:
            for key in ("rank", "blocks", "assigned_rows", "batches", "processed_rows"):
                row[key] = int(row[key])
        array_id = selected.get("array_id", "unsubmitted")
        job_id = f"{array_id}_{index}"
        artifact = Path("artifacts/sisa_score_dim") / task.run_id
        checkpoint_path = Path("checkpoints/MerRec_Action") / (task.run_id + ".log")
        slurm_path = Path("logs") / f"unirank-sisa-score-dim-{job_id}.out"
        checkpoint = sources.read(checkpoint_path) or ""
        slurm = sources.read(slurm_path) or ""
        record = audit_run(task=task, array_id=array_id, account=accounts.get(job_id, {}),
                           manifest=sources.json(artifact / "manifest.json") or {},
                           outcome=sources.json(artifact / "outcome.json") or {},
                           checkpoint=checkpoint, slurm=slurm, expected_coverage=expected_coverage,
                           expected_sources=expected_sources)
        record.update(checkpoint_log=str(checkpoint_path), slurm_log=str(slurm_path))
        if not selected:
            record["state"] = "NOT_SUBMITTED"
        progress = re.findall(r"INFO (?:Train|Epoch|\[Epoch)[^\r\n]*", checkpoint)
        bars = [line for line in slurm.splitlines()
                if re.search(r"\d+%\|.*\|\s*\d+/\d+", line)]
        record["latest_progress"] = bars[-1] if bars else (progress[-1] if progress else "")
        runs.append(record)
    runs.sort(key=lambda row: (row["score_dim"], row["seed"]))
    summary = summarize_dimensions(runs)
    attempt_history = sources.json(directory / "notes/attempt_history.json") or []
    return dict(generated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                study=runner.STUDY, model="HyFormer", dataset="MerRec_Action", seed=runner.SEED,
                runs=runs, summary=summary, comparisons=comparisons(summary), sources=list(sources.records.values()),
                report_reference=dict(score_dim=16, statistic="three_run_mean", seeds=list(REFERENCE_SEEDS)),
                training_protocol_reference_job="548166_1",
                attempt_history=attempt_history,
                accounting=receipt, complete=all(row["eligible"] for row in summary),
                metric_definition="Final Test only; equal mean of five labels per run. Report r16 as equal mean of all three historical runs; other dimensions have one run each.",
                coverage_definition="Per-rank block, row and batch counts; not sample identity hashes.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=runner.DIRECTORY / "results")
    args = parser.parse_args()
    evidence = collect()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["score_dim", "seed", "job_task_id", "attempt", "state", "eligible", "world_size", "per_gpu_batch",
              "global_batch", "gradient_checkpointing",
              "comparison_scope", "macro_auc", "macro_logloss",
              "total_parameters", "dense_parameters", "checkpoint_log", "slurm_log", "issues"]
    rows = [{key: "; ".join(row[key]) if key == "issues" else row.get(key) for key in fields}
            for row in evidence["runs"]]
    write_csv(args.output / "runs.csv", rows, fields)
    labels = [dict(score_dim=row["score_dim"], seed=row["seed"], job_task_id=row["job_task_id"], label=label, **values)
              for row in evidence["runs"] for label, values in row["metrics"].items()]
    write_csv(args.output / "run_label_metrics.csv", labels,
              ["score_dim", "seed", "job_task_id", "label", "AUC", "logloss"])
    summary_fields = ["score_dim", "n_runs", "statistic", "seeds", "job_task_ids", "eligible",
                      "macro_auc", "macro_logloss", "comparison_scope"]
    summary_rows = [{key: ";".join(map(str, row[key])) if key in {"seeds", "job_task_ids"}
                     else row[key] for key in summary_fields} for row in evidence["summary"]]
    write_csv(args.output / "summary.csv", summary_rows, summary_fields)
    label_rows = [dict(score_dim=row["score_dim"], n_runs=row["n_runs"], statistic=row["statistic"],
                       seeds=";".join(map(str, row["seeds"])), label=label, **values)
                  for row in evidence["summary"] for label, values in row["metrics"].items()]
    write_csv(args.output / "label_metrics.csv", label_rows,
              ["score_dim", "n_runs", "statistic", "seeds", "label", "AUC", "logloss"])
    write_csv(args.output / "comparisons.csv", evidence["comparisons"],
              ["score_dim", "reference_dim", "label", "comparison_scope", "delta_auc", "delta_logloss"])
    print(json.dumps(dict(generated_at=evidence["generated_at"], complete=evidence["complete"],
                          runs=[{key: row.get(key) for key in
                                 ("score_dim", "job_task_id", "state", "eligible", "macro_auc",
                                  "macro_logloss", "latest_progress")} for row in evidence["runs"]]),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
