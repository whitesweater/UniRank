#!/usr/bin/env python3
"""Collect and validate the 38-task SISA expansion matrix on HPC3/ACD."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.audit_sisa_expansion_baselines import EXPERIMENTS
from scripts.collect_sisa_native_strict_results import parse_test_metrics


COMPLETION_MARKER = "SISA_ACD_EXPANSION_COMPLETE"
HARDWARE_PATTERN = re.compile(
    r"ACD_EXPANSION_HARDWARE gpu_count=(\d+) gpu_name=(.*?) host=(\S+)"
)
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|"
    r"NCCL[^\n]*\b(?:error|timed?\s*out|timeout)\b|"
    r"\bKilled\b|\bNaN\b|ACD_EXPANSION_FAILED",
    re.IGNORECASE,
)
CALIBRATION_TASKS = frozenset({0, 8, 18, 28})


@dataclass(frozen=True)
class ExpansionTask:
    task_id: int
    model: str
    dataset: str
    setting: str


def expansion_task(task_id: int) -> ExpansionTask:
    if task_id < 0 or task_id >= 38:
        raise ValueError(f"task {task_id} is outside the expansion matrix")
    experiment = EXPERIMENTS[task_id // 2]
    model, dataset = experiment.split("_", 1)
    return ExpansionTask(
        task_id=task_id,
        model=model,
        dataset=dataset,
        setting="sisa" if task_id % 2 else "baseline",
    )


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_slurm_log(repository: Path, array_job: int, task_id: int) -> Path:
    """Resolve a task log even when a retry used a different Slurm job name."""
    log_dir = repository / "logs"
    preferred = log_dir / f"unirank-sisa-expansion-acd-{array_job}_{task_id}.out"
    if preferred.is_file():
        return preferred

    candidates = sorted(log_dir.glob(f"*-{array_job}_{task_id}.out"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        relative_candidates = ", ".join(
            str(path.relative_to(repository)) for path in candidates
        )
        raise RuntimeError(
            f"ambiguous Slurm logs for {array_job}_{task_id}: {relative_candidates}"
        )
    return preferred


def scheduler_outcome(job_task_id: str) -> tuple[str, str, str]:
    accounting = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            job_task_id,
            "--format=State,ExitCode",
            "--parsable2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in accounting.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 2 and fields[0]:
            return fields[0].split("+")[0], fields[1], "sacct"

    control = subprocess.run(
        ["scontrol", "show", "job", job_task_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if control.returncode == 0:
        state_match = re.search(r"\bJobState=(\S+)", control.stdout)
        exit_match = re.search(r"\bExitCode=(\S+)", control.stdout)
        return (
            state_match.group(1) if state_match else "UNKNOWN",
            exit_match.group(1) if exit_match else "UNKNOWN",
            "scontrol",
        )
    return "UNKNOWN", "UNKNOWN", "unavailable"


def protocol_valid(task: ExpansionTask, slurm_text: str, checkpoint_text: str) -> bool:
    required_slurm = (
        "ACD_EXPANSION_HARDWARE gpu_count=4",
        "EXPANSION_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 "
        "accumulation_steps=1 epochs=1 seed=20262027 bf16=true",
        f"setting={task.setting}",
        COMPLETION_MARKER,
    )
    required_checkpoint = (
        '"batch_size": "8192"',
        '"accumulation_steps": "1"',
        '"epochs": "1"',
        '"seed": "20262027"',
        '"max_len": "100"',
        '"world_size": "4"',
        '"enable_bf16": "True"',
        "DDP blocked training: True",
    )
    if not all(item in slurm_text for item in required_slurm):
        return False
    if not all(item in checkpoint_text for item in required_checkpoint):
        return False
    has_sisa = '"sisa_enabled": "True"' in checkpoint_text
    return has_sisa == (task.setting == "sisa")


def finite_metrics(metrics: dict[str, dict[str, float]]) -> bool:
    return bool(metrics) and all(
        math.isfinite(value)
        for task_metrics in metrics.values()
        for value in task_metrics.values()
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--calibration-job", type=int, required=True)
    parser.add_argument("--main-job", type=int, required=True)
    parser.add_argument(
        "--job-override",
        action="append",
        default=[],
        metavar="TASK_ID=ARRAY_JOB_ID",
        help="Use a retry array job for one logical task; may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/sisa_expansion_acd/results"),
    )
    args = parser.parse_args()

    job_overrides: dict[int, int] = {}
    for value in args.job_override:
        try:
            task_text, job_text = value.split("=", 1)
            task_id = int(task_text)
            array_job = int(job_text)
        except ValueError as error:
            raise SystemExit(
                f"invalid --job-override {value!r}; expected TASK_ID=ARRAY_JOB_ID"
            ) from error
        expansion_task(task_id)
        if task_id in job_overrides:
            raise SystemExit(f"duplicate --job-override for task {task_id}")
        job_overrides[task_id] = array_job

    repository = args.repository.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repository / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for task_id in range(38):
        task = expansion_task(task_id)
        array_job = job_overrides.get(
            task_id,
            args.calibration_job if task_id in CALIBRATION_TASKS else args.main_job,
        )
        job_task_id = f"{array_job}_{task_id}"
        slurm_log = resolve_slurm_log(repository, array_job, task_id)
        checkpoint_log = (
            repository
            / "checkpoints"
            / task.dataset
            / (
                f"SISA_expansion_{task.model}_{task.dataset}_"
                f"{task.setting}_seed20262027.log"
            )
        )
        slurm_text = read_text(slurm_log)
        checkpoint_text = read_text(checkpoint_log)
        metrics = parse_test_metrics(checkpoint_text)
        hardware_match = HARDWARE_PATTERN.search(slurm_text)
        gpu_count = int(hardware_match.group(1)) if hardware_match else 0
        gpu_name = hardware_match.group(2) if hardware_match else ""
        host = hardware_match.group(3) if hardware_match else ""
        state, exit_code, state_source = scheduler_outcome(job_task_id)
        has_completion = COMPLETION_MARKER in slurm_text
        if state == "UNKNOWN" and has_completion:
            state, exit_code, state_source = (
                "COMPLETED_MARKER",
                "0:0 (success marker)",
                "success_marker",
            )
        valid_protocol = protocol_valid(task, slurm_text, checkpoint_text)
        metrics_finite = finite_metrics(metrics)
        scheduler_success = (
            state.startswith("COMPLETED") and exit_code.startswith("0:0")
        )
        complete = (
            scheduler_success
            and has_completion
            and valid_protocol
            and metrics_finite
            and gpu_count == 4
            and bool(gpu_name)
        )
        run_rows.append(
            {
                "task_id": task_id,
                "model": task.model,
                "dataset": task.dataset,
                "setting": task.setting,
                "array_job": array_job,
                "job_task_id": job_task_id,
                "state": state,
                "exit_code": exit_code,
                "state_source": state_source,
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
                "host": host,
                "protocol_valid": valid_protocol,
                "completion_marker": has_completion,
                "finite_metrics": metrics_finite,
                "error_count": len(ERROR_PATTERN.findall(slurm_text + checkpoint_text)),
                "complete": complete,
                "slurm_log": str(slurm_log.relative_to(repository)),
                "checkpoint_log": str(checkpoint_log.relative_to(repository)),
                "metrics_json": json.dumps(metrics, sort_keys=True),
            }
        )
        for label, values in metrics.items():
            metric_rows.append(
                {
                    "task_id": task_id,
                    "model": task.model,
                    "dataset": task.dataset,
                    "setting": task.setting,
                    "gpu_name": gpu_name,
                    "label": label,
                    "logloss": values.get("logloss", ""),
                    "AUC": values.get("AUC", ""),
                }
            )

    pair_hardware_errors: list[str] = []
    pair_metric_errors: list[str] = []
    summary_rows: list[dict[str, object]] = []
    metrics_by_task_label = {
        (int(row["task_id"]), str(row["label"])): row for row in metric_rows
    }
    for baseline_task_id in range(0, 38, 2):
        baseline_run = run_rows[baseline_task_id]
        sisa_run = run_rows[baseline_task_id + 1]
        if baseline_run["gpu_name"] != sisa_run["gpu_name"]:
            pair_hardware_errors.append(
                f"tasks {baseline_task_id}/{baseline_task_id + 1}: "
                f"GPU mismatch {baseline_run['gpu_name']!r} vs {sisa_run['gpu_name']!r}"
            )
        baseline_labels = {
            label
            for task_id, label in metrics_by_task_label
            if task_id == baseline_task_id
        }
        sisa_labels = {
            label
            for task_id, label in metrics_by_task_label
            if task_id == baseline_task_id + 1
        }
        if baseline_labels != sisa_labels:
            pair_metric_errors.append(
                f"tasks {baseline_task_id}/{baseline_task_id + 1}: label mismatch"
            )
        for label in sorted(baseline_labels & sisa_labels):
            baseline_metric = metrics_by_task_label[(baseline_task_id, label)]
            sisa_metric = metrics_by_task_label[(baseline_task_id + 1, label)]
            baseline_auc = float(baseline_metric["AUC"])
            sisa_auc = float(sisa_metric["AUC"])
            baseline_logloss = float(baseline_metric["logloss"])
            sisa_logloss = float(sisa_metric["logloss"])
            summary_rows.append(
                {
                    "model": baseline_run["model"],
                    "dataset": baseline_run["dataset"],
                    "label": label,
                    "baseline_auc": baseline_auc,
                    "sisa_auc": sisa_auc,
                    "delta_auc": sisa_auc - baseline_auc,
                    "baseline_logloss": baseline_logloss,
                    "sisa_logloss": sisa_logloss,
                    "delta_logloss": sisa_logloss - baseline_logloss,
                }
            )

    write_csv(output_dir / "runs.csv", run_rows, list(run_rows[0]))
    write_csv(
        output_dir / "metrics.csv",
        metric_rows,
        [
            "task_id",
            "model",
            "dataset",
            "setting",
            "gpu_name",
            "label",
            "logloss",
            "AUC",
        ],
    )
    write_csv(
        output_dir / "paired_summary.csv",
        summary_rows,
        [
            "model",
            "dataset",
            "label",
            "baseline_auc",
            "sisa_auc",
            "delta_auc",
            "baseline_logloss",
            "sisa_logloss",
            "delta_logloss",
        ],
    )

    completed = sum(bool(row["complete"]) for row in run_rows)
    auc_deltas = [float(row["delta_auc"]) for row in summary_rows]
    mean_auc_delta = sum(auc_deltas) / len(auc_deltas) if auc_deltas else math.nan
    summary_lines = [
        "# UniRank SISA HPC3/ACD expansion result snapshot",
        "",
        f"- Complete tasks: {completed}/38",
        f"- Paired label metrics: {len(summary_rows)}",
        f"- Mean delta AUC: {mean_auc_delta:+.6f}",
        f"- GPU-pair errors: {len(pair_hardware_errors)}",
        f"- Metric-pair errors: {len(pair_metric_errors)}",
        "",
    ]
    if pair_hardware_errors or pair_metric_errors:
        summary_lines.extend(
            ["## Pairing errors", ""]
            + [f"- {error}" for error in pair_hardware_errors + pair_metric_errors]
            + [""]
        )
    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    print(
        f"SISA_ACD_EXPANSION_COLLECT complete={completed}/38 "
        f"paired_labels={len(summary_rows)} mean_delta_auc={mean_auc_delta:+.6f} "
        f"gpu_pair_errors={len(pair_hardware_errors)} "
        f"metric_pair_errors={len(pair_metric_errors)} output_dir={output_dir}"
    )
    if completed != 38 or pair_hardware_errors or pair_metric_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
