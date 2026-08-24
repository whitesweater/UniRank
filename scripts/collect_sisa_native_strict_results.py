#!/usr/bin/env python3
"""Collect and validate the 32-task strict UniRank/SISA matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


MODELS = ("OneTrans", "HiFormer", "RankMixer", "Zenith")
DATASETS = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "Taobao_Action",
    "MerRec_Action",
)
CALIBRATION_TASK = 4
RTX_TASKS = frozenset(
    (0, 1, 5, 10, 11, 14, 15, 16, 17, 20, 21, 26, 27, 30, 31)
)
L40_TASKS = frozenset(
    (2, 3, 6, 7, 8, 9, 12, 13, 18, 19, 22, 23, 24, 25, 28, 29)
)
TEST_MARKER = "******** Test evaluation ********"
COMPLETION_MARKER = "SISA_NATIVE_STRICT_COMPLETE"
ERROR_PATTERN = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|OutOfMemory|oom-kill|"
    r"NCCL|Killed|Segmentation fault|"
    r"(?:^|[^A-Za-z0-9_])(?:nan|inf)(?:[^A-Za-z0-9_]|$)",
    flags=re.IGNORECASE | re.MULTILINE,
)
TASK_METRIC_PATTERN = re.compile(r"\[Task:\s*([^\]]+)\]\[Metrics\]\s*(.*)")
METRIC_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class LogicalTask:
    task_id: int
    model: str
    dataset: str
    setting: str


def logical_task(task_id: int) -> LogicalTask:
    if not 0 <= task_id < 32:
        raise ValueError(f"task id must be in [0, 31], got {task_id}")
    setting_index = task_id % 2
    cell_index = task_id // 2
    dataset_index = cell_index % len(DATASETS)
    model_index = cell_index // len(DATASETS)
    return LogicalTask(
        task_id=task_id,
        model=MODELS[model_index],
        dataset=DATASETS[dataset_index],
        setting="sisa" if setting_index else "baseline",
    )


def assignment(
    task_id: int,
    calibration_job: int,
    rtx_job: int,
    l40_job: int,
) -> tuple[int, str]:
    if task_id == CALIBRATION_TASK:
        return calibration_job, "rtx4090"
    if task_id in RTX_TASKS:
        return rtx_job, "rtx4090"
    if task_id in L40_TASKS:
        return l40_job, "l40s"
    raise ValueError(f"task {task_id} has no strict assignment")


def parse_job_overrides(values: list[str]) -> dict[int, int]:
    """Parse repeatable TASK_ID=ARRAY_JOB_ID retry mappings."""
    overrides: dict[int, int] = {}
    for value in values:
        try:
            task_text, job_text = value.split("=", 1)
            task_id = int(task_text)
            array_job = int(job_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid job override {value!r}; expected TASK_ID=ARRAY_JOB_ID"
            ) from error
        logical_task(task_id)
        if array_job <= 0:
            raise ValueError(f"array job id must be positive, got {array_job}")
        overrides[task_id] = array_job
    return overrides


def parse_gpu_type_overrides(values: list[str]) -> dict[int, str]:
    """Parse repeatable TASK_ID=GPU_TYPE retry mappings."""
    overrides: dict[int, str] = {}
    for value in values:
        try:
            task_text, gpu_type = value.split("=", 1)
            task_id = int(task_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid GPU override {value!r}; expected TASK_ID=GPU_TYPE"
            ) from error
        logical_task(task_id)
        if gpu_type not in {"rtx4090", "l40s"}:
            raise ValueError(f"unsupported GPU type {gpu_type!r}")
        overrides[task_id] = gpu_type
    return overrides


def parse_test_metrics(checkpoint_text: str) -> dict[str, dict[str, float]]:
    if TEST_MARKER not in checkpoint_text:
        return {}
    section = checkpoint_text.rsplit(TEST_MARKER, 1)[1]
    parsed: dict[str, dict[str, float]] = {}
    for line in section.splitlines():
        match = TASK_METRIC_PATTERN.search(line)
        if match is None:
            continue
        label = match.group(1).strip()
        metrics = {
            name: float(value)
            for name, value in METRIC_PATTERN.findall(match.group(2))
        }
        if metrics:
            parsed[label] = metrics
    return parsed


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def slurm_state(job_task_id: str) -> tuple[str, str]:
    completed = subprocess.run(
        ["scontrol", "show", "job", job_task_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "UNKNOWN", "UNKNOWN"
    state_match = re.search(r"\bJobState=(\S+)", completed.stdout)
    exit_match = re.search(r"\bExitCode=(\S+)", completed.stdout)
    return (
        state_match.group(1) if state_match else "UNKNOWN",
        exit_match.group(1) if exit_match else "UNKNOWN",
    )


def effective_slurm_outcome(
    state: str,
    exit_code: str,
    has_completion_marker: bool,
) -> tuple[str, str, str]:
    """Keep successful evidence after Slurm purges a completed job record.

    HPC01 can remove completed jobs from ``scontrol`` within one monitor
    interval. The strict marker is printed only after ``torchrun`` returns
    successfully under ``set -euo pipefail``, so it is durable success
    evidence when the scheduler record is no longer available.
    """
    if state == "UNKNOWN" and exit_code == "UNKNOWN" and has_completion_marker:
        return "COMPLETED_MARKER", "0:0 (success marker)", "success_marker"
    return state, exit_code, "scontrol"


def protocol_valid(
    logical: LogicalTask,
    slurm_text: str,
    checkpoint_text: str,
) -> bool:
    required_slurm = (
        "STRICT_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 "
        "accumulation_steps=1 epochs=1 seed=20262027 bf16=true",
        f"setting={logical.setting}",
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
    return has_sisa == (logical.setting == "sisa")


def metrics_are_finite(metrics: dict[str, dict[str, float]]) -> bool:
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


def write_summary(
    path: Path,
    run_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
) -> None:
    completed = sum(bool(row["complete"]) for row in run_rows)
    lines = [
        "# UniRank SISA strict result snapshot",
        "",
        f"Structurally complete logical tasks: **{completed}/32**.",
        "",
        "Exploratory single-GPU results are intentionally excluded.",
        "",
        "| Model | Dataset | Label | Baseline AUC | SISA AUC | Delta AUC | "
        "Baseline logloss | SISA logloss | Delta logloss |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_key: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in metric_rows:
        key = (str(row["model"]), str(row["dataset"]), str(row["label"]))
        by_key.setdefault(key, {})[str(row["setting"])] = row
    for (model, dataset, label), settings in sorted(by_key.items()):
        if set(settings) != {"baseline", "sisa"}:
            continue
        baseline = settings["baseline"]
        sisa = settings["sisa"]
        baseline_auc = float(baseline["AUC"])
        sisa_auc = float(sisa["AUC"])
        baseline_logloss = float(baseline["logloss"])
        sisa_logloss = float(sisa["logloss"])
        lines.append(
            f"| {model} | {dataset} | {label} | {baseline_auc:.6f} | "
            f"{sisa_auc:.6f} | {sisa_auc - baseline_auc:+.6f} | "
            f"{baseline_logloss:.6f} | {sisa_logloss:.6f} | "
            f"{sisa_logloss - baseline_logloss:+.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--calibration-job", type=int, default=14806)
    parser.add_argument("--rtx-job", type=int, default=14810)
    parser.add_argument("--l40-job", type=int, default=14811)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sisa_native_strict"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--job-override",
        action="append",
        default=[],
        metavar="TASK_ID=ARRAY_JOB_ID",
        help="Use a retry array job for one logical task (repeatable).",
    )
    parser.add_argument(
        "--gpu-type-override",
        action="append",
        default=[],
        metavar="TASK_ID=GPU_TYPE",
        help="Record the actual GPU type for a retried logical task.",
    )
    args = parser.parse_args()
    job_overrides = parse_job_overrides(args.job_override)
    gpu_type_overrides = parse_gpu_type_overrides(args.gpu_type_override)

    repository = args.repository.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repository / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for task_id in range(32):
        logical = logical_task(task_id)
        array_job, gpu_type = assignment(
            task_id,
            args.calibration_job,
            args.rtx_job,
            args.l40_job,
        )
        array_job = job_overrides.get(task_id, array_job)
        gpu_type = gpu_type_overrides.get(task_id, gpu_type)
        job_task_id = f"{array_job}_{task_id}"
        slurm_log = (
            repository
            / "logs"
            / f"unirank-sisa-strict-{array_job}_{task_id}.out"
        )
        checkpoint_log = (
            repository
            / "checkpoints"
            / logical.dataset
            / (
                f"SISA_strict_{logical.model}_{logical.dataset}_"
                f"{logical.setting}_seed20262027.log"
            )
        )
        slurm_text = read_text(slurm_log)
        checkpoint_text = read_text(checkpoint_log)
        metrics = parse_test_metrics(checkpoint_text)
        error_count = len(ERROR_PATTERN.findall(slurm_text + "\n" + checkpoint_text))
        has_completion = COMPLETION_MARKER in slurm_text
        state, exit_code = slurm_state(job_task_id)
        state, exit_code, status_evidence = effective_slurm_outcome(
            state,
            exit_code,
            has_completion,
        )
        valid_protocol = protocol_valid(logical, slurm_text, checkpoint_text)
        finite_metrics = metrics_are_finite(metrics)
        slurm_success = (
            state == "COMPLETED" and exit_code == "0:0"
        ) or state == "COMPLETED_MARKER"
        complete = (
            slurm_success
            and has_completion
            and valid_protocol
            and finite_metrics
            and error_count == 0
        )
        run_rows.append(
            {
                "task_id": task_id,
                "model": logical.model,
                "dataset": logical.dataset,
                "setting": logical.setting,
                "gpu_type": gpu_type,
                "job_task_id": job_task_id,
                "state": state,
                "exit_code": exit_code,
                "status_evidence": status_evidence,
                "protocol_valid": valid_protocol,
                "completion_marker": has_completion,
                "finite_metrics": finite_metrics,
                "error_count": error_count,
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
                    "model": logical.model,
                    "dataset": logical.dataset,
                    "setting": logical.setting,
                    "gpu_type": gpu_type,
                    "job_task_id": job_task_id,
                    "label": label,
                    "logloss": values.get("logloss", ""),
                    "AUC": values.get("AUC", ""),
                }
            )

    run_fields = list(run_rows[0])
    metric_fields = [
        "task_id",
        "model",
        "dataset",
        "setting",
        "gpu_type",
        "job_task_id",
        "label",
        "logloss",
        "AUC",
    ]
    write_csv(output_dir / "runs.csv", run_rows, run_fields)
    write_csv(output_dir / "metrics.csv", metric_rows, metric_fields)
    write_summary(output_dir / "summary.md", run_rows, metric_rows)

    complete_count = sum(bool(row["complete"]) for row in run_rows)
    print(f"strict_complete={complete_count}/32 output_dir={output_dir}")
    if complete_count != 32 and not args.allow_incomplete:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
