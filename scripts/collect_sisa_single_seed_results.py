#!/usr/bin/env python3
"""Collect and validate one 16-task SISA single-seed study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sisa_single_seed_tasks import EXPERIMENTS, SingleSeedTask, single_seed_task


SEED = 20262028
DATALOADER_SEED = 2027
SISA_PARAMETER_SEED = 20260822
COMPLETION_MARKER = "SISA_SINGLE_SEED_COMPLETE"
TEST_MARKER = "******** Test evaluation ********"
OUTPUT_DIRECTORY = Path("experiments/sisa_single_seed20262028/results")

HARDWARE_PATTERN = re.compile(
    r"SISA_SINGLE_SEED_HARDWARE gpu_count=(\d+) gpu_name=(.*?) host=(\S+)"
)
TASK_METRIC_PATTERN = re.compile(r"\[Task:\s*([^\]]+)\]\[Metrics\]\s*(.*)")
METRIC_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class Protocol:
    name: str
    world_size: int
    per_gpu_batch: int

    @property
    def global_batch(self) -> int:
        return self.world_size * self.per_gpu_batch


PRIMARY_PROTOCOL = Protocol("ws2_bs16384_acc1", 2, 16384)
OOM_FALLBACK_PROTOCOL = Protocol("ws4_bs8192_acc1_oom_fallback", 4, 8192)
PROTOCOLS = {
    protocol.name: protocol
    for protocol in (PRIMARY_PROTOCOL, OOM_FALLBACK_PROTOCOL)
}


@dataclass(frozen=True)
class RunAssignment:
    task_id: int
    array_job: int
    protocol: Protocol
    attempt: int

    @property
    def job_task_id(self) -> str:
        return f"{self.array_job}_{self.task_id}"


def parse_run_overrides(values: Sequence[str]) -> dict[int, RunAssignment]:
    """Parse repeatable TASK=JOB:PROTOCOL:ATTEMPT retry selections."""
    overrides: dict[int, RunAssignment] = {}
    for value in values:
        try:
            task_text, run_text = value.split("=", 1)
            job_text, protocol_name, attempt_text = run_text.split(":")
            task_id = int(task_text)
            array_job = int(job_text)
            attempt = int(attempt_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid run override {value!r}; expected "
                "TASK=JOB:PROTOCOL:ATTEMPT"
            ) from error

        single_seed_task(task_id)
        if array_job <= 0:
            raise ValueError(f"array job id must be positive, got {array_job}")
        if protocol_name not in PROTOCOLS:
            supported = ", ".join(sorted(PROTOCOLS))
            raise ValueError(
                f"unsupported protocol {protocol_name!r}; expected one of {supported}"
            )
        if attempt <= 0:
            raise ValueError(f"attempt must be positive, got {attempt}")
        if task_id in overrides:
            raise ValueError(f"duplicate run override for task {task_id}")
        overrides[task_id] = RunAssignment(
            task_id=task_id,
            array_job=array_job,
            protocol=PROTOCOLS[protocol_name],
            attempt=attempt,
        )
    return overrides


def run_id(
    task: SingleSeedTask,
    assignment: RunAssignment,
    *,
    seed: int = SEED,
) -> str:
    return (
        f"SISA_single_seed_{task.model}_{task.dataset}_sisa_seed{seed}_"
        f"{assignment.protocol.name}_attempt{assignment.attempt}"
    )


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_slurm_log(
    repository: Path,
    assignment: RunAssignment,
    *,
    seed: int = SEED,
) -> Path:
    """Resolve either the standard two-GPU log or a manually named retry log."""
    log_directory = repository / "logs"
    job_name = (
        f"unirank-sisa-seed{seed}-ws2"
        if assignment.protocol == PRIMARY_PROTOCOL
        else f"unirank-sisa-seed{seed}-oom4"
    )
    preferred = (
        log_directory
        / f"{job_name}-{assignment.array_job}_{assignment.task_id}.out"
    )
    if preferred.is_file():
        return preferred

    candidates = sorted(
        log_directory.glob(
            f"*-{assignment.array_job}_{assignment.task_id}.out"
        )
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        relative = ", ".join(
            str(candidate.relative_to(repository)) for candidate in candidates
        )
        raise RuntimeError(
            f"ambiguous Slurm logs for {assignment.job_task_id}: {relative}"
        )
    return preferred


def scheduler_outcome(job_task_id: str) -> tuple[str, str, str]:
    """Return normalized state and exit code from Slurm accounting."""
    try:
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
    except OSError:
        return "UNKNOWN", "UNKNOWN", "unavailable"

    for line in accounting.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2 or not fields[0]:
            continue
        state = fields[0].split("+", 1)[0].split(maxsplit=1)[0]
        return state, fields[1] or "UNKNOWN", "sacct"
    return "UNKNOWN", "UNKNOWN", "sacct_missing"


def effective_scheduler_outcome(
    state: str,
    exit_code: str,
    source: str,
    has_completion_marker: bool,
) -> tuple[str, str, str]:
    """Use the durable success marker only if Slurm has purged the job record."""
    if state == "UNKNOWN" and has_completion_marker:
        return "COMPLETED_MARKER", "0:0 (success marker)", "success_marker"
    return state, exit_code, source


def parse_test_metrics(checkpoint_text: str) -> dict[str, dict[str, float]]:
    """Parse metrics only from the final test section, never validation output."""
    if TEST_MARKER not in checkpoint_text:
        return {}
    test_section = checkpoint_text.rsplit(TEST_MARKER, 1)[1]
    metrics_by_label: dict[str, dict[str, float]] = {}
    for line in test_section.splitlines():
        task_match = TASK_METRIC_PATTERN.search(line)
        if task_match is None:
            continue
        metrics = {
            name: float(value)
            for name, value in METRIC_PATTERN.findall(task_match.group(2))
        }
        if metrics:
            metrics_by_label[task_match.group(1).strip()] = metrics
    return metrics_by_label


def metrics_valid(metrics: dict[str, dict[str, float]]) -> bool:
    return bool(metrics) and all(
        {"AUC", "logloss"}.issubset(values)
        and math.isfinite(values["AUC"])
        and math.isfinite(values["logloss"])
        for values in metrics.values()
    )


def protocol_valid(
    task: SingleSeedTask,
    assignment: RunAssignment,
    slurm_text: str,
    checkpoint_text: str,
    *,
    seed: int = SEED,
    dataloader_seed: int = DATALOADER_SEED,
    sisa_parameter_seed: int = SISA_PARAMETER_SEED,
) -> bool:
    protocol = assignment.protocol
    expected_run_id = run_id(task, assignment, seed=seed)
    required_slurm = (
        f"SISA_SINGLE_SEED_PROTOCOL protocol={protocol.name} "
        f"world_size={protocol.world_size} "
        f"per_gpu_batch={protocol.per_gpu_batch} "
        f"global_batch={protocol.global_batch} accumulation_steps=1 "
        f"epochs=1 seed={seed} dataloader_seed={dataloader_seed} "
        f"sisa_parameter_seed={sisa_parameter_seed} bf16=true",
        f"task_id={task.task_id} ",
        f"job_id={assignment.job_task_id} ",
        f"experiment={task.experiment} setting=sisa "
        f"study=sisa_single_seed{seed} run_id={expected_run_id} ",
        f"attempt={assignment.attempt} ",
    )
    required_checkpoint = (
        f'"batch_size": "{protocol.per_gpu_batch}"',
        '"accumulation_steps": "1"',
        '"epochs": "1"',
        f'"seed": "{seed}"',
        f'"dataloader_seed": "{dataloader_seed}"',
        f'"sisa_parameter_seed": "{sisa_parameter_seed}"',
        f'"world_size": "{protocol.world_size}"',
        f'"model": "{task.model}"',
        f'"dataset_id": "{task.dataset}"',
        f'"model_id": "{expected_run_id}"',
        '"max_len": "100"',
        '"enable_bf16": "True"',
        '"sisa_enabled": "True"',
        '"sisa_score_dim": "16"',
        '"sisa_lambda_init": "0.1"',
        '"sisa_score_scale": "1.0"',
        "DDP blocked training: True",
    )
    return all(value in slurm_text for value in required_slurm) and all(
        value in checkpoint_text for value in required_checkpoint
    )


def classify_errors(
    slurm_text: str,
    checkpoint_text: str,
    scheduler_state: str,
) -> list[str]:
    """Return stable, actionable failure classes in priority order."""
    combined = f"{slurm_text}\n{checkpoint_text}"
    classes: list[str] = []

    def add(name: str, pattern: str) -> None:
        if re.search(pattern, combined, flags=re.IGNORECASE | re.MULTILINE):
            classes.append(name)

    if scheduler_state == "OUT_OF_MEMORY":
        classes.append("oom")
    else:
        add("oom", r"CUDA out of memory|OutOfMemoryError|oom-kill")
    add(
        "nccl",
        r"NCCL (?:error|failure|WARN)|"
        r"nccl(?:System|Internal|UnhandledCuda|Remote)Error|"
        r"ProcessGroupNCCL[^\n]*(?:error|failed|watchdog|timed?\s*out)",
    )
    if scheduler_state == "TIMEOUT":
        classes.append("timeout")
    else:
        add("timeout", r"DUE TO TIME LIMIT|time limit exceeded")
    if scheduler_state in {"NODE_FAIL", "BOOT_FAIL"}:
        classes.append("node_failure")
    if scheduler_state == "PREEMPTED":
        classes.append("preempted")
    if scheduler_state == "CANCELLED":
        classes.append("cancelled")
    add("killed", r"(?:^|\s)Killed(?:\s|$)|SIGKILL|Segmentation fault")
    add(
        "numerical",
        r"(?:^|[^A-Za-z0-9_])(?:nan|inf)(?:[^A-Za-z0-9_]|$)",
    )
    add("traceback", r"^Traceback \(most recent call last\):")

    failed_states = {
        "FAILED",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
    }
    if scheduler_state in failed_states and not classes:
        classes.append("scheduler_failure")
    return list(dict.fromkeys(classes))


def relative_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def collect_results(
    repository: Path,
    main_job: int,
    overrides: dict[int, RunAssignment],
    scheduler_lookup: Callable[[str], tuple[str, str, str]] | None = None,
    *,
    seed: int = SEED,
    dataloader_seed: int = DATALOADER_SEED,
    sisa_parameter_seed: int = SISA_PARAMETER_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if main_job <= 0:
        raise ValueError(f"main job id must be positive, got {main_job}")
    if len(EXPERIMENTS) != 16:
        raise RuntimeError(
            f"single-seed task mapping changed: expected 16, got {len(EXPERIMENTS)}"
        )
    if scheduler_lookup is None:
        scheduler_lookup = scheduler_outcome

    run_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for task_id in range(len(EXPERIMENTS)):
        task = single_seed_task(task_id)
        assignment = overrides.get(
            task_id,
            RunAssignment(task_id, main_job, PRIMARY_PROTOCOL, 1),
        )
        expected_run_id = run_id(task, assignment, seed=seed)
        slurm_log = resolve_slurm_log(
            repository,
            assignment,
            seed=seed,
        )
        checkpoint_log = (
            repository
            / "checkpoints"
            / task.dataset
            / f"{expected_run_id}.log"
        )
        slurm_text = read_text(slurm_log)
        checkpoint_text = read_text(checkpoint_log)
        metrics = parse_test_metrics(checkpoint_text)
        has_completion = (
            f"{COMPLETION_MARKER} task_id={task_id} "
            f"run_id={expected_run_id} protocol={assignment.protocol.name}"
            in slurm_text
        )
        state, exit_code, state_source = scheduler_lookup(
            assignment.job_task_id
        )
        state, exit_code, state_source = effective_scheduler_outcome(
            state,
            exit_code,
            state_source,
            has_completion,
        )
        hardware_match = HARDWARE_PATTERN.search(slurm_text)
        gpu_count = int(hardware_match.group(1)) if hardware_match else 0
        gpu_name = hardware_match.group(2) if hardware_match else ""
        host = hardware_match.group(3) if hardware_match else ""
        h100_valid = (
            gpu_count == assignment.protocol.world_size
            and "H100" in gpu_name.upper()
        )
        valid_protocol = protocol_valid(
            task,
            assignment,
            slurm_text,
            checkpoint_text,
            seed=seed,
            dataloader_seed=dataloader_seed,
            sisa_parameter_seed=sisa_parameter_seed,
        )
        final_metrics_valid = metrics_valid(metrics)
        error_classes = classify_errors(slurm_text, checkpoint_text, state)
        scheduler_success = (
            state == "COMPLETED" and exit_code.startswith("0:0")
        ) or state == "COMPLETED_MARKER"

        incomplete_reasons: list[str] = []
        if not scheduler_success:
            incomplete_reasons.append("scheduler_not_completed")
        if not slurm_text:
            incomplete_reasons.append("slurm_log_missing")
        if not checkpoint_text:
            incomplete_reasons.append("checkpoint_log_missing")
        if not has_completion:
            incomplete_reasons.append("completion_marker_missing")
        if not valid_protocol:
            incomplete_reasons.append("protocol_invalid")
        if not h100_valid:
            incomplete_reasons.append("hardware_invalid")
        if not final_metrics_valid:
            incomplete_reasons.append("final_metrics_invalid")
        incomplete_reasons.extend(f"error:{name}" for name in error_classes)
        complete = not incomplete_reasons

        run_rows.append(
            {
                "task_id": task_id,
                "model": task.model,
                "dataset": task.dataset,
                "setting": "sisa",
                "seed": seed,
                "dataloader_seed": dataloader_seed,
                "sisa_parameter_seed": sisa_parameter_seed,
                "protocol": assignment.protocol.name,
                "attempt": assignment.attempt,
                "array_job": assignment.array_job,
                "job_task_id": assignment.job_task_id,
                "state": state,
                "exit_code": exit_code,
                "state_source": state_source,
                "world_size": assignment.protocol.world_size,
                "per_gpu_batch": assignment.protocol.per_gpu_batch,
                "global_batch": assignment.protocol.global_batch,
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
                "host": host,
                "h100_valid": h100_valid,
                "protocol_valid": valid_protocol,
                "completion_marker": has_completion,
                "finite_test_metrics": final_metrics_valid,
                "error_class": error_classes[0] if error_classes else "none",
                "error_classes": ";".join(error_classes),
                "error_count": len(error_classes),
                "incomplete_reasons": ";".join(incomplete_reasons),
                "complete": complete,
                "slurm_log": relative_path(slurm_log, repository),
                "checkpoint_log": relative_path(checkpoint_log, repository),
                "metrics_json": json.dumps(metrics, sort_keys=True),
            }
        )
        for label, values in sorted(metrics.items()):
            metric_rows.append(
                {
                    "task_id": task_id,
                    "model": task.model,
                    "dataset": task.dataset,
                    "setting": "sisa",
                    "seed": seed,
                    "protocol": assignment.protocol.name,
                    "attempt": assignment.attempt,
                    "job_task_id": assignment.job_task_id,
                    "gpu_name": gpu_name,
                    "label": label,
                    "AUC": values.get("AUC", ""),
                    "logloss": values.get("logloss", ""),
                }
            )
    return run_rows, metric_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    run_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    *,
    seed: int = SEED,
    dataloader_seed: int = DATALOADER_SEED,
    sisa_parameter_seed: int = SISA_PARAMETER_SEED,
) -> None:
    def format_metric(value: object) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return ""
        return f"{numeric_value:.6f}" if math.isfinite(numeric_value) else str(value)

    completed = sum(bool(row["complete"]) for row in run_rows)
    protocol_counts = Counter(str(row["protocol"]) for row in run_rows)
    error_counts = Counter(
        str(error)
        for row in run_rows
        for error in str(row["error_classes"]).split(";")
        if error
    )
    lines = [
        f"# SISA single-seed {seed} result snapshot",
        "",
        f"- Complete tasks: **{completed}/16**",
        f"- Final label metrics: **{len(metric_rows)}**",
        f"- Experiment seed: **`{seed}`** (one experiment seed only)",
        f"- Internal RNG substreams: dataloader `{dataloader_seed}`, "
        f"SISA parameters `{sisa_parameter_seed}`; these are not additional experiment seeds",
        "- Protocol assignments: "
        + ", ".join(
            f"`{name}`={count}" for name, count in sorted(protocol_counts.items())
        ),
        "- Error classes: "
        + (
            ", ".join(
                f"`{name}`={count}" for name, count in sorted(error_counts.items())
            )
            if error_counts
            else "none"
        ),
        "",
        "## Runs",
        "",
        "| Task | Model | Dataset | Protocol | Attempt | Job | State | GPU | "
        "Complete | Error / incomplete evidence |",
        "|---:|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in run_rows:
        evidence = str(row["incomplete_reasons"]) or "none"
        lines.append(
            f"| {row['task_id']} | {row['model']} | {row['dataset']} | "
            f"{row['protocol']} | {row['attempt']} | {row['job_task_id']} | "
            f"{row['state']} | {row['gpu_name'] or 'unknown'} | "
            f"{row['complete']} | {evidence} |"
        )

    lines.extend(
        [
            "",
            "## Final test metrics",
            "",
            "| Task | Model | Dataset | Label | AUC | Logloss | Protocol |",
            "|---:|---|---|---|---:|---:|---|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['task_id']} | {row['model']} | {row['dataset']} | "
            f"{row['label']} | {format_metric(row['AUC'])} | "
            f"{format_metric(row['logloss'])} | {row['protocol']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(
    output_directory: Path,
    run_rows: list[dict[str, object]],
    metric_rows: list[dict[str, object]],
    *,
    seed: int = SEED,
    dataloader_seed: int = DATALOADER_SEED,
    sisa_parameter_seed: int = SISA_PARAMETER_SEED,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / "runs.csv", run_rows, list(run_rows[0]))
    metric_fields = [
        "task_id",
        "model",
        "dataset",
        "setting",
        "seed",
        "protocol",
        "attempt",
        "job_task_id",
        "gpu_name",
        "label",
        "AUC",
        "logloss",
    ]
    write_csv(output_directory / "metrics.csv", metric_rows, metric_fields)
    write_summary(
        output_directory / "summary.md",
        run_rows,
        metric_rows,
        seed=seed,
        dataloader_seed=dataloader_seed,
        sisa_parameter_seed=sisa_parameter_seed,
    )


def validate_seed(value: int, name: str) -> int:
    if not 0 <= value < 2**32 - 1:
        raise ValueError(f"{name} must be in [0, {2**32 - 2}], got {value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--main-job", type=int, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataloader-seed", type=int, default=DATALOADER_SEED)
    parser.add_argument(
        "--sisa-parameter-seed",
        type=int,
        default=SISA_PARAMETER_SEED,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Result directory relative to the repository; defaults to the seed-specific study path.",
    )
    parser.add_argument(
        "--run-override",
        action="append",
        default=[],
        metavar="TASK=JOB:PROTOCOL:ATTEMPT",
        help="Select the successful retry for one logical task; may be repeated.",
    )
    args = parser.parse_args(argv)
    try:
        seed = validate_seed(args.seed, "seed")
        dataloader_seed = validate_seed(args.dataloader_seed, "dataloader seed")
        sisa_parameter_seed = validate_seed(
            args.sisa_parameter_seed,
            "SISA parameter seed",
        )
        expected_output = Path(f"experiments/sisa_single_seed{seed}/results")
        relative_output = args.output_directory or expected_output
        if relative_output.is_absolute() or ".." in relative_output.parts:
            raise ValueError(
                "--output-directory must be a repository-relative path without '..'"
            )
        if relative_output != expected_output:
            raise ValueError(
                "--output-directory must match the seed-specific path: "
                f"{expected_output}"
            )
        overrides = parse_run_overrides(args.run_override)
        repository = args.repository.resolve()
        run_rows, metric_rows = collect_results(
            repository,
            args.main_job,
            overrides,
            seed=seed,
            dataloader_seed=dataloader_seed,
            sisa_parameter_seed=sisa_parameter_seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    output_directory = repository / relative_output
    write_results(
        output_directory,
        run_rows,
        metric_rows,
        seed=seed,
        dataloader_seed=dataloader_seed,
        sisa_parameter_seed=sisa_parameter_seed,
    )
    complete_count = sum(bool(row["complete"]) for row in run_rows)
    print(
        f"SISA_SINGLE_SEED_COLLECT complete={complete_count}/16 "
        f"metrics={len(metric_rows)} output_dir={output_directory}"
    )
    return 0 if complete_count == 16 else 1


if __name__ == "__main__":
    raise SystemExit(main())
