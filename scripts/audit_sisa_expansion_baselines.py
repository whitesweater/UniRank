#!/usr/bin/env python3
"""Compare SISA expansion baselines against the official UniRank logs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from scripts.collect_sisa_native_strict_results import parse_test_metrics


EXPERIMENTS = (
    "OneTrans_TencentGR_10M_Action",
    "HiFormer_TencentGR_10M_Action",
    "RankMixer_TencentGR_10M_Action",
    "Zenith_TencentGR_10M_Action",
    "UniMixer_QK_Video_Action",
    "UniMixer_KuaiRand_Video_Action",
    "UniMixer_TencentGR_10M_Action",
    "UniMixer_Taobao_Action",
    "UniMixer_MerRec_Action",
    "HyFormer_QK_Video_Action",
    "HyFormer_KuaiRand_Video_Action",
    "HyFormer_TencentGR_10M_Action",
    "HyFormer_Taobao_Action",
    "HyFormer_MerRec_Action",
    "UltraHSTU_QK_Video_Action",
    "UltraHSTU_KuaiRand_Video_Action",
    "UltraHSTU_TencentGR_10M_Action",
    "UltraHSTU_Taobao_Action",
    "UltraHSTU_MerRec_Action",
)


@dataclass(frozen=True)
class BaselineTask:
    task_id: int
    model: str
    dataset: str


def baseline_task(task_id: int) -> BaselineTask:
    if task_id < 0 or task_id >= 38 or task_id % 2:
        raise ValueError(f"task {task_id} is not an expansion baseline task")
    experiment = EXPERIMENTS[task_id // 2]
    model, dataset = experiment.split("_", 1)
    return BaselineTask(task_id=task_id, model=model, dataset=dataset)


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    return parse_test_metrics(path.read_text(encoding="utf-8", errors="replace"))


def compare_baseline(
    repository: Path,
    task: BaselineTask,
    auc_tolerance: float,
) -> tuple[list[dict[str, object]], list[str]]:
    official_log = (
        repository
        / "benchmark"
        / task.dataset
        / f"{task.model}_{task.dataset}.log"
    )
    local_log = (
        repository
        / "checkpoints"
        / task.dataset
        / (
            f"SISA_expansion_{task.model}_{task.dataset}_"
            "baseline_seed20262027.log"
        )
    )
    official = read_metrics(official_log)
    local = read_metrics(local_log)
    errors: list[str] = []
    rows: list[dict[str, object]] = []

    if not official:
        errors.append(f"task {task.task_id}: missing official metrics: {official_log}")
    if not local:
        errors.append(f"task {task.task_id}: missing local metrics: {local_log}")
    if errors:
        return rows, errors

    if set(official) != set(local):
        errors.append(
            f"task {task.task_id}: label mismatch official={sorted(official)} "
            f"local={sorted(local)}"
        )

    for label in sorted(set(official) & set(local)):
        official_auc = official[label].get("AUC")
        local_auc = local[label].get("AUC")
        if official_auc is None or local_auc is None:
            errors.append(f"task {task.task_id} label {label}: missing AUC")
            continue
        delta = local_auc - official_auc
        within_tolerance = math.isfinite(delta) and abs(delta) <= auc_tolerance
        rows.append(
            {
                "task_id": task.task_id,
                "model": task.model,
                "dataset": task.dataset,
                "label": label,
                "official_auc": official_auc,
                "local_auc": local_auc,
                "delta_auc": delta,
                "abs_delta_auc": abs(delta),
                "auc_tolerance": auc_tolerance,
                "within_tolerance": within_tolerance,
                "official_log": official_log.relative_to(repository),
                "local_log": local_log.relative_to(repository),
            }
        )
        if not within_tolerance:
            errors.append(
                f"task {task.task_id} {task.model}/{task.dataset}/{label}: "
                f"AUC delta {delta:+.6f} exceeds {auc_tolerance:.6f}"
            )
    return rows, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", type=int, action="append", default=[])
    parser.add_argument("--auc-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/sisa_expansion_acd/results/baseline_audit.csv"
        ),
    )
    args = parser.parse_args()
    if args.auc_tolerance < 0:
        raise SystemExit("--auc-tolerance must be non-negative")

    repository = args.repository.resolve()
    task_ids = args.task_id or list(range(0, 38, 2))
    tasks = [baseline_task(task_id) for task_id in task_ids]

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for task in tasks:
        task_rows, task_errors = compare_baseline(
            repository,
            task,
            args.auc_tolerance,
        )
        rows.extend(task_rows)
        errors.extend(task_errors)

    output = args.output
    if not output.is_absolute():
        output = repository / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "model",
        "dataset",
        "label",
        "official_auc",
        "local_auc",
        "delta_auc",
        "abs_delta_auc",
        "auc_tolerance",
        "within_tolerance",
        "official_log",
        "local_log",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    max_abs_delta = max((float(row["abs_delta_auc"]) for row in rows), default=math.nan)
    passed = len(errors) == 0
    print(
        f"SISA_EXPANSION_BASELINE_AUDIT passed={passed} tasks={len(tasks)} "
        f"labels={len(rows)} tolerance={args.auc_tolerance:.6f} "
        f"max_abs_delta={max_abs_delta:.6f} output={output}"
    )
    if errors:
        for error in errors:
            print(f"BASELINE_AUDIT_ERROR {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
