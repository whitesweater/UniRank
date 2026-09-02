#!/usr/bin/env python3
"""Build a canonical portable report artifact for a same-protocol SISA seed pair."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo


DATASET_LABELS = {
    "QK_Video_Action": "QK-Video",
    "KuaiRand_Video_Action": "KuaiRand",
    "TencentGR_10M_Action": "TencentGR",
    "MerRec_Action": "MerRec",
}
EXPECTED_PROTOCOL = "ws2_bs16384_acc1"


def report_paths(
    reference_seed: int,
    candidate_seed: int,
) -> tuple[Path, Path, Path, Path]:
    if reference_seed == candidate_seed:
        raise ValueError("reference and candidate seeds must differ")
    for name, seed in (
        ("reference seed", reference_seed),
        ("candidate seed", candidate_seed),
    ):
        if not 0 <= seed < 2**32 - 1:
            raise ValueError(f"{name} must be in [0, {2**32 - 2}], got {seed}")
    reference_study = Path(f"experiments/sisa_single_seed{reference_seed}")
    candidate_study = Path(f"experiments/sisa_single_seed{candidate_seed}")
    comparison = candidate_study / f"comparison_vs_seed{reference_seed}"
    return reference_study, candidate_study, comparison, comparison / "report"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def relative(repository: Path, path: Path) -> str:
    return path.resolve().relative_to(repository).as_posix()


def source(
    repository: Path,
    source_id: str,
    label: str,
    path: Path,
    *,
    sql: str | None = None,
    description: str | None = None,
    tables_used: Sequence[str] | None = None,
    filters: Sequence[str] = (),
    metric_definitions: Sequence[str] = (),
) -> dict[str, object]:
    source_path = relative(repository, path)
    if sql is None:
        reader = "read_json_auto" if path.suffix == ".json" else "read_csv_auto"
        sql = f"SELECT * FROM {reader}('{source_path}')"
    return {
        "id": source_id,
        "label": label,
        "path": source_path,
        "query": {
            "engine": "duckdb",
            "sql": sql,
            "description": description or label,
            "tables_used": list(tables_used) if tables_used is not None else [source_path],
            "filters": list(filters),
            "metric_definitions": list(metric_definitions),
        },
    }


def milli(value: object) -> float:
    return round(float(value) * 1000, 3)


def round6(value: object) -> float:
    return round(float(value), 6)


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def validate_inputs(
    runs: list[dict[str, str]],
    reference_metrics: list[dict[str, str]],
    candidate_metrics: list[dict[str, str]],
    reference_seed: int,
    candidate_seed: int,
) -> None:
    if len(runs) != 16:
        raise ValueError(f"candidate run audit must contain 16 rows, found {len(runs)}")
    if not all(truthy(row.get("complete")) for row in runs):
        failed = [row.get("task_id", "?") for row in runs if not truthy(row.get("complete"))]
        raise ValueError(f"candidate run audit is incomplete for tasks {failed}")
    if {int(row["seed"]) for row in runs} != {candidate_seed}:
        raise ValueError("candidate run audit contains a different experiment seed")
    if {row["protocol"] for row in runs} != {EXPECTED_PROTOCOL}:
        raise ValueError("candidate run audit contains a different protocol")
    if not all(truthy(row.get("h100_valid")) for row in runs):
        raise ValueError("candidate run audit contains a non-H100 allocation")

    for label, rows, seed in (
        ("reference metrics", reference_metrics, reference_seed),
        ("candidate metrics", candidate_metrics, candidate_seed),
    ):
        if len(rows) != 68:
            raise ValueError(f"{label} must contain 68 rows, found {len(rows)}")
        if {int(row["seed"]) for row in rows} != {seed}:
            raise ValueError(f"{label} contains a different experiment seed")
        if {row["protocol"] for row in rows} != {EXPECTED_PROTOCOL}:
            raise ValueError(f"{label} contains a different protocol")


def build_artifact(
    repository: Path,
    reference_seed: int,
    candidate_seed: int,
) -> tuple[dict[str, object], Path, str, str]:
    reference_rel, candidate_rel, comparison_rel, report_rel = report_paths(
        reference_seed,
        candidate_seed,
    )
    reference_study = repository / reference_rel
    candidate_study = repository / candidate_rel
    comparison = repository / comparison_rel
    report_dir = repository / report_rel

    reference_metrics_path = reference_study / "results" / "metrics.csv"
    candidate_metrics_path = candidate_study / "results" / "metrics.csv"
    candidate_runs_path = candidate_study / "results" / "runs.csv"
    headline_path = comparison / "headline.json"
    unit_path = comparison / "unit_summary.csv"
    model_path = comparison / "model_summary.csv"
    dataset_path = comparison / "dataset_summary.csv"
    label_path = comparison / "label_deltas.csv"
    ranking_path = comparison / "ranking_stability.csv"

    reference_metrics = read_csv(reference_metrics_path)
    candidate_metrics = read_csv(candidate_metrics_path)
    runs = read_csv(candidate_runs_path)
    validate_inputs(
        runs,
        reference_metrics,
        candidate_metrics,
        reference_seed,
        candidate_seed,
    )
    headline = read_json(headline_path)
    units_raw = read_csv(unit_path)
    models_raw = read_csv(model_path)
    datasets_raw = read_csv(dataset_path)
    labels_raw = read_csv(label_path)
    rankings_raw = read_csv(ranking_path)
    if len(units_raw) != 16 or len(labels_raw) != 68 or len(rankings_raw) != 17:
        raise ValueError("comparison outputs must contain 16 cells, 68 labels, and 17 rankings")

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    title = (
        "UniRank SISA 单一新 seed 复现实验："
        f"{candidate_seed} 对 {reference_seed} 同协议比较"
    )
    retry_tasks = sum(int(row["attempt"]) > 1 for row in runs)
    array_jobs = sorted({row["array_job"] for row in runs})

    headline_rows = [
        {
            "completed_tasks": 16,
            "retry_tasks": retry_tasks,
            "cell_count": int(headline["cell_count"]),
            "label_count": int(headline["label_count"]),
            "cell_macro_delta_auc_milli": milli(headline["mean_delta_auc_cell_macro"]),
            "positive_auc_cells": int(headline["positive_auc_cells"]),
            "cell_macro_delta_logloss_milli": milli(
                headline["mean_delta_logloss_cell_macro"]
            ),
            "same_best_model_tasks": int(headline["same_best_model_tasks"]),
            "ranking_tasks": int(headline["ranking_tasks"]),
        }
    ]

    unit_rows = [
        {
            "unit": f"{row['model']} · {DATASET_LABELS[row['dataset']]}",
            "model": row["model"],
            "dataset": DATASET_LABELS[row["dataset"]],
            "labels": int(row["labels"]),
            "delta_auc_milli": milli(row["mean_delta_auc"]),
            "delta_logloss_milli": milli(row["mean_delta_logloss"]),
            "auc_improved": f"{row['auc_improved']}/{row['labels']}",
            "logloss_improved": f"{row['logloss_improved']}/{row['labels']}",
        }
        for row in units_raw
    ]
    unit_rows.sort(key=lambda row: float(row["delta_auc_milli"]), reverse=True)

    model_rows = [
        {
            "model": row["model"],
            "cells": int(row["cells"]),
            "labels": int(row["labels"]),
            "cell_macro_delta_auc_milli": milli(row["unit_macro_delta_auc"]),
            "positive_auc_cells": f"{row['positive_auc_cells']}/{row['cells']}",
            "cell_macro_delta_logloss_milli": milli(row["unit_macro_delta_logloss"]),
            "lower_logloss_cells": f"{row['lower_logloss_cells']}/{row['cells']}",
        }
        for row in models_raw
    ]
    dataset_rows = [
        {
            "dataset": DATASET_LABELS[row["dataset"]],
            "cells": int(row["cells"]),
            "labels": int(row["labels"]),
            "cell_macro_delta_auc_milli": milli(row["unit_macro_delta_auc"]),
            "positive_auc_cells": f"{row['positive_auc_cells']}/{row['cells']}",
            "cell_macro_delta_logloss_milli": milli(row["unit_macro_delta_logloss"]),
            "lower_logloss_cells": f"{row['lower_logloss_cells']}/{row['cells']}",
        }
        for row in datasets_raw
    ]
    label_rows = [
        {
            "model": row["model"],
            "dataset": DATASET_LABELS[row["dataset"]],
            "label": row["label"],
            "reference_auc": round6(row["previous_auc"]),
            "candidate_auc": round6(row["new_auc"]),
            "delta_auc_milli": milli(row["delta_auc"]),
            "abs_delta_auc_milli": abs(milli(row["delta_auc"])),
            "reference_logloss": round6(row["previous_logloss"]),
            "candidate_logloss": round6(row["new_logloss"]),
            "delta_logloss_milli": milli(row["delta_logloss"]),
        }
        for row in labels_raw
    ]
    label_rows.sort(key=lambda row: float(row["abs_delta_auc_milli"]), reverse=True)
    ranking_rows = [
        {
            "dataset": DATASET_LABELS[row["dataset"]],
            "label": row["label"],
            "reference_best_model": row["previous_best_model"],
            "candidate_best_model": row["new_best_model"],
            "same_best_model": "是" if truthy(row["same_best_model"]) else "否",
        }
        for row in rankings_raw
    ]
    protocol_rows = [
        {
            "round": "参考轮",
            "experiment_seed": reference_seed,
            "world_size": 2,
            "gpus_per_task": 2,
            "gpu": "H100 80GB",
            "per_gpu_batch": 16384,
            "global_batch": 32768,
            "accumulation": 1,
        },
        {
            "round": "候选轮",
            "experiment_seed": candidate_seed,
            "world_size": 2,
            "gpus_per_task": 2,
            "gpu": "H100 80GB",
            "per_gpu_batch": 16384,
            "global_batch": 32768,
            "accumulation": 1,
        },
    ]

    report_builder_path = repository / "scripts" / "build_sisa_seed_pair_report.py"
    candidate_runs_rel = relative(repository, candidate_runs_path)
    headline_rel = relative(repository, headline_path)
    unit_rel = relative(repository, unit_path)
    model_rel = relative(repository, model_path)
    dataset_rel = relative(repository, dataset_path)
    label_rel = relative(repository, label_path)
    ranking_rel = relative(repository, ranking_path)
    dataset_case = (
        "CASE dataset "
        "WHEN 'QK_Video_Action' THEN 'QK-Video' "
        "WHEN 'KuaiRand_Video_Action' THEN 'KuaiRand' "
        "WHEN 'TencentGR_10M_Action' THEN 'TencentGR' "
        "WHEN 'MerRec_Action' THEN 'MerRec' END"
    )
    sources = [
        source(
            repository,
            "candidate_runs",
            "候选轮 16-task 完成审计",
            candidate_runs_path,
            sql=(
                "SELECT count(*) FILTER (WHERE complete) AS completed_tasks, "
                "count(*) FILTER (WHERE attempt > 1) AS retry_tasks "
                f"FROM read_csv_auto('{candidate_runs_rel}', header = true) "
                f"WHERE seed = {candidate_seed} AND protocol = '{EXPECTED_PROTOCOL}'"
            ),
            filters=(
                f"seed = {candidate_seed}",
                f"protocol = '{EXPECTED_PROTOCOL}'",
            ),
            metric_definitions=(
                "completed_tasks = count of rows whose complete field is true",
                "retry_tasks = count of completed logical tasks whose selected attempt is greater than 1",
            ),
        ),
        source(repository, "reference_metrics", "参考 seed 的 68 个最终测试指标", reference_metrics_path),
        source(repository, "candidate_metrics", "候选 seed 的 68 个最终测试指标", candidate_metrics_path),
        source(
            repository,
            "comparison_headline",
            "同协议 seed-pair 对比摘要",
            headline_path,
            sql=(
                "SELECT cell_count, label_count, "
                "1000 * mean_delta_auc_cell_macro AS cell_macro_delta_auc_milli, "
                "positive_auc_cells, "
                "1000 * mean_delta_logloss_cell_macro AS cell_macro_delta_logloss_milli, "
                "same_best_model_tasks, ranking_tasks "
                f"FROM read_json_auto('{headline_rel}')"
            ),
            metric_definitions=(
                "cell_macro_delta_auc_milli = 1000 times the equal-weight mean of the 16 cell-level mean AUC deltas",
                "cell_macro_delta_logloss_milli = 1000 times the equal-weight mean of the 16 cell-level mean logloss deltas",
                "same_best_model_tasks = dataset-label tasks whose top-AUC model is unchanged between the two seeds",
            ),
        ),
        source(
            repository,
            "unit_summary",
            "16 个模型×数据集单元差值",
            unit_path,
            sql=(
                f"SELECT model || ' · ' || {dataset_case} AS unit, model, "
                f"{dataset_case} AS dataset, labels, "
                "1000 * mean_delta_auc AS delta_auc_milli, "
                "1000 * mean_delta_logloss AS delta_logloss_milli, "
                "auc_improved || '/' || labels AS auc_improved, "
                "logloss_improved || '/' || labels AS logloss_improved "
                f"FROM read_csv_auto('{unit_rel}', header = true)"
            ),
            metric_definitions=(
                "delta_auc_milli = 1000 times the equal-weight mean label AUC delta within one model-dataset cell",
                "delta_logloss_milli = 1000 times the equal-weight mean label logloss delta within one model-dataset cell",
            ),
        ),
        source(
            repository,
            "model_summary",
            "模型维度汇总",
            model_path,
            sql=(
                "SELECT model, cells, labels, "
                "1000 * unit_macro_delta_auc AS cell_macro_delta_auc_milli, "
                "positive_auc_cells || '/' || cells AS positive_auc_cells, "
                "1000 * unit_macro_delta_logloss AS cell_macro_delta_logloss_milli, "
                "lower_logloss_cells || '/' || cells AS lower_logloss_cells "
                f"FROM read_csv_auto('{model_rel}', header = true)"
            ),
            metric_definitions=(
                "cell_macro deltas give each of the four dataset cells equal weight within a model",
            ),
        ),
        source(
            repository,
            "dataset_summary",
            "数据集维度汇总",
            dataset_path,
            sql=(
                f"SELECT {dataset_case} AS dataset, cells, labels, "
                "1000 * unit_macro_delta_auc AS cell_macro_delta_auc_milli, "
                "positive_auc_cells || '/' || cells AS positive_auc_cells, "
                "1000 * unit_macro_delta_logloss AS cell_macro_delta_logloss_milli, "
                "lower_logloss_cells || '/' || cells AS lower_logloss_cells "
                f"FROM read_csv_auto('{dataset_rel}', header = true)"
            ),
            metric_definitions=(
                "cell_macro deltas give each of the four model cells equal weight within a dataset",
            ),
        ),
        source(
            repository,
            "label_deltas",
            "68 个标签级差值",
            label_path,
            sql=(
                f"SELECT model, {dataset_case} AS dataset, label, "
                "previous_auc AS reference_auc, new_auc AS candidate_auc, "
                "1000 * delta_auc AS delta_auc_milli, "
                "abs(1000 * delta_auc) AS abs_delta_auc_milli, "
                "previous_logloss AS reference_logloss, new_logloss AS candidate_logloss, "
                "1000 * delta_logloss AS delta_logloss_milli "
                f"FROM read_csv_auto('{label_rel}', header = true)"
            ),
            metric_definitions=(
                "delta_auc = candidate seed AUC minus reference seed AUC",
                "delta_logloss = candidate seed logloss minus reference seed logloss",
            ),
        ),
        source(
            repository,
            "ranking_stability",
            "17 个任务的最优模型稳定性",
            ranking_path,
            sql=(
                f"SELECT {dataset_case} AS dataset, label, "
                "previous_best_model AS reference_best_model, "
                "new_best_model AS candidate_best_model, "
                "CASE WHEN same_best_model THEN '是' ELSE '否' END AS same_best_model "
                f"FROM read_csv_auto('{ranking_rel}', header = true)"
            ),
            metric_definitions=(
                "same_best_model is true when the top-AUC model matches between the reference and candidate seeds",
            ),
        ),
        source(
            repository,
            "comparison_code",
            "同协议 seed-pair 指标对齐与聚合代码",
            repository / "scripts" / "compare_sisa_seed_pair_results.py",
        ),
        source(
            repository,
            "report_builder",
            "报告快照构建代码",
            report_builder_path,
            sql=(
                "SELECT * FROM (VALUES "
                f"('参考轮', {reference_seed}, 2, 2, 'H100 80GB', 16384, 32768, 1), "
                f"('候选轮', {candidate_seed}, 2, 2, 'H100 80GB', 16384, 32768, 1)"
                ") AS protocol(round, experiment_seed, world_size, gpus_per_task, gpu, "
                "per_gpu_batch, global_batch, accumulation)"
            ),
            tables_used=(),
            metric_definitions=(
                "global_batch = world_size times per_gpu_batch times accumulation",
            ),
        ),
    ]

    cards = [
        {
            "id": "completed_tasks",
            "description": "正式 collector 审计通过的候选轮模型×数据集任务数。",
            "dataset": "headline",
            "sourceId": "candidate_runs",
            "metrics": [{"label": "完成任务", "field": "completed_tasks", "format": "number"}],
        },
        {
            "id": "retry_tasks",
            "description": "因训练后脚本错误而使用独立 attempt 路径补跑的任务数。",
            "dataset": "headline",
            "sourceId": "candidate_runs",
            "metrics": [{"label": "补跑任务", "field": "retry_tasks", "format": "number"}],
        },
        {
            "id": "delta_auc",
            "description": "每个 cell 内标签等权，再对 16 个 cells 等权；正值表示候选 seed 更高。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [
                {
                    "label": "Cell-macro 平均 ΔAUC（×10⁻³）",
                    "field": "cell_macro_delta_auc_milli",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "delta_logloss",
            "description": "候选 seed 减参考 seed；负值表示候选 seed 的 logloss 更低。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [
                {
                    "label": "Cell-macro 平均 ΔLogloss（×10⁻³）",
                    "field": "cell_macro_delta_logloss_milli",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "ranking_stability",
            "description": "参考轮和候选轮中 AUC 第一名模型保持相同的数据集×标签任务数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [
                {"label": "第一名保持不变", "field": "same_best_model_tasks", "format": "number"}
            ],
        },
    ]

    charts = [
        {
            "id": "unit_auc_deltas",
            "title": "16 个模型×数据集单元的平均 AUC 差值",
            "subtitle": f"seed {candidate_seed} 减 seed {reference_seed}；单位为 10⁻³",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "同协议下，哪些模型×数据集单元对 seed 变化最敏感？",
            "rationale": "排序横向条形图便于比较 16 个长标签单元及正负方向。",
            "dataset": "unit_deltas",
            "sourceId": "unit_summary",
            "palette": {"kind": "diverging", "midpoint": 0},
            "encodings": {
                "x": {"field": "unit", "type": "nominal", "aggregate": "none", "label": "模型 × 数据集"},
                "y": {"field": "delta_auc_milli", "type": "quantitative", "aggregate": "none", "label": "平均 ΔAUC（×10⁻³）"},
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "模型"},
                    {"field": "dataset", "type": "nominal", "label": "数据集"},
                    {"field": "delta_auc_milli", "type": "quantitative", "label": "平均 ΔAUC（×10⁻³）"},
                    {"field": "delta_logloss_milli", "type": "quantitative", "label": "平均 ΔLogloss（×10⁻³）"},
                ],
            },
            "xAxisTitle": "模型 × 数据集",
            "yAxisTitle": "平均 ΔAUC（×10⁻³）",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 16,
        }
    ]

    tables = [
        {
            "id": "unit_results",
            "title": "16 个单元的同协议 seed 差值",
            "subtitle": "每行先对该模型×数据集的全部标签等权；Δ 为候选 seed 减参考 seed",
            "dataset": "unit_deltas",
            "sourceId": "unit_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "delta_auc_milli", "label": "平均 ΔAUC（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "auc_improved", "label": "AUC 提升标签", "type": "text"},
                {"field": "delta_logloss_milli", "label": "平均 ΔLogloss（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "logloss_improved", "label": "Logloss 改善标签", "type": "text"},
            ],
        },
        {
            "id": "model_results",
            "title": "按模型汇总",
            "subtitle": "四个数据集等权；AUC 正值更高，logloss 负值更低",
            "dataset": "model_summary",
            "sourceId": "model_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "cell_macro_delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "cell_macro_delta_auc_milli", "label": "Cell-macro ΔAUC（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "positive_auc_cells", "label": "正向 cells", "type": "text"},
                {"field": "cell_macro_delta_logloss_milli", "label": "Cell-macro ΔLogloss（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "lower_logloss_cells", "label": "较低 Logloss cells", "type": "text"},
            ],
        },
        {
            "id": "dataset_results",
            "title": "按数据集汇总",
            "subtitle": "每个数据集覆盖四个模型；AUC 与 logloss 方向需分别解读",
            "dataset": "dataset_summary",
            "sourceId": "dataset_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "cell_macro_delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "cell_macro_delta_auc_milli", "label": "Cell-macro ΔAUC（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "positive_auc_cells", "label": "正向 cells", "type": "text"},
                {"field": "cell_macro_delta_logloss_milli", "label": "Cell-macro ΔLogloss（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "lower_logloss_cells", "label": "较低 Logloss cells", "type": "text"},
            ],
        },
        {
            "id": "largest_label_moves",
            "title": "标签级 AUC 差值绝对值最大的结果",
            "subtitle": "按 |ΔAUC| 排序；完整 68 行保留在报告快照中",
            "dataset": "label_deltas",
            "sourceId": "label_deltas",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "abs_delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "label", "label": "标签", "type": "text"},
                {"field": "delta_auc_milli", "label": "ΔAUC（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "abs_delta_auc_milli", "label": "|ΔAUC|（×10⁻³）", "type": "number", "format": "number"},
                {"field": "delta_logloss_milli", "label": "ΔLogloss（×10⁻³）", "type": "number", "format": "number", "movement": True},
            ],
        },
        {
            "id": "ranking_table",
            "title": "数据集×标签任务的最优模型稳定性",
            "subtitle": "比较 17 个任务中 AUC 第一名模型是否保持一致",
            "dataset": "ranking_stability",
            "sourceId": "ranking_stability",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "same_best_model", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "label", "label": "标签", "type": "text"},
                {"field": "reference_best_model", "label": "参考轮第一名", "type": "text"},
                {"field": "candidate_best_model", "label": "候选轮第一名", "type": "text"},
                {"field": "same_best_model", "label": "是否一致", "type": "text"},
            ],
        },
        {
            "id": "protocol_table",
            "title": "两轮训练协议对照",
            "subtitle": "除 experiment seed 及其派生随机流外，硬件与训练协议保持一致",
            "dataset": "protocol_comparison",
            "sourceId": "report_builder",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "experiment_seed", "direction": "asc"},
            "columns": [
                {"field": "round", "label": "轮次", "type": "text"},
                {"field": "experiment_seed", "label": "实验 seed", "type": "number", "format": "number"},
                {"field": "gpu", "label": "GPU", "type": "text"},
                {"field": "world_size", "label": "World size", "type": "number", "format": "number"},
                {"field": "per_gpu_batch", "label": "每卡 batch", "type": "number", "format": "number"},
                {"field": "global_batch", "label": "Global batch", "type": "number", "format": "number"},
            ],
        },
    ]

    best_unit = unit_rows[0]
    worst_unit = unit_rows[-1]
    best_label = headline["largest_auc_improvement"]
    worst_label = headline["largest_auc_decline"]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 技术摘要\n\n"
                f"这是一次只新增 **1 个实验 seed（{candidate_seed}）** 的 16-task 重跑，"
                f"并与同协议参考 seed {reference_seed} 对齐比较。68 个标签的 cell-macro "
                f"平均 ΔAUC 为 **{float(headline['mean_delta_auc_cell_macro']):+.6f}**，"
                f"平均 Δlogloss 为 **{float(headline['mean_delta_logloss_cell_macro']):+.6f}**；"
                f"AUC 第一名模型在 **{headline['same_best_model_tasks']}/{headline['ranking_tasks']}** "
                "个数据集×标签任务上保持一致。两轮各只有一个点估计，因此这些结果用于描述"
                "同协议重复性，不构成方差或显著性估计。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["completed_tasks", "retry_tasks", "delta_auc", "delta_logloss", "ranking_stability"],
        },
        {
            "id": "execution_result",
            "type": "markdown",
            "sourceId": "candidate_runs",
            "body": (
                "## 16 个正式结果均通过完成性审计\n\n"
                f"候选轮最终使用 Slurm array/job **{', '.join(array_jobs)}**。正式 collector 只接收 "
                f"`COMPLETED / 0:0`、协议与 seed 匹配、有限测试指标、完成标记和非空 checkpoint "
                f"同时成立的记录；共有 **{retry_tasks}** 个任务使用 attempt 2 补跑，失败 attempt "
                "保留在独立文件名中用于审计，但不进入正式指标。"
            ),
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 平均差值必须与胜负数量和排名稳定性一起解读\n\n"
                f"AUC 提升标签为 **{headline['auc_improved_labels']}/{headline['label_count']}**，"
                f"正向 cells 为 **{headline['positive_auc_cells']}/{headline['cell_count']}**；"
                f"logloss 改善标签为 **{headline['logloss_improved_labels']}/{headline['label_count']}**。"
                "均值可能受少数极值拉动，不能单独代表所有模型×数据集组合的稳定性。"
            ),
        },
        {"id": "unit_chart", "type": "chart", "chartId": "unit_auc_deltas"},
        {
            "id": "unit_interpretation",
            "type": "markdown",
            "sourceId": "unit_summary",
            "body": (
                "## 单元级差值呈现模型×数据集交互\n\n"
                f"最大正向单元是 **{best_unit['unit']}**（ΔAUC {float(best_unit['delta_auc_milli']) / 1000:+.6f}），"
                f"最大负向单元是 **{worst_unit['unit']}**（ΔAUC {float(worst_unit['delta_auc_milli']) / 1000:+.6f}）。"
                "条形图按单元展示正负方向，表格保留 AUC 与 logloss 的精确差值。"
            ),
        },
        {"id": "unit_table", "type": "table", "tableId": "unit_results"},
        {
            "id": "model_finding",
            "type": "markdown",
            "sourceId": "model_summary",
            "body": "## 模型汇总用于识别跨数据集的一致方向\n\n每个模型对四个数据集等权；该聚合避免标签数较多的数据集自动获得更高权重。",
        },
        {"id": "model_table", "type": "table", "tableId": "model_results"},
        {
            "id": "dataset_finding",
            "type": "markdown",
            "sourceId": "dataset_summary",
            "body": "## 数据集汇总用于识别跨模型的共同敏感性\n\n每个数据集对四个模型等权；AUC 与 logloss 方向分别报告，避免把排序能力变化误写成综合质量变化。",
        },
        {"id": "dataset_table", "type": "table", "tableId": "dataset_results"},
        {
            "id": "largest_moves",
            "type": "markdown",
            "sourceId": "label_deltas",
            "body": (
                "## 标签级极值界定平均数的解释边界\n\n"
                f"最大 AUC 提升为 **{best_label['model']}–{DATASET_LABELS[str(best_label['dataset'])]} "
                f"`{best_label['label']}`**（Δ {float(best_label['delta_auc']):+.6f}）；"
                f"最大回退为 **{worst_label['model']}–{DATASET_LABELS[str(worst_label['dataset'])]} "
                f"`{worst_label['label']}`**（Δ {float(worst_label['delta_auc']):+.6f}）。"
            ),
        },
        {"id": "largest_moves_table", "type": "table", "tableId": "largest_label_moves"},
        {
            "id": "ranking_finding",
            "type": "markdown",
            "sourceId": "ranking_stability",
            "body": (
                "## 模型第一名稳定性提供均值之外的视角\n\n"
                f"17 个数据集×标签任务中有 **{headline['same_best_model_tasks']}** 个保持同一 AUC 第一名模型。"
                "排名变化只说明两个点估计之间的顺序改变，不能解释改变概率。"
            ),
        },
        {"id": "ranking_table_block", "type": "table", "tableId": "ranking_table"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "comparison_code",
            "body": (
                "## 范围、指标与比较基线\n\n"
                "矩阵固定为 4 个模型 × 4 个数据集，共 16 个 cells、17 个数据集×标签任务和 "
                "68 个模型×数据集×标签指标。`ΔAUC = 候选 seed − 参考 seed`，正值更高；"
                "`Δlogloss = 候选 seed − 参考 seed`，负值更低。Cell-macro 先在每个 cell 内"
                "对标签等权，再对 16 个 cells 等权。"
            ),
        },
        {
            "id": "experimental_design",
            "type": "markdown",
            "sourceId": "report_builder",
            "body": (
                "## 实验设计只新增一个 experiment seed\n\n"
                f"参考轮 experiment seed 为 {reference_seed}，候选轮只新增 seed {candidate_seed}。"
                "每任务均为 2×H100 80GB、world size 2、每卡 batch 16384、accumulation 1、"
                "global batch 32768、1 epoch。Dataloader 与 SISA 初始化随机流随 experiment seed "
                "按既定规则推进，它们是同一次候选实验内部的随机子流，不是额外实验轮次。"
            ),
        },
        {"id": "protocol_table_block", "type": "table", "tableId": "protocol_table"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "comparison_code",
            "body": (
                "## 对齐、聚合与验证方法\n\n"
                "比较脚本要求两份 `metrics.csv` 各含 68 个唯一键、seed 与协议完全匹配，并对 "
                "`(model, dataset, label)` 做一一对齐；随后计算标签级差值、cell 均值、模型/数据集"
                "汇总和第一名稳定性。正式结果生成前另行校验 16 个任务的 Slurm 状态、完成标记、"
                "有限指标、H100 证据和 checkpoint。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限性、不确定性与稳健性边界\n\n"
                "这是两个同协议点估计的描述性比较，样本量不足以估计跨 seed 方差、置信区间或"
                "显著性。虽然 batch、world size、硬件和 drop-last 形状相同，随机数据顺序、SISA "
                "参数初始化和训练轨迹会按设计不同；任务可能也由不同 H100 节点执行。因此差值可"
                "用于定位 seed 敏感组合，但不能被解释为确定增益或退化。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 建议的结果使用方式\n\n"
                "1. 将 seed 20262028 与 20262029 的 checkpoints、运行审计和指标作为两个独立快照保存。\n"
                "2. 正式引用时同时报告 AUC、logloss、cell-macro 聚合方式和两轮差值，不只引用单个极值。\n"
                "3. 对标签级极值和第一名变化的任务做定向复核；在没有更多同协议 seed 前，不报告方差或显著性。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 后续研究问题\n\n"
                "- AUC 与 logloss 方向不一致的组合是否主要来自概率校准变化？\n"
                "- 标签级极值是否集中在低基率任务或特定模型结构？\n"
                "- 第一名变化是否发生在模型 AUC 本来就非常接近的任务上？"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": f"单一候选 experiment seed {candidate_seed} 与同协议参考 seed {reference_seed} 的完成审计和重复性比较。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": headline_rows,
            "unit_deltas": unit_rows,
            "model_summary": model_rows,
            "dataset_summary": dataset_rows,
            "label_deltas": label_rows,
            "ranking_stability": ranking_rows,
            "protocol_comparison": protocol_rows,
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
    chart_map = (
        "# Chart map\n\n"
        "| Section | Question | Family / type | Fields | Claim | Palette |\n"
        "|---|---|---|---|---|---|\n"
        "| 单元级差值 | 哪些模型×数据集单元对 seed 最敏感？ | Comparison / horizontalBar | unit, delta_auc_milli | 显示 16 cells 的方向与幅度 | diverging, midpoint 0 |\n"
    )
    source_notes = (
        "# Report source notes\n\n"
        "- Audience: technical.\n"
        "- Delivery mode: portable HTML from the canonical artifact contract.\n"
        "- Required structure mapping: title; technical summary; visual findings; scope and definitions; experimental design; methodology; limitations; next steps; further questions.\n"
        "- Comparison basis: two completed studies with protocol ws2_bs16384_acc1.\n"
        "- Statistical boundary: descriptive two-point comparison only; no variance or significance estimate.\n"
        "- The only quantitative visual is a 16-category horizontal bar because the primary visual question is category comparison; exact lookup remains in tables.\n"
    )
    return artifact, report_dir, chart_map, source_notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--reference-seed", type=int, required=True)
    parser.add_argument("--candidate-seed", type=int, required=True)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    try:
        artifact, report_dir, chart_map, source_notes = build_artifact(
            repository,
            args.reference_seed,
            args.candidate_seed,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "chart_map.md").write_text(chart_map, encoding="utf-8")
    (report_dir / "source_notes.md").write_text(source_notes, encoding="utf-8")
    print(f"WROTE {report_dir / 'artifact.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
