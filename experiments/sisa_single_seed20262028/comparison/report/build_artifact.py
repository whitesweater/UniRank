#!/usr/bin/env python3
"""Build the canonical portable report artifact for the SISA repeatability run."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPORT_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = REPORT_DIR.parent
STUDY_DIR = COMPARISON_DIR.parent
PROJECT_ROOT = STUDY_DIR.parents[1]

DATASET_LABELS = {
    "QK_Video_Action": "QK-Video",
    "KuaiRand_Video_Action": "KuaiRand",
    "TencentGR_10M_Action": "TencentGR",
    "MerRec_Action": "MerRec",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def source(source_id: str, label: str, path: Path) -> dict[str, object]:
    return {"id": source_id, "label": label, "path": relative(path)}


def sql_source(
    source_id: str,
    label: str,
    path: Path,
    table_name: str,
    description: str,
    generated_at: str,
) -> dict[str, object]:
    value = source(source_id, label, path)
    value["query"] = {
        "engine": "sqlite",
        "language": "sql",
        "sql": f"SELECT * FROM {table_name}",
        "description": description,
        "executed_at": generated_at,
        "tables_used": [table_name],
    }
    return value


def materialize_with_sqlite(
    table_name: str, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Use the same SQL exposed in source metadata to produce snapshot rows."""
    if not rows:
        raise ValueError(f"cannot materialize empty report dataset: {table_name}")
    columns = list(rows[0])
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            f'CREATE TABLE "{table_name}" ({quoted_columns})'
        )
        connection.executemany(
            f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
            [[row[column] for column in columns] for row in rows],
        )
        result = connection.execute(f"SELECT * FROM {table_name}").fetchall()
    finally:
        connection.close()
    return [dict(row) for row in result]


def round6(value: object) -> float:
    return round(float(value), 6)


def milli(value: object) -> float:
    return round(float(value) * 1000, 3)


def build_artifact() -> dict[str, object]:
    headline_path = COMPARISON_DIR / "headline.json"
    unit_path = COMPARISON_DIR / "unit_summary.csv"
    model_path = COMPARISON_DIR / "model_summary.csv"
    dataset_path = COMPARISON_DIR / "dataset_summary.csv"
    label_path = COMPARISON_DIR / "label_deltas.csv"
    ranking_path = COMPARISON_DIR / "ranking_stability.csv"
    protocol_audit_path = STUDY_DIR / "results" / "protocol_audit.md"
    new_runs_path = STUDY_DIR / "results" / "runs.csv"
    new_metrics_path = STUDY_DIR / "results" / "metrics.csv"
    strict_runs_path = PROJECT_ROOT / "experiments/sisa_native_strict/results/runs.csv"
    strict_metrics_path = PROJECT_ROOT / "experiments/sisa_native_strict/results/metrics.csv"
    expansion_runs_path = PROJECT_ROOT / "experiments/sisa_expansion_acd/results/runs.csv"
    expansion_metrics_path = PROJECT_ROOT / "experiments/sisa_expansion_acd/results/metrics.csv"
    comparison_script_path = PROJECT_ROOT / "scripts/compare_sisa_single_seed_results.py"
    artifact_builder_path = REPORT_DIR / "build_artifact.py"

    headline = read_json(headline_path)
    units_raw = read_csv(unit_path)
    models_raw = read_csv(model_path)
    datasets_raw = read_csv(dataset_path)
    labels_raw = read_csv(label_path)
    rankings_raw = read_csv(ranking_path)
    new_runs = read_csv(new_runs_path)

    if len(new_runs) != 16 or not all(row["complete"] == "True" for row in new_runs):
        raise ValueError("report requires 16/16 complete new runs")
    if len(labels_raw) != 68 or len(units_raw) != 16:
        raise ValueError("report requires 68 aligned labels across 16 cells")

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    title = "UniRank SISA 单种子复现实验：2×H100 完成审计与旧结果对比"

    sources = [
        source("new_runs", "本轮 16-task 运行审计", new_runs_path),
        source("new_metrics", "本轮 68 个最终测试指标", new_metrics_path),
        sql_source(
            "protocol_audit",
            "本轮协议、资源与样本覆盖审计",
            protocol_audit_path,
            "report_coverage",
            "读取由协议审计复核的样本覆盖快照；完成卡片使用同一审计来源。",
            generated_at,
        ),
        source("strict_runs", "旧 strict 运行审计", strict_runs_path),
        source("strict_metrics", "旧 strict SISA 指标", strict_metrics_path),
        source("expansion_runs", "旧 expansion 运行审计", expansion_runs_path),
        source("expansion_metrics", "旧 expansion SISA 指标", expansion_metrics_path),
        source("comparison_script", "旧新指标选择、对齐与聚合代码", comparison_script_path),
        sql_source(
            "artifact_builder",
            "报告快照构建代码",
            artifact_builder_path,
            "report_protocol_comparison",
            "读取报告构建器依据旧新 run 审计整理的协议对照行。",
            generated_at,
        ),
        sql_source(
            "comparison_headline",
            "旧新对比摘要",
            headline_path,
            "report_headline",
            "读取经 68 个唯一标签键对齐后生成的 headline 指标。",
            generated_at,
        ),
        sql_source(
            "label_deltas",
            "68 个标签级旧新差值",
            label_path,
            "report_label_deltas",
            "读取 68 个模型×数据集×标签的旧新 AUC 与 logloss 差值。",
            generated_at,
        ),
        sql_source(
            "unit_summary",
            "16 个模型×数据集单元汇总",
            unit_path,
            "report_unit_deltas",
            "读取标签内等权后得到的 16 个模型×数据集单元差值。",
            generated_at,
        ),
        sql_source(
            "model_summary",
            "模型维度汇总",
            model_path,
            "report_model_summary",
            "读取按模型汇总的 cell-macro 与标签等权差值。",
            generated_at,
        ),
        sql_source(
            "dataset_summary",
            "数据集维度汇总",
            dataset_path,
            "report_dataset_summary",
            "读取按数据集汇总的 cell-macro 差值。",
            generated_at,
        ),
        sql_source(
            "ranking_stability",
            "17 个数据集×标签任务的最优模型稳定性",
            ranking_path,
            "report_ranking_stability",
            "读取旧新两轮每个数据集×标签任务的 AUC 第一名模型。",
            generated_at,
        ),
    ]

    headline_rows = [
        {
            "completed_tasks": 16,
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

    unit_rows = []
    for row in units_raw:
        dataset = DATASET_LABELS[row["dataset"]]
        unit_rows.append(
            {
                "unit": f"{row['model']} · {dataset}",
                "model": row["model"],
                "dataset": dataset,
                "labels": int(row["labels"]),
                "delta_auc_milli": milli(row["mean_delta_auc"]),
                "delta_logloss_milli": milli(row["mean_delta_logloss"]),
                "auc_improved": f"{row['auc_improved']}/{row['labels']}",
                "logloss_improved": f"{row['logloss_improved']}/{row['labels']}",
                "joint_improved": f"{row['joint_improved']}/{row['labels']}",
                "previous_source": row["previous_source"],
            }
        )
    unit_rows.sort(key=lambda row: float(row["delta_auc_milli"]), reverse=True)

    model_rows = []
    for row in models_raw:
        model_rows.append(
            {
                "model": row["model"],
                "cells": int(row["cells"]),
                "labels": int(row["labels"]),
                "cell_macro_delta_auc_milli": milli(row["unit_macro_delta_auc"]),
                "positive_auc_cells": f"{row['positive_auc_cells']}/{row['cells']}",
                "label_weighted_delta_auc_milli": milli(row["label_weighted_delta_auc"]),
                "cell_macro_delta_logloss_milli": milli(
                    row["unit_macro_delta_logloss"]
                ),
                "lower_logloss_cells": f"{row['lower_logloss_cells']}/{row['cells']}",
            }
        )

    dataset_rows = []
    for row in datasets_raw:
        dataset_rows.append(
            {
                "dataset": DATASET_LABELS[row["dataset"]],
                "cells": int(row["cells"]),
                "labels": int(row["labels"]),
                "cell_macro_delta_auc_milli": milli(row["unit_macro_delta_auc"]),
                "positive_auc_cells": f"{row['positive_auc_cells']}/{row['cells']}",
                "cell_macro_delta_logloss_milli": milli(
                    row["unit_macro_delta_logloss"]
                ),
                "lower_logloss_cells": f"{row['lower_logloss_cells']}/{row['cells']}",
            }
        )

    label_rows = []
    for row in labels_raw:
        label_rows.append(
            {
                "model": row["model"],
                "dataset": DATASET_LABELS[row["dataset"]],
                "label": row["label"],
                "previous_auc": round6(row["previous_auc"]),
                "new_auc": round6(row["new_auc"]),
                "delta_auc_milli": milli(row["delta_auc"]),
                "abs_delta_auc_milli": abs(milli(row["delta_auc"])),
                "previous_logloss": round6(row["previous_logloss"]),
                "new_logloss": round6(row["new_logloss"]),
                "delta_logloss_milli": milli(row["delta_logloss"]),
                "auc_direction": row["auc_direction"],
                "logloss_direction": row["logloss_direction"],
                "previous_source": row["previous_source"],
            }
        )
    label_rows.sort(key=lambda row: abs(float(row["delta_auc_milli"])), reverse=True)

    ranking_rows = []
    for row in rankings_raw:
        ranking_rows.append(
            {
                "dataset": DATASET_LABELS[row["dataset"]],
                "label": row["label"],
                "previous_best_model": row["previous_best_model"],
                "previous_best_auc": round6(row["previous_best_auc"]),
                "new_best_model": row["new_best_model"],
                "new_best_auc": round6(row["new_best_auc"]),
                "same_best_model": row["same_best_model"],
            }
        )

    protocol_rows = [
        {
            "round": "旧结果（seed 20262027）",
            "cells": 16,
            "base_seed": 20262027,
            "dataloader_seed": "2026（归档代码默认值）",
            "sisa_parameter_seed": "20260821（归档代码默认值）",
            "gpus_per_task": 4,
            "hardware": "7 cells H100；7 cells L40S；2 cells RTX 4090",
            "world_size": 4,
            "per_gpu_batch": 8192,
            "global_batch": 32768,
            "accumulation": 1,
            "epochs": 1,
        },
        {
            "round": "本轮（seed 20262028）",
            "cells": 16,
            "base_seed": 20262028,
            "dataloader_seed": "2027（日志显式记录）",
            "sisa_parameter_seed": "20260822（日志显式记录）",
            "gpus_per_task": 2,
            "hardware": "16 cells 均为 H100 80GB",
            "world_size": 2,
            "per_gpu_batch": 16384,
            "global_batch": 32768,
            "accumulation": 1,
            "epochs": 1,
        },
    ]

    coverage_rows = [
        {"dataset": "MerRec", "split": "train", "assigned_rows": 144108235, "previous_processed": 144072704, "previous_coverage_percent": 99.975344, "new_processed": 144048128, "new_coverage_percent": 99.958290, "added_dropped_rows": 24576, "coverage_delta_pp": -0.017054},
        {"dataset": "MerRec", "split": "valid", "assigned_rows": 16515570, "previous_processed": 16482304, "previous_coverage_percent": 99.798578, "new_processed": 16449536, "new_coverage_percent": 99.600171, "added_dropped_rows": 32768, "coverage_delta_pp": -0.198407},
        {"dataset": "MerRec", "split": "test", "assigned_rows": 11681154, "previous_processed": 11640832, "previous_coverage_percent": 99.654812, "new_processed": 11599872, "new_coverage_percent": 99.304161, "added_dropped_rows": 40960, "coverage_delta_pp": -0.350650},
        {"dataset": "KuaiRand", "split": "train", "assigned_rows": 256110233, "previous_processed": 255959040, "previous_coverage_percent": 99.940966, "new_processed": 255836160, "new_coverage_percent": 99.892986, "added_dropped_rows": 122880, "coverage_delta_pp": -0.047979},
        {"dataset": "KuaiRand", "split": "valid", "assigned_rows": 34334869, "previous_processed": 34308096, "previous_coverage_percent": 99.922024, "new_processed": 34258944, "new_coverage_percent": 99.778869, "added_dropped_rows": 49152, "coverage_delta_pp": -0.143155},
        {"dataset": "KuaiRand", "split": "test", "assigned_rows": 33019342, "previous_processed": 32980992, "previous_coverage_percent": 99.883856, "new_processed": 32964608, "new_coverage_percent": 99.834237, "added_dropped_rows": 16384, "coverage_delta_pp": -0.049619},
        {"dataset": "QK-Video", "split": "train", "assigned_rows": 394647318, "previous_processed": 394526720, "previous_coverage_percent": 99.969442, "new_processed": 394428416, "new_coverage_percent": 99.944532, "added_dropped_rows": 98304, "coverage_delta_pp": -0.024909},
        {"dataset": "QK-Video", "split": "valid", "assigned_rows": 50079027, "previous_processed": 50053120, "previous_coverage_percent": 99.948268, "new_processed": 50036736, "new_coverage_percent": 99.915551, "added_dropped_rows": 16384, "coverage_delta_pp": -0.032716},
        {"dataset": "QK-Video", "split": "test", "assigned_rows": 48579958, "previous_processed": 48562176, "previous_coverage_percent": 99.963396, "new_processed": 48529408, "new_coverage_percent": 99.895945, "added_dropped_rows": 32768, "coverage_delta_pp": -0.067452},
        {"dataset": "TencentGR", "split": "train", "assigned_rows": 586117494, "previous_processed": 586047488, "previous_coverage_percent": 99.988056, "new_processed": 585957376, "new_coverage_percent": 99.972682, "added_dropped_rows": 90112, "coverage_delta_pp": -0.015374},
        {"dataset": "TencentGR", "split": "valid", "assigned_rows": 17846140, "previous_processed": 17809408, "previous_coverage_percent": 99.794174, "new_processed": 17793024, "new_coverage_percent": 99.702367, "added_dropped_rows": 16384, "coverage_delta_pp": -0.091807},
        {"dataset": "TencentGR", "split": "test", "assigned_rows": 153243512, "previous_processed": 153206784, "previous_coverage_percent": 99.976033, "new_processed": 153157632, "new_coverage_percent": 99.943958, "added_dropped_rows": 49152, "coverage_delta_pp": -0.032074},
    ]

    cards = [
        {
            "id": "completed_tasks",
            "description": "本轮 16 个模型×数据集任务均以首次尝试完成，且无 OOM、NCCL、NaN 或 traceback。",
            "dataset": "headline",
            "sourceId": "protocol_audit",
            "metrics": [
                {"label": "完成任务", "field": "completed_tasks", "format": "number"}
            ],
        },
        {
            "id": "cell_macro_auc",
            "description": "先在每个模型×数据集单元内对标签等权，再对 16 个单元等权；正值表示本轮 AUC 更高。",
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
            "id": "positive_cells",
            "description": "16 个模型×数据集单元中，平均 AUC 高于旧结果的单元数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [
                {"label": "AUC 为正的 cells", "field": "positive_auc_cells", "format": "number"}
            ],
        },
        {
            "id": "cell_macro_logloss",
            "description": "新结果减旧结果；logloss 越低越好，因此正值表示总体变差。",
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
            "description": "17 个数据集×标签任务中，旧新两轮 AUC 第一名模型相同的任务数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [
                {"label": "最优模型保持不变", "field": "same_best_model_tasks", "format": "number"}
            ],
        },
    ]

    charts = [
        {
            "id": "unit_auc_deltas",
            "title": "16 个模型×数据集单元的平均 AUC 变化",
            "subtitle": "本轮减旧结果；正值表示本轮更高，9/16 个单元为正",
            "type": "horizontalBar",
            "dataset": "unit_deltas",
            "sourceId": "unit_summary",
            "encodings": {
                "x": {
                    "field": "unit",
                    "type": "nominal",
                    "aggregate": "none",
                    "label": "模型 × 数据集",
                },
                "y": {
                    "field": "delta_auc_milli",
                    "type": "quantitative",
                    "aggregate": "none",
                    "label": "平均 ΔAUC（×10⁻³）",
                },
                "tooltip": [
                    {"field": "model", "type": "nominal", "label": "模型"},
                    {"field": "dataset", "type": "nominal", "label": "数据集"},
                    {
                        "field": "delta_auc_milli",
                        "type": "quantitative",
                        "label": "平均 ΔAUC（×10⁻³）",
                    },
                    {"field": "auc_improved", "type": "nominal", "label": "AUC 提升标签"},
                    {
                        "field": "delta_logloss_milli",
                        "type": "quantitative",
                        "label": "平均 ΔLogloss（×10⁻³）",
                    },
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
            "title": "16 个单元的旧新差值",
            "subtitle": "每行先对该模型×数据集的全部标签等权；Δ 均为本轮减旧结果",
            "dataset": "unit_deltas",
            "sourceId": "unit_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {
                    "field": "delta_auc_milli",
                    "label": "平均 ΔAUC（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {"field": "auc_improved", "label": "AUC 提升标签", "type": "text"},
                {
                    "field": "delta_logloss_milli",
                    "label": "平均 ΔLogloss（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {"field": "logloss_improved", "label": "Logloss 改善标签", "type": "text"},
            ],
        },
        {
            "id": "model_results",
            "title": "按模型汇总",
            "subtitle": "Cell-macro 对四个数据集等权；标签等权列对每个模型的 17 个标签等权",
            "dataset": "model_summary",
            "sourceId": "model_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "cell_macro_delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {
                    "field": "cell_macro_delta_auc_milli",
                    "label": "Cell-macro ΔAUC（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {"field": "positive_auc_cells", "label": "正向 cells", "type": "text"},
                {
                    "field": "label_weighted_delta_auc_milli",
                    "label": "标签等权 ΔAUC（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "cell_macro_delta_logloss_milli",
                    "label": "Cell-macro ΔLogloss（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
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
                {"field": "labels", "label": "标签数", "type": "number", "format": "number"},
                {
                    "field": "cell_macro_delta_auc_milli",
                    "label": "Cell-macro ΔAUC（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {"field": "positive_auc_cells", "label": "正向 cells", "type": "text"},
                {
                    "field": "cell_macro_delta_logloss_milli",
                    "label": "Cell-macro ΔLogloss（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {"field": "lower_logloss_cells", "label": "较低 Logloss cells", "type": "text"},
            ],
        },
        {
            "id": "largest_label_moves",
            "title": "标签级 AUC 变化幅度最大的结果",
            "subtitle": "按 |ΔAUC| 从大到小显示；完整 68 行均保留在数据快照中",
            "dataset": "label_deltas",
            "sourceId": "label_deltas",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "abs_delta_auc_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "label", "label": "标签", "type": "text"},
                {
                    "field": "delta_auc_milli",
                    "label": "ΔAUC（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "abs_delta_auc_milli",
                    "label": "|ΔAUC|（×10⁻³）",
                    "type": "number",
                    "format": "number",
                },
                {
                    "field": "delta_logloss_milli",
                    "label": "ΔLogloss（×10⁻³）",
                    "type": "number",
                    "format": "number",
                    "movement": True,
                },
            ],
        },
        {
            "id": "protocol_comparison",
            "title": "旧结果与本轮 seed / 硬件对照",
            "subtitle": "Seed bundle、每任务 GPU 数和硬件组成同时改变",
            "dataset": "protocol_comparison",
            "sourceId": "artifact_builder",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "base_seed", "direction": "asc"},
            "columns": [
                {"field": "round", "label": "轮次", "type": "text"},
                {"field": "base_seed", "label": "Base seed", "type": "number", "format": "number"},
                {"field": "dataloader_seed", "label": "Dataloader seed", "type": "text"},
                {"field": "sisa_parameter_seed", "label": "SISA 参数 seed", "type": "text"},
                {"field": "gpus_per_task", "label": "每任务 GPU", "type": "number", "format": "number"},
                {"field": "hardware", "label": "硬件", "type": "text"},
            ],
        },
        {
            "id": "batch_protocol_comparison",
            "title": "旧结果与本轮 DDP / batch 对照",
            "subtitle": "名义 global batch、epoch 与 accumulation 不变，但 world size 与每卡 batch 改变",
            "dataset": "protocol_comparison",
            "sourceId": "artifact_builder",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "base_seed", "direction": "asc"},
            "columns": [
                {"field": "round", "label": "轮次", "type": "text"},
                {"field": "base_seed", "label": "Base seed", "type": "number", "format": "number"},
                {"field": "world_size", "label": "World size", "type": "number", "format": "number"},
                {"field": "per_gpu_batch", "label": "每卡 batch", "type": "number", "format": "number"},
                {"field": "global_batch", "label": "Global batch", "type": "number", "format": "number"},
                {"field": "accumulation", "label": "Accumulation", "type": "number", "format": "number"},
            ],
        },
        {
            "id": "coverage_audit",
            "title": "旧新 blocked dataloader 样本覆盖对照",
            "subtitle": "12/12 个 split 的分配行数相同，但 2×16384 协议处理的行数均更少",
            "dataset": "coverage",
            "sourceId": "protocol_audit",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "coverage_delta_pp", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "split", "label": "Split", "type": "text"},
                {"field": "previous_coverage_percent", "label": "旧覆盖率（%）", "type": "number", "format": "number"},
                {"field": "new_coverage_percent", "label": "本轮覆盖率（%）", "type": "number", "format": "number"},
                {"field": "added_dropped_rows", "label": "新增丢弃行数", "type": "number", "format": "number"},
                {"field": "coverage_delta_pp", "label": "覆盖率 Δ（pp）", "type": "number", "format": "number", "movement": True},
            ],
        },
        {
            "id": "ranking_table",
            "title": "数据集×标签任务的最优模型稳定性",
            "subtitle": "17 个任务中 11 个保持同一 AUC 第一名模型",
            "dataset": "ranking_stability",
            "sourceId": "ranking_stability",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "same_best_model", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "label", "label": "标签", "type": "text"},
                {"field": "previous_best_model", "label": "旧第一名", "type": "text"},
                {"field": "new_best_model", "label": "本轮第一名", "type": "text"},
                {"field": "same_best_model", "label": "是否一致", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": "## 技术摘要\n\n本轮与旧 SISA 结果精确对齐 **68 个标签、16 个模型×数据集单元**。Cell-macro 平均 ΔAUC 为 **+0.001619**，其中 **9/16** 个单元为正；但 cell-macro 平均 Δlogloss 为 **+0.001001**，只有 **4/16** 个单元的 logloss 更低。AUC 的点估计略向上，而校准损失与标签胜率并不支持‘整体一致改善’这一表述。",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "completed_tasks",
                "cell_macro_auc",
                "positive_cells",
                "cell_macro_logloss",
                "ranking_stability",
            ],
        },
        {
            "id": "execution_result",
            "type": "markdown",
            "sourceId": "protocol_audit",
            "body": "## 16/16 个 2×H100 任务首次尝试完成\n\nSlurm array `548166` 从 2026-08-25 21:01:04 运行至 2026-08-26 00:48:19，墙钟 **03:47:15**。16 个任务全部为 `COMPLETED / 0:0`，没有重试，也没有触发 4 卡 OOM fallback；错误审计未发现 CUDA OOM、NCCL、NaN、traceback 或 killed process。峰值采样显存为 **71,847 MiB**，低于 H100 的 81,559 MiB 采样容量。",
        },
        {
            "id": "key_finding_auc",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": "## AUC 均值小幅上移，但不是多数标签普遍提升\n\n标签等权平均 ΔAUC 为 **+0.000833**，但标签胜负为 **32 升 / 36 降**；中位数 ΔAUC 接近零（−0.000072）。**48/68** 个标签的变化落在 ±0.005 内，说明总体均值由少数较大正向变化拉高，不能只用平均值概括稳定性。",
        },
        {"id": "unit_chart", "type": "chart", "chartId": "unit_auc_deltas"},
        {
            "id": "unit_chart_interpretation",
            "type": "markdown",
            "sourceId": "unit_summary",
            "body": "## 单元级变化集中在 TencentGR 与少数模型组合\n\n最大正向单元是 Zenith–TencentGR（+0.014694），其次是 HyFormer–TencentGR（+0.011973）；最大回退是 RankMixer–TencentGR（−0.007284）和 HyFormer–MerRec（−0.005258）。同一数据集内方向并不统一，说明这些点估计更像模型×数据集交互，而非新协议带来的统一增益。",
        },
        {"id": "unit_table", "type": "table", "tableId": "unit_results"},
        {
            "id": "key_finding_loss",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": "## Logloss 稳定性弱于 AUC\n\n标签等权平均 Δlogloss 为 **+0.000812**（正值更差），只有 **23/68** 个标签改善，45 个变差；同时 AUC 与 logloss 都改善的标签为 19 个，而两者都退化的标签为 32 个。因此结果解读应同时保留排序能力与校准损失，不能把 AUC 的正均值外推为综合质量提升。",
        },
        {
            "id": "dataset_finding",
            "type": "markdown",
            "sourceId": "dataset_summary",
            "body": "## 数据集层面存在明显分化\n\nTencentGR 的 cell-macro ΔAUC 最大（+0.005861），但平均 Δlogloss 同时变差 +0.002259；MerRec 的 ΔAUC 为 +0.002035。KuaiRand 与 QK-Video 的平均 ΔAUC 分别为 −0.001028 和 −0.000392。",
        },
        {"id": "dataset_table", "type": "table", "tableId": "dataset_results"},
        {
            "id": "model_finding",
            "type": "markdown",
            "sourceId": "model_summary",
            "body": "## 模型结论依赖聚合权重\n\n按四个数据集等权，HiFormer 的平均 ΔAUC 为 +0.003345，Zenith 为 +0.002334，HyFormer 为 +0.001215，RankMixer 为 −0.000418。按 17 个标签等权时，HyFormer、RankMixer、Zenith 的符号与幅度会改变，因此正式引用必须注明采用 cell-macro 还是标签等权。",
        },
        {"id": "model_table", "type": "table", "tableId": "model_results"},
        {
            "id": "largest_moves_finding",
            "type": "markdown",
            "sourceId": "label_deltas",
            "body": "## 极值说明均值受少数标签驱动\n\n最大 AUC 提升为 Zenith–TencentGR `is_conversion`，从 0.866268 升至 0.888088（Δ +0.021820）；最大回退为 HyFormer–MerRec `Offer`，从 0.745556 降至 0.731796（Δ −0.013760）。这两个极值都远大于标签级中位变化。",
        },
        {"id": "largest_moves_table", "type": "table", "tableId": "largest_label_moves"},
        {
            "id": "ranking_finding",
            "type": "markdown",
            "sourceId": "ranking_stability",
            "body": "## 最优模型在 11/17 个数据集×标签任务上保持不变\n\n六个任务的 AUC 第一名发生变化，主要集中在 KuaiRand、MerRec 与 TencentGR；QK-Video 的四个标签第一名全部保持不变。排名稳定性提供了均值之外的另一个视角，但仍只有两轮点估计，不能据此估计模型排序概率。",
        },
        {"id": "ranking_detail", "type": "table", "tableId": "ranking_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "comparison_script",
            "body": "## 范围、数据与指标定义\n\n比较矩阵固定为 HiFormer、HyFormer、RankMixer、Zenith × QK-Video、KuaiRand、TencentGR、MerRec，共 16 个单元、17 个数据集×标签任务和 68 个模型×数据集×标签指标。`ΔAUC = 本轮 − 旧结果`，正值更好；`Δlogloss = 本轮 − 旧结果`，负值更好。标签等权对 68 行直接求均值；cell-macro 先在单元内对标签等权，再对 16 个单元等权。",
        },
        {
            "id": "experimental_design",
            "type": "markdown",
            "sourceId": "comparison_script",
            "body": "## 实验设计与旧结果选择\n\n旧结果不是 `benchmark/` baseline，而是两个 finalized SISA collector 的无重叠并集：`sisa_native_strict` 提供 9 cells / 45 labels，`sisa_expansion_acd` 提供 7 cells / 23 labels。旧新 68 个 `(model, dataset, label)` 键一一对齐，无重复或缺失。",
        },
        {"id": "protocol_table", "type": "table", "tableId": "protocol_comparison"},
        {"id": "batch_protocol_table", "type": "table", "tableId": "batch_protocol_comparison"},
        {
            "id": "coverage_heading",
            "type": "markdown",
            "sourceId": "protocol_audit",
            "body": "## 旧新测试指标并非基于完全相同的 drop-last 样本子集\n\n12/12 个 train/valid/test split 的分配前行数一致，但 2-rank、per-GPU batch 16384 改变了 blocked dataloader 的 `drop_last` 边界，使 12/12 个 split 的实际处理行数减少。最大覆盖率差异是 MerRec test：**99.654812% → 99.304161%（−0.350650 pp）**。因此旧新最终测试指标不只是训练轨迹不同，评价样本子集也不完全相同。",
        },
        {"id": "coverage_table", "type": "table", "tableId": "coverage_audit"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "comparison_script",
            "body": "## 汇总与验证方法\n\n比较脚本从三个最终 `metrics.csv` 中选择旧 SISA 子集，校验键唯一且完全相等，验证数值有限，再计算标签级差值、单元级均值、模型/数据集汇总和最优模型稳定性。任务完成性、GPU、seed、batch、world size、错误信号与最终指标由 run collector 和协议审计独立检查。",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## 局限性、不确定性与稳健性边界\n\n这不是纯 seed 实验：base seed、dataloader seed、SISA 参数 seed、world size、每卡 batch、硬件组成和 `drop_last` 样本边界同时改变；MerRec 还统一采用了回归验证过的低峰值 Adagrad 兼容路径。旧结果中 7 cells 使用 H100、7 cells 使用 L40S、2 cells 使用 RTX 4090，本轮 16 cells 全部使用 H100。两轮都只有一个点估计，没有方差、置信区间或统计显著性；因此本报告只能描述探索性重复性，不能把差值归因于随机种子或 2 卡协议。",
        },
        {
            "id": "recommended_next_steps",
            "type": "markdown",
            "body": "## 建议下一步\n\n1. 将本轮结果作为 2×H100、batch 16384 协议的完成快照，不覆盖旧 4 卡结果。\n2. 后续同协议新增 seed 时，固定硬件、world size、每卡 batch、dataloader seed 生成规则与 optimizer 路径，再计算方差和置信区间。\n3. 优先复核变化最大的 Zenith–TencentGR、RankMixer–TencentGR 与 HyFormer–MerRec，以区分偶然波动和可重复的模型×数据集交互。\n4. 继续保留 2 卡为默认申请；仅对真实 OOM 任务切回 4 卡，并把 fallback 结果单独标记为不同协议。",
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": "## 后续研究问题\n\n- 腾讯数据上 AUC 增益与 logloss 退化并存，是否来自分数尺度或概率校准变化？\n- Zenith–TencentGR 的大幅正向变化能否在相同 2 卡协议下复现？\n- HyFormer–MerRec `Offer` 与 RankMixer–TencentGR 的回退是 seed 敏感，还是更大单卡 batch 的交互？\n- 若增加相同协议的第三个点估计，模型排序稳定率和 cell-macro 方差会如何变化？",
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "16 个 SISA 单种子任务的 2×H100 完成审计、与旧 seed/protocol 结果的探索性对比及可比性边界。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
    }

    datasets = {
        "headline": materialize_with_sqlite("report_headline", headline_rows),
        "unit_deltas": materialize_with_sqlite("report_unit_deltas", unit_rows),
        "model_summary": materialize_with_sqlite("report_model_summary", model_rows),
        "dataset_summary": materialize_with_sqlite(
            "report_dataset_summary", dataset_rows
        ),
        "label_deltas": materialize_with_sqlite(
            "report_label_deltas", label_rows
        ),
        "ranking_stability": materialize_with_sqlite(
            "report_ranking_stability", ranking_rows
        ),
        "protocol_comparison": materialize_with_sqlite(
            "report_protocol_comparison", protocol_rows
        ),
        "coverage": materialize_with_sqlite("report_coverage", coverage_rows),
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": datasets,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def main() -> None:
    artifact = build_artifact()
    output_path = REPORT_DIR / "artifact.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
