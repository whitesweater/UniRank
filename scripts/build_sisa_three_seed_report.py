#!/usr/bin/env python3
"""Build the canonical portable report for paper/baseline/three-seed SISA results."""

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
    "TencentGR_10M_Action": "TAAC-25 / TencentGR",
    "MerRec_Action": "MerRec",
}
MODEL_ORDER = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
SEEDS = (20262027, 20262028, 20262029)
STUDY = Path("experiments/sisa_three_seed_unified")


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
    url: str | None = None,
) -> dict[str, object]:
    source_path = relative(repository, path)
    if sql is None:
        reader = "read_json_auto" if path.suffix == ".json" else "read_csv_auto"
        sql = f"SELECT * FROM {reader}('{source_path}')"
    query: dict[str, object] = {
        "engine": "duckdb",
        "sql": sql,
        "description": description or label,
        "tables_used": list(tables_used) if tables_used is not None else [source_path],
        "filters": list(filters),
        "metric_definitions": list(metric_definitions),
    }
    if url is not None:
        query["url"] = url
    return {"id": source_id, "label": label, "path": source_path, "query": query}


def milli(value: object) -> float:
    return round(float(value) * 1000, 3)


def round6(value: object) -> float:
    return round(float(value), 6)


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def validate_inputs(
    headline: dict[str, object],
    cells: Sequence[dict[str, str]],
    labels: Sequence[dict[str, str]],
    protocols: Sequence[dict[str, str]],
) -> None:
    if headline.get("experiment_seeds") != list(SEEDS):
        raise ValueError("headline must contain the three experiment seeds 20262027/28/29")
    if int(headline["cell_count"]) != 16 or len(cells) != 16:
        raise ValueError("unified comparison must contain 16 cells")
    if int(headline["label_count"]) != 68 or len(labels) != 68:
        raise ValueError("unified comparison must contain 68 labels")
    if len(protocols) != 5:
        raise ValueError("protocol comparison must contain paper, baseline, and three SISA rows")
    cell_keys = {(row["model"], row["dataset"]) for row in cells}
    label_keys = {(row["model"], row["dataset"], row["label"]) for row in labels}
    if len(cell_keys) != 16 or len(label_keys) != 68:
        raise ValueError("unified comparison contains duplicate semantic keys")


def build_artifact(
    repository: Path,
) -> tuple[dict[str, object], Path, str, str]:
    study_dir = repository / STUDY
    result_dir = study_dir / "results"
    report_dir = study_dir / "report"
    paper_path = study_dir / "sources" / "paper_table.csv"
    headline_path = result_dir / "headline.json"
    cell_path = result_dir / "cell_summary.csv"
    label_path = result_dir / "label_comparison.csv"
    model_path = result_dir / "model_summary.csv"
    dataset_path = result_dir / "dataset_summary.csv"
    protocol_path = result_dir / "protocols.csv"

    headline = read_json(headline_path)
    cells_raw = read_csv(cell_path)
    labels_raw = read_csv(label_path)
    models_raw = read_csv(model_path)
    datasets_raw = read_csv(dataset_path)
    protocols_raw = read_csv(protocol_path)
    validate_inputs(headline, cells_raw, labels_raw, protocols_raw)

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    title = "UniRank SISA 三-seed 统一对比报告"

    headline_rows = [
        {
            "sisa_seed_count": 3,
            "cell_count": int(headline["cell_count"]),
            "label_count": int(headline["label_count"]),
            "delta_auc_vs_baseline_milli": milli(
                headline["cell_macro_delta_auc_vs_baseline"]
            ),
            "positive_auc_cells_vs_baseline": int(
                headline["positive_auc_cells_vs_baseline"]
            ),
            "delta_auc_vs_paper_milli": milli(headline["cell_macro_delta_auc_vs_paper"]),
            "delta_logloss_vs_baseline_milli": milli(
                headline["cell_macro_delta_logloss_vs_baseline"]
            ),
            "auc_labels_all_seeds_better_baseline": int(
                headline["auc_labels_all_seeds_better_baseline"]
            ),
            "same_top_model_all_three_seeds": int(
                headline["same_top_model_all_three_seeds"]
            ),
            "ranking_tasks": int(headline["ranking_tasks"]),
            "median_cell_seed_std_auc_milli": milli(
                headline["median_cell_seed_std_auc"]
            ),
        }
    ]

    cell_rows = []
    for row in cells_raw:
        cell_rows.append(
            {
                "unit": f"{row['model']} · {DATASET_LABELS[row['dataset']]}",
                "model": row["model"],
                "dataset": DATASET_LABELS[row["dataset"]],
                "labels": int(row["labels"]),
                "paper_auc": round6(row["paper_mean_auc"]),
                "baseline_auc": round6(row["baseline_mean_auc"]),
                "sisa_20262027_auc": round6(row["sisa_20262027_mean_auc"]),
                "sisa_20262028_auc": round6(row["sisa_20262028_mean_auc"]),
                "sisa_20262029_auc": round6(row["sisa_20262029_mean_auc"]),
                "sisa_three_seed_auc": round6(row["sisa_three_seed_mean_auc"]),
                "sisa_seed_std_auc_milli": milli(row["sisa_seed_std_auc"]),
                "sisa_seed_range_auc_milli": milli(row["sisa_seed_range_auc"]),
                "delta_auc_vs_baseline_milli": milli(row["sisa_delta_auc_vs_baseline"]),
                "delta_auc_vs_paper_milli": milli(row["sisa_delta_auc_vs_paper"]),
                "auc_labels_mean_better_baseline": int(
                    row["auc_labels_mean_better_baseline"]
                ),
                "auc_labels_all_seeds_better_baseline": int(
                    row["auc_labels_all_seeds_better_baseline"]
                ),
                "paper_logloss": round6(row["paper_mean_logloss"]),
                "baseline_logloss": round6(row["baseline_mean_logloss"]),
                "sisa_20262027_logloss": round6(row["sisa_20262027_mean_logloss"]),
                "sisa_20262028_logloss": round6(row["sisa_20262028_mean_logloss"]),
                "sisa_20262029_logloss": round6(row["sisa_20262029_mean_logloss"]),
                "sisa_three_seed_logloss": round6(
                    row["sisa_three_seed_mean_logloss"]
                ),
                "delta_logloss_vs_baseline_milli": milli(
                    row["sisa_delta_logloss_vs_baseline"]
                ),
                "delta_logloss_vs_paper_milli": milli(
                    row["sisa_delta_logloss_vs_paper"]
                ),
            }
        )
    cell_rows.sort(key=lambda row: float(row["delta_auc_vs_baseline_milli"]), reverse=True)

    cell_by_key = {(row["model"], row["dataset"]): row for row in cell_rows}
    long_table_numeric_fields = (
        "sisa_20262027_auc",
        "sisa_20262028_auc",
        "sisa_20262029_auc",
        "sisa_three_seed_auc",
        "sisa_seed_std_auc_milli",
        "baseline_auc",
        "delta_auc_vs_baseline_milli",
    )
    three_seed_long_rows = []
    for model in MODEL_ORDER:
        for dataset in DATASET_LABELS.values():
            long_row = dict(cell_by_key[(model, dataset)])
            for field in long_table_numeric_fields:
                long_row[field] = f"{float(long_row[field]):.5f}"
            three_seed_long_rows.append(long_row)
    auc_matrix_rows = []
    for dataset in DATASET_LABELS.values():
        matrix_row = {"dataset": dataset}
        for model in MODEL_ORDER:
            cell = cell_by_key[(model, dataset)]
            mean_auc = float(cell["sisa_three_seed_auc"])
            delta_auc = float(cell["delta_auc_vs_baseline_milli"]) / 1000
            direction = "↑" if delta_auc > 0 else "↓" if delta_auc < 0 else "→"
            matrix_row[model] = f"{mean_auc:.5f} {direction} {delta_auc:+.5f}"
        auc_matrix_rows.append(matrix_row)

    label_rows = []
    for row in labels_raw:
        label_rows.append(
            {
                "model": row["model"],
                "dataset": DATASET_LABELS[row["dataset"]],
                "label": row["label"],
                "paper_auc": round6(row["paper_auc"]),
                "baseline_auc": round6(row["baseline_auc"]),
                "sisa_20262027_auc": round6(row["sisa_20262027_auc"]),
                "sisa_20262028_auc": round6(row["sisa_20262028_auc"]),
                "sisa_20262029_auc": round6(row["sisa_20262029_auc"]),
                "sisa_three_seed_auc": round6(row["sisa_mean_auc"]),
                "sisa_seed_std_auc_milli": milli(row["sisa_std_auc"]),
                "delta_auc_vs_baseline_milli": milli(
                    row["sisa_mean_delta_auc_vs_baseline"]
                ),
                "delta_auc_vs_paper_milli": milli(row["sisa_mean_delta_auc_vs_paper"]),
                "abs_delta_auc_vs_baseline_milli": abs(
                    milli(row["sisa_mean_delta_auc_vs_baseline"])
                ),
                "all_seeds_beat_baseline": (
                    "是" if truthy(row["auc_all_seeds_beat_baseline"]) else "否"
                ),
                "sisa_three_seed_logloss": round6(row["sisa_mean_logloss"]),
                "delta_logloss_vs_baseline_milli": milli(
                    row["sisa_mean_delta_logloss_vs_baseline"]
                ),
            }
        )
    label_rows.sort(
        key=lambda row: float(row["abs_delta_auc_vs_baseline_milli"]), reverse=True
    )

    def dimension_rows(raw: Sequence[dict[str, str]], dimension: str) -> list[dict[str, object]]:
        rows = []
        for row in raw:
            display = DATASET_LABELS[row[dimension]] if dimension == "dataset" else row[dimension]
            rows.append(
                {
                    dimension: display,
                    "cells": int(row["cells"]),
                    "labels": int(row["labels"]),
                    "paper_auc": round6(row["cell_macro_paper_auc"]),
                    "baseline_auc": round6(row["cell_macro_baseline_auc"]),
                    "sisa_three_seed_auc": round6(row["cell_macro_sisa_auc"]),
                    "delta_auc_vs_baseline_milli": milli(
                        row["cell_macro_delta_auc_vs_baseline"]
                    ),
                    "delta_auc_vs_paper_milli": milli(
                        row["cell_macro_delta_auc_vs_paper"]
                    ),
                    "positive_auc_cells_vs_baseline": int(
                        row["positive_auc_cells_vs_baseline"]
                    ),
                    "median_seed_std_auc_milli": milli(row["median_seed_std_auc"]),
                    "delta_logloss_vs_baseline_milli": milli(
                        row["cell_macro_delta_logloss_vs_baseline"]
                    ),
                }
            )
        return rows

    model_rows = dimension_rows(models_raw, "model")
    dataset_rows = dimension_rows(datasets_raw, "dataset")
    protocol_rows = [
        {
            "series": row["series"],
            "experiment_seed": row["experiment_seed"],
            "internal_rng_seeds": row["internal_rng_seeds"],
            "gpu_protocol": row["gpu_protocol"],
            "per_gpu_batch": row["per_gpu_batch"],
            "global_batch": row["global_batch"],
            "epochs": int(row["epochs"]),
            "sequence_length": int(row["sequence_length"]),
            "comparability": row["comparability"],
        }
        for row in protocols_raw
    ]

    result_inputs = [
        relative(repository, paper_path),
        "experiments/sisa_native_strict/results/metrics.csv",
        "experiments/sisa_expansion_acd/results/metrics.csv",
        "experiments/sisa_single_seed20262028/results/metrics.csv",
        "experiments/sisa_single_seed20262029/results/metrics.csv",
    ]
    headline_rel = relative(repository, headline_path)
    cell_rel = relative(repository, cell_path)
    label_rel = relative(repository, label_path)
    model_rel = relative(repository, model_path)
    dataset_rel = relative(repository, dataset_path)
    protocol_rel = relative(repository, protocol_path)
    sources = [
        source(
            repository,
            "paper_table",
            "UniRank 论文 Table 2 的 68 个匹配指标",
            paper_path,
            description="UniRank comprehensive benchmark table, sequence length 100",
            filters=("models = HiFormer, HyFormer, RankMixer, Zenith", "datasets = four report datasets"),
            metric_definitions=("AUC higher is better", "binary logloss lower is better"),
            url="https://arxiv.org/abs/2607.19987",
        ),
        source(
            repository,
            "comparison_headline",
            "论文、baseline 与三-seed SISA 汇总",
            headline_path,
            sql=(
                "SELECT *, 3 AS sisa_seed_count, "
                "1000 * cell_macro_delta_auc_vs_baseline AS delta_auc_vs_baseline_milli, "
                "positive_auc_cells_vs_baseline, "
                "1000 * cell_macro_delta_auc_vs_paper AS delta_auc_vs_paper_milli, "
                "1000 * cell_macro_delta_logloss_vs_baseline AS delta_logloss_vs_baseline_milli, "
                "auc_labels_all_seeds_better_baseline, same_top_model_all_three_seeds, ranking_tasks "
                f"FROM read_json_auto('{headline_rel}')"
            ),
            tables_used=result_inputs,
            metric_definitions=(
                "cell-macro = equal-weight mean over labels within each cell, then equal-weight mean over 16 cells",
                "three-seed SISA mean = arithmetic mean of experiment seeds 20262027, 20262028, and 20262029",
                "AUC delta = SISA three-seed mean minus comparator",
                "logloss delta = SISA three-seed mean minus comparator; negative is better",
            ),
        ),
        source(
            repository,
            "cell_summary",
            "16 个模型×数据集单元的统一比较",
            cell_path,
            sql=f"SELECT * FROM read_csv_auto('{cell_rel}', header = true)",
            tables_used=result_inputs,
            metric_definitions=(
                "paper and baseline values are label means within one model-dataset cell",
                "SISA seed standard deviation is the sample standard deviation of three cell-level seed means",
            ),
        ),
        source(
            repository,
            "label_comparison",
            "68 个标签的论文/baseline/三-seed 明细",
            label_path,
            sql=f"SELECT * FROM read_csv_auto('{label_rel}', header = true)",
            tables_used=result_inputs,
        ),
        source(
            repository,
            "model_summary",
            "按模型的 cell-macro 汇总",
            model_path,
            sql=f"SELECT * FROM read_csv_auto('{model_rel}', header = true)",
            tables_used=(cell_rel,),
        ),
        source(
            repository,
            "dataset_summary",
            "按数据集的 cell-macro 汇总",
            dataset_path,
            sql=f"SELECT * FROM read_csv_auto('{dataset_rel}', header = true)",
            tables_used=(cell_rel,),
        ),
        source(
            repository,
            "protocols",
            "论文、baseline 与三次 SISA 的协议边界",
            protocol_path,
            sql=f"SELECT * FROM read_csv_auto('{protocol_rel}', header = true)",
            metric_definitions=(
                "experiment seed counts a complete SISA training round",
                "dataloader and SISA parameter seeds are internal RNG streams, not additional experiment seeds",
            ),
        ),
        source(
            repository,
            "comparison_code",
            "三-seed 对齐、聚合与校验代码",
            repository / "scripts" / "compare_sisa_three_seed_results.py",
            sql="SELECT 'scripts/compare_sisa_three_seed_results.py' AS source_file",
            tables_used=("scripts/compare_sisa_three_seed_results.py",),
        ),
        source(
            repository,
            "report_builder",
            "统一报告快照构建代码",
            repository / "scripts" / "build_sisa_three_seed_report.py",
            sql="SELECT 'scripts/build_sisa_three_seed_report.py' AS source_file",
            tables_used=("scripts/build_sisa_three_seed_report.py",),
        ),
    ]

    cards = [
        {
            "id": "seed_count",
            "description": "三次完整 SISA 实验轮次；不含内部 dataloader/SISA 参数随机流。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "SISA experiment seeds", "field": "sisa_seed_count", "format": "number"}],
        },
        {
            "id": "auc_vs_baseline",
            "description": "三-seed SISA cell-macro AUC 减本地复现 baseline。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "ΔAUC vs baseline（×10⁻³）", "field": "delta_auc_vs_baseline_milli", "format": "number", "signed": True}],
        },
        {
            "id": "positive_cells",
            "description": "SISA 三-seed mean AUC 高于本地 baseline 的模型×数据集单元数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "AUC 正向 cells", "field": "positive_auc_cells_vs_baseline", "format": "number"}],
        },
        {
            "id": "auc_vs_paper",
            "description": "三-seed SISA cell-macro AUC 减论文四位小数表值。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "ΔAUC vs paper（×10⁻³）", "field": "delta_auc_vs_paper_milli", "format": "number", "signed": True}],
        },
        {
            "id": "logloss_vs_baseline",
            "description": "三-seed SISA cell-macro logloss 减 baseline；负值更好。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "ΔLogloss vs baseline（×10⁻³）", "field": "delta_logloss_vs_baseline_milli", "format": "number", "signed": True}],
        },
        {
            "id": "robust_labels",
            "description": "三个 SISA seed 的 AUC 都高于 baseline 的标签数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "三轮均胜 baseline 的标签", "field": "auc_labels_all_seeds_better_baseline", "format": "number"}],
        },
        {
            "id": "ranking_stability",
            "description": "三次 SISA 中 AUC 第一名模型完全一致的 dataset×label 任务数。",
            "dataset": "headline",
            "sourceId": "comparison_headline",
            "metrics": [{"label": "三轮第一名一致", "field": "same_top_model_all_three_seeds", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "cell_auc_vs_baseline",
            "title": "16 个单元的 SISA 三-seed mean AUC 相对 baseline 差值",
            "subtitle": "每个单元先对标签等权；单位为 10⁻³，正值表示 SISA 更高",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "哪些模型×数据集单元稳定地贡献了 SISA 相对 baseline 的 AUC 变化？",
            "rationale": "16 个长标签且包含正负差值，排序横向条形图最适合比较方向和幅度。",
            "dataset": "cell_comparison",
            "sourceId": "cell_summary",
            "palette": {"kind": "diverging", "midpoint": 0},
            "labels": {"values": "auto"},
            "encodings": {
                "x": {"field": "unit", "type": "nominal", "aggregate": "none", "label": "模型 × 数据集"},
                "y": {"field": "delta_auc_vs_baseline_milli", "type": "quantitative", "aggregate": "none", "label": "ΔAUC（×10⁻³）"},
                "tooltip": [
                    {"field": "paper_auc", "type": "quantitative", "label": "论文 AUC"},
                    {"field": "baseline_auc", "type": "quantitative", "label": "Baseline AUC"},
                    {"field": "sisa_three_seed_auc", "type": "quantitative", "label": "SISA 三-seed mean AUC"},
                    {"field": "sisa_seed_std_auc_milli", "type": "quantitative", "label": "Seed sample SD（×10⁻³）"},
                ],
            },
            "xAxisTitle": "模型 × 数据集",
            "yAxisTitle": "ΔAUC（×10⁻³）",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 16,
        },
        {
            "id": "paper_delta_vs_seed_range",
            "title": "单元级论文 AUC 差值与三-seed 波动范围",
            "subtitle": "16 个模型×数据集单元；横轴为 SISA mean−paper，纵轴为三 seed 最大值−最小值，均为 10⁻³",
            "type": "scatter",
            "intent": "relationship",
            "question": "相对论文的 AUC 提升是否集中在三-seed 波动较大的单元？",
            "rationale": "16 个同粒度观测足以检查增益幅度与 seed 波动范围的关系，并保留单元身份。",
            "dataset": "cell_comparison",
            "sourceId": "cell_summary",
            "palette": {"kind": "categorical", "name": "dataset"},
            "legend": {"position": "bottom", "title": "数据集"},
            "encodings": {
                "x": {"field": "delta_auc_vs_paper_milli", "type": "quantitative", "aggregate": "none", "label": "ΔAUC vs paper（×10⁻³）"},
                "y": {"field": "sisa_seed_range_auc_milli", "type": "quantitative", "aggregate": "none", "label": "三-seed AUC range（×10⁻³）"},
                "color": {"field": "dataset", "type": "nominal", "label": "数据集"},
                "tooltip": [
                    {"field": "unit", "type": "nominal", "label": "单元"},
                    {"field": "delta_auc_vs_paper_milli", "type": "quantitative", "label": "ΔAUC vs paper（×10⁻³）"},
                    {"field": "sisa_seed_range_auc_milli", "type": "quantitative", "label": "Seed range（×10⁻³）"},
                    {"field": "delta_auc_vs_baseline_milli", "type": "quantitative", "label": "ΔAUC vs baseline（×10⁻³）"},
                ],
            },
            "xAxisTitle": "ΔAUC vs paper（×10⁻³）",
            "yAxisTitle": "三-seed AUC range（×10⁻³）",
            "valueFormat": "number",
            "layout": "full",
            "maxRows": 16,
        },
    ]

    auc_columns = [
        {"field": "model", "label": "模型", "type": "text"},
        {"field": "dataset", "label": "数据集", "type": "text"},
        {"field": "paper_auc", "label": "论文 AUC", "type": "number", "format": "number"},
        {"field": "baseline_auc", "label": "Baseline AUC", "type": "number", "format": "number"},
        {"field": "sisa_20262027_auc", "label": "SISA 20262027", "type": "number", "format": "number"},
        {"field": "sisa_20262028_auc", "label": "SISA 20262028", "type": "number", "format": "number"},
        {"field": "sisa_20262029_auc", "label": "SISA 20262029", "type": "number", "format": "number"},
        {"field": "sisa_three_seed_auc", "label": "SISA 3-seed mean", "type": "number", "format": "number"},
        {"field": "sisa_seed_std_auc_milli", "label": "Seed SD（×10⁻³）", "type": "number", "format": "number"},
        {"field": "delta_auc_vs_baseline_milli", "label": "Δ vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
        {"field": "delta_auc_vs_paper_milli", "label": "Δ vs paper（×10⁻³）", "type": "number", "format": "number", "movement": True},
    ]
    tables = [
        {
            "id": "auc_matrix_table",
            "title": "三-seed mean AUC 二维矩阵",
            "subtitle": "行是数据集、列是模型；每格为三-seed mean AUC，箭头后为相对 local baseline 的原始尺度 ΔAUC",
            "dataset": "auc_matrix",
            "sourceId": "cell_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "dataset", "direction": "asc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "HiFormer", "label": "HiFormer", "type": "text"},
                {"field": "HyFormer", "label": "HyFormer", "type": "text"},
                {"field": "RankMixer", "label": "RankMixer", "type": "text"},
                {"field": "Zenith", "label": "Zenith", "type": "text"},
            ],
        },
        {
            "id": "three_seed_long_table",
            "title": "每个模型×数据集的三-seed AUC 明细",
            "subtitle": "先按模型分组；每个模型内固定依次为 QK-Video、KuaiRand、TAAC-25 / TencentGR、MerRec",
            "dataset": "three_seed_long",
            "sourceId": "cell_summary",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "model", "direction": "asc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "sisa_20262027_auc", "label": "20262027", "type": "text", "align": "right"},
                {"field": "sisa_20262028_auc", "label": "20262028", "type": "text", "align": "right"},
                {"field": "sisa_20262029_auc", "label": "20262029", "type": "text", "align": "right"},
                {"field": "sisa_three_seed_auc", "label": "Mean", "type": "text", "align": "right"},
                {"field": "sisa_seed_std_auc_milli", "label": "SD（×10⁻³）", "type": "text", "align": "right"},
                {"field": "baseline_auc", "label": "Baseline", "type": "text", "align": "right"},
                {"field": "delta_auc_vs_baseline_milli", "label": "Δ（×10⁻³）", "type": "text", "align": "right", "movement": True},
            ],
        },
        {
            "id": "cell_auc_table",
            "title": "论文、baseline 与三次 SISA 的 16-cell AUC 对照",
            "subtitle": "每行对该模型×数据集的全部标签等权；SD 为三个 experiment seed 的样本标准差",
            "dataset": "cell_comparison",
            "sourceId": "cell_summary",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "delta_auc_vs_baseline_milli", "direction": "desc"},
            "columns": auc_columns,
        },
        {
            "id": "cell_logloss_table",
            "title": "论文、baseline 与三次 SISA 的 16-cell logloss 对照",
            "subtitle": "每行对标签等权；Δ 为 SISA 三-seed mean 减比较项，负值更低",
            "dataset": "cell_comparison",
            "sourceId": "cell_summary",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "delta_logloss_vs_baseline_milli", "direction": "asc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "paper_logloss", "label": "论文 Logloss", "type": "number", "format": "number"},
                {"field": "baseline_logloss", "label": "Baseline Logloss", "type": "number", "format": "number"},
                {"field": "sisa_20262027_logloss", "label": "SISA 20262027", "type": "number", "format": "number"},
                {"field": "sisa_20262028_logloss", "label": "SISA 20262028", "type": "number", "format": "number"},
                {"field": "sisa_20262029_logloss", "label": "SISA 20262029", "type": "number", "format": "number"},
                {"field": "sisa_three_seed_logloss", "label": "SISA 3-seed mean", "type": "number", "format": "number"},
                {"field": "delta_logloss_vs_baseline_milli", "label": "Δ vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "delta_logloss_vs_paper_milli", "label": "Δ vs paper（×10⁻³）", "type": "number", "format": "number", "movement": True},
            ],
        },
        {
            "id": "model_table",
            "title": "按模型的 cell-macro 汇总",
            "subtitle": "每个模型的四个数据集等权",
            "dataset": "model_summary",
            "sourceId": "model_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "delta_auc_vs_baseline_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "paper_auc", "label": "论文 AUC", "type": "number", "format": "number"},
                {"field": "baseline_auc", "label": "Baseline AUC", "type": "number", "format": "number"},
                {"field": "sisa_three_seed_auc", "label": "SISA 3-seed mean", "type": "number", "format": "number"},
                {"field": "delta_auc_vs_baseline_milli", "label": "ΔAUC vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "positive_auc_cells_vs_baseline", "label": "正向 cells", "type": "number", "format": "number"},
                {"field": "median_seed_std_auc_milli", "label": "Median seed SD（×10⁻³）", "type": "number", "format": "number"},
                {"field": "delta_logloss_vs_baseline_milli", "label": "ΔLogloss vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
            ],
        },
        {
            "id": "dataset_table",
            "title": "按数据集的 cell-macro 汇总",
            "subtitle": "每个数据集的四个模型等权",
            "dataset": "dataset_summary",
            "sourceId": "dataset_summary",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "delta_auc_vs_baseline_milli", "direction": "desc"},
            "columns": [
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "paper_auc", "label": "论文 AUC", "type": "number", "format": "number"},
                {"field": "baseline_auc", "label": "Baseline AUC", "type": "number", "format": "number"},
                {"field": "sisa_three_seed_auc", "label": "SISA 3-seed mean", "type": "number", "format": "number"},
                {"field": "delta_auc_vs_baseline_milli", "label": "ΔAUC vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "positive_auc_cells_vs_baseline", "label": "正向 cells", "type": "number", "format": "number"},
                {"field": "median_seed_std_auc_milli", "label": "Median seed SD（×10⁻³）", "type": "number", "format": "number"},
                {"field": "delta_logloss_vs_baseline_milli", "label": "ΔLogloss vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
            ],
        },
        {
            "id": "label_table",
            "title": "68 个标签的完整 AUC 审计明细",
            "subtitle": "按 |SISA 三-seed mean−baseline| 降序；完整结果保留在报告快照中",
            "dataset": "label_comparison",
            "sourceId": "label_comparison",
            "layout": "full",
            "density": "dense",
            "defaultSort": {"field": "abs_delta_auc_vs_baseline_milli", "direction": "desc"},
            "columns": [
                {"field": "model", "label": "模型", "type": "text"},
                {"field": "dataset", "label": "数据集", "type": "text"},
                {"field": "label", "label": "标签", "type": "text"},
                {"field": "paper_auc", "label": "论文 AUC", "type": "number", "format": "number"},
                {"field": "baseline_auc", "label": "Baseline AUC", "type": "number", "format": "number"},
                {"field": "sisa_three_seed_auc", "label": "SISA 3-seed mean", "type": "number", "format": "number"},
                {"field": "sisa_seed_std_auc_milli", "label": "Seed SD（×10⁻³）", "type": "number", "format": "number"},
                {"field": "delta_auc_vs_baseline_milli", "label": "Δ vs baseline（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "abs_delta_auc_vs_baseline_milli", "label": "|Δ vs baseline|（×10⁻³）", "type": "number", "format": "number"},
                {"field": "delta_auc_vs_paper_milli", "label": "Δ vs paper（×10⁻³）", "type": "number", "format": "number", "movement": True},
                {"field": "all_seeds_beat_baseline", "label": "三轮均胜 baseline", "type": "text"},
            ],
        },
        {
            "id": "protocol_table",
            "title": "论文、baseline 与三次 SISA 的口径和协议",
            "subtitle": "只有 seed 20262028 与 20262029 是完全同协议的两轮 SISA",
            "dataset": "protocols",
            "sourceId": "protocols",
            "layout": "full",
            "density": "comfortable",
            "defaultSort": {"field": "series", "direction": "asc"},
            "columns": [
                {"field": "series", "label": "系列", "type": "text"},
                {"field": "experiment_seed", "label": "Experiment seed", "type": "text"},
                {"field": "internal_rng_seeds", "label": "内部 RNG seed", "type": "text"},
                {"field": "gpu_protocol", "label": "GPU / 协议", "type": "text"},
                {"field": "per_gpu_batch", "label": "每卡 batch", "type": "text"},
                {"field": "global_batch", "label": "Global batch", "type": "text"},
                {"field": "comparability", "label": "可比性", "type": "text"},
            ],
        },
    ]

    best = headline["largest_cell_auc_gain_vs_baseline"]
    worst = headline["largest_cell_auc_decline_vs_baseline"]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 技术摘要\n\n"
                "统一口径覆盖 **4 个模型 × 4 个数据集、16 个 cells、68 个标签**。"
                f"SISA 的三个 experiment seed 是 **{SEEDS[0]}、{SEEDS[1]}、{SEEDS[2]}**；"
                "内部 dataloader seed 和 SISA parameter seed 不计作额外实验轮次。三-seed SISA 相对本地 baseline 的 "
                f"cell-macro ΔAUC 为 **{float(headline['cell_macro_delta_auc_vs_baseline']):+.6f}**，"
                f"相对论文表值为 **{float(headline['cell_macro_delta_auc_vs_paper']):+.6f}**；"
                f"AUC 正向 cells 为 **{headline['positive_auc_cells_vs_baseline']}/16**，"
                f"cell-macro Δlogloss 为 **{float(headline['cell_macro_delta_logloss_vs_baseline']):+.6f}**（负值更好）。"
                "由于第一轮与后两轮协议不同，三点均值、样本标准差和范围只作描述性结果。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["seed_count", "auc_vs_baseline", "positive_cells", "auc_vs_paper", "logloss_vs_baseline", "robust_labels", "ranking_stability"]},
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## SISA 的平均 AUC 优势覆盖 15/16 个单元，但跨 seed 排名并不完全固定\n\n"
                f"三-seed mean AUC 在 **{headline['auc_labels_mean_better_baseline']}/68** 个标签上高于 baseline，"
                f"其中 **{headline['auc_labels_all_seeds_better_baseline']}/68** 个标签是三轮全部高于 baseline。"
                f"17 个 dataset×label 任务中只有 **{headline['same_top_model_all_three_seeds']}/17** 个任务的 AUC 第一名模型三轮一致。"
                "因此统一均值适合总结整体方向，但不应替代逐 cell 和逐标签查看。"
            ),
        },
        {
            "id": "auc_matrix_finding",
            "type": "markdown",
            "sourceId": "cell_summary",
            "body": (
                "## 二维矩阵让模型×数据集的升降方向一眼可见\n\n"
                "每格第一个数是三个 experiment seed 的 mean AUC；箭头后的第二个数是该均值减 local baseline 的原始尺度 ΔAUC。"
                "**↑** 表示高于 baseline，**↓** 表示低于 baseline；矩阵只汇报 AUC，避免把 logloss 的相反优劣方向混进同一格。"
            ),
        },
        {"id": "auc_matrix_table_block", "type": "table", "tableId": "auc_matrix_table"},
        {
            "id": "three_seed_long_finding",
            "type": "markdown",
            "sourceId": "cell_summary",
            "body": (
                "## 三个 seed 的逐 cell 明细用于检查均值来自哪里\n\n"
                "长表先把同一模型的四个数据集聚在一起；每组固定按 QK-Video、KuaiRand、TAAC-25 / TencentGR、MerRec 排列。"
                "数值列依次给出 seed 20262027、20262028、20262029、三-seed mean、sample SD、local baseline 和最终差值。"
                "这里的 SD 仍是描述性统计，因为第一轮和后两轮的 GPU/每卡 batch 协议不同。"
            ),
        },
        {"id": "three_seed_long_table_block", "type": "table", "tableId": "three_seed_long_table"},
        {"id": "baseline_chart_block", "type": "chart", "chartId": "cell_auc_vs_baseline"},
        {
            "id": "baseline_chart_interpretation",
            "type": "markdown",
            "sourceId": "cell_summary",
            "body": (
                "## 单元差值显示增益集中在特定模型×数据集组合\n\n"
                f"相对 baseline 的最大 cell AUC 增益来自 **{best['model']}–{DATASET_LABELS[str(best['dataset'])]}** "
                f"（Δ {float(best['delta_auc']):+.6f}），唯一负向单元是 **{worst['model']}–{DATASET_LABELS[str(worst['dataset'])]}** "
                f"（Δ {float(worst['delta_auc']):+.6f}）。横条比较方向和幅度；下表保留论文、baseline 与三个 seed 的精确均值。"
            ),
        },
        {"id": "cell_auc_table_block", "type": "table", "tableId": "cell_auc_table"},
        {
            "id": "paper_comparison",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 论文对照的四位小数精度足以做方向比较，不支持过细归因\n\n"
                f"本地 baseline 与论文 AUC 的标签级平均绝对差为 **{float(headline['mean_abs_baseline_delta_auc_vs_paper']):.6f}**，"
                f"**{headline['baseline_auc_within_0_01_of_paper']}/68** 个标签在 ±0.01 内。"
                "论文值来自 Table 2 的四位小数发布值；scatter 将 SISA 相对论文的 cell 差值与三-seed range 放在同一粒度比较。"
            ),
        },
        {"id": "paper_scatter_block", "type": "chart", "chartId": "paper_delta_vs_seed_range"},
        {
            "id": "paper_scatter_interpretation",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 增益幅度与 seed 波动需要同时观察\n\n"
                f"16 个 cells 的 seed sample SD 中位数为 **{float(headline['median_cell_seed_std_auc']):.6f}**，"
                f"最大 seed range 为 **{float(headline['max_cell_seed_range_auc']):.6f}**。"
                "右上方单元同时具备较高论文增益与较大 seed 波动，不能仅凭三点均值断言稳定改进。"
            ),
        },
        {
            "id": "model_finding",
            "type": "markdown",
            "sourceId": "model_summary",
            "body": "## 模型汇总对四个数据集等权\n\nCell-macro 避免 KuaiRand 的六个标签比 TencentGR 的两个标签自动获得更高权重；模型表同时保留论文、baseline、SISA 均值与 seed 波动。",
        },
        {"id": "model_table_block", "type": "table", "tableId": "model_table"},
        {
            "id": "dataset_finding",
            "type": "markdown",
            "sourceId": "dataset_summary",
            "body": "## 数据集汇总对四个模型等权\n\n该切面用于查看跨模型是否呈现共同方向；它不替代逐标签结果，尤其不能掩盖低基率标签的 logloss 差异。",
        },
        {"id": "dataset_table_block", "type": "table", "tableId": "dataset_table"},
        {
            "id": "logloss_finding",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## Logloss 整体下降，但需要按标签基率分别解释\n\n"
                f"三-seed SISA 相对 baseline 的 cell-macro Δlogloss 为 **{float(headline['cell_macro_delta_logloss_vs_baseline']):+.6f}**，"
                f"**{headline['lower_logloss_cells_vs_baseline']}/16** 个 cells 更低。"
                "由于不同标签的 logloss 尺度受正例率影响，报告不额外绘制跨标签绝对 logloss 图，而以 16-cell 精确表展示。"
            ),
        },
        {"id": "logloss_table_block", "type": "table", "tableId": "cell_logloss_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "comparison_headline",
            "body": (
                "## 范围、指标与比较定义\n\n"
                "语义键固定为 `(model, dataset, label)`；AUC 越高越好，binary logloss 越低越好。"
                "SISA 三-seed mean 是三个完整 experiment seed 的算术平均；seed SD 是三个 seed 在同一 cell 上的样本标准差。"
                "Cell-macro 先在 cell 内对标签等权，再对 cells 等权。论文列保留 Table 2 的四位小数，baseline 列是本地 seed 20262027 的 non-SISA 复现。"
            ),
        },
        {
            "id": "experimental_design",
            "type": "markdown",
            "sourceId": "protocols",
            "body": (
                "## 实验设计包含三个 SISA experiment seed，而不是三个新增 seed\n\n"
                "seed 20262027 是原 4-GPU 实验轮次；seed 20262028 和 20262029 是两次 2×H100 轮次。"
                "三轮 global batch 都是 32768，但第一轮采用每卡 batch 8192，后两轮采用每卡 batch 16384，"
                "因此 blocked dataloader 的 drop-last 边界和硬件条件存在混杂。内部 dataloader/SISA parameter seed 只属于对应实验轮次。"
            ),
        },
        {"id": "protocol_table_block", "type": "table", "tableId": "protocol_table"},
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "comparison_code",
            "body": (
                "## 对齐、聚合与完整性校验\n\n"
                "论文表、本地 baseline 和三个 SISA 结果必须各自提供 68 个唯一语义键和有限 AUC/logloss。"
                "seed 20262027 按 9 个 strict cells 加 7 个 expansion cells 组合；seed 20262028/20262029 直接读取各自正式 collector 结果，"
                "并要求 16/16 runs 完成、协议字段匹配且 H100 证据有效。统计量仅包括均值、样本标准差、范围、胜负计数和排名稳定性，没有显著性检验。"
            ),
        },
        {
            "id": "label_detail",
            "type": "markdown",
            "sourceId": "label_comparison",
            "body": "## 68 个标签明细用于复核聚合结论\n\n完整表按相对 baseline 的 AUC 绝对差排序，保留论文、baseline、三-seed mean、sample SD、两类差值和三轮是否全部胜出。",
        },
        {"id": "label_table_block", "type": "table", "tableId": "label_table"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 局限性、不确定性与稳健性边界\n\n"
                "三-seed 汇总只有三个点，其中 seed 20262027 与后两轮不是同协议；因此 sample SD 混合了随机种子、GPU/协议和 drop-last 覆盖差异。"
                "论文值只有四位小数，也限制了细粒度差值的解释。当前结果支持探索性方向、异常定位和统一汇报，"
                "不支持纯 seed 方差、置信区间、统计显著性或因果归因。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 建议的统一汇报规则\n\n"
                "1. 正式展示时始终把论文、local baseline、三个 SISA seed 和三-seed mean 放在同一张明细表中。\n"
                "2. 主结论使用 cell-macro AUC/logloss，并同时给出 16-cell 方向、三轮均胜标签数和 seed range。\n"
                "3. 将三点统计标为 descriptive；只有 seed 20262028 与 20262029 可称为同协议重复。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 后续研究问题\n\n"
                "- 唯一负向 cell 的退化是否集中在某一标签和模型结构？\n"
                "- 三轮第一名不一致的 9 个任务，模型间 AUC 间隔是否小于 seed 波动？\n"
                "- AUC 改善但 logloss 变差的标签是否需要单独做校准分析？"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "论文表值、本地 baseline 与 SISA experiment seeds 20262027/20262028/20262029 的统一技术汇报。",
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
            "cell_comparison": cell_rows,
            "auc_matrix": auc_matrix_rows,
            "three_seed_long": three_seed_long_rows,
            "label_comparison": label_rows,
            "model_summary": model_rows,
            "dataset_summary": dataset_rows,
            "protocols": protocol_rows,
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
        "| Section | Question | Family / type | Fields | Supported claim | Palette |\n"
        "|---|---|---|---|---|---|\n"
        "| SISA vs baseline | 哪些 16-cell 单元贡献 AUC 变化？ | Comparison / horizontalBar | unit, delta_auc_vs_baseline_milli | 展示方向和幅度 | diverging, midpoint 0 |\n"
        "| Paper delta vs variability | 论文增益是否伴随高 seed 波动？ | Relationship / scatter | delta_auc_vs_paper_milli, sisa_seed_range_auc_milli, dataset | 识别高增益/高波动单元 | categorical by dataset |\n\n"
        "The first visual uses ranked bars because labels are long and signed. The second uses 16 same-grain observations, meeting the scatter sufficiency gate.\n"
    )
    source_notes = (
        "# Report source notes\n\n"
        "- Audience: technical.\n"
        "- Delivery mode: portable HTML from the canonical artifact contract.\n"
        "- Required structure mapping: title; technical summary; key visual findings; scope/definitions; experimental design; methodology; limitations; reporting rules; further questions.\n"
        "- Paper source: UniRank arXiv:2607.19987 Table 2, four-decimal values, materialized in sources/paper_table.csv. TAAC-25 maps to TencentGR_10M_Action.\n"
        "- Baseline source: local non-SISA seed20262027 results selected from strict and expansion archives.\n"
        "- SISA sources: seed20262027 is the matching strict+expansion union; seed20262028 and seed20262029 are complete 2×H100 studies.\n"
        "- Statistical boundary: descriptive three-point mean/sample-SD/range only because seed20262027 differs in hardware and per-GPU batch.\n"
        "- Exact lookup additions: a 4×4 AUC matrix shows three-seed mean first and raw-scale delta vs baseline second; a 16-row long table shows all three seed values, mean, sample SD, baseline, and delta.\n"
        "- Reader pagination: the portable delivery wrapper uses a 16-row table page so the four complete model groups remain visible together.\n"
        "- Omitted visual: logloss remains a table because label base rates create heterogeneous absolute scales; a cross-label chart would overstate comparability.\n"
        "- Repeated chart-family audit: no repetition; one comparison bar and one relationship scatter.\n"
    )
    return artifact, report_dir, chart_map, source_notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    try:
        artifact, report_dir, chart_map, source_notes = build_artifact(repository)
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
