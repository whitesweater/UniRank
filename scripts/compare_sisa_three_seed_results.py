#!/usr/bin/env python3
"""Unify paper, local baseline, and three SISA experiment seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
DATASETS = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "TencentGR_10M_Action",
    "MerRec_Action",
)
SEEDS = (20262027, 20262028, 20262029)
KEY_FIELDS = ("model", "dataset", "label")
STRICT_MODELS = frozenset({"HiFormer", "RankMixer", "Zenith"})
STRICT_DATASETS = frozenset(
    {"QK_Video_Action", "KuaiRand_Video_Action", "MerRec_Action"}
)
EXPECTED_LABELS = 68
EXPECTED_CELLS = 16
EPSILON = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def metric_key(row: dict[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)  # type: ignore[return-value]


def finite_metric(row: dict[str, object]) -> bool:
    return all(math.isfinite(float(row[field])) for field in ("AUC", "logloss"))


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot calculate a mean over zero values")
    return statistics.fmean(materialized)


def index_unique(
    rows: Iterable[dict[str, object]], source_name: str
) -> dict[tuple[str, str, str], dict[str, object]]:
    indexed: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        key = metric_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {source_name} metric key: {key}")
        if not finite_metric(row):
            raise ValueError(f"non-finite {source_name} metric for {key}")
        indexed[key] = row
    return indexed


def select_legacy_metrics(
    strict_rows: Iterable[dict[str, str]],
    expansion_rows: Iterable[dict[str, str]],
    setting: str,
) -> list[dict[str, object]]:
    if setting not in {"baseline", "sisa"}:
        raise ValueError(f"unsupported legacy setting: {setting}")
    selected: list[dict[str, object]] = []
    for row in strict_rows:
        if (
            row.get("setting") == setting
            and row.get("model") in STRICT_MODELS
            and row.get("dataset") in STRICT_DATASETS
        ):
            normalized: dict[str, object] = dict(row)
            normalized.update(
                seed=20262027,
                protocol="ws4_bs8192_acc1",
                source_bundle="sisa_native_strict",
                gpu_name=row.get("gpu_type", "unknown"),
            )
            selected.append(normalized)

    for row in expansion_rows:
        model = row.get("model")
        dataset = row.get("dataset")
        if (
            row.get("setting") == setting
            and model in MODELS
            and dataset in DATASETS
            and (model == "HyFormer" or dataset == "TencentGR_10M_Action")
        ):
            normalized = dict(row)
            normalized.update(
                seed=20262027,
                protocol="ws4_bs8192_acc1",
                source_bundle="sisa_expansion_acd",
                gpu_name=row.get("gpu_name", "unknown"),
            )
            selected.append(normalized)
    return selected


def validate_seed_rows(
    rows: Sequence[dict[str, str]], seed: int, protocol: str
) -> None:
    if len(rows) != EXPECTED_LABELS:
        raise ValueError(f"seed {seed} must contain 68 metrics, found {len(rows)}")
    if {int(row["seed"]) for row in rows} != {seed}:
        raise ValueError(f"seed {seed} metrics contain a different experiment seed")
    if {row["protocol"] for row in rows} != {protocol}:
        raise ValueError(f"seed {seed} metrics contain a different protocol")
    index_unique(rows, f"seed {seed}")


def validate_run_audit(rows: Sequence[dict[str, str]], seed: int) -> None:
    if len(rows) != EXPECTED_CELLS:
        raise ValueError(f"seed {seed} run audit must contain 16 rows")
    if {int(row["seed"]) for row in rows} != {seed}:
        raise ValueError(f"seed {seed} run audit contains another experiment seed")
    if {row["protocol"] for row in rows} != {"ws2_bs16384_acc1"}:
        raise ValueError(f"seed {seed} run audit contains another protocol")
    if not all(truthy(row.get("complete")) for row in rows):
        raise ValueError(f"seed {seed} run audit contains incomplete tasks")
    if not all(truthy(row.get("h100_valid")) for row in rows):
        raise ValueError(f"seed {seed} run audit contains a non-H100 allocation")


def validate_aligned_sources(
    paper: dict[tuple[str, str, str], dict[str, object]],
    baseline: dict[tuple[str, str, str], dict[str, object]],
    seeds: dict[int, dict[tuple[str, str, str], dict[str, object]]],
) -> None:
    expected_keys = set(paper)
    if len(expected_keys) != EXPECTED_LABELS:
        raise ValueError(f"paper table must contain 68 unique keys, found {len(expected_keys)}")
    expected_cells = {(key[0], key[1]) for key in expected_keys}
    if expected_cells != {(model, dataset) for model in MODELS for dataset in DATASETS}:
        raise ValueError("paper table does not contain the expected 16 cells")
    for name, indexed in (("baseline", baseline), *[(f"seed {s}", seeds[s]) for s in SEEDS]):
        if set(indexed) != expected_keys:
            missing = sorted(expected_keys - set(indexed))
            extra = sorted(set(indexed) - expected_keys)
            raise ValueError(f"{name} keys do not align; missing={missing}, extra={extra}")


def descriptive(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    if len(values) != 3:
        raise ValueError(f"three-seed summary requires exactly 3 values, found {len(values)}")
    return (
        statistics.fmean(values),
        statistics.stdev(values),
        min(values),
        max(values),
        max(values) - min(values),
    )


def build_label_rows(
    paper: dict[tuple[str, str, str], dict[str, object]],
    baseline: dict[tuple[str, str, str], dict[str, object]],
    seeds: dict[int, dict[tuple[str, str, str], dict[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(paper):
        paper_row = paper[key]
        baseline_row = baseline[key]
        auc_values = [float(seeds[seed][key]["AUC"]) for seed in SEEDS]
        logloss_values = [float(seeds[seed][key]["logloss"]) for seed in SEEDS]
        auc_mean, auc_std, auc_min, auc_max, auc_range = descriptive(auc_values)
        ll_mean, ll_std, ll_min, ll_max, ll_range = descriptive(logloss_values)
        paper_auc = float(paper_row["AUC"])
        paper_ll = float(paper_row["logloss"])
        baseline_auc = float(baseline_row["AUC"])
        baseline_ll = float(baseline_row["logloss"])
        row: dict[str, object] = {
            "model": key[0],
            "dataset": key[1],
            "label": key[2],
            "legacy_source": baseline_row["source_bundle"],
            "paper_auc": paper_auc,
            "baseline_auc": baseline_auc,
            "baseline_delta_auc_vs_paper": baseline_auc - paper_auc,
            "sisa_mean_auc": auc_mean,
            "sisa_std_auc": auc_std,
            "sisa_min_auc": auc_min,
            "sisa_max_auc": auc_max,
            "sisa_range_auc": auc_range,
            "sisa_mean_delta_auc_vs_baseline": auc_mean - baseline_auc,
            "sisa_mean_delta_auc_vs_paper": auc_mean - paper_auc,
            "auc_mean_beats_baseline": auc_mean > baseline_auc + EPSILON,
            "auc_mean_beats_paper": auc_mean > paper_auc + EPSILON,
            "auc_all_seeds_beat_baseline": min(auc_values) > baseline_auc + EPSILON,
            "auc_all_seeds_beat_paper": min(auc_values) > paper_auc + EPSILON,
            "paper_logloss": paper_ll,
            "baseline_logloss": baseline_ll,
            "baseline_delta_logloss_vs_paper": baseline_ll - paper_ll,
            "sisa_mean_logloss": ll_mean,
            "sisa_std_logloss": ll_std,
            "sisa_min_logloss": ll_min,
            "sisa_max_logloss": ll_max,
            "sisa_range_logloss": ll_range,
            "sisa_mean_delta_logloss_vs_baseline": ll_mean - baseline_ll,
            "sisa_mean_delta_logloss_vs_paper": ll_mean - paper_ll,
            "logloss_mean_beats_baseline": ll_mean < baseline_ll - EPSILON,
            "logloss_mean_beats_paper": ll_mean < paper_ll - EPSILON,
            "logloss_all_seeds_beat_baseline": max(logloss_values) < baseline_ll - EPSILON,
            "logloss_all_seeds_beat_paper": max(logloss_values) < paper_ll - EPSILON,
        }
        for seed, auc, logloss in zip(SEEDS, auc_values, logloss_values):
            row[f"sisa_{seed}_auc"] = auc
            row[f"sisa_{seed}_logloss"] = logloss
        rows.append(row)
    return rows


def build_cell_rows(label_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in label_rows:
        grouped[(str(row["model"]), str(row["dataset"]))].append(row)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in DATASETS:
            group = grouped[(model, dataset)]
            seed_auc = {
                seed: mean(float(row[f"sisa_{seed}_auc"]) for row in group)
                for seed in SEEDS
            }
            seed_ll = {
                seed: mean(float(row[f"sisa_{seed}_logloss"]) for row in group)
                for seed in SEEDS
            }
            sisa_auc_mean, sisa_auc_std, _, _, sisa_auc_range = descriptive(
                list(seed_auc.values())
            )
            sisa_ll_mean, sisa_ll_std, _, _, sisa_ll_range = descriptive(
                list(seed_ll.values())
            )
            paper_auc = mean(float(row["paper_auc"]) for row in group)
            baseline_auc = mean(float(row["baseline_auc"]) for row in group)
            paper_ll = mean(float(row["paper_logloss"]) for row in group)
            baseline_ll = mean(float(row["baseline_logloss"]) for row in group)
            row: dict[str, object] = {
                "model": model,
                "dataset": dataset,
                "labels": len(group),
                "legacy_source": group[0]["legacy_source"],
                "paper_mean_auc": paper_auc,
                "baseline_mean_auc": baseline_auc,
                "sisa_three_seed_mean_auc": sisa_auc_mean,
                "sisa_seed_std_auc": sisa_auc_std,
                "sisa_seed_range_auc": sisa_auc_range,
                "sisa_delta_auc_vs_baseline": sisa_auc_mean - baseline_auc,
                "sisa_delta_auc_vs_paper": sisa_auc_mean - paper_auc,
                "auc_labels_mean_better_baseline": sum(
                    bool(row["auc_mean_beats_baseline"]) for row in group
                ),
                "auc_labels_mean_better_paper": sum(
                    bool(row["auc_mean_beats_paper"]) for row in group
                ),
                "auc_labels_all_seeds_better_baseline": sum(
                    bool(row["auc_all_seeds_beat_baseline"]) for row in group
                ),
                "paper_mean_logloss": paper_ll,
                "baseline_mean_logloss": baseline_ll,
                "sisa_three_seed_mean_logloss": sisa_ll_mean,
                "sisa_seed_std_logloss": sisa_ll_std,
                "sisa_seed_range_logloss": sisa_ll_range,
                "sisa_delta_logloss_vs_baseline": sisa_ll_mean - baseline_ll,
                "sisa_delta_logloss_vs_paper": sisa_ll_mean - paper_ll,
                "logloss_labels_mean_better_baseline": sum(
                    bool(row["logloss_mean_beats_baseline"]) for row in group
                ),
                "logloss_labels_mean_better_paper": sum(
                    bool(row["logloss_mean_beats_paper"]) for row in group
                ),
                "logloss_labels_all_seeds_better_baseline": sum(
                    bool(row["logloss_all_seeds_beat_baseline"]) for row in group
                ),
            }
            for seed in SEEDS:
                row[f"sisa_{seed}_mean_auc"] = seed_auc[seed]
                row[f"sisa_{seed}_mean_logloss"] = seed_ll[seed]
            rows.append(row)
    return rows


def summarize_cells(
    cell_rows: Sequence[dict[str, object]], dimension: str, values: Sequence[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in values:
        group = [row for row in cell_rows if row[dimension] == value]
        labels = sum(int(row["labels"]) for row in group)
        rows.append(
            {
                dimension: value,
                "cells": len(group),
                "labels": labels,
                "cell_macro_paper_auc": mean(float(row["paper_mean_auc"]) for row in group),
                "cell_macro_baseline_auc": mean(
                    float(row["baseline_mean_auc"]) for row in group
                ),
                "cell_macro_sisa_auc": mean(
                    float(row["sisa_three_seed_mean_auc"]) for row in group
                ),
                "cell_macro_delta_auc_vs_baseline": mean(
                    float(row["sisa_delta_auc_vs_baseline"]) for row in group
                ),
                "cell_macro_delta_auc_vs_paper": mean(
                    float(row["sisa_delta_auc_vs_paper"]) for row in group
                ),
                "positive_auc_cells_vs_baseline": sum(
                    float(row["sisa_delta_auc_vs_baseline"]) > EPSILON for row in group
                ),
                "positive_auc_cells_vs_paper": sum(
                    float(row["sisa_delta_auc_vs_paper"]) > EPSILON for row in group
                ),
                "median_seed_std_auc": statistics.median(
                    float(row["sisa_seed_std_auc"]) for row in group
                ),
                "max_seed_range_auc": max(float(row["sisa_seed_range_auc"]) for row in group),
                "cell_macro_paper_logloss": mean(
                    float(row["paper_mean_logloss"]) for row in group
                ),
                "cell_macro_baseline_logloss": mean(
                    float(row["baseline_mean_logloss"]) for row in group
                ),
                "cell_macro_sisa_logloss": mean(
                    float(row["sisa_three_seed_mean_logloss"]) for row in group
                ),
                "cell_macro_delta_logloss_vs_baseline": mean(
                    float(row["sisa_delta_logloss_vs_baseline"]) for row in group
                ),
                "cell_macro_delta_logloss_vs_paper": mean(
                    float(row["sisa_delta_logloss_vs_paper"]) for row in group
                ),
                "lower_logloss_cells_vs_baseline": sum(
                    float(row["sisa_delta_logloss_vs_baseline"]) < -EPSILON
                    for row in group
                ),
                "lower_logloss_cells_vs_paper": sum(
                    float(row["sisa_delta_logloss_vs_paper"]) < -EPSILON for row in group
                ),
            }
        )
    return rows


def top_model_stability(label_rows: Sequence[dict[str, object]]) -> tuple[int, int]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in label_rows:
        grouped[(str(row["dataset"]), str(row["label"]))].append(row)
    stable = 0
    for group in grouped.values():
        winners = {
            max(group, key=lambda row: float(row[f"sisa_{seed}_auc"]))["model"]
            for seed in SEEDS
        }
        stable += len(winners) == 1
    return stable, len(grouped)


def build_headline(
    label_rows: Sequence[dict[str, object]], cell_rows: Sequence[dict[str, object]]
) -> dict[str, object]:
    stable, ranking_tasks = top_model_stability(label_rows)
    best = max(cell_rows, key=lambda row: float(row["sisa_delta_auc_vs_baseline"]))
    worst = min(cell_rows, key=lambda row: float(row["sisa_delta_auc_vs_baseline"]))
    return {
        "experiment_seeds": list(SEEDS),
        "cell_count": len(cell_rows),
        "label_count": len(label_rows),
        "cell_macro_delta_auc_vs_baseline": mean(
            float(row["sisa_delta_auc_vs_baseline"]) for row in cell_rows
        ),
        "cell_macro_delta_auc_vs_paper": mean(
            float(row["sisa_delta_auc_vs_paper"]) for row in cell_rows
        ),
        "positive_auc_cells_vs_baseline": sum(
            float(row["sisa_delta_auc_vs_baseline"]) > EPSILON for row in cell_rows
        ),
        "positive_auc_cells_vs_paper": sum(
            float(row["sisa_delta_auc_vs_paper"]) > EPSILON for row in cell_rows
        ),
        "auc_labels_mean_better_baseline": sum(
            bool(row["auc_mean_beats_baseline"]) for row in label_rows
        ),
        "auc_labels_all_seeds_better_baseline": sum(
            bool(row["auc_all_seeds_beat_baseline"]) for row in label_rows
        ),
        "cell_macro_delta_logloss_vs_baseline": mean(
            float(row["sisa_delta_logloss_vs_baseline"]) for row in cell_rows
        ),
        "cell_macro_delta_logloss_vs_paper": mean(
            float(row["sisa_delta_logloss_vs_paper"]) for row in cell_rows
        ),
        "lower_logloss_cells_vs_baseline": sum(
            float(row["sisa_delta_logloss_vs_baseline"]) < -EPSILON for row in cell_rows
        ),
        "lower_logloss_cells_vs_paper": sum(
            float(row["sisa_delta_logloss_vs_paper"]) < -EPSILON for row in cell_rows
        ),
        "logloss_labels_mean_better_baseline": sum(
            bool(row["logloss_mean_beats_baseline"]) for row in label_rows
        ),
        "median_cell_seed_std_auc": statistics.median(
            float(row["sisa_seed_std_auc"]) for row in cell_rows
        ),
        "max_cell_seed_range_auc": max(
            float(row["sisa_seed_range_auc"]) for row in cell_rows
        ),
        "mean_abs_baseline_delta_auc_vs_paper": mean(
            abs(float(row["baseline_delta_auc_vs_paper"])) for row in label_rows
        ),
        "baseline_auc_within_0_01_of_paper": sum(
            abs(float(row["baseline_delta_auc_vs_paper"])) <= 0.01
            for row in label_rows
        ),
        "same_top_model_all_three_seeds": stable,
        "ranking_tasks": ranking_tasks,
        "largest_cell_auc_gain_vs_baseline": {
            "model": best["model"],
            "dataset": best["dataset"],
            "delta_auc": best["sisa_delta_auc_vs_baseline"],
        },
        "largest_cell_auc_decline_vs_baseline": {
            "model": worst["model"],
            "dataset": worst["dataset"],
            "delta_auc": worst["sisa_delta_auc_vs_baseline"],
        },
    }


def protocol_rows() -> list[dict[str, object]]:
    return [
        {
            "series": "论文表值",
            "experiment_seed": "论文单次结果",
            "internal_rng_seeds": "未用于三-seed 计数",
            "gpu_protocol": "论文报告口径",
            "per_gpu_batch": "未在表中展示",
            "global_batch": "未在表中展示",
            "epochs": 1,
            "sequence_length": 100,
            "comparability": "四位小数发布值",
        },
        {
            "series": "本地 baseline",
            "experiment_seed": 20262027,
            "internal_rng_seeds": "dataloader=2026; SISA 参数不适用（旧默认值仅作审计）",
            "gpu_protocol": "4 GPU；L40S/RTX4090/H100 混合，cell 内 baseline/SISA 配对",
            "per_gpu_batch": 8192,
            "global_batch": 32768,
            "epochs": 1,
            "sequence_length": 100,
            "comparability": "本地复现基线，不是新增 seed",
        },
        {
            "series": "SISA seed 20262027",
            "experiment_seed": 20262027,
            "internal_rng_seeds": "dataloader=2026; SISA parameter=20260821（旧默认值推断）",
            "gpu_protocol": "4 GPU；L40S/RTX4090/H100 混合",
            "per_gpu_batch": 8192,
            "global_batch": 32768,
            "epochs": 1,
            "sequence_length": 100,
            "comparability": "与后两轮存在协议/硬件/drop_last 混杂",
        },
        {
            "series": "SISA seed 20262028",
            "experiment_seed": 20262028,
            "internal_rng_seeds": "dataloader=2027; SISA parameter=20260822",
            "gpu_protocol": "2×H100 80GB",
            "per_gpu_batch": 16384,
            "global_batch": 32768,
            "epochs": 1,
            "sequence_length": 100,
            "comparability": "与 seed 20262029 同协议",
        },
        {
            "series": "SISA seed 20262029",
            "experiment_seed": 20262029,
            "internal_rng_seeds": "dataloader=2028; SISA parameter=20260823",
            "gpu_protocol": "2×H100 80GB",
            "per_gpu_batch": 16384,
            "global_batch": 32768,
            "epochs": 1,
            "sequence_length": 100,
            "comparability": "与 seed 20262028 同协议；task 0 使用同协议 attempt2",
        },
    ]


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, headline: dict[str, object]) -> None:
    lines = [
        "# 论文 / baseline / 三-seed SISA 统一结果",
        "",
        "SISA 三次实验 seed 为 **20262027、20262028、20262029**。内部 dataloader seed 和 "
        "SISA parameter seed 只是各轮 RNG 子流，不计作额外实验 seed。",
        "",
        f"- 对齐范围：**{headline['cell_count']}** 个模型×数据集单元、**{headline['label_count']}** 个标签。",
        f"- 三-seed SISA 相对本地 baseline 的 cell-macro ΔAUC：**{float(headline['cell_macro_delta_auc_vs_baseline']):+.6f}**；"
        f"正向单元 **{headline['positive_auc_cells_vs_baseline']}/{headline['cell_count']}**。",
        f"- 三-seed SISA 相对论文表值的 cell-macro ΔAUC：**{float(headline['cell_macro_delta_auc_vs_paper']):+.6f}**；"
        f"正向单元 **{headline['positive_auc_cells_vs_paper']}/{headline['cell_count']}**。",
        f"- 三-seed SISA 相对 baseline 的 cell-macro Δlogloss：**{float(headline['cell_macro_delta_logloss_vs_baseline']):+.6f}** "
        "（负值更好）。",
        f"- 三轮最优 AUC 模型完全一致：**{headline['same_top_model_all_three_seeds']}/{headline['ranking_tasks']}** 个数据集×标签任务。",
        "",
        "seed 20262027 使用 4 卡混合硬件和每卡 batch 8192；seed 20262028/20262029 使用 "
        "2×H100 和每卡 batch 16384。三点均值、样本标准差和范围只能作为探索性描述，不能解释为纯 seed 方差或显著性检验。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=Path("experiments/sisa_three_seed_unified/sources/paper_table.csv"),
    )
    parser.add_argument(
        "--strict-metrics",
        type=Path,
        default=Path("experiments/sisa_native_strict/results/metrics.csv"),
    )
    parser.add_argument(
        "--expansion-metrics",
        type=Path,
        default=Path("experiments/sisa_expansion_acd/results/metrics.csv"),
    )
    parser.add_argument(
        "--seed-20262028-metrics",
        type=Path,
        default=Path("experiments/sisa_single_seed20262028/results/metrics.csv"),
    )
    parser.add_argument(
        "--seed-20262029-metrics",
        type=Path,
        default=Path("experiments/sisa_single_seed20262029/results/metrics.csv"),
    )
    parser.add_argument(
        "--seed-20262028-runs",
        type=Path,
        default=Path("experiments/sisa_single_seed20262028/results/runs.csv"),
    )
    parser.add_argument(
        "--seed-20262029-runs",
        type=Path,
        default=Path("experiments/sisa_single_seed20262029/results/runs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/sisa_three_seed_unified/results"),
    )
    args = parser.parse_args(argv)
    repository = args.repository.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    paper_rows = read_csv(resolve(args.paper_table))
    strict_rows = read_csv(resolve(args.strict_metrics))
    expansion_rows = read_csv(resolve(args.expansion_metrics))
    seed28_rows = read_csv(resolve(args.seed_20262028_metrics))
    seed29_rows = read_csv(resolve(args.seed_20262029_metrics))
    validate_seed_rows(seed28_rows, 20262028, "ws2_bs16384_acc1")
    validate_seed_rows(seed29_rows, 20262029, "ws2_bs16384_acc1")
    validate_run_audit(read_csv(resolve(args.seed_20262028_runs)), 20262028)
    validate_run_audit(read_csv(resolve(args.seed_20262029_runs)), 20262029)

    baseline_rows = select_legacy_metrics(strict_rows, expansion_rows, "baseline")
    seed27_rows = select_legacy_metrics(strict_rows, expansion_rows, "sisa")
    if len(baseline_rows) != EXPECTED_LABELS or len(seed27_rows) != EXPECTED_LABELS:
        raise ValueError("legacy source selection must produce 68 baseline and 68 SISA rows")

    paper = index_unique(paper_rows, "paper")
    baseline = index_unique(baseline_rows, "baseline")
    seeds = {
        20262027: index_unique(seed27_rows, "seed 20262027"),
        20262028: index_unique(seed28_rows, "seed 20262028"),
        20262029: index_unique(seed29_rows, "seed 20262029"),
    }
    validate_aligned_sources(paper, baseline, seeds)

    label_rows = build_label_rows(paper, baseline, seeds)
    cell_rows = build_cell_rows(label_rows)
    model_rows = summarize_cells(cell_rows, "model", MODELS)
    dataset_rows = summarize_cells(cell_rows, "dataset", DATASETS)
    headline = build_headline(label_rows, cell_rows)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "label_comparison.csv", label_rows)
    write_csv(output_dir / "cell_summary.csv", cell_rows)
    write_csv(output_dir / "model_summary.csv", model_rows)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "protocols.csv", protocol_rows())
    (output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(output_dir / "summary.md", headline)
    print(
        "SISA_THREE_SEED_COMPARISON "
        f"labels={len(label_rows)} cells={len(cell_rows)} "
        f"delta_auc_vs_baseline={headline['cell_macro_delta_auc_vs_baseline']:+.6f} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
