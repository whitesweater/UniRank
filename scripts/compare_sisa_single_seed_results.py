#!/usr/bin/env python3
"""Compare the new SISA seed bundle with the finalized previous SISA results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
DATASETS = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "TencentGR_10M_Action",
    "MerRec_Action",
)
STRICT_MODELS = frozenset({"HiFormer", "RankMixer", "Zenith"})
STRICT_DATASETS = frozenset(
    {"QK_Video_Action", "KuaiRand_Video_Action", "MerRec_Action"}
)
KEY_FIELDS = ("model", "dataset", "label")
EPSILON = 1e-12


def seed_study_paths(
    seed: int,
    new_metrics: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    if not 0 <= seed < 2**32 - 1:
        raise ValueError(f"seed must be in [0, {2**32 - 2}], got {seed}")
    expected_root = Path(f"experiments/sisa_single_seed{seed}")
    expected_metrics = expected_root / "results/metrics.csv"
    expected_output = expected_root / "comparison"
    selected_metrics = new_metrics or expected_metrics
    selected_output = output_dir or expected_output
    if selected_metrics != expected_metrics:
        raise ValueError(
            f"--new-metrics must match the seed-specific path: {expected_metrics}"
        )
    if selected_output != expected_output:
        raise ValueError(
            f"--output-dir must match the seed-specific path: {expected_output}"
        )
    return selected_metrics, selected_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def metric_key(row: dict[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[field]) for field in KEY_FIELDS)  # type: ignore[return-value]


def select_previous_metrics(
    strict_rows: Iterable[dict[str, str]],
    expansion_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in strict_rows:
        if (
            row.get("setting") == "sisa"
            and row.get("model") in STRICT_MODELS
            and row.get("dataset") in STRICT_DATASETS
        ):
            normalized = dict(row)
            normalized["previous_source"] = "sisa_native_strict"
            normalized["previous_gpu"] = row.get("gpu_type", "unknown")
            normalized["previous_protocol"] = "ws4_bs8192_acc1_seed20262027_mixed_gpu"
            selected.append(normalized)

    for row in expansion_rows:
        model = row.get("model")
        dataset = row.get("dataset")
        if (
            row.get("setting") == "sisa"
            and model in MODELS
            and dataset in DATASETS
            and (model == "HyFormer" or dataset == "TencentGR_10M_Action")
        ):
            normalized = dict(row)
            normalized["previous_source"] = "sisa_expansion_acd"
            normalized["previous_gpu"] = row.get("gpu_name", "unknown")
            normalized["previous_protocol"] = "ws4_bs8192_acc1_seed20262027_h100"
            selected.append(normalized)
    return selected


def index_unique(
    rows: Iterable[dict[str, str]], source_name: str
) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = metric_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {source_name} metric key: {key}")
        indexed[key] = row
    return indexed


def direction(delta: float, *, lower_is_better: bool = False) -> str:
    if abs(delta) <= EPSILON:
        return "tie"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "declined"


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot calculate a mean over zero values")
    return statistics.fmean(materialized)


def compare_metrics(
    new_rows: Iterable[dict[str, str]],
    previous_rows: Iterable[dict[str, str]],
) -> list[dict[str, object]]:
    new_index = index_unique(new_rows, "new")
    previous_index = index_unique(previous_rows, "previous")
    new_keys = set(new_index)
    previous_keys = set(previous_index)
    if new_keys != previous_keys:
        missing = sorted(new_keys - previous_keys)
        extra = sorted(previous_keys - new_keys)
        raise ValueError(
            f"metric keys do not align; missing_previous={missing}, extra_previous={extra}"
        )

    rows: list[dict[str, object]] = []
    for key in sorted(new_keys):
        new = new_index[key]
        previous = previous_index[key]
        previous_auc = float(previous["AUC"])
        new_auc = float(new["AUC"])
        previous_logloss = float(previous["logloss"])
        new_logloss = float(new["logloss"])
        values = (previous_auc, new_auc, previous_logloss, new_logloss)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite metric for {key}: {values}")
        delta_auc = new_auc - previous_auc
        delta_logloss = new_logloss - previous_logloss
        auc_direction = direction(delta_auc)
        logloss_direction = direction(delta_logloss, lower_is_better=True)
        if auc_direction == "improved" and logloss_direction == "improved":
            joint_direction = "both_improved"
        elif auc_direction == "declined" and logloss_direction == "declined":
            joint_direction = "both_declined"
        else:
            joint_direction = "mixed"
        rows.append(
            {
                "model": key[0],
                "dataset": key[1],
                "label": key[2],
                "previous_source": previous["previous_source"],
                "previous_protocol": previous["previous_protocol"],
                "previous_gpu": previous["previous_gpu"],
                "previous_seed": int(previous.get("seed", 20262027)),
                "new_protocol": new["protocol"],
                "new_seed": int(new["seed"]),
                "previous_auc": previous_auc,
                "new_auc": new_auc,
                "delta_auc": delta_auc,
                "previous_logloss": previous_logloss,
                "new_logloss": new_logloss,
                "delta_logloss": delta_logloss,
                "auc_direction": auc_direction,
                "logloss_direction": logloss_direction,
                "joint_direction": joint_direction,
            }
        )
    return rows


def summarize_units(label_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in label_rows:
        grouped[(str(row["model"]), str(row["dataset"]))].append(row)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in DATASETS:
            group = grouped.get((model, dataset), [])
            if not group:
                raise ValueError(f"missing model-dataset unit: {(model, dataset)}")
            auc_deltas = [float(row["delta_auc"]) for row in group]
            logloss_deltas = [float(row["delta_logloss"]) for row in group]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "labels": len(group),
                    "mean_delta_auc": mean(auc_deltas),
                    "median_delta_auc": statistics.median(auc_deltas),
                    "mean_abs_delta_auc": mean(abs(value) for value in auc_deltas),
                    "auc_improved": sum(value > EPSILON for value in auc_deltas),
                    "auc_declined": sum(value < -EPSILON for value in auc_deltas),
                    "mean_delta_logloss": mean(logloss_deltas),
                    "logloss_improved": sum(value < -EPSILON for value in logloss_deltas),
                    "logloss_worsened": sum(value > EPSILON for value in logloss_deltas),
                    "joint_improved": sum(
                        row["joint_direction"] == "both_improved" for row in group
                    ),
                    "previous_source": group[0]["previous_source"],
                }
            )
    return rows


def summarize_dimension(
    unit_rows: Sequence[dict[str, object]],
    label_rows: Sequence[dict[str, object]],
    dimension: str,
    ordered_values: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in ordered_values:
        units = [row for row in unit_rows if row[dimension] == value]
        labels = [row for row in label_rows if row[dimension] == value]
        rows.append(
            {
                dimension: value,
                "cells": len(units),
                "labels": len(labels),
                "unit_macro_delta_auc": mean(float(row["mean_delta_auc"]) for row in units),
                "positive_auc_cells": sum(
                    float(row["mean_delta_auc"]) > EPSILON for row in units
                ),
                "label_weighted_delta_auc": mean(
                    float(row["delta_auc"]) for row in labels
                ),
                "unit_macro_delta_logloss": mean(
                    float(row["mean_delta_logloss"]) for row in units
                ),
                "lower_logloss_cells": sum(
                    float(row["mean_delta_logloss"]) < -EPSILON for row in units
                ),
            }
        )
    return rows


def summarize_rankings(label_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in label_rows:
        grouped[(str(row["dataset"]), str(row["label"]))].append(row)
    rows: list[dict[str, object]] = []
    for (dataset, label), group in sorted(grouped.items()):
        previous_best = max(group, key=lambda row: float(row["previous_auc"]))
        new_best = max(group, key=lambda row: float(row["new_auc"]))
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "previous_best_model": previous_best["model"],
                "previous_best_auc": previous_best["previous_auc"],
                "new_best_model": new_best["model"],
                "new_best_auc": new_best["new_auc"],
                "same_best_model": previous_best["model"] == new_best["model"],
            }
        )
    return rows


def build_headline(
    label_rows: Sequence[dict[str, object]],
    unit_rows: Sequence[dict[str, object]],
    ranking_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    auc_deltas = [float(row["delta_auc"]) for row in label_rows]
    logloss_deltas = [float(row["delta_logloss"]) for row in label_rows]
    best = max(label_rows, key=lambda row: float(row["delta_auc"]))
    worst = min(label_rows, key=lambda row: float(row["delta_auc"]))
    return {
        "label_count": len(label_rows),
        "cell_count": len(unit_rows),
        "mean_delta_auc_label_weighted": mean(auc_deltas),
        "mean_delta_auc_cell_macro": mean(
            float(row["mean_delta_auc"]) for row in unit_rows
        ),
        "median_delta_auc": statistics.median(auc_deltas),
        "mean_abs_delta_auc": mean(abs(value) for value in auc_deltas),
        "median_abs_delta_auc": statistics.median(abs(value) for value in auc_deltas),
        "max_abs_delta_auc": max(abs(value) for value in auc_deltas),
        "auc_improved_labels": sum(value > EPSILON for value in auc_deltas),
        "auc_declined_labels": sum(value < -EPSILON for value in auc_deltas),
        "positive_auc_cells": sum(
            float(row["mean_delta_auc"]) > EPSILON for row in unit_rows
        ),
        "auc_within_0_001": sum(abs(value) <= 0.001 for value in auc_deltas),
        "auc_within_0_005": sum(abs(value) <= 0.005 for value in auc_deltas),
        "auc_within_0_010": sum(abs(value) <= 0.010 for value in auc_deltas),
        "mean_delta_logloss_label_weighted": mean(logloss_deltas),
        "mean_delta_logloss_cell_macro": mean(
            float(row["mean_delta_logloss"]) for row in unit_rows
        ),
        "logloss_improved_labels": sum(value < -EPSILON for value in logloss_deltas),
        "logloss_worsened_labels": sum(value > EPSILON for value in logloss_deltas),
        "lower_logloss_cells": sum(
            float(row["mean_delta_logloss"]) < -EPSILON for row in unit_rows
        ),
        "joint_improved_labels": sum(
            row["joint_direction"] == "both_improved" for row in label_rows
        ),
        "joint_declined_labels": sum(
            row["joint_direction"] == "both_declined" for row in label_rows
        ),
        "same_best_model_tasks": sum(bool(row["same_best_model"]) for row in ranking_rows),
        "ranking_tasks": len(ranking_rows),
        "previous_source_counts": dict(
            Counter(str(row["previous_source"]) for row in label_rows)
        ),
        "largest_auc_improvement": {
            field: best[field]
            for field in ("model", "dataset", "label", "delta_auc")
        },
        "largest_auc_decline": {
            field: worst[field]
            for field in ("model", "dataset", "label", "delta_auc")
        },
    }


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


def write_summary(
    path: Path,
    headline: dict[str, object],
    model_rows: Sequence[dict[str, object]],
    dataset_rows: Sequence[dict[str, object]],
    comparison_note: str | None = None,
) -> None:
    if comparison_note is None:
        comparison_note = (
            "The candidate seed has a small positive mean AUC shift but weaker "
            "logloss consistency. Because GPU topology, per-GPU batch, "
            "blocked-loader drop boundaries, dataloader seed, and SISA "
            "initialization seed also changed, these are descriptive "
            "repeatability deltas, not a pure seed effect or significance estimate."
        )
    lines = [
        "# SISA seed/protocol comparison snapshot",
        "",
        comparison_note,
        "",
        f"- Aligned labels: **{headline['label_count']}** across **{headline['cell_count']}** cells",
        f"- Label-weighted mean ΔAUC: **{float(headline['mean_delta_auc_label_weighted']):+.6f}**",
        f"- Cell-macro mean ΔAUC: **{float(headline['mean_delta_auc_cell_macro']):+.6f}**",
        f"- AUC improved: **{headline['auc_improved_labels']}/{headline['label_count']}** labels and "
        f"**{headline['positive_auc_cells']}/{headline['cell_count']}** cells",
        f"- Label-weighted mean Δlogloss: **{float(headline['mean_delta_logloss_label_weighted']):+.6f}** "
        "(negative is better)",
        f"- Logloss improved: **{headline['logloss_improved_labels']}/{headline['label_count']}** labels and "
        f"**{headline['lower_logloss_cells']}/{headline['cell_count']}** cells",
        f"- AUC changes within ±0.005: **{headline['auc_within_0_005']}/{headline['label_count']}**",
        f"- Same top model for each dataset-label task: **{headline['same_best_model_tasks']}/{headline['ranking_tasks']}**",
        "",
        "## Model-level cell-macro deltas",
        "",
        "| Model | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['cells']} | {row['labels']} | "
            f"{float(row['unit_macro_delta_auc']):+.6f} | {row['positive_auc_cells']}/{row['cells']} | "
            f"{float(row['unit_macro_delta_logloss']):+.6f} | {row['lower_logloss_cells']}/{row['cells']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-level cell-macro deltas",
            "",
            "| Dataset | Cells | Labels | Mean ΔAUC | Positive cells | Mean Δlogloss | Lower-logloss cells |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['cells']} | {row['labels']} | "
            f"{float(row['unit_macro_delta_auc']):+.6f} | {row['positive_auc_cells']}/{row['cells']} | "
            f"{float(row['unit_macro_delta_logloss']):+.6f} | {row['lower_logloss_cells']}/{row['cells']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=20262028)
    parser.add_argument(
        "--new-metrics",
        type=Path,
        default=None,
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
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    repository = args.repository.resolve()

    try:
        new_metrics_path, output_path = seed_study_paths(
            args.seed,
            args.new_metrics,
            args.output_dir,
        )
    except ValueError as error:
        parser.error(str(error))

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repository / path

    new_rows = read_csv(resolve(new_metrics_path))
    observed_seeds = {int(row["seed"]) for row in new_rows}
    if observed_seeds != {args.seed}:
        parser.error(
            f"new metrics seed mismatch: expected {args.seed}, found {sorted(observed_seeds)}"
        )
    previous_rows = select_previous_metrics(
        read_csv(resolve(args.strict_metrics)), read_csv(resolve(args.expansion_metrics))
    )
    label_rows = compare_metrics(new_rows, previous_rows)
    if len(label_rows) != 68:
        raise SystemExit(f"expected 68 aligned label metrics, found {len(label_rows)}")
    unit_rows = summarize_units(label_rows)
    ranking_rows = summarize_rankings(label_rows)
    model_rows = summarize_dimension(unit_rows, label_rows, "model", MODELS)
    dataset_rows = summarize_dimension(unit_rows, label_rows, "dataset", DATASETS)
    headline = build_headline(label_rows, unit_rows, ranking_rows)

    output_dir = resolve(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "label_deltas.csv", label_rows)
    write_csv(output_dir / "unit_summary.csv", unit_rows)
    write_csv(output_dir / "model_summary.csv", model_rows)
    write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(output_dir / "ranking_stability.csv", ranking_rows)
    (output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(output_dir / "summary.md", headline, model_rows, dataset_rows)
    print(
        "SISA_SEED_COMPARISON "
        f"labels={len(label_rows)} cells={len(unit_rows)} "
        f"mean_delta_auc={headline['mean_delta_auc_label_weighted']:+.6f} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
