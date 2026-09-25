"""Transpose the existing 20-cell display to model rows and dataset columns."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
MODELS = ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith")
DATASETS = (
    ("QK_Video_Action", "QK-Video"),
    ("KuaiRand_Video_Action", "KuaiRand"),
    ("TencentGR_10M_Action", "TAAC-25"),
    ("MerRec_Action", "MerRec"),
)
SEEDS = (20262027, 20262028, 20262029)


def display(value: str) -> str:
    return f"{float(value):.7f}"


def signed(value: str) -> str:
    numeric = float(value)
    return f"{numeric:+.7f}" if abs(numeric) >= 0.00000005 else "0.0000000"


def mark(value: float) -> str:
    if value > 0.00000005:
        return "🟢"
    if value < -0.00000005:
        return "🔴"
    return "⚪"


def cell_text(row: dict[str, str], metric: str) -> str:
    auc = metric == "auc"
    result = [f"Baseline {display(row[f'baseline_{metric}'])}"]
    retry_seed = str(row.get("retry_seed") or "")
    latest_mean = display(row[f"latest_three_seed_mean_{metric}"])
    improvement = (
        row["auc_delta_vs_baseline"] if auc else
        row["logloss_improvement_vs_baseline"]
    )
    result.append(f"三种子均值 {latest_mean} {mark(float(improvement))} {signed(improvement)}")
    if retry_seed:
        seed = int(retry_seed)
        assert seed in SEEDS
        original = display(row[f"original_{seed}_{metric}"])
        latest = display(row[f"latest_{seed}_{metric}"])
        label = str(seed)[-2:]
        favorable_change = float(latest) - float(original)
        if not auc:
            favorable_change = -favorable_change
        result.append(f"{label} {original} → {latest} {mark(favorable_change)}")
    fourth = row.get(f"seed30_{metric}")
    if fourth:
        result.append(f"🔵 30 {display(fourth)}")
    return "\n".join(result)


def matrix_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed = {(r["model"], r["dataset_id"]): r for r in rows}
    assert len(rows) == 20 and len(indexed) == 20
    assert set(indexed) == {(m, d) for m in MODELS for d, _ in DATASETS}
    matrix = []
    for model in MODELS:
        displayed = {"model": model}
        for dataset, _ in DATASETS:
            row = indexed[model, dataset]
            for metric in ("auc", "logloss"):
                displayed[f"{dataset}_{metric}"] = cell_text(row, metric)
        matrix.append(displayed)
    return matrix


def transpose(artifact: dict, matrix: list[dict[str, str]]) -> dict:
    assert len(matrix) == 5
    artifact["snapshot"]["datasets"]["integrated_matrix"] = matrix
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    blocks = {block["id"]: block for block in artifact["manifest"]["blocks"]}
    for metric, label in (("auc", "AUC"), ("logloss", "LogLoss")):
        table = tables[f"integrated_{metric}_table"]
        table["dataset"] = "integrated_matrix"
        table["title"] = f"{label}：模型为行，数据集为列"
        table["subtitle"] = (
            "每格保留 Baseline、最新三种子均值及同种子多次运行；"
            "只有两组有新增 seed30"
        )
        table["columns"] = [
            {"field": "model", "label": "模型", "type": "text"},
            *[
                {"field": f"{dataset}_{metric}", "label": name, "type": "text"}
                for dataset, name in DATASETS
            ],
        ]
    blocks["integrated_auc_explanation"]["body"] = (
        "## AUC：模型为行，数据集为列\n\n"
        "每格列出 Baseline、最新三种子均值和相对 Baseline 的变化。"
        "同一种子有两次有效运行时用“原值 → 复跑值”完整展示，即使两次完全一致；"
        "🟢 表示 AUC 提升，🔴 表示下降，⚪ 表示一致；"
        "🔵 是仅两组有记录的 seed30，它不进入三种子均值。窄屏可左右滑动表格。"
    )
    blocks["integrated_logloss_explanation"]["body"] = (
        "## LogLoss：同样按模型行和数据集列查看\n\n"
        "每格显示 Baseline、最新三种子均值以及存在复跑时的原值与新值。"
        "较 Baseline 的改善定义为 Baseline 减最新三种子均值；"
        "🟢 表示 LogLoss 降低，🔴 表示升高，⚪ 表示一致；🔵 标出新增 seed30。"
        "OneTrans–TAAC-25 seed29 复跑后为 0.0447520，仍高于该模型 seed27 的 0.0334015"
        "和 seed28 的 0.0420805。"
    )
    return artifact


def main() -> None:
    source = ROOT / "experiments/sisa_three_seed_unified/results/integrated_cells.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    artifact_path = REPORT / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    updated = transpose(artifact, matrix_rows(rows))
    artifact_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Transposed two main tables to five model rows by four dataset columns.")


if __name__ == "__main__":
    main()
