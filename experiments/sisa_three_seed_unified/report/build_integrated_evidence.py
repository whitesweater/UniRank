"""Add the accepted 20-cell and completed retry evidence to the existing report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MODELS = ("HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith")
DATASETS = (
    "QK_Video_Action",
    "KuaiRand_Video_Action",
    "TencentGR_10M_Action",
    "MerRec_Action",
)
DATASET_NAMES = {
    "QK_Video_Action": "QK-Video",
    "KuaiRand_Video_Action": "KuaiRand",
    "TencentGR_10M_Action": "TAAC-25",
    "MerRec_Action": "MerRec",
}
LABEL_COUNTS = dict(zip(DATASETS, (4, 6, 2, 5)))
SEEDS = (20262027, 20262028, 20262029)
STUDY = Path("experiments/sisa_three_seed_unified")
RETRY = Path("experiments/sisa_seed_variability_20260923")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_integrated_rows(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    older = read_csv(root / STUDY / "results/cell_summary.csv")
    one_trans = read_csv(
        root / "experiments/onetrans_seed_completion/results/selected_macro_summary.csv"
    )
    attempts = read_csv(root / RETRY / "results/final_comparison.csv")
    assert len(older) == 16 and len(one_trans) == 4 and len(attempts) == 10
    baseline = {(r["model"], r["dataset"]): r for r in older}
    baseline.update({("OneTrans", r["dataset"]): r for r in one_trans})
    assert len(baseline) == 20
    assert set(baseline) == {(m, d) for m in MODELS for d in DATASETS}

    repeated = {}
    fourth = {}
    jobs = set()
    for attempt in attempts:
        key = attempt["model"], attempt["dataset"]
        assert key in baseline
        assert attempt["job_task_id"] not in jobs
        jobs.add(attempt["job_task_id"])
        if attempt["category"] == "同种子复跑":
            seed = int(attempt["run_seed"])
            assert seed in SEEDS and int(attempt["reference_seed"]) == seed
            assert (key, seed) not in repeated
            repeated[key, seed] = attempt
        elif attempt["category"] == "新种子实验":
            assert int(attempt["run_seed"]) == 20262030 and key not in fourth
            fourth[key] = attempt
        else:
            raise ValueError(f"unexpected attempt category: {attempt['category']}")
    assert len(repeated) == 8 and len(fourth) == 2
    assert set(fourth) == {
        ("HyFormer", "MerRec_Action"), ("HiFormer", "MerRec_Action")
    }

    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            key = model, dataset
            source = baseline[key]
            row: dict[str, object] = {
                "model": model,
                "dataset_id": dataset,
                "dataset": DATASET_NAMES[dataset],
                "labels": LABEL_COUNTS[dataset],
                "has_retry": any((key, seed) in repeated for seed in SEEDS),
                "has_seed30": key in fourth,
                "retry_seed": next(
                    (seed for seed in SEEDS if (key, seed) in repeated), None
                ),
            }
            for metric in ("auc", "logloss"):
                base_column = (
                    f"baseline_mean_{metric}" if model != "OneTrans"
                    else f"baseline_{metric}"
                )
                base_value = float(source[base_column])
                assert math.isfinite(base_value)
                row[f"baseline_{metric}"] = base_value
                original_values = []
                latest_values = []
                for seed in SEEDS:
                    source_column = (
                        f"sisa_{seed}_mean_{metric}" if model != "OneTrans"
                        else f"sisa_{seed}_{metric}"
                    )
                    original = float(source[source_column])
                    assert math.isfinite(original)
                    selected = repeated.get((key, seed))
                    latest = original
                    if selected:
                        assert abs(float(selected[f"reference_{metric}"]) - original) < 1e-12
                        latest = float(selected[f"run_{metric}"])
                        row[f"latest_job_{seed}"] = selected["job_task_id"]
                    row[f"original_{seed}_{metric}"] = original
                    row[f"latest_{seed}_{metric}"] = latest
                    original_values.append(original)
                    latest_values.append(latest)
                row[f"original_three_seed_mean_{metric}"] = statistics.fmean(original_values)
                row[f"latest_three_seed_mean_{metric}"] = statistics.fmean(latest_values)
                if key in fourth:
                    trial = fourth[key]
                    assert abs(float(trial[f"reference_{metric}"]) -
                               row[f"original_{int(trial['reference_seed'])}_{metric}"]) < 1e-12
                    row[f"seed30_{metric}"] = float(trial[f"run_{metric}"])
                    row["seed30_job"] = trial["job_task_id"]
                else:
                    row[f"seed30_{metric}"] = None
            row["auc_delta_latest_vs_original"] = (
                row["latest_three_seed_mean_auc"] - row["original_three_seed_mean_auc"]
            )
            row["auc_delta_vs_baseline"] = (
                row["latest_three_seed_mean_auc"] - row["baseline_auc"]
            )
            row["logloss_improvement_vs_original"] = (
                row["original_three_seed_mean_logloss"] -
                row["latest_three_seed_mean_logloss"]
            )
            row["logloss_improvement_vs_baseline"] = (
                row["baseline_logloss"] - row["latest_three_seed_mean_logloss"]
            )
            rows.append(row)
    assert sum(r["labels"] for r in rows) == 85
    return rows


def signed(value: float) -> str:
    return f"{value:+.7f}" if abs(value) >= 0.00000005 else "0.0000000"


def display_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    displayed = []
    for row in rows:
        visible = {
            "model": row["model"],
            "dataset": row["dataset"],
            "retry_seed": str(row["retry_seed"])[-2:] if row["retry_seed"] else "—",
        }
        for metric in ("auc", "logloss"):
            visible[f"baseline_{metric}"] = f"{row[f'baseline_{metric}']:.7f}"
            for seed in SEEDS:
                visible[f"latest_{seed}_{metric}"] = f"{row[f'latest_{seed}_{metric}']:.7f}"
            visible[f"seed30_{metric}"] = (
                f"{row[f'seed30_{metric}']:.7f}" if row["has_seed30"] else ""
            )
            visible[f"original_three_seed_mean_{metric}"] = (
                f"{row[f'original_three_seed_mean_{metric}']:.7f}"
            )
            visible[f"latest_three_seed_mean_{metric}"] = (
                f"{row[f'latest_three_seed_mean_{metric}']:.7f}"
            )
        visible["auc_delta_vs_baseline"] = signed(row["auc_delta_vs_baseline"])
        visible["auc_delta_latest_vs_original"] = signed(
            row["auc_delta_latest_vs_original"]
        )
        visible["logloss_improvement_vs_baseline"] = signed(
            row["logloss_improvement_vs_baseline"]
        )
        visible["logloss_improvement_vs_original"] = signed(
            row["logloss_improvement_vs_original"]
        )
        displayed.append(visible)
    return displayed


def attempt_display_rows(root: Path) -> list[dict[str, str]]:
    result = []
    for row in read_csv(root / RETRY / "results/final_comparison.csv"):
        result.append({
            "model": row["model"],
            "dataset": DATASET_NAMES[row["dataset"]],
            "type": row["category"],
            "reference_seed": row["reference_seed"],
            "run_seed": row["run_seed"],
            "reference_auc": f"{float(row['reference_auc']):.7f}",
            "run_auc": f"{float(row['run_auc']):.7f}",
            "delta_auc": signed(float(row["delta_auc"])),
            "reference_logloss": f"{float(row['reference_logloss']):.7f}",
            "run_logloss": f"{float(row['run_logloss']):.7f}",
            "logloss_improvement": signed(-float(row["delta_logloss"])),
            "job_task_id": row["job_task_id"],
        })
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_source(source_id: str, label: str, path: str, inputs: list[str]) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header = true)",
            "description": label,
            "tables_used": inputs,
            "filters": ["five accepted models", "four accepted datasets"],
            "metric_definitions": [
                "cell AUC and LogLoss equal the arithmetic mean of test labels",
                "latest seed value selects the most recent audited completed attempt for that logical seed regardless of its metric",
                "the seed20262030 values are shown separately and excluded from the comparable three-seed mean",
                "positive logloss improvement means lower LogLoss",
            ],
        },
    }


def metric_table(metric: str) -> dict:
    is_auc = metric == "auc"
    suffix = "AUC" if is_auc else "LogLoss"
    improvement = (
        "auc_delta_vs_baseline" if is_auc else "logloss_improvement_vs_baseline"
    )
    latest_change = (
        "auc_delta_latest_vs_original" if is_auc else "logloss_improvement_vs_original"
    )
    columns = [
        {"field": "model", "label": "模型", "type": "text"},
        {"field": "dataset", "label": "数据集", "type": "text"},
        {"field": f"baseline_{metric}", "label": "Baseline", "type": "text", "align": "right"},
    ]
    columns.extend({"field": f"latest_{seed}_{metric}", "label": f"{str(seed)[-2:]} 最新",
                    "type": "text", "align": "right"} for seed in SEEDS)
    columns.extend([
        {"field": f"seed30_{metric}", "label": "30 新种子", "type": "text", "align": "right"},
        {"field": f"original_three_seed_mean_{metric}", "label": "原三种子均值",
         "type": "text", "align": "right"},
        {"field": f"latest_three_seed_mean_{metric}", "label": "最新三种子均值",
         "type": "text", "align": "right"},
        {"field": improvement, "label": "较 Baseline 改善", "type": "text",
         "align": "right", "movement": True},
        {"field": latest_change, "label": "较原三种子变化", "type": "text",
         "align": "right", "movement": True},
        {"field": "retry_seed", "label": "复跑种子", "type": "text", "align": "center"},
    ])
    return {
        "id": f"integrated_{metric}_table",
        "title": f"五模型×四数据集：{suffix} 的最终观测",
        "subtitle": "20 个组合；27/28/29 列取各逻辑种子的最新核验运行；30 单列且不进入三种子均值",
        "dataset": "integrated_cells",
        "sourceId": "integrated_20_cells_source",
        "layout": "full",
        "density": "dense",
        "columns": columns,
    }


def augment_artifact(artifact: dict, rows: list[dict[str, object]],
                     attempts: list[dict[str, str]]) -> dict:
    assert len(rows) == 20 and len(attempts) == 10
    manifest = artifact["manifest"]
    assert not any(b["id"] == "integrated_auc_table_block" for b in manifest["blocks"])
    original_blocks = manifest["blocks"][:]
    original_title = original_blocks[0]
    assert original_title["id"] == "title"
    origin_delta = statistics.fmean(
        r["original_three_seed_mean_auc"] - r["baseline_auc"] for r in rows
    )
    latest_delta = statistics.fmean(r["auc_delta_vs_baseline"] for r in rows)
    positive = sum(r["auc_delta_vs_baseline"] > 0 for r in rows)
    latest_logloss_improvement = statistics.fmean(
        r["logloss_improvement_vs_baseline"] for r in rows
    )
    title = "UniRank SISA 五模型整体结果与复跑对照"
    manifest["title"] = title
    manifest["description"] = (
        "20 个正式模型×数据集单元；三种子原记录、最新核验运行、复跑与新增种子。"
    )
    manifest["generatedAt"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    artifact["snapshot"]["generatedAt"] = manifest["generatedAt"]
    old_headline = dict(original_title, body=f"# {title}")
    summary = (
        "## 最新 20 组结果\n\n"
        f"五个模型×四个数据集已组成 **20 组、85 个标签**。按每个逻辑种子最新核验运行计算，"
        f"三种子 cell-macro AUC 相对历史单种子 Baseline 为 **{latest_delta:+.7f}**，"
        f"其中 **{positive}/20** 组高于 Baseline；对应 LogLoss 改善为 "
        f"**{latest_logloss_improvement:+.7f}**（正值表示降低）。"
        f"复跑前的 20 组三种子 AUC 差值为 **{origin_delta:+.7f}**。"
        "本轮针对低值组合选择性复跑，最新运行均值仅供查看复算后的整体观测，"
        "不按测试表现择优替换原始三种子证据。"
    )
    new_blocks = [
        old_headline,
        {"id": "integrated_summary", "type": "markdown",
         "sourceId": "integrated_20_cells_source", "body": summary},
        {"id": "integrated_auc_explanation", "type": "markdown",
         "sourceId": "integrated_20_cells_source", "body":
         "## AUC：原三种子与最新运行在同一行\n\n"
         "27/28/29 列是各逻辑种子的最新有效 AUC；“原三种子均值”保留首次选定结果，"
         "“最新三种子均值”以固定的最新完成运行规则重算。复跑种子列标出有更新的行。"
         "只有 HyFormer–MerRec 与 HiFormer–MerRec 有第四种子 30，它以蓝色单列，"
         "不进入其他模型的三种子均值。绿色表示 AUC 上升，红色表示下降，灰色表示零变化。"},
        {"id": "integrated_auc_table_block", "type": "table",
         "tableId": "integrated_auc_table"},
        {"id": "integrated_logloss_explanation", "type": "markdown",
         "sourceId": "integrated_20_cells_source", "body":
         "## LogLoss：较低更好，改善方向已统一\n\n"
         "这里的两列变化都定义为“旧值减新值”，因此绿色正值代表 LogLoss 降低，"
         "红色负值代表升高。OneTrans–TAAC-25 seed29 修复后 LogLoss 为 0.0447520，"
         "仍高于该模型 seed27 的 0.0334015 和 seed28 的 0.0420805。"
         "Zenith–TAAC-25 复跑 AUC 上升，但 LogLoss 升高。"},
        {"id": "integrated_logloss_table_block", "type": "table",
         "tableId": "integrated_logloss_table"},
        {"id": "integrated_attempts_explanation", "type": "markdown",
         "sourceId": "integrated_attempts_source", "body":
         "## 10 次新增有效运行的原值与新值\n\n"
         "八个同种子逻辑任务全部完成，另有两个 seed20262030 新种子。"
         "OneTrans–TAAC-25 的首次复跑 645228_5 在训练前因计算节点缺少 Git 失败；"
         "表中仅列修复后完成的 645445_5。两项同种子完全一致的 MerRec 组合"
         "新增 seed30 后方向相反：HyFormer AUC 降低，HiFormer AUC 升高。"
         "这张表只描述这些选定组合的变化，不把较高的测试值自动升格为论文主结果。"},
        {"id": "integrated_attempts_table_block", "type": "table",
         "tableId": "integrated_attempts_table"},
        {"id": "historical_16_scope", "type": "markdown", "body":
         "## 历史 16 组报告（原始快照）\n\n"
         "以下保留原四模型、68 标签的三种子报告及图表，数值口径停留在原始运行。"
         "上述 20 组新表包含 OneTrans 和本轮复跑。请按各自的范围阅读，"
         "不要把以下历史 16 组汇总当作当前 20 组汇总。"},
    ]
    manifest["blocks"] = new_blocks + original_blocks[1:]
    assert manifest["blocks"][len(new_blocks):] == original_blocks[1:]

    manifest["tables"].extend([metric_table("auc"), metric_table("logloss"), {
        "id": "integrated_attempts_table",
        "title": "本轮复跑与新种子逐任务对照",
        "subtitle": "10 次有效运行；AUC 差值为本次减参考，LogLoss 改善为参考减本次",
        "dataset": "integrated_attempts",
        "sourceId": "integrated_attempts_source",
        "layout": "full",
        "density": "dense",
        "columns": [
            {"field": "type", "label": "类型", "type": "text"},
            {"field": "model", "label": "模型", "type": "text"},
            {"field": "dataset", "label": "数据集", "type": "text"},
            {"field": "reference_seed", "label": "参考种子", "type": "text"},
            {"field": "run_seed", "label": "本次种子", "type": "text"},
            {"field": "reference_auc", "label": "参考 AUC", "type": "text", "align": "right"},
            {"field": "run_auc", "label": "本次 AUC", "type": "text", "align": "right"},
            {"field": "delta_auc", "label": "ΔAUC", "type": "text", "align": "right", "movement": True},
            {"field": "reference_logloss", "label": "参考 LogLoss", "type": "text", "align": "right"},
            {"field": "run_logloss", "label": "本次 LogLoss", "type": "text", "align": "right"},
            {"field": "logloss_improvement", "label": "LogLoss 改善", "type": "text", "align": "right", "movement": True},
            {"field": "job_task_id", "label": "作业", "type": "text"},
        ],
    }])
    artifact["snapshot"]["datasets"]["integrated_cells"] = display_rows(rows)
    artifact["snapshot"]["datasets"]["integrated_attempts"] = attempts
    cell_path = "experiments/sisa_three_seed_unified/results/integrated_cells.csv"
    attempts_path = "experiments/sisa_seed_variability_20260923/results/final_comparison.csv"
    sources = [
        new_source("integrated_20_cells_source", "20 组原三种子与最新有效运行复算", cell_path,
                   ["experiments/sisa_three_seed_unified/results/cell_summary.csv",
                    "experiments/onetrans_seed_completion/results/selected_macro_summary.csv",
                    attempts_path]),
        new_source("integrated_attempts_source", "10 次有效复跑及新增种子的审计对照",
                   attempts_path, [attempts_path,
                                   "experiments/sisa_seed_variability_20260923/audit/final_result_verification.json"]),
    ]
    manifest["sources"].extend(sources)
    artifact["sources"].extend(sources)
    return artifact


def repair_static_tables(report_html: str, artifact: dict) -> str:
    """Restore exact display strings and every row in the no-script HTML tables.

    The packaged fallback rounds numeric-looking text to two decimals and caps
    static tables at 15 rows. The enhanced reader reads the exact snapshot.
    """
    target_ids = {
        "integrated_auc_table", "integrated_logloss_table", "integrated_attempts_table"
    }
    tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
    for table_id in target_ids:
        table = tables[table_id]
        columns = table["columns"]
        rows = artifact["snapshot"]["datasets"][table["dataset"]]
        parts = [f'<table><caption>{html.escape(table["title"])}</caption><thead><tr>']
        parts.extend(
            f'<th scope="col">{html.escape(column["label"])}</th>' for column in columns
        )
        parts.append("</tr></thead><tbody>")
        for row in rows:
            parts.append("<tr>")
            for column in columns:
                raw = row.get(column["field"])
                value = "—" if raw is None or raw == "" else str(raw)
                classes = []
                if column.get("align") == "right":
                    classes.append("portable-table-number")
                elif column.get("align") == "center":
                    classes.append("portable-table-center")
                if raw is None or raw == "":
                    classes.append("portable-table-empty")
                if column.get("movement"):
                    if value.startswith("+"):
                        classes.append("portable-table-positive")
                    elif value.startswith("-"):
                        classes.append("portable-table-negative")
                attributes = f' class="{" ".join(classes)}"' if classes else ""
                parts.append(f"<td{attributes}>{html.escape(value)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        replacement = "".join(parts)
        section_pattern = re.compile(
            r'<section\b[^>]*\bdata-table-id="' + re.escape(table_id) +
            r'"[^>]*>.*?</section>', re.DOTALL
        )

        def replace_section(match: re.Match[str]) -> str:
            section, changed = re.subn(
                r"<table\b[^>]*>.*?</table>", lambda _: replacement,
                match.group(), count=1, flags=re.DOTALL,
            )
            if changed != 1:
                raise ValueError(f"static table not found for {table_id}")
            return re.sub(
                r'<p class="portable-table-note">.*?</p>',
                lambda _: f'<p class="portable-table-note">{len(rows)} results</p>',
                section, count=1, flags=re.DOTALL,
            )

        report_html, count = section_pattern.subn(replace_section, report_html)
        if count != 1:
            raise ValueError(f"expected one static section for {table_id}, got {count}")
    return report_html


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    artifact_path = root / STUDY / "report/artifact.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-table-fallback", action="store_true")
    args = parser.parse_args()
    if args.repair_table_fallback:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        sys.stdout.write(repair_static_tables(sys.stdin.read(), artifact))
        return
    rows = build_integrated_rows(root)
    attempts = attempt_display_rows(root)
    write_csv(root / STUDY / "results/integrated_cells.csv", rows)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    augment_artifact(artifact, rows, attempts)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Integrated 20 cells, {sum(r['has_retry'] for r in rows)} retry cells, "
          f"and {sum(r['has_seed30'] for r in rows)} fourth-seed cells.")


if __name__ == "__main__":
    main()
