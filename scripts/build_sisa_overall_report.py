#!/usr/bin/env python3
"""Add model/dataset/action detail to the canonical SISA overall report."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import html
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Sequence


MODEL_ORDER = (
    "UltraHSTU",
    "HiFormer",
    "HyFormer",
    "OneTrans",
    "UniMixer",
    "RankMixer",
    "Zenith",
)
ACCEPTED_MODELS = ("HiFormer", "HyFormer", "RankMixer", "Zenith")
ACCEPTED_DATASETS = frozenset(
    {
        "QK_Video_Action",
        "KuaiRand_Video_Action",
        "TencentGR_10M_Action",
        "MerRec_Action",
    }
)
DATASET_SPECS = (
    {
        "dataset": "QK_Video_Action",
        "display": "QK-Video",
        "id": "qk_video",
        "labels": ("click", "follow", "like", "share"),
    },
    {
        "dataset": "KuaiRand_Video_Action",
        "display": "KuaiRand",
        "id": "kuairand",
        "labels": (
            "is_click",
            "is_follow",
            "is_like",
            "is_comment",
            "is_forward",
            "long_view",
        ),
    },
    {
        "dataset": "Taobao_Action",
        "display": "Taobao",
        "id": "taobao",
        "labels": ("is_click", "cart", "fav", "buy"),
    },
    {
        "dataset": "MerRec_Action",
        "display": "MerRec",
        "id": "merrec",
        "labels": ("Like", "Cart", "Offer", "Checkout", "Purchase"),
    },
    {
        "dataset": "TencentGR_10M_Action",
        "display": "TencentGR",
        "id": "tencentgr",
        "labels": ("is_click", "is_conversion"),
    },
)
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value: object, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value for {context}: {value}")
    return number


def action_key(row: dict[str, object]) -> tuple[str, str, str]:
    return str(row["model"]), str(row["dataset"]), str(row["label"])


def build_action_rows(repository: Path) -> list[dict[str, object]]:
    strict_path = (
        repository / "experiments/sisa_native_strict/results/metrics.csv"
    )
    expansion_path = (
        repository
        / "experiments/sisa_expansion_acd/results/paired_summary.csv"
    )
    strict_pairs: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in read_csv(strict_path):
        setting = row["setting"]
        if setting not in {"baseline", "sisa"}:
            continue
        key = action_key(row)
        pair = strict_pairs.setdefault(
            key,
            {"model": key[0], "dataset": key[1], "label": key[2]},
        )
        if setting in pair:
            raise ValueError(f"duplicate strict {setting} metric: {key}")
        pair[setting] = {
            "auc": finite(row["AUC"], f"{key} {setting} AUC"),
            "logloss": finite(
                row["logloss"], f"{key} {setting} logloss"
            ),
        }

    combined: list[dict[str, object]] = []
    for key, pair in strict_pairs.items():
        if "baseline" not in pair or "sisa" not in pair:
            raise ValueError(f"incomplete strict baseline/SISA pair: {key}")
        baseline = pair["baseline"]
        sisa = pair["sisa"]
        if not isinstance(baseline, dict) or not isinstance(sisa, dict):
            raise TypeError(f"invalid strict pair payload: {key}")
        combined.append(
            {
                "model": pair["model"],
                "dataset": pair["dataset"],
                "label": pair["label"],
                "baseline_auc": baseline["auc"],
                "sisa_auc": sisa["auc"],
                "baseline_logloss": baseline["logloss"],
                "sisa_logloss": sisa["logloss"],
                "source": "DIGAI Lab strict",
            }
        )

    for row in read_csv(expansion_path):
        key = action_key(row)
        combined.append(
            {
                "model": key[0],
                "dataset": key[1],
                "label": key[2],
                "baseline_auc": finite(row["baseline_auc"], f"{key} baseline AUC"),
                "sisa_auc": finite(row["sisa_auc"], f"{key} SISA AUC"),
                "baseline_logloss": finite(
                    row["baseline_logloss"], f"{key} baseline logloss"
                ),
                "sisa_logloss": finite(
                    row["sisa_logloss"], f"{key} SISA logloss"
                ),
                "source": "HPC3 expansion",
            }
        )

    indexed: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in combined:
        key = action_key(row)
        if key in indexed:
            raise ValueError(f"duplicate combined action metric: {key}")
        baseline_auc = float(row["baseline_auc"])
        sisa_auc = float(row["sisa_auc"])
        baseline_logloss = float(row["baseline_logloss"])
        sisa_logloss = float(row["sisa_logloss"])
        indexed[key] = {
            **row,
            "delta_auc": sisa_auc - baseline_auc,
            "delta_logloss": sisa_logloss - baseline_logloss,
            "evidence_scope": (
                "accepted"
                if row["model"] in ACCEPTED_MODELS
                and row["dataset"] in ACCEPTED_DATASETS
                else "diagnostic"
            ),
        }

    ordered: list[dict[str, object]] = []
    for spec in DATASET_SPECS:
        for label in spec["labels"]:
            for model in MODEL_ORDER:
                key = (model, str(spec["dataset"]), str(label))
                if key not in indexed:
                    raise ValueError(f"missing combined action metric: {key}")
                ordered.append(indexed.pop(key))
    if indexed:
        raise ValueError(f"unexpected combined action metrics: {sorted(indexed)}")
    if len(ordered) != 147:
        raise ValueError(
            f"combined report must contain 147 action rows, found {len(ordered)}"
        )
    query_path = repository / "experiments/sisa_native_strict/migration/action_breakdown.sql"
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        queried = [dict(row) for row in connection.execute(
            query_path.read_text(encoding="utf-8"),
            {"action_rows_json": json.dumps(ordered, ensure_ascii=False)},
        )]
    if queried != ordered:
        raise ValueError("action presentation query changed the audited rows")
    return queried


def action_table(
    spec: dict[str, object], scope: str, row_count: int
) -> dict[str, object]:
    labels = tuple(spec["labels"])
    scope_label = "Accepted 证据" if scope == "accepted" else "Diagnostic 附录"
    return {
        "id": f"{scope}_action_{spec['id']}_table",
        "title": f"{spec['display']}：{scope_label}逐动作结果",
        "subtitle": (
            f"覆盖 {len(labels)} 个动作类别、{row_count} 个模型×类别配对；"
            "所有指标固定显示 6 位小数；Δ=SISA−baseline，"
            "正 ΔAUC、负 ΔLogloss 表示改善"
        ),
        "dataset": f"{scope}_action_{spec['id']}",
        "sourceId": "action_breakdown",
        "layout": "full",
        "density": "dense",
        "defaultSort": {"field": "label", "direction": "asc"},
        "columns": [
            {"field": "label", "label": "动作类别", "type": "text"},
            {"field": "model", "label": "模型", "type": "text"},
            {
                "field": "baseline_auc_display",
                "label": "Baseline AUC",
                "type": "text",
                "align": "right",
            },
            {
                "field": "sisa_auc_display",
                "label": "SISA AUC",
                "type": "text",
                "align": "right",
            },
            {
                "field": "delta_auc_display",
                "label": "ΔAUC",
                "type": "text",
                "align": "right",
            },
            {
                "field": "baseline_logloss_display",
                "label": "Baseline Logloss",
                "type": "text",
                "align": "right",
            },
            {
                "field": "sisa_logloss_display",
                "label": "SISA Logloss",
                "type": "text",
                "align": "right",
            },
            {
                "field": "delta_logloss_display",
                "label": "ΔLogloss",
                "type": "text",
                "align": "right",
            },
            {"field": "source", "label": "运行来源", "type": "text"},
        ],
    }


def replace_sources(
    sources: list[dict[str, object]], source: dict[str, object]
) -> None:
    sources[:] = [item for item in sources if item.get("id") != source["id"]]
    sources.append(source)


def update_artifact(
    artifact_path: Path, rows: Sequence[dict[str, object]]
) -> None:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    source = {
        "id": "action_breakdown",
        "label": "由既有 strict/expansion 结果生成的逐动作报告视图",
        "path": "experiments/sisa_native_strict/migration/action_breakdown.sql",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": artifact_path.with_name("action_breakdown.sql").read_text(encoding="utf-8")
            if artifact_path.with_name("action_breakdown.sql").exists()
            else (Path(__file__).resolve().parents[1] / "experiments/sisa_native_strict/migration/action_breakdown.sql").read_text(encoding="utf-8"),
            "description": "Python 只读对齐既有 CSV；将完整配对行绑定为 :action_rows_json，执行展示查询，不修改实验数据",
            "tables_used": [
                "experiments/sisa_native_strict/results/metrics.csv",
                "experiments/sisa_expansion_acd/results/paired_summary.csv",
            ],
            "metric_definitions": [
                "每行粒度为 model × dataset × label",
                "ΔAUC = SISA AUC − baseline AUC；正值更好",
                "ΔLogloss = SISA logloss − baseline logloss；负值更好",
            ],
        },
    }
    replace_sources(artifact["manifest"]["sources"], source)
    replace_sources(artifact["sources"], source)

    action_prefixes = ("action_", "accepted_action_", "diagnostic_action_")
    tables = [
        table
        for table in artifact["manifest"]["tables"]
        if not str(table.get("id", "")).startswith(action_prefixes)
    ]
    try:
        table_index = next(
            index
            for index, table in enumerate(tables)
            if table.get("id") == "migration_inventory"
        )
    except StopIteration as error:
        raise ValueError("migration_inventory table anchor is missing") from error
    action_tables = []
    for scope in ("accepted", "diagnostic"):
        for spec in DATASET_SPECS:
            row_count = sum(
                row["dataset"] == spec["dataset"]
                and row["evidence_scope"] == scope
                for row in rows
            )
            if row_count:
                action_tables.append(action_table(spec, scope, row_count))
    tables[table_index:table_index] = action_tables
    artifact["manifest"]["tables"] = tables

    blocks = [
        block
        for block in artifact["manifest"]["blocks"]
        if block.get("id") != "evidence_scope_boundary"
        and not str(block.get("id", "")).startswith(
            ("action_", "accepted_action_", "diagnostic_action_")
        )
    ]
    try:
        block_index = next(
            index
            for index, block in enumerate(blocks)
            if block.get("id") == "strict_unit_table"
        )
    except StopIteration as error:
        raise ValueError("strict_unit_table block anchor is missing") from error
    accepted_specs = [
        spec for spec in DATASET_SPECS if spec["dataset"] in ACCEPTED_DATASETS
    ]
    action_blocks = [
        {
            "id": "accepted_action_breakdown_intro",
            "type": "markdown",
            "sourceId": "action_breakdown",
            "body": (
                "## Accepted 16-cell 证据：从数据集总体下钻到动作类别\n\n"
                "下列 4 张表展示 accepted 范围内的 **68 个模型×数据集×动作类别配对**，"
                "不再只看模型×数据集均值。QK-Video 包含 "
                "`click/follow/like/share`，KuaiRand 包含 6 类行为，MerRec 包含 "
                "`Like/Cart/Offer/Checkout/Purchase`，TencentGR 包含 "
                "`is_click/is_conversion`。每行同时给出 baseline、SISA、ΔAUC 和 "
                "ΔLogloss；运行来源列只用于说明数据归档位置，不改变 accepted 状态。"
            ),
        },
        *(
            {
                "id": f"accepted_action_{spec['id']}_block",
                "type": "table",
                "tableId": f"accepted_action_{spec['id']}_table",
            }
            for spec in accepted_specs
        ),
        {
            "id": "diagnostic_action_breakdown_intro",
            "type": "markdown",
            "sourceId": "action_breakdown",
            "body": (
                "## Diagnostic 附录：额外模型与 Taobao 的逐动作记录\n\n"
                "以下 **79 个模型×数据集×动作类别配对** 不参与论文汇总或实验结论，"
                "仅用于运行诊断和历史追溯。它们与上方 68 条 accepted 证据保持独立。"
            ),
        },
        *(
            {
                "id": f"diagnostic_action_{spec['id']}_block",
                "type": "table",
                "tableId": f"diagnostic_action_{spec['id']}_table",
            }
            for spec in DATASET_SPECS
        ),
    ]
    blocks[block_index + 1 : block_index + 1] = action_blocks
    blocks.insert(1, {
        "id": "evidence_scope_boundary",
        "type": "markdown",
        "body": (
            "## 历史诊断档案与当前论文证据范围\n\n"
            "本报告保留早期 70-task / 35-cell 混合矩阵，相关总量、均值、极值与解读"
            "均为历史诊断记录，不作为当前论文证据。当前正式范围仅为 HiFormer、"
            "HyFormer、RankMixer、Zenith × QK-Video、KuaiRand、TAAC-25、MerRec。"
            "下方 Accepted 16-cell 的 68 条逐动作结果与 Diagnostic 79 条记录分别展示；"
            "完整论文汇总采用独立报告 sisa_paper_report/report.md。"
        ),
    })
    artifact["manifest"]["blocks"] = blocks

    datasets = artifact["snapshot"]["datasets"]
    for dataset_id in list(datasets):
        if dataset_id.startswith(
            ("action_", "accepted_action_", "diagnostic_action_")
        ):
            del datasets[dataset_id]
    for scope in ("accepted", "diagnostic"):
        for spec in DATASET_SPECS:
            scoped_rows = [
                row
                for row in rows
                if row["dataset"] == spec["dataset"]
                and row["evidence_scope"] == scope
            ]
            if not scoped_rows:
                continue
            datasets[f"{scope}_action_{spec['id']}"] = [
                {
                    "label": row["label"],
                    "model": row["model"],
                    "baseline_auc": round(float(row["baseline_auc"]), 6),
                    "sisa_auc": round(float(row["sisa_auc"]), 6),
                    "delta_auc": round(float(row["delta_auc"]), 6),
                    "baseline_auc_display": f'{float(row["baseline_auc"]):.6f}',
                    "sisa_auc_display": f'{float(row["sisa_auc"]):.6f}',
                    "delta_auc_display": f'{float(row["delta_auc"]):+.6f}',
                    "baseline_logloss": round(
                        float(row["baseline_logloss"]), 6
                    ),
                    "sisa_logloss": round(float(row["sisa_logloss"]), 6),
                    "delta_logloss": round(float(row["delta_logloss"]), 6),
                    "baseline_logloss_display": (
                        f'{float(row["baseline_logloss"]):.6f}'
                    ),
                    "sisa_logloss_display": (
                        f'{float(row["sisa_logloss"]):.6f}'
                    ),
                    "delta_logloss_display": (
                        f'{float(row["delta_logloss"]):+.6f}'
                    ),
                    "source": row["source"],
                }
                for row in scoped_rows
            ]

    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fallback_action_html(rows: Sequence[dict[str, object]]) -> str:
    parts = ['<!-- unirank-action-breakdown-start -->']
    scopes = (
        (
            "accepted",
            "Accepted 16-cell 证据：从数据集总体下钻到动作类别",
            "68 个 accepted 模型×数据集×动作类别配对",
            [
                spec
                for spec in DATASET_SPECS
                if spec["dataset"] in ACCEPTED_DATASETS
            ],
        ),
        (
            "diagnostic",
            "Diagnostic 附录：额外模型与 Taobao 的逐动作记录",
            "79 个诊断配对；不参与论文汇总或实验结论",
            list(DATASET_SPECS),
        ),
    )
    for scope, heading, summary, specs in scopes:
        parts.extend(
            [
                '<div class="portable-block portable-layout-full" '
                f'data-artifact-block-id="{scope}_action_breakdown_intro" '
                'data-artifact-block-type="markdown" data-layout="full">',
                '<section class="portable-markdown">',
                f"<h2>{html.escape(heading)}</h2>",
                f"<p>{html.escape(summary)}。每行同时给出 baseline、SISA、ΔAUC 和 "
                "ΔLogloss；正 ΔAUC、负 ΔLogloss 表示改善。</p>",
                "</section></div>",
            ]
        )
        for spec in specs:
            dataset_rows = [
                row
                for row in rows
                if row["dataset"] == spec["dataset"]
                and row["evidence_scope"] == scope
            ]
            if not dataset_rows:
                continue
            scope_label = (
                "Accepted 证据" if scope == "accepted" else "Diagnostic 附录"
            )
            title = f"{spec['display']}：{scope_label}逐动作结果"
            table_id = f"{scope}_action_{spec['id']}_table"
            parts.extend(
                [
                    '<div class="portable-block portable-layout-full" '
                    f'data-artifact-block-id="{scope}_action_{spec["id"]}_block" '
                    'data-artifact-block-type="table" data-layout="full">',
                    '<section class="portable-content-card portable-table-card" '
                    f'data-artifact-id="{table_id}" data-artifact-kind="table" '
                    f'data-table-id="{table_id}">',
                    f'<header class="portable-visual-header"><h2>{html.escape(title)}</h2>'
                    f"<p>覆盖 {len(tuple(spec['labels']))} 个动作类别、"
                    f"{len(dataset_rows)} 个模型×类别配对；所有指标固定显示 6 位小数；"
                    "Δ=SISA−baseline。</p></header>",
                    '<div class="portable-inline-source portable-source-summary" '
                    'data-source-id="action_breakdown"><div '
                    'class="portable-inline-source-content portable-source-summary-content">'
                    "<strong>Source: 既有 strict/expansion 逐标签结果</strong>"
                    '<span class="portable-source-meta">Files: '
                    "experiments/sisa_native_strict/results/metrics.csv; "
                    "experiments/sisa_expansion_acd/results/paired_summary.csv"
                    "</span></div></div>",
                    '<div class="portable-table-source-region"><div class="portable-table-scroll">',
                    f"<table><caption>{html.escape(title)}</caption><thead><tr>",
                    "<th scope=\"col\">动作类别</th><th scope=\"col\">模型</th>"
                    '<th scope="col" class="portable-table-number">Baseline AUC</th>'
                    '<th scope="col" class="portable-table-number">SISA AUC</th>'
                    '<th scope="col" class="portable-table-number">ΔAUC</th>'
                    '<th scope="col" class="portable-table-number">Baseline Logloss</th>'
                    '<th scope="col" class="portable-table-number">SISA Logloss</th>'
                    '<th scope="col" class="portable-table-number">ΔLogloss</th>'
                    '<th scope="col">来源</th></tr></thead><tbody>',
                ]
            )
            for row in dataset_rows:
                delta_auc = float(row["delta_auc"])
                auc_class = (
                    " portable-table-positive"
                    if delta_auc > 0
                    else " portable-table-negative"
                    if delta_auc < 0
                    else ""
                )
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(row['label']))}</td>"
                    f"<td>{html.escape(str(row['model']))}</td>"
                    f'<td class="portable-table-number">{float(row["baseline_auc"]):.6f}</td>'
                    f'<td class="portable-table-number">{float(row["sisa_auc"]):.6f}</td>'
                    f'<td class="portable-table-number{auc_class}">{delta_auc:+.6f}</td>'
                    f'<td class="portable-table-number">{float(row["baseline_logloss"]):.6f}</td>'
                    f'<td class="portable-table-number">{float(row["sisa_logloss"]):.6f}</td>'
                    f'<td class="portable-table-number">{float(row["delta_logloss"]):+.6f}</td>'
                    f"<td>{html.escape(str(row['source']))}</td>"
                    "</tr>"
                )
            parts.extend(
                [
                    "</tbody></table></div></div>",
                    f'<p class="portable-table-note">{len(dataset_rows)} results</p>',
                    "</section></div>",
                ]
            )
    parts.append("<!-- unirank-action-breakdown-end -->")
    return "\n".join(parts)


def update_embedded_report(
    report_path: Path,
    artifact_path: Path,
    rows: Sequence[dict[str, object]],
) -> bool:
    """Refresh the enhanced payload and semantic fallback when HTML exists.

    The Node delivery step still performs the canonical full rebuild. This update keeps
    the checked-in report usable on machines where Node is unavailable.
    """

    if not report_path.is_file():
        return False
    report = report_path.read_text(encoding="utf-8")
    payload_pattern = re.compile(
        r'(<template id="data-analytics-portable-artifact-payload-source"[^>]*>\s*)'
        r"(.*?)"
        r"(\s*</template>)",
        re.DOTALL,
    )
    match = payload_pattern.search(report)
    if match is None:
        raise ValueError("report HTML has no embedded artifact payload")
    encoded = re.sub(r"\s+", "", match.group(2))
    payload = json.loads(gzip.decompress(base64.b64decode(encoded)))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for key in ("surface", "manifest", "snapshot", "sources"):
        payload[key] = artifact[key]
    payload_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    compressed = gzip.compress(payload_bytes, mtime=0)
    encoded_payload = base64.b64encode(compressed).decode("ascii")
    wrapped = "\n".join(
        encoded_payload[index : index + 76]
        for index in range(0, len(encoded_payload), 76)
    )
    report = payload_pattern.sub(
        lambda current: f"{current.group(1)}{wrapped}{current.group(3)}",
        report,
        count=1,
    )

    runtime_pattern = re.compile(
        r'(<template id="data-analytics-portable-reader-runtime-source"[^>]*>\s*)'
        r"(.*?)"
        r"(\s*</template>)",
        re.DOTALL,
    )
    runtime_match = runtime_pattern.search(report)
    if runtime_match is None:
        raise ValueError("report HTML has no embedded reader runtime")
    runtime_encoded = re.sub(r"\s+", "", runtime_match.group(2))
    runtime = gzip.decompress(base64.b64decode(runtime_encoded)).decode("utf-8")
    page_size_pattern = re.compile(
        r"([A-Za-z_$][\w$]*=520,[A-Za-z_$][\w$]*=)(?:15|28)"
        r"(,[A-Za-z_$][\w$]*=10,[A-Za-z_$][\w$]*=144,"
        r"[A-Za-z_$][\w$]*=16,[A-Za-z_$][\w$]*=24,"
        r"[A-Za-z_$][\w$]*=24,[A-Za-z_$][\w$]*=420)"
    )
    runtime, replacements = page_size_pattern.subn(r"\g<1>28\g<2>", runtime)
    if replacements != 1:
        raise ValueError(
            "expected one embedded reader table page-size constant, "
            f"found {replacements}"
        )
    runtime_compressed = gzip.compress(runtime.encode("utf-8"), mtime=0)
    runtime_base64 = base64.b64encode(runtime_compressed).decode("ascii")
    runtime_wrapped = "\n".join(
        runtime_base64[index : index + 76]
        for index in range(0, len(runtime_base64), 76)
    )
    report = runtime_pattern.sub(
        lambda current: (
            f"{current.group(1)}{runtime_wrapped}{current.group(3)}"
        ),
        report,
        count=1,
    )

    marker_pattern = re.compile(
        r"<!-- unirank-action-breakdown-start -->.*?"
        r"<!-- unirank-action-breakdown-end -->\s*",
        re.DOTALL,
    )
    report = marker_pattern.sub("", report)
    if 'data-artifact-block-id="accepted_action_breakdown_intro"' not in report:
        anchor = (
            '<div class="portable-block portable-layout-full" '
            'data-artifact-block-id="scope_definitions"'
        )
        if anchor not in report:
            raise ValueError("report HTML has no scope_definitions fallback anchor")
        report = report.replace(
            anchor,
            f"{fallback_action_html(rows)}\n{anchor}",
            1,
        )
    action_style_pattern = re.compile(
        r'\s*<style id="unirank-action-breakdown-style">.*?</style>',
        re.DOTALL,
    )
    report = action_style_pattern.sub("", report)
    action_style = (
        '<style id="unirank-action-breakdown-style">\n'
        '  [data-artifact-id*="_action_"] .data-table { '
        "min-width: 1120px; }\n"
        "</style>"
    )
    if "</head>" not in report:
        raise ValueError("report HTML has no </head> marker")
    report = report.replace("</head>", f"{action_style}\n</head>", 1)
    scope_pattern = re.compile(
        r"<!-- unirank-evidence-scope-start -->.*?<!-- unirank-evidence-scope-end -->\s*",
        re.DOTALL,
    )
    report = scope_pattern.sub("", report)
    scope = next(block for block in artifact["manifest"]["blocks"] if block["id"] == "evidence_scope_boundary")
    heading, paragraph = str(scope["body"]).split("\n\n", 1)
    scope_html = (
        '<!-- unirank-evidence-scope-start -->\n'
        '<section data-artifact-block-id="evidence_scope_boundary">'
        f'<h2>{html.escape(heading.removeprefix("## "))}</h2>'
        f'<p>{html.escape(paragraph)}</p></section>\n'
        '<!-- unirank-evidence-scope-end -->\n'
    )
    anchor = '<div class="portable-block portable-layout-full" data-artifact-block-id="technical_summary"'
    if anchor not in report:
        raise ValueError("report HTML has no technical summary fallback anchor")
    report = report.replace(anchor, scope_html + anchor, 1)
    report_path.write_text(report, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    migration_dir = (
        repository / "experiments/sisa_native_strict/migration"
    )
    rows = build_action_rows(repository)
    artifact_path = migration_dir / "artifact.json"
    update_artifact(artifact_path, rows)
    embedded_report_updated = update_embedded_report(
        migration_dir / "report.html",
        artifact_path,
        rows,
    )
    print(
        "SISA_OVERALL_REPORT "
        f"action_rows={len(rows)} datasets={len(DATASET_SPECS)} "
        f"embedded_report_updated={embedded_report_updated} "
        f"artifact={artifact_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
