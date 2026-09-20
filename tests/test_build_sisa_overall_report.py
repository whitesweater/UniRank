from __future__ import annotations

import base64
import gzip
import json
import re
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_sisa_overall_report import (
    ACCEPTED_MODELS,
    DATASET_SPECS,
    build_action_rows,
    restore_precise_table_fallbacks,
    update_artifact,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MIGRATION_DIR = REPOSITORY / "experiments/sisa_native_strict/migration"


class SisaOverallReportTest(unittest.TestCase):
    def test_fallback_keeps_exact_display_strings_and_all_rows(self):
        rows = [
            {"label": f"label<{index}>", "delta_display": "+0.000228"}
            for index in range(20)
        ]
        rows[-1]["delta_display"] = "-0.000000"
        artifact = {
            "manifest": {"tables": [{
                "id": "precision_table", "title": "AUC & Logloss",
                "subtitle": "所有指标固定显示 6 位小数", "dataset": "precision",
                "columns": [
                    {"field": "label", "label": "Label", "type": "text"},
                    {"field": "delta_display", "label": "Delta", "type": "text", "align": "right"},
                ],
            }]},
            "snapshot": {"datasets": {"precision": rows}},
        }
        report = (
            '<section data-table-id="precision_table"><header><h2>Old</h2></header>'
            '<div data-source-id="audited-source"></div>'
            '<table><tr><td>0</td></tr></table>'
            '<p class="portable-table-note">15 of 20 results</p></section>'
            '<section id="unrelated">Keep this section</section>'
        )
        updated = restore_precise_table_fallbacks(report, artifact)
        self.assertIn("+0.000228", updated)
        self.assertIn("-0.000000", updated)
        self.assertIn("label&lt;19&gt;", updated)
        self.assertIn("AUC &amp; Logloss", updated)
        self.assertIn("所有指标固定显示 6 位小数", updated)
        self.assertEqual(updated.count("<tr>"), 21)
        self.assertIn("20 results", updated)
        self.assertIn('data-source-id="audited-source"', updated)
        self.assertTrue(updated.endswith('<section id="unrelated">Keep this section</section>'))
        self.assertEqual(restore_precise_table_fallbacks(updated, artifact), updated)

    def test_action_rows_cover_every_model_dataset_label_pair(self):
        rows = build_action_rows(REPOSITORY)

        self.assertEqual(len(rows), 147)
        self.assertEqual(
            Counter(str(row["dataset"]) for row in rows),
            {
                "QK_Video_Action": 28,
                "KuaiRand_Video_Action": 42,
                "Taobao_Action": 28,
                "MerRec_Action": 35,
                "TencentGR_10M_Action": 14,
            },
        )
        self.assertEqual(
            len(
                {
                    (row["model"], row["dataset"], row["label"])
                    for row in rows
                }
            ),
            147,
        )
        self.assertEqual(
            Counter(str(row["evidence_scope"]) for row in rows),
            {"accepted": 68, "diagnostic": 79},
        )
        qk_click = next(
            row
            for row in rows
            if (row["model"], row["dataset"], row["label"])
            == ("OneTrans", "QK_Video_Action", "click")
        )
        self.assertAlmostEqual(float(qk_click["baseline_auc"]), 0.935674)
        self.assertAlmostEqual(float(qk_click["sisa_auc"]), 0.935902)
        self.assertAlmostEqual(float(qk_click["delta_auc"]), 0.000228)

    def test_artifact_update_is_idempotent_and_separates_evidence_scope(self):
        rows = build_action_rows(REPOSITORY)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "artifact.json"
            shutil.copyfile(MIGRATION_DIR / "artifact.json", artifact_path)
            update_artifact(artifact_path, rows)
            update_artifact(artifact_path, rows)
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        expected_table_ids = {
            f"accepted_action_{spec['id']}_table"
            for spec in DATASET_SPECS
            if spec["dataset"] != "Taobao_Action"
        } | {
            f"diagnostic_action_{spec['id']}_table" for spec in DATASET_SPECS
        }
        table_ids = [table["id"] for table in artifact["manifest"]["tables"]]
        self.assertTrue(expected_table_ids.issubset(table_ids))
        self.assertTrue(
            all(
                table_ids.count(table_id) == 1
                for table_id in expected_table_ids
            )
        )
        source_ids = [source["id"] for source in artifact["sources"]]
        self.assertEqual(source_ids.count("action_breakdown"), 1)
        block_ids = [block["id"] for block in artifact["manifest"]["blocks"]]
        self.assertLess(
            block_ids.index("strict_unit_table"),
            block_ids.index("accepted_action_breakdown_intro"),
        )
        self.assertLess(
            block_ids.index("diagnostic_action_tencentgr_block"),
            block_ids.index("scope_definitions"),
        )

    def test_materialized_report_has_auc_and_logloss_by_action(self):
        artifact = json.loads(
            (MIGRATION_DIR / "artifact.json").read_text(encoding="utf-8")
        )
        expected_counts = {
            "accepted_action_qk_video": 16,
            "accepted_action_kuairand": 24,
            "accepted_action_merrec": 20,
            "accepted_action_tencentgr": 8,
            "diagnostic_action_qk_video": 12,
            "diagnostic_action_kuairand": 18,
            "diagnostic_action_taobao": 28,
            "diagnostic_action_merrec": 15,
            "diagnostic_action_tencentgr": 6,
        }
        self.assertEqual(
            {
                dataset: len(artifact["snapshot"]["datasets"][dataset])
                for dataset in expected_counts
            },
            expected_counts,
        )

        tables = {
            table["id"]: table for table in artifact["manifest"]["tables"]
        }
        taobao_columns = {
            column["field"]
            for column in tables["diagnostic_action_taobao_table"]["columns"]
        }
        self.assertTrue(
            {
                "label",
                "model",
                "baseline_auc_display",
                "sisa_auc_display",
                "delta_auc_display",
                "baseline_logloss_display",
                "sisa_logloss_display",
                "delta_logloss_display",
                "source",
            }.issubset(taobao_columns)
        )
        precision_fields = {
            "baseline_auc_display",
            "sisa_auc_display",
            "delta_auc_display",
            "baseline_logloss_display",
            "sisa_logloss_display",
            "delta_logloss_display",
        }
        for column in tables["diagnostic_action_taobao_table"]["columns"]:
            if column["field"] in precision_fields:
                self.assertEqual(column["type"], "text")
                self.assertEqual(column["align"], "right")

        qk_rows = artifact["snapshot"]["datasets"][
            "diagnostic_action_qk_video"
        ]
        qk_click = next(
            row
            for row in qk_rows
            if row["model"] == "OneTrans" and row["label"] == "click"
        )
        self.assertEqual(qk_click["baseline_auc_display"], "0.935674")
        self.assertEqual(qk_click["sisa_auc_display"], "0.935902")
        self.assertEqual(qk_click["delta_auc_display"], "+0.000228")
        unsigned_pattern = re.compile(r"^-?\d+\.\d{6}$")
        signed_pattern = re.compile(r"^[+-]\d+\.\d{6}$")
        for dataset in expected_counts:
            for row in artifact["snapshot"]["datasets"][dataset]:
                for field in (
                    "baseline_auc_display",
                    "sisa_auc_display",
                    "baseline_logloss_display",
                    "sisa_logloss_display",
                ):
                    self.assertRegex(row[field], unsigned_pattern)
                for field in ("delta_auc_display", "delta_logloss_display"):
                    self.assertRegex(row[field], signed_pattern)
        self.assertEqual(
            {
                row["label"]
                for row in artifact["snapshot"]["datasets"][
                    "diagnostic_action_taobao"
                ]
            },
            {"is_click", "cart", "fav", "buy"},
        )
        accepted_models = {
            row["model"]
            for dataset, rows in artifact["snapshot"]["datasets"].items()
            if dataset.startswith("accepted_action_")
            for row in rows
        }
        self.assertEqual(accepted_models, set(ACCEPTED_MODELS))
        self.assertNotIn(
            "accepted_action_taobao", artifact["snapshot"]["datasets"]
        )

        html = (MIGRATION_DIR / "report.html").read_text(encoding="utf-8")
        self.assertIn(
            'data-artifact-block-id="accepted_action_breakdown_intro"', html
        )
        self.assertIn(
            'data-artifact-block-id="diagnostic_action_breakdown_intro"', html
        )
        self.assertIn("Taobao：Diagnostic 附录逐动作结果", html)
        self.assertIn("所有指标固定显示 6 位小数", html)
        self.assertIn("+0.000228", html)
        payload_match = re.search(
            r'<template id="data-analytics-portable-artifact-payload-source"'
            r"[^>]*>\s*(.*?)\s*</template>",
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(payload_match)
        assert payload_match is not None
        payload = json.loads(
            gzip.decompress(
                base64.b64decode(re.sub(r"\s+", "", payload_match.group(1)))
            )
        )
        self.assertEqual(
            len(payload["snapshot"]["datasets"]["accepted_action_kuairand"]),
            24,
        )
        self.assertEqual(
            len(payload["snapshot"]["datasets"]["diagnostic_action_taobao"]),
            28,
        )
        runtime_match = re.search(
            r'<template id="data-analytics-portable-reader-runtime-source"'
            r"[^>]*>\s*(.*?)\s*</template>",
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(runtime_match)
        assert runtime_match is not None
        runtime = gzip.decompress(
            base64.b64decode(
                re.sub(r"\s+", "", runtime_match.group(1))
            )
        ).decode("utf-8")
        page_size_pattern = re.compile(
            r"([A-Za-z_$][\w$]*=520,[A-Za-z_$][\w$]*=)28"
            r"(,[A-Za-z_$][\w$]*=10,[A-Za-z_$][\w$]*=144,"
            r"[A-Za-z_$][\w$]*=16,[A-Za-z_$][\w$]*=24,"
            r"[A-Za-z_$][\w$]*=24,[A-Za-z_$][\w$]*=420)"
        )
        self.assertEqual(len(page_size_pattern.findall(runtime)), 1)


if __name__ == "__main__":
    unittest.main()
