from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_sisa_three_seed_report import build_artifact, source, validate_inputs


REPOSITORY = Path(__file__).resolve().parents[1]


class SisaThreeSeedReportTest(unittest.TestCase):
    def test_build_artifact_contains_unified_sources_and_rows(self):
        artifact, report_dir, chart_map, source_notes = build_artifact(REPOSITORY)
        self.assertEqual(
            artifact["manifest"]["title"], "UniRank SISA 三-seed 统一对比报告"
        )
        self.assertEqual(report_dir.name, "report")
        self.assertEqual(len(artifact["snapshot"]["datasets"]["cell_comparison"]), 16)
        self.assertEqual(len(artifact["snapshot"]["datasets"]["auc_matrix"]), 4)
        long_rows = artifact["snapshot"]["datasets"]["three_seed_long"]
        self.assertEqual(len(long_rows), 16)
        self.assertEqual(
            [(row["model"], row["dataset"]) for row in long_rows[:4]],
            [
                ("HiFormer", "QK-Video"),
                ("HiFormer", "KuaiRand"),
                ("HiFormer", "TAAC-25 / TencentGR"),
                ("HiFormer", "MerRec"),
            ],
        )
        self.assertEqual(
            [row["model"] for row in long_rows[::4]],
            ["HiFormer", "HyFormer", "RankMixer", "Zenith"],
        )
        self.assertEqual(len(artifact["snapshot"]["datasets"]["label_comparison"]), 68)
        self.assertEqual(len(artifact["manifest"]["charts"]), 2)
        tables = {table["id"]: table for table in artifact["manifest"]["tables"]}
        self.assertIn("auc_matrix_table", tables)
        self.assertIn("three_seed_long_table", tables)
        self.assertEqual(tables["three_seed_long_table"]["dataset"], "three_seed_long")
        self.assertEqual(
            tables["three_seed_long_table"]["defaultSort"],
            {"field": "model", "direction": "asc"},
        )
        matrix_row = artifact["snapshot"]["datasets"]["auc_matrix"][0]
        self.assertEqual(set(matrix_row), {"dataset", "HiFormer", "HyFormer", "RankMixer", "Zenith"})
        self.assertRegex(matrix_row["HiFormer"], r"^0\.\d{5} [↑↓→] [+-]0\.\d{5}$")
        long_fields = {column["field"] for column in tables["three_seed_long_table"]["columns"]}
        self.assertTrue(
            {
                "sisa_20262027_auc",
                "sisa_20262028_auc",
                "sisa_20262029_auc",
                "sisa_three_seed_auc",
                "baseline_auc",
                "delta_auc_vs_baseline_milli",
            }.issubset(long_fields)
        )
        for field in (
            "sisa_20262027_auc",
            "sisa_20262028_auc",
            "sisa_20262029_auc",
            "sisa_three_seed_auc",
            "sisa_seed_std_auc_milli",
            "baseline_auc",
            "delta_auc_vs_baseline_milli",
        ):
            self.assertRegex(long_rows[0][field], r"^-?\d+\.\d{5}$")
        long_columns = {
            column["field"]: column
            for column in tables["three_seed_long_table"]["columns"]
        }
        self.assertEqual(long_columns["sisa_three_seed_auc"]["type"], "text")
        self.assertEqual(long_columns["sisa_three_seed_auc"]["align"], "right")
        self.assertEqual(
            [column["label"] for column in tables["three_seed_long_table"]["columns"]],
            [
                "模型",
                "数据集",
                "20262027",
                "20262028",
                "20262029",
                "Mean",
                "SD（×10⁻³）",
                "Baseline",
                "Δ（×10⁻³）",
            ],
        )
        self.assertIn("paper_table", {item["id"] for item in artifact["sources"]})
        self.assertIn("scatter", chart_map)
        self.assertIn("descriptive", source_notes)

    def test_validation_rejects_missing_cells(self):
        headline = {
            "experiment_seeds": [20262027, 20262028, 20262029],
            "cell_count": 16,
            "label_count": 68,
        }
        with self.assertRaisesRegex(ValueError, "16 cells"):
            validate_inputs(headline, [], [{}] * 68, [{}] * 5)

    def test_source_query_is_runnable_and_relative(self):
        path = REPOSITORY / "experiments/sisa_three_seed_unified/results/cell_summary.csv"
        metadata = source(REPOSITORY, "cells", "Cells", path)
        self.assertIn("read_csv_auto('experiments/", metadata["query"]["sql"])
        self.assertFalse(str(metadata["path"]).startswith("/"))

    def test_delivery_runtime_keeps_all_cell_rows_on_one_page(self):
        wrapper = (
            REPOSITORY
            / "experiments/sisa_three_seed_unified/report/deliver_report.mjs"
        ).read_text(encoding="utf-8")
        self.assertIn("showAllCellRowsOnOnePage", wrapper)
        self.assertIn('return html.replace(tableCardPageSizePattern, "$116$2")', wrapper)
        self.assertIn('min-width: 936px !important', wrapper)
        self.assertIn('--ds-report-content-max-width: min(1180px, calc(100vw - 48px))', wrapper)
        self.assertIn('.report-stack-item-markdown', wrapper)


if __name__ == "__main__":
    unittest.main()
