"""Guard the 20-cell merged result against accidental seed or source selection."""

import csv
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "experiments/sisa_three_seed_unified/report/build_integrated_evidence.py"
TRANSPOSE = ROOT / "experiments/sisa_three_seed_unified/report/transpose_main_tables.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_integrated_evidence", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_transpose():
    spec = importlib.util.spec_from_file_location("transpose_main_tables", TRANSPOSE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SISAIntegratedFinalReportTest(unittest.TestCase):
    def test_accepted_matrix_has_twenty_unique_cells_and_only_two_fourth_seeds(self):
        rows = load_builder().build_integrated_rows(ROOT)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({(r["model"], r["dataset_id"]) for r in rows}), 20)
        self.assertEqual(sum(r["has_seed30"] for r in rows), 2)
        self.assertEqual({r["model"] for r in rows},
                         {"HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith"})

    def test_latest_attempt_uses_all_eight_valid_retries_and_keeps_original(self):
        rows = load_builder().build_integrated_rows(ROOT)
        index = {(r["model"], r["dataset_id"]): r for r in rows}
        one = index["OneTrans", "TencentGR_10M_Action"]
        self.assertAlmostEqual(one["original_20262029_auc"], 0.8559670, places=7)
        self.assertAlmostEqual(one["latest_20262029_auc"], 0.8597095, places=7)
        self.assertEqual(one["latest_job_20262029"], "645445_5")
        self.assertAlmostEqual(one["latest_20262029_logloss"], 0.0447520, places=7)
        hy = index["HyFormer", "MerRec_Action"]
        self.assertAlmostEqual(hy["latest_20262028_auc"], 0.8008724, places=7)
        self.assertAlmostEqual(hy["seed30_auc"], 0.7989772, places=7)
        self.assertAlmostEqual(hy["latest_three_seed_mean_auc"],
                               hy["original_three_seed_mean_auc"], places=12)
        self.assertEqual(sum(r["has_retry"] for r in rows), 8)

    def test_logloss_improvement_sign_and_missing_seed30_are_explicit(self):
        rows = load_builder().build_integrated_rows(ROOT)
        index = {(r["model"], r["dataset_id"]): r for r in rows}
        zenith = index["Zenith", "TencentGR_10M_Action"]
        self.assertLess(zenith["latest_three_seed_mean_logloss"] -
                        zenith["original_three_seed_mean_logloss"], 0.002)
        self.assertLess(zenith["logloss_improvement_vs_original"], 0)
        self.assertFalse(index["RankMixer", "MerRec_Action"]["has_seed30"])
        self.assertIsNone(index["RankMixer", "MerRec_Action"]["seed30_auc"])

    def test_model_rows_and_dataset_columns_keep_latest_seed_values(self):
        source = ROOT / "experiments/sisa_three_seed_unified/results/integrated_cells.csv"
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        matrix = load_transpose().matrix_rows(rows)
        self.assertEqual(len(matrix), 5)
        self.assertEqual([row["model"] for row in matrix],
                         ["HiFormer", "HyFormer", "OneTrans", "RankMixer", "Zenith"])
        self.assertEqual(set(matrix[0]), {
            "model", "QK_Video_Action_auc", "KuaiRand_Video_Action_auc",
            "TencentGR_10M_Action_auc", "MerRec_Action_auc",
            "QK_Video_Action_logloss", "KuaiRand_Video_Action_logloss",
            "TencentGR_10M_Action_logloss", "MerRec_Action_logloss",
        })
        self.assertIn("0.7989772", matrix[1]["MerRec_Action_auc"])
        self.assertIn("28 0.8008724 → 0.8008724", matrix[1]["MerRec_Action_auc"])
        self.assertIn("0.8597095", matrix[2]["TencentGR_10M_Action_auc"])
        self.assertIn("29 0.8559670 → 0.8597095", matrix[2]["TencentGR_10M_Action_auc"])
        self.assertIn("0.0447520", matrix[2]["TencentGR_10M_Action_logloss"])


if __name__ == "__main__":
    unittest.main()
