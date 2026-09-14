"""Protect paper evidence scope and independent log reconstruction."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_sisa_paper_evidence.py"
SPEC = importlib.util.spec_from_file_location("archived_paper_evidence", SCRIPT_PATH)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)
accepted = EVIDENCE.accepted
parse_test_metrics = EVIDENCE.parse_test_metrics
summarize = EVIDENCE.summarize
validate_keys = EVIDENCE.validate_keys


class PaperEvidenceTest(unittest.TestCase):
    def test_scope_requires_both_model_and_dataset(self):
        self.assertTrue(accepted({"model": "HiFormer", "dataset": "QK_Video_Action"}))
        self.assertFalse(accepted({"model": "OneTrans", "dataset": "QK_Video_Action"}))
        self.assertFalse(accepted({"model": "HiFormer", "dataset": "Taobao_Action"}))

    def test_test_metrics_do_not_use_validation_values(self):
        text = "[Task: click][Metrics] AUC: 0.99 - logloss: 0.01\n"
        text += "******** Test evaluation ********\n"
        text += "[Task: click][Metrics] logloss: 0.2 - AUC: 0.8\n"
        self.assertEqual(parse_test_metrics(text), {"click": {"AUC": 0.8, "logloss": 0.2}})
        with self.assertRaises(ValueError):
            parse_test_metrics("[Task: click][Metrics] AUC: 0.9 - logloss: 0.1")

    def test_appended_sessions_select_last_explicit_test(self):
        text = "Test evaluation\n[Task: click][Metrics] AUC: 0.9 - logloss: 0.1\n"
        text += "Test evaluation\n[Task: click][Metrics] AUC: 0.8 - logloss: 0.2\n"
        self.assertEqual(parse_test_metrics(text)["click"]["AUC"], 0.8)

    def test_rejects_duplicate_and_missing_task_keys(self):
        row = {"model": "HiFormer", "dataset": "QK_Video_Action", "label": "click"}
        with self.assertRaises(ValueError):
            validate_keys([row, row])
        with self.assertRaises(ValueError):
            validate_keys([row])

    def test_cell_macro_does_not_weight_by_label_count(self):
        rows = []
        for dataset, count, gain in [("QK_Video_Action", 4, 0.01), ("TencentGR_10M_Action", 2, 0.03)]:
            for i in range(count):
                rows.append({"model": "HiFormer", "dataset": dataset, "label": str(i),
                             "baseline_auc": 0.7, "sisa_20262027_auc": 0.7 + gain,
                             "baseline_logloss": 0.2, "sisa_20262027_logloss": 0.19})
        result = summarize(rows, 20262027)
        self.assertAlmostEqual(result["cell_macro_delta_auc"], 0.02)
        self.assertAlmostEqual(result["label_macro_delta_auc"], 1 / 60)


if __name__ == "__main__":
    unittest.main()
