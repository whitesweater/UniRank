"""Definitions, fixed-bin boundaries and formal paired quality export checks."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import auc, average_precision_score, log_loss, precision_recall_curve

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TESTS = Path(__file__).resolve().parent
for path in (SCRIPTS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import analyze_prediction_quality as QUALITY
import test_hiformer_group_analysis as PAIR_FIXTURE


class PredictionQualityTest(unittest.TestCase):
    def test_ap_is_uninterpolated_sklearn_ap_not_trapezoidal_pr_area(self):
        y, p = np.array([0, 1, 1]), np.array([.3, .2, .1])
        result, _ = QUALITY.quality_metrics(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        self.assertAlmostEqual(result["AP"], average_precision_score(y, p))
        self.assertNotAlmostEqual(result["AP"], auc(recall, precision))

    def test_logloss_uses_float64_clipping_and_brier_keeps_original_scores(self):
        y, p = np.array([1, 0]), np.array([0, 1], dtype=np.float32)
        result, _ = QUALITY.quality_metrics(y, p)
        self.assertEqual(result["Brier"], 1)
        self.assertAlmostEqual(result["Logloss"], log_loss(y, p.astype(np.float64)))
        self.assertGreater(result["Logloss"], log_loss(y, p) + 10)

    def test_fixed_bins_keep_empty_bins_and_all_boundaries(self):
        edges = np.array([0., 1e-6, .01, .1, .5, 1.])
        y, p = np.array([0, 1, 0, 1]), np.array([0., 1e-6, .1, 1.])
        rows = QUALITY.calibration_table(y, p, edges)
        self.assertEqual([row["N"] for row in rows], [1, 1, 0, 1, 1])
        self.assertIsNone(rows[2]["positive_rate"])
        self.assertIsNone(rows[2]["mean_prediction"])
        self.assertEqual(rows[-1]["closure"], "[lower,upper]")
        self.assertEqual(rows[-1]["mean_prediction"], 1)
        self.assertEqual(sum(row["positive_count"] for row in rows), 2)

    def test_invalid_probabilities_and_empty_inputs_are_rejected(self):
        for probabilities in ([float("nan"), .2], [-.1, .2], [1.1, .2]):
            with self.assertRaises(ValueError):
                QUALITY.quality_metrics(np.array([0, 1]), np.array(probabilities))
        with self.assertRaises(ValueError):
            QUALITY.quality_metrics(np.array([]), np.array([]))

    def pair(self, root):
        fixture = PAIR_FIXTURE.HiFormerGroupAnalysisTest(methodName="runTest")
        a = fixture.build_export(root / "before", "A")
        e = fixture.build_export(root / "after", "E")
        for directory in (a, e):
            path = directory / "complete.json"
            complete = json.loads(path.read_text())
            table = pq.read_table(next(directory.glob("*.parquet")))
            for label in QUALITY.DATASETS["QK_Video_Action"]["labels"]:
                result, _ = QUALITY.quality_metrics(table[label].to_numpy(), table[label + "_pred"].to_numpy())
                complete["metrics"][label] = {"AUC": result["ROC_AUC"], "logloss": result["Logloss"]}
            path.write_text(json.dumps(complete))
        return a, e

    def test_small_formal_pair_produces_complete_supplement_and_empty_bin_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, e = self.pair(root)
            small = dict(QUALITY.DATASETS["QK_Video_Action"], rows=16384)
            output = root / "quality"
            with patch.dict(QUALITY.DATASETS, {"QK_Video_Action": small}):
                QUALITY.analyze("QK_Video_Action", a, e, output)
            complete = json.loads((output / "complete.json").read_text())
            self.assertTrue(complete["paired_sample_identity_verified"])
            self.assertTrue(complete["global_metric_recheck_passed"])
            with (output / "calibration_bins.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2 * 4 * (len(QUALITY.BIN_EDGES) - 1))
            self.assertTrue(any(row["N"] == "0" and row["mean_prediction"] == "" for row in rows))
            with (output / "paired_deltas.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(float(row["delta_AP"]) == 0 for row in rows))

    def test_tampered_row_labels_cannot_pass_a_rehashed_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, e = self.pair(Path(tmp))
            path = next(e.glob("*.parquet"))
            table = pq.read_table(path)
            table = table.set_column(table.schema.get_field_index("follow"), "follow", pa.array(np.zeros(len(table), np.uint8)))
            pq.write_table(table, path)
            receipt = json.loads(path.with_suffix(".json").read_text())
            receipt["sha256"] = QUALITY.sha256(path)
            path.with_suffix(".json").write_text(json.dumps(receipt))
            with patch.dict(QUALITY.DATASETS, {"QK_Video_Action": dict(QUALITY.DATASETS["QK_Video_Action"], rows=16384)}):
                with self.assertRaisesRegex(ValueError, "identity"):
                    QUALITY.read_pair(a, e, "QK_Video_Action")

    def test_smoke_is_rejected_before_quality_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, e = self.pair(Path(tmp))
            path = a / "complete.json"
            complete = json.loads(path.read_text())
            complete["smoke_only"] = True
            path.write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError, "Formal"):
                QUALITY.inspect_export(a, "QK_Video_Action", "A")


if __name__ == "__main__":
    unittest.main()
