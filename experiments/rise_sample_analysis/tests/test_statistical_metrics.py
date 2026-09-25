import importlib.util
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


PATH = Path(__file__).resolve().parents[1] / "scripts/statistical_metrics.py"
SPEC = importlib.util.spec_from_file_location("rise_statistical_metrics", PATH)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class StatisticalMetricsTest(unittest.TestCase):
    def test_auc_matches_pair_enumeration_with_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0])
        scores = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.8, 0.2])
        difference = scores[y == 1, None] - scores[None, y == 0]
        expected = np.mean((difference > 0) + 0.5 * (difference == 0))
        actual, influence = METRICS.auc_and_influence(y, scores)
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(actual, roc_auc_score(y, scores))
        self.assertAlmostEqual(influence.sum(), 0)

    def test_influence_matches_weighted_auc_derivative(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0])
        scores = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.8, 0.2])
        value, influence = METRICS.auc_and_influence(y, scores)
        epsilon = 1e-5
        for index in range(len(y)):
            weight = np.ones(len(y)); weight[index] += epsilon
            derivative = (roc_auc_score(y, scores, sample_weight=weight) - value) / epsilon
            self.assertAlmostEqual(derivative, influence[index], places=6)

    def test_identical_paired_scores_have_zero_difference_uncertainty(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0])
        scores = np.array([0.1, 0.5, 0.5, 0.5, 0.2, 0.8, 0.2])
        result, _, influence = METRICS.paired_auc_difference(
            y, scores, scores, np.array([1, 1, 2, 2, 3, 3, 4]))
        self.assertEqual(result["estimate"], 0)
        self.assertEqual(result["cluster_se"], 0)
        self.assertTrue(np.all(influence == 0))

    def test_group_contrast_keeps_shared_user_covariance(self):
        result = METRICS.group_difference_interval(
            (0.02, np.array([1, 2, 3]), np.array([0.01, -0.01, 0])),
            (0.01, np.array([1, 2, 3]), np.array([0.01, -0.01, 0])))
        self.assertAlmostEqual(result["estimate"], 0.01)
        self.assertEqual(result["cluster_se"], 0)

    def test_pair_decomposition_includes_cross_group_and_single_class_groups(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([0.4, 0.7, 0.4, 0.2, 0.1, 0.8])
        groups = np.array([0, 0, 1, 1, 2, 2])
        parts = METRICS.auc_pair_decomposition(y, scores, groups)
        self.assertAlmostEqual(sum(x["weight"] for x in parts), 1)
        self.assertAlmostEqual(sum(x["contribution"] for x in parts), roc_auc_score(y, scores))
        self.assertTrue(any(x["pair_auc"] is None for x in parts))
        self.assertTrue(all(x["contribution"] == 0 for x in parts if x["pair_auc"] is None))

    def test_single_class_auc_is_rejected(self):
        with self.assertRaises(ValueError):
            METRICS.auc_and_influence(np.ones(3), np.arange(3))

    def test_logloss_agrees_with_sklearn_float32_clipping(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0, 1, 0.1, 0.9], dtype=np.float32)
        self.assertAlmostEqual(float(METRICS.binary_logloss_rows(y, p).mean()), log_loss(y, p))


if __name__ == "__main__":
    unittest.main()
