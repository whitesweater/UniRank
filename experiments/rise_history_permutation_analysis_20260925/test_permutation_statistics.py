"""Independent formula tests; includes an explicit user-resampling oracle."""
import unittest

import numpy as np

from permutation_statistics import (bce, clusters, cluster_sums, bootstrap_weights,
                                    paired_cluster_summary, diagnostic_auc)


class StatisticsTests(unittest.TestCase):
    def test_bce_clipping_matches_previous_probe(self):
        labels = np.array([[0, 1], [1, 0]])
        p = np.array([[0, 1], [0, 1]], dtype=np.float64)
        expected = np.array([[-np.log1p(-1e-7), -np.log(1-1e-7)],
                             [-np.log(1e-7), -np.log1p(-(1-1e-7))]])
        np.testing.assert_array_equal(bce(labels, p), expected)
        with self.assertRaises(ValueError):
            bce(labels, p + 2)
        with self.assertRaises(ValueError):
            bce(labels + 1, p)

    def test_user_cluster_bootstrap_by_explicit_row_resampling(self):
        users = np.array([101, 202, 202, 202])
        deltas = np.array([-1., 2., -1., 2.])
        ids, inverse, counts = clusters(users)
        sums = cluster_sums(deltas, inverse, len(ids))
        draws = np.array([[2, 0], [1, 1], [0, 2], [1, 1]])
        expected_rows, expected_users = [], []
        for multiplicities in draws:
            sampled_rows, sampled_user_means = [], []
            for user, multiplier in zip(ids, multiplicities):
                for _ in range(multiplier):
                    selected = deltas[users == user]
                    sampled_rows.extend(selected.tolist())
                    sampled_user_means.append(float(selected.mean()))
            expected_rows.append(np.mean(sampled_rows))
            expected_users.append(np.mean(sampled_user_means))
        result = paired_cluster_summary(sums, counts, draws)[0]
        self.assertEqual(result["bce_difference"], .5)
        self.assertEqual(result["user_balanced_difference"], 0.)
        np.testing.assert_allclose([result["ci_low"], result["ci_high"]], np.quantile(expected_rows, [.025, .975]))
        np.testing.assert_allclose([result["user_balanced_ci_low"], result["user_balanced_ci_high"]], np.quantile(expected_users, [.025, .975]))

    def test_null_contrast_and_sign_reversal(self):
        counts = np.array([2., 3., 4.])
        draws = bootstrap_weights(3)
        sums = np.array([[-.2, .1], [.3, -.3], [.1, .7]])
        forward = paired_cluster_summary(sums, counts, draws)
        reverse = paired_cluster_summary(-sums, counts, draws)
        for a, b in zip(forward, reverse):
            self.assertAlmostEqual(a["bce_difference"], -b["bce_difference"])
            self.assertAlmostEqual(a["ci_low"], -b["ci_high"])
            self.assertAlmostEqual(a["user_balanced_ci_low"], -b["user_balanced_ci_high"])
        null = paired_cluster_summary(np.zeros((3, 1)), counts, draws)[0]
        self.assertTrue(all(value == 0 for value in null.values()))

    def test_bootstrap_exact_prior_rng_recipe(self):
        expected = np.random.default_rng(20260925).multinomial(107, np.full(107, 1/107), size=2000)
        actual = bootstrap_weights(107)
        np.testing.assert_array_equal(actual, expected)

    def test_auc_ties_and_single_class(self):
        self.assertEqual(diagnostic_auc([0, 1, 0, 1], [.1, .4, .4, .8]), .875)
        self.assertEqual(diagnostic_auc([0, 1, 0, 1], [.9, .6, .6, .2]), .125)
        self.assertIsNone(diagnostic_auc([1, 1], [.4, .5]))


if __name__ == "__main__":
    unittest.main()
