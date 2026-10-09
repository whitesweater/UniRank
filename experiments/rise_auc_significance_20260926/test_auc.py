"""Small independent oracles; no real-data bootstrap outcomes are inspected."""
import sys
sys.dont_write_bytecode = True
import unittest
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from auc_core import prepare_auc, summarize_interval, draw_user_weights


class WeightedAUCTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        self.p = np.array([.1, .4, .4, .8, .4, .1, .8, .8], dtype=np.float32)
        self.users = np.array([0, 1, 1, 2, 0, 2, 3, 3])

    def test_ties_weighted_roc_and_explicit_user_replication(self):
        weights = np.array([[1, 1, 1, 1], [2, 0, 1, 1], [0, 3, 0, 1], [4, 0, 0, 0]])
        for kind in ('dense', 'sparse'):
            actual, _, _ = prepare_auc(self.y, self.p, self.users, 4, kind).evaluate(weights.T)
            for index, w in enumerate(weights):
                row_weight = w[self.users]
                expected = roc_auc_score(self.y, self.p, sample_weight=row_weight)
                indices = np.repeat(np.arange(len(self.y)), row_weight)
                repeated = roc_auc_score(self.y[indices], self.p[indices])
                self.assertAlmostEqual(actual[index], expected, places=14)
                self.assertAlmostEqual(actual[index], repeated, places=14)

    def test_noninteger_weights_against_sklearn(self):
        w = np.array([.1, 1.5, 2.2, 0.])
        actual, _, _ = prepare_auc(self.y, self.p, self.users, 4).evaluate(w)
        self.assertAlmostEqual(actual[0], roc_auc_score(self.y, self.p, sample_weight=w[self.users]), places=14)

    def test_class_drop_produces_nan_not_label_dropping(self):
        users = np.array([0, 0, 1, 1])
        y = np.array([0, 0, 1, 1])
        group = prepare_auc(y, np.array([.2, .3, .6, .8]), users, 2)
        result, pos, neg = group.evaluate(np.array([[2, 0, 1], [0, 2, 1]]))
        self.assertTrue(np.isnan(result[:2]).all())
        for user_weights in (np.array([2, 0]), np.array([0, 2])):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    sklearn_value = roc_auc_score(y, [.2, .3, .6, .8], sample_weight=user_weights[users])
                except ValueError:
                    sklearn_value = np.nan
            self.assertTrue(np.isnan(sklearn_value))
        self.assertEqual(pos[0], 0)
        self.assertEqual(neg[1], 0)
        macro = np.mean(np.column_stack([result, np.ones(3)]), axis=1)
        self.assertTrue(np.isnan(macro[:2]).all())
        self.assertEqual(macro[2], 1.)
        interval = summarize_interval(macro, 0.)
        self.assertEqual(interval['invalid_draws'], 2)
        self.assertIsNone(interval['ci95'])
        self.assertIsNone(interval['ci99_375_bonferroni8'])

    def test_randomized_exact_integer_counts(self):
        rng = np.random.default_rng(7)
        users = rng.integers(0, 12, 100)
        y = rng.integers(0, 2, 100)
        p = (rng.integers(0, 8, 100) / 8).astype(np.float32)
        weights = rng.multinomial(12, np.full(12, 1/12), size=40)
        dense = prepare_auc(y, p, users, 12, 'dense').evaluate(weights.T)[0]
        sparse = prepare_auc(y, p, users, 12, 'sparse').evaluate(weights.T)[0]
        np.testing.assert_allclose(dense, sparse, rtol=0, atol=2e-15)
        for i, w in enumerate(weights):
            self.assertAlmostEqual(dense[i], roc_auc_score(y, p, sample_weight=w[users]), places=14)

    def test_macro_inside_draw_and_paired_sign(self):
        weights = np.concatenate([w for _, w in draw_user_weights(4, 100, 20260926, 17)])
        full1 = prepare_auc(self.y, self.p, self.users, 4).evaluate(weights.T)[0]
        rms1 = prepare_auc(self.y, 1-self.p, self.users, 4).evaluate(weights.T)[0]
        full2 = prepare_auc(1-self.y, 1-self.p, self.users, 4).evaluate(weights.T)[0]
        rms2 = prepare_auc(1-self.y, self.p, self.users, 4).evaluate(weights.T)[0]
        delta = np.mean(np.column_stack([full1, full2]), axis=1) - np.mean(np.column_stack([rms1, rms2]), axis=1)
        expected = ((full1-rms1) + (full2-rms2)) / 2
        np.testing.assert_allclose(delta, expected, rtol=0, atol=2e-15)
        np.testing.assert_allclose(-delta, -expected, rtol=0, atol=2e-15)

    def test_batching_does_not_change_draw_stream(self):
        expected = np.random.default_rng(20260926).multinomial(7, np.full(7, 1/7), size=113)
        actual = np.concatenate([w for _, w in draw_user_weights(7, 113, 20260926, 19)])
        np.testing.assert_array_equal(actual, expected)

    def test_adjusted_interval_and_no_p(self):
        draws = np.linspace(-.001, .003, 10000)
        result = summarize_interval(draws, .001)
        np.testing.assert_allclose(result['ci95'], np.quantile(draws, [.025, .975]))
        np.testing.assert_allclose(result['ci99_375_bonferroni8'], np.quantile(draws, [.003125, .996875]))
        self.assertEqual(result['adjusted_direction'], 'unresolved')
        self.assertIsNone(result['p_value'])
        self.assertEqual(summarize_interval(draws + .01, .011)['adjusted_direction'], 'Full_higher_AUC')
        self.assertEqual(summarize_interval(draws - .01, -.009)['adjusted_direction'], 'RMS_higher_AUC')


if __name__ == '__main__':
    unittest.main()
