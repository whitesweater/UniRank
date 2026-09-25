import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('auc_structure', SCRIPTS / 'analyze_auc_structure.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def brute_auc(y, scores):
    positive, negative = scores[y == 1], scores[y == 0]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative)))


class AUCStructureTest(unittest.TestCase):
    def test_pair_decomposition_reconciles_ties_and_single_class_groups(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        scores = np.array([.1, .5, .5, .9, .9, .1])
        groups = np.array([0, 0, 1, 1, 2, 2])
        rows = analysis.exact_pair_decomposition(y, scores, groups, group_count=4)
        self.assertEqual(len(rows), 16)
        self.assertAlmostEqual(sum(r['weight'] for r in rows), 1)
        self.assertAlmostEqual(sum(r['contribution'] for r in rows), brute_auc(y, scores))
        for row in rows:
            if row['pair_count'] == 0:
                self.assertIsNone(row['pair_auc'])
                self.assertEqual(row['contribution'], 0)

    def test_cross_group_ranking_changes_global_auc_with_fixed_within_auc(self):
        y = np.array([1, 0, 1, 0])
        groups = np.array([0, 0, 1, 1])
        before, after = np.array([.9, .8, .7, .6]), np.array([.9, .4, .7, .6])
        a = analysis.exact_pair_decomposition(y, before, groups, 2)
        b = analysis.exact_pair_decomposition(y, after, groups, 2)
        within = sum(q['contribution'] - p['contribution'] for p, q in zip(a, b) if p['within_group'])
        total = sum(q['contribution'] - p['contribution'] for p, q in zip(a, b))
        self.assertEqual(within, 0)
        self.assertEqual(total, .25)

    def test_exact_weighted_auc_matches_explicit_user_replication(self):
        y = np.array([1, 0, 0, 1, 0, 1])
        scores = np.array([.5, .2, .5, .9, .9, .2])
        users = np.array([0, 0, 1, 1, 2, 2])
        prepared = analysis.prepare_weighted_auc(y, scores, users)
        for weights in [np.array([1, 1, 1]), np.array([2, 0, 1]), np.array([0, 1, 2])]:
            idx = np.repeat(np.arange(len(y)), weights[users])
            actual, positives, negatives = analysis.exact_weighted_auc(prepared, weights)
            self.assertAlmostEqual(actual, brute_auc(y[idx], scores[idx]))
            self.assertEqual(positives + negatives, len(idx))

    def test_user_resampling_retains_unequal_cluster_sizes_and_invalid_classes(self):
        y = np.array([1, 0, 0, 0])
        users = np.array([0, 1, 1, 1])
        prepared = analysis.prepare_weighted_auc(y, np.array([.9, .1, .3, .5]), users)
        auc, positives, negatives = analysis.exact_weighted_auc(prepared, np.array([0, 2]))
        self.assertIsNone(auc)
        self.assertEqual((positives, negatives), (0, 6))
        auc, positives, negatives = analysis.exact_weighted_auc(prepared, np.array([2, 0]))
        self.assertIsNone(auc)
        self.assertEqual((positives, negatives), (2, 0))

    def test_shared_user_covariance_cancels_identical_effect_influences(self):
        interval, users, influence = analysis.align_cluster_difference(
            (.05, np.array([1, 2]), np.array([.1, -.1])),
            (.02, np.array([1, 2]), np.array([.1, -.1])))
        self.assertAlmostEqual(interval['estimate'], .03)
        self.assertEqual(interval['cluster_se'], 0)
        np.testing.assert_array_equal(influence, [0, 0])

    def test_union_users_preserves_partial_overlap(self):
        interval, users, influence = analysis.align_cluster_difference(
            (.05, np.array([1, 2]), np.array([.1, -.1])),
            (.02, np.array([2, 3]), np.array([.2, -.2])))
        np.testing.assert_array_equal(users, [1, 2, 3])
        np.testing.assert_allclose(influence, [.1, -.3, .2])
        self.assertAlmostEqual(interval['cluster_se'], np.sqrt(1.5 * (.01 + .09 + .04)))

    def test_contrast_masks_exclude_undefined_and_respect_length_restriction(self):
        features = {'effective_length': np.array([100, 100, 100, 100, 20, 0]),
                    'c1_entropy': np.array([np.log(6), np.log(4), np.nan, np.log(2), np.log(6), 0]),
                    'target_c1_history_fraction': np.array([.25, 1, np.nan, 0, .2, 0])}
        contrasts = analysis.contrast_masks(features)
        np.testing.assert_array_equal(np.flatnonzero(contrasts[0][-1]), [4])
        np.testing.assert_array_equal(np.flatnonzero(contrasts[1][-2]), [0])
        np.testing.assert_array_equal(np.flatnonzero(contrasts[1][-1]), [1])
        np.testing.assert_array_equal(np.flatnonzero(contrasts[2][-2]), [0])
        np.testing.assert_array_equal(np.flatnonzero(contrasts[2][-1]), [1])


if __name__ == '__main__':
    unittest.main()
