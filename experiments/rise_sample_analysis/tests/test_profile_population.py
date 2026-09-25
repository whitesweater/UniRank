import importlib.util
import unittest
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/profile_group_population.py'
spec = importlib.util.spec_from_file_location('population_profile', SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfilePopulationTest(unittest.TestCase):
    def test_entity_quantiles_weight_entities_not_interactions(self):
        counts = np.array([0, 1, 1, 1, 1, 2, 100], dtype=np.uint64)
        summary, values, weights = profile.frequency_profile(counts)
        self.assertEqual(summary['distinct_observed_entities'], 6)
        self.assertEqual(summary['target_rows'], 106)
        self.assertEqual((summary['q25'], summary['q75']), (1, 2))
        expected = np.quantile(counts[counts > 0], [.25, .75], method='inverted_cdf')
        np.testing.assert_array_equal(profile.weighted_quantile(values, weights, [.25, .75]), expected)

    def test_tied_frequency_cutpoints_keep_empty_middle_group(self):
        features = self._features()
        groups = list(profile.group_dimensions(features, {'user': (1, 1), 'item': (1, 1)}))
        dimension, memberships, names = groups[1]
        self.assertEqual(dimension, 'item_train_target_frequency')
        self.assertIn(2, names)
        self.assertFalse((memberships == 2).any())
        self.assertEqual(memberships[0], 0)

    def test_length100_restriction_and_undefined_categories(self):
        features = self._features()
        groups = dict((name, (values, names)) for name, values, names in profile.group_dimensions(
            features, {'user': (1, 2), 'item': (1, 2)}))
        np.testing.assert_array_equal(groups['effective_length'][0], [0, 1, 4, 4, 4, 4])
        np.testing.assert_array_equal(groups['c1_entropy_at_length100'][0], [-1, -1, 0, 1, 2, 4])
        np.testing.assert_array_equal(groups['candidate_category_support_at_length100'][0], [-1, -1, 0, 1, 2, 4])

    def test_positive_counts_include_all_five_labels_and_zero_groups(self):
        features = self._features()
        labels = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
        groups = {}
        profile.update_groups(groups, features, labels, {'user': (1, 1), 'item': (1, 1)})
        n = sum(v['rows'] for k, v in groups.items() if k[0] == 'effective_length')
        self.assertEqual(n, 6)
        np.testing.assert_array_equal(groups[('effective_length', 4, '100')]['positive'], [1, 1, 2, 2, 2])
        self.assertEqual(groups[('effective_length', 2, '21-50')]['rows'], 0)

    def test_generated_output_mtime_drift_is_recorded_but_input_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'output.parquet'
            path.write_bytes(b'example')
            expected = {'path': str(path), 'bytes': 7, 'mtime_ns': path.stat().st_mtime_ns - 1}
            with self.assertRaisesRegex(ValueError, 'source changed'):
                profile.verify_source_identity(expected)
            discrepancy = profile.verify_source_identity(expected, allow_output_mtime_drift=True)
            self.assertEqual(discrepancy['mtime_difference_ns'], 1)
            expected['bytes'] = 8
            with self.assertRaisesRegex(ValueError, 'size changed'):
                profile.verify_source_identity(expected, allow_output_mtime_drift=True)

    def test_independent_scalar_entropy_reconstructs_ambiguous_target_count(self):
        actual = profile.independent_window_features([0, 1, 2, 2, 3, 4], 3,
                                                       {0: 0, 1: 1, 2: 2, 3: 309, 4: 0})
        self.assertEqual(actual['effective_length'], 5)
        self.assertEqual(actual['c1_known_count'], 3)
        self.assertEqual(actual['c1_ambiguous_count'], 1)
        self.assertEqual(actual['target_c1_history_count'], 1)
        self.assertIsNone(actual['target_c1_history_fraction'])
        self.assertAlmostEqual(actual['c1_entropy'], -np.log(1 / 3) / 3 - 2 * np.log(2 / 3) / 3)

    def test_reordered_physical_ids_rejected_even_when_count_matches(self):
        features = {'part_id': np.array([0, 0]), 'row_offset': np.array([1, 0])}
        with self.assertRaisesRegex(ValueError, 'identity'):
            profile.validate_alignment(features, {}, 0, 2)

    def _features(self):
        return {'effective_length': np.array([0, 10, 100, 100, 100, 100]),
                'item_train_target_count': np.array([0, 1, 1, 1, 2, 4]),
                'user_train_target_count': np.array([1, 1, 1, 2, 4, 0]),
                'c1_entropy': np.array([np.nan, .2, np.nan, 0, np.log(2), np.log(6)]),
                'target_c1_history_fraction': np.array([np.nan, .2, np.nan, 0, .25, 1.])}


if __name__ == '__main__':
    unittest.main()
