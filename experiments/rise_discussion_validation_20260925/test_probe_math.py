import unittest
import numpy as np
from probe_math import crosspart_templates, crosspart_donor_indices, match_moments, moments, bce


class ProbeControlsTest(unittest.TestCase):
    def setUp(self):
        self.parts = np.repeat(np.arange(8), 3)
        self.delta = np.random.default_rng(8).normal(size=(24, 5))
        self.mask = np.ones_like(self.delta, dtype=bool)

    def test_crossfit_excludes_recipient_part(self):
        raw, matched, receipts = crosspart_templates(self.delta, self.parts, self.mask)
        expected = self.delta[self.parts != 0].mean(axis=0)
        expected = np.tile(expected - expected.mean(), (3, 1)) + self.delta[:3].mean(1, keepdims=True)
        np.testing.assert_allclose(raw[self.parts == 0], expected)
        changed = self.delta.copy()
        changed[self.parts == 0] += 1000
        raw_changed, _, _ = crosspart_templates(changed, self.parts, self.mask)
        np.testing.assert_allclose(raw_changed[self.parts == 0] - 1000, raw[self.parts == 0], atol=1e-12)
        self.assertEqual(receipts[0]['peer_rows'], 21)
        for a, b in zip(moments(matched), moments(self.delta)):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_donor_is_reproducible_bijection_excluding_own_part(self):
        donor = crosspart_donor_indices(self.parts, 20260925)
        np.testing.assert_array_equal(donor, crosspart_donor_indices(self.parts, 20260925))
        np.testing.assert_array_equal(np.sort(donor), np.arange(24))
        self.assertTrue(np.all(self.parts[donor] != self.parts))
        matched = match_moments(self.delta[donor], self.delta)
        for a, b in zip(moments(matched), moments(self.delta)):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_flat_source_and_input_drift_rejected(self):
        with self.assertRaises(ValueError):
            match_moments(np.ones((24, 5)), self.delta)
        bad = self.mask.copy(); bad[0, 0] = False
        with self.assertRaises(ValueError):
            crosspart_templates(self.delta, self.parts, bad)

    def test_bce_uses_both_outcomes_and_probability_bounds(self):
        np.testing.assert_allclose(bce(np.array([0, 1]), np.array([.2, .8])), -np.log(.8))
        with self.assertRaises(ValueError):
            bce(np.array([0]), np.array([1.1]))


if __name__ == '__main__':
    unittest.main()
