import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('native_auc_structure', SCRIPTS / 'analyze_native_auc_structure.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeAUCStructureTest(unittest.TestCase):
    def test_native_control_is_explicit_and_reconstruction_handles_empty_pairs(self):
        y = np.array([1, 0, 1, 0])
        a = np.array([.9, .8, .7, .6])
        e = np.array([.9, .4, .7, .6])
        group = np.array([0, 0, 1, 1])
        rows = module.native_pair_table(y, a, e, group)
        self.assertEqual(len(rows), 25)
        self.assertAlmostEqual(sum(r['A_contribution'] for r in rows), .75)
        self.assertAlmostEqual(sum(r['E_contribution'] for r in rows), 1.)
        self.assertAlmostEqual(sum(r['delta_contribution'] for r in rows), .25)
        self.assertNotIn('B_pair_auc', rows[0])
        self.assertTrue(any(r['A_pair_auc'] is None for r in rows))

    def test_equal_scores_receive_half_credit_with_explicit_native_role(self):
        rows = module.native_pair_table(np.array([1, 0]), np.array([.3, .3]),
                                       np.array([.4, .4]), np.array([4, 4]))
        self.assertEqual(sum(r['A_contribution'] for r in rows), .5)
        self.assertEqual(sum(r['delta_contribution'] for r in rows), 0)


if __name__ == '__main__':
    unittest.main()
