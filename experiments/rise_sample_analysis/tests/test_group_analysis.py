from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_groups import macro_summary, summarize_group, unique_sorted_keys, frequency_groups


class GroupAnalysisTest(unittest.TestCase):
    def test_macro_preserves_perfect_label_dependence(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        users = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        predictions = {
            "B": np.array([.1, .4, .6, .8, .3, .2, .7, .5]),
            "C": np.array([.1, .3, .5, .7, .4, .2, .6, .8]),
            "D": np.array([.2, .4, .7, .8, .1, .3, .6, .5]),
            "E": np.array([.1, .6, .3, .8, .2, .4, .7, .5]),
        }
        result = summarize_group(y, predictions, users)
        _, effects = macro_summary([result] * 5)
        individual = next(x for x in result[1] if x["comparison"] == "E-B")
        macro = next(x for x in effects if x["comparison"] == "E-B")
        self.assertGreater(individual["auc_cluster_se"], 0)
        self.assertAlmostEqual(macro["delta_auc"], individual["delta_auc"])
        self.assertAlmostEqual(macro["auc_cluster_se"], individual["auc_cluster_se"])

    def test_single_class_groups_retain_support_without_auc(self):
        points, effects, _ = summarize_group(np.ones(4), {"B": np.ones(4) * .3, "E": np.ones(4) * .4},
                                            np.array([1, 1, 2, 2]))
        self.assertTrue(all(point["AUC"] is None for point in points))
        self.assertIsNone(effects[0]["delta_auc"])
        self.assertLess(effects[0]["delta_logloss"], 0)

    def test_macro_does_not_silently_drop_undefined_label(self):
        good = summarize_group(np.array([0, 1, 0, 1]),
                               {"B": np.array([.2, .7, .3, .8]), "E": np.array([.1, .8, .2, .9])},
                               np.array([1, 1, 2, 2]))
        missing = summarize_group(np.ones(4), {"B": np.ones(4) * .3, "E": np.ones(4) * .4},
                                  np.array([1, 1, 2, 2]))
        points, effects = macro_summary([good, missing])
        self.assertTrue(all(point["AUC"] is None for point in points))
        self.assertIsNone(effects[0]["delta_auc"])

    def test_physical_identity_rejects_duplicate_and_unsorted_rows(self):
        for offset in ([0, 0], [2, 1]):
            with self.assertRaises(ValueError):
                unique_sorted_keys({"part_id": np.array([0, 0]), "row_offset": np.array(offset)})

    def test_frequency_zero_is_separate_and_ties_do_not_invent_groups(self):
        group, _ = frequency_groups(np.array([0, 1, 2, 3]), (1, 1))
        np.testing.assert_array_equal(group, [0, 1, 3, 3])

    def test_group_logloss_uses_official_float64_evaluation_clipping(self):
        y = np.array([1, 0, 1, 0])
        scores = np.array([0, 1, .2, .8], dtype=np.float32)
        points, _, _ = summarize_group(y, {"B": scores, "E": scores}, np.array([1, 1, 2, 2]))
        self.assertAlmostEqual(points[0]["logloss"], log_loss(y, scores.astype(np.float64)))


if __name__ == "__main__":
    unittest.main()
