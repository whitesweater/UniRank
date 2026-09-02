from __future__ import annotations

import unittest
from pathlib import Path

from scripts.compare_sisa_seed_pair_results import (
    seed_pair_paths,
    validate_study_rows,
)


class SisaSeedPairComparisonTest(unittest.TestCase):
    def test_seed_pair_paths_are_candidate_scoped(self):
        reference, candidate, output = seed_pair_paths(20262028, 20262029)
        self.assertEqual(
            reference,
            Path("experiments/sisa_single_seed20262028/results/metrics.csv"),
        )
        self.assertEqual(
            candidate,
            Path("experiments/sisa_single_seed20262029/results/metrics.csv"),
        )
        self.assertEqual(
            output,
            Path(
                "experiments/sisa_single_seed20262029/"
                "comparison_vs_seed20262028"
            ),
        )

    def test_seed_pair_must_use_different_seeds(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            seed_pair_paths(20262029, 20262029)

    def test_study_rows_require_expected_seed_and_protocol(self):
        rows = [
            {
                "seed": "20262029",
                "protocol": "ws2_bs16384_acc1",
            }
            for _ in range(68)
        ]
        validate_study_rows(rows, 20262029, "candidate")
        rows[0]["seed"] = "20262028"
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            validate_study_rows(rows, 20262029, "candidate")


if __name__ == "__main__":
    unittest.main()
