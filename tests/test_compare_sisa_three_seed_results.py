from __future__ import annotations

import unittest
from pathlib import Path

from scripts.compare_sisa_three_seed_results import (
    descriptive,
    index_unique,
    read_csv,
    select_legacy_metrics,
    validate_aligned_sources,
)


REPOSITORY = Path(__file__).resolve().parents[1]


class SisaThreeSeedComparisonTest(unittest.TestCase):
    def test_paper_and_three_seed_key_space_aligns(self):
        paper = index_unique(
            read_csv(
                REPOSITORY
                / "experiments/sisa_three_seed_unified/sources/paper_table.csv"
            ),
            "paper",
        )
        strict = read_csv(
            REPOSITORY / "experiments/sisa_native_strict/results/metrics.csv"
        )
        expansion = read_csv(
            REPOSITORY / "experiments/sisa_expansion_acd/results/metrics.csv"
        )
        baseline = index_unique(
            select_legacy_metrics(strict, expansion, "baseline"), "baseline"
        )
        seed27 = index_unique(
            select_legacy_metrics(strict, expansion, "sisa"), "seed 20262027"
        )
        seed28 = index_unique(
            read_csv(
                REPOSITORY
                / "experiments/sisa_single_seed20262028/results/metrics.csv"
            ),
            "seed 20262028",
        )
        seed29 = index_unique(
            read_csv(
                REPOSITORY
                / "experiments/sisa_single_seed20262029/results/metrics.csv"
            ),
            "seed 20262029",
        )

        validate_aligned_sources(
            paper,
            baseline,
            {20262027: seed27, 20262028: seed28, 20262029: seed29},
        )
        self.assertEqual(len(paper), 68)
        self.assertEqual(len({(key[0], key[1]) for key in paper}), 16)

    def test_descriptive_uses_sample_standard_deviation(self):
        average, sample_std, minimum, maximum, value_range = descriptive(
            [1.0, 2.0, 3.0]
        )
        self.assertEqual(average, 2.0)
        self.assertEqual(sample_std, 1.0)
        self.assertEqual((minimum, maximum, value_range), (1.0, 3.0, 2.0))

    def test_legacy_sources_are_split_nine_plus_seven_cells(self):
        strict = read_csv(
            REPOSITORY / "experiments/sisa_native_strict/results/metrics.csv"
        )
        expansion = read_csv(
            REPOSITORY / "experiments/sisa_expansion_acd/results/metrics.csv"
        )
        selected = select_legacy_metrics(strict, expansion, "sisa")
        strict_cells = {
            (row["model"], row["dataset"])
            for row in selected
            if row["source_bundle"] == "sisa_native_strict"
        }
        expansion_cells = {
            (row["model"], row["dataset"])
            for row in selected
            if row["source_bundle"] == "sisa_expansion_acd"
        }
        self.assertEqual(len(strict_cells), 9)
        self.assertEqual(len(expansion_cells), 7)
        self.assertEqual(len(selected), 68)


if __name__ == "__main__":
    unittest.main()
