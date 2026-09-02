from __future__ import annotations

import unittest
from pathlib import Path

from scripts.build_sisa_seed_pair_report import report_paths, source, validate_inputs


class SisaSeedPairReportTest(unittest.TestCase):
    def test_report_paths_are_candidate_scoped(self):
        reference, candidate, comparison, report = report_paths(20262028, 20262029)
        self.assertEqual(reference, Path("experiments/sisa_single_seed20262028"))
        self.assertEqual(candidate, Path("experiments/sisa_single_seed20262029"))
        self.assertEqual(
            comparison,
            Path("experiments/sisa_single_seed20262029/comparison_vs_seed20262028"),
        )
        self.assertEqual(report, comparison / "report")

    def test_validation_requires_complete_candidate_runs(self):
        runs = [
            {
                "task_id": str(task),
                "complete": "True",
                "seed": "20262029",
                "protocol": "ws2_bs16384_acc1",
                "h100_valid": "True",
            }
            for task in range(16)
        ]
        reference = [
            {"seed": "20262028", "protocol": "ws2_bs16384_acc1"}
            for _ in range(68)
        ]
        candidate = [
            {"seed": "20262029", "protocol": "ws2_bs16384_acc1"}
            for _ in range(68)
        ]
        validate_inputs(runs, reference, candidate, 20262028, 20262029)
        runs[0]["complete"] = "False"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_inputs(runs, reference, candidate, 20262028, 20262029)

    def test_source_includes_runnable_query_provenance(self):
        repository = Path("/tmp/unirank")
        path = repository / "results" / "metrics.csv"
        metadata = source(repository, "metrics", "Metrics", path)
        self.assertEqual(metadata["query"]["engine"], "duckdb")
        self.assertIn("read_csv_auto('results/metrics.csv')", metadata["query"]["sql"])
        self.assertEqual(metadata["query"]["tables_used"], ["results/metrics.csv"])


if __name__ == "__main__":
    unittest.main()
