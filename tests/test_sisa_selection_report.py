"""Protect the evidence and three-distinct-seed selection contract."""

import importlib.util
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "experiments/sisa_three_seed_unified/report/build_selector.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_selector", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SISASelectionReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_builder().collect_catalog(ROOT)

    def test_exact_accepted_matrix_and_one_baseline_per_cell(self):
        cells = self.catalog["cells"]
        self.assertEqual(len(cells), 20)
        self.assertEqual(len({(c["model"], c["dataset"]) for c in cells}), 20)
        self.assertEqual(sum(len(c["baseline"]["labels"]) for c in cells), 85)
        self.assertTrue(all(c["baseline"]["seed"] == 20262027 for c in cells))

    def test_run_options_keep_retries_distinct_and_exclude_failed_attempt(self):
        cells = self.catalog["cells"]
        options = [run for cell in cells for run in cell["runs"]]
        self.assertEqual(len(options), 70)
        self.assertEqual(len({run["id"] for run in options}), 70)
        self.assertNotIn("645228_5", {run["job"] for run in options})
        self.assertIn("645445_5", {run["job"] for run in options})
        self.assertEqual(sum(run["kind"] == "rerun" for run in options), 8)
        self.assertEqual(sum(run["kind"] == "new_seed" for run in options), 2)

    def test_default_selection_has_three_distinct_seeds_and_reconciles(self):
        for cell in self.catalog["cells"]:
            selected = load_builder().default_selection(cell)
            self.assertEqual(set(selected), {"20262027", "20262028", "20262029"})
            runs = {run["id"]: run for run in cell["runs"]}
            auc = statistics.fmean(runs[selected[str(seed)]]["auc"]
                                   for seed in (20262027, 20262028, 20262029))
            self.assertAlmostEqual(auc, cell["original_three_seed_auc"], places=12)
            self.assertEqual(len({runs[run_id]["seed"] for run_id in selected.values()}), 3)


if __name__ == "__main__":
    unittest.main()
