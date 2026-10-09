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
        self.assertEqual(len(options), 82)
        self.assertEqual(len({run["id"] for run in options}), 82)
        self.assertNotIn("645228_5", {run["job"] for run in options})
        self.assertIn("645445_5", {run["job"] for run in options})
        self.assertEqual(sum(run["kind"] == "rerun" for run in options), 8)
        self.assertEqual(sum(run["kind"] == "new_seed" for run in options), 2)
        self.assertEqual(sum(run["kind"] == "target_only" for run in options), 12)

    def test_default_selection_uses_top_auc_per_seed_and_three_distinct_seeds(self):
        for cell in self.catalog["cells"]:
            selected = load_builder().default_selection(cell)
            self.assertEqual(len(selected), 3)
            runs = {run["id"]: run for run in cell["runs"]}
            allowed = [run for run in cell["runs"]
                       if cell["model"] != "HiFormer" or run["kind"] == "target_only"]
            best_by_seed = {}
            for run in allowed:
                prior = best_by_seed.get(run["seed"])
                if prior is None or run["auc"] > prior["auc"]:
                    best_by_seed[run["seed"]] = run
            expected = sorted(best_by_seed.values(), key=lambda run: -run["auc"])[:3]
            self.assertEqual(set(selected.values()), {run["id"] for run in expected})
            self.assertEqual(len({runs[run_id]["seed"] for run_id in selected.values()}), 3)
            if cell["model"] == "HiFormer":
                self.assertTrue(all(runs[run_id]["kind"] == "target_only"
                                    for run_id in selected.values()))

    def test_new_seed_only_enters_when_it_beats_an_existing_seed(self):
        cells = {(cell["model"], cell["dataset"]): cell for cell in self.catalog["cells"]}
        hy = cells["HyFormer", "MerRec_Action"]
        selected = load_builder().default_selection(hy)
        self.assertNotIn("20262030", selected)
        self.assertEqual(selected["20262028"],
                         next(run["id"] for run in hy["runs"]
                              if run["seed"] == 20262028 and run["kind"] == "original"))
        zenith = cells["Zenith", "MerRec_Action"]
        self.assertEqual(load_builder().default_selection(zenith)["20262029"],
                         next(run["id"] for run in zenith["runs"]
                              if run["seed"] == 20262029 and run["kind"] == "rerun"))


if __name__ == "__main__":
    unittest.main()
