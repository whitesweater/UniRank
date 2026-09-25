import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_matched_baseline.py"
SPEC = importlib.util.spec_from_file_location("rise_matched_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MatchedBaselineTest(unittest.TestCase):
    def test_only_treatment_and_output_identity_change(self):
        for task in MODULE.TASKS:
            reference = MODULE.reference_params(task)
            effective = MODULE.baseline_params(task, attempt=1)
            differences = {k for k in reference.keys() | effective.keys()
                           if reference.get(k) != effective.get(k)}
            self.assertEqual(differences, {"sisa_enabled", "model_id", "model_root"})
            self.assertFalse(effective["sisa_enabled"])
            self.assertEqual(effective["world_size"], 2)
            self.assertEqual(effective["batch_size"], 16384)
            self.assertEqual(effective["seed"], 20262028)

    def test_variant_and_optimizer_protocol_are_not_borrowed(self):
        merrec, qk, taac = [MODULE.baseline_params(task, 1) for task in MODULE.TASKS]
        self.assertEqual(merrec["sparse_adagrad_chunk_size"], 16777216)
        self.assertNotIn("sparse_adagrad_chunk_size", qk)
        self.assertNotIn("sparse_adagrad_chunk_size", taac)
        self.assertEqual(taac["group_id"], "user_index")

    def test_baseline_command_does_not_enable_auxiliary_branch(self):
        command = MODULE.training_command(Path("/tmp/config"), "example")
        self.assertIn("--nproc_per_node=2", command)
        self.assertNotIn("--sisa-enabled", command)
        self.assertNotIn("--sisa-variant", command)

    def test_attempt_must_be_positive(self):
        with self.assertRaises(ValueError):
            MODULE.baseline_params(MODULE.TASKS[0], 0)


if __name__ == "__main__":
    unittest.main()
