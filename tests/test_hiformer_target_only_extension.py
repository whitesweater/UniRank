"""CPU checks for the three-dataset HiFormer placement follow-up."""

from copy import deepcopy
import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model_zoo.HiFormer import HiFormer
from unirank.features import feature_map_from_config
from unirank.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_hiformer_target_only_extension.py"
SPEC = importlib.util.spec_from_file_location("hiformer_target_only_extension_test", ENTRY)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class HiFormerTargetOnlyExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".hiformer-target-only-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.fixture_root = Path(self.temporary.name)
        self.params = {}
        self.logs = {}
        for task in RUN.TASKS:
            dataset = task["dataset"]
            params = load_config(str(ROOT / "config"), f"HiFormer_{dataset}")
            params.update(seed=20262028, world_size=2, batch_size=16384,
                          dataloader_seed=2027, sisa_parameter_seed=20260822,
                          sisa_enabled=True, sisa_score_dim=16, sisa_lambda_init=.1,
                          sisa_score_scale=1., enable_bf16=True, enable_torch_compile=True)
            self.params[dataset] = params
            dense = task["expected_dense"] + 62988
            path = self.fixture_root / f"{dataset}.log"
            path.write_text("INFO Params: " + json.dumps(params) + "\n"
                            f"Number of dense parameters: {dense}.\n"
                            "Number of embedding parameters: 100.\n"
                            f"Total number of parameters: {dense + 100}.\n"
                            "******** Test evaluation ********\n")
            self.logs[dataset] = path
        self.reference_patch = patch.object(
            RUN._BASE, "reference_log", side_effect=lambda task: self.logs[task["dataset"]])
        self.reference_patch.start()
        self.addCleanup(self.reference_patch.stop)

    def test_exact_scope_worker_entry_and_isolation(self):
        self.assertEqual([task["dataset"] for task in RUN.TASKS],
                         ["MerRec_Action", "KuaiRand_Video_Action", "TencentGR_10M_Action"])
        self.assertEqual([task["reference_task"] for task in RUN.TASKS],
                         ["548166_2", "548166_6", "548166_14"])
        self.assertTrue(all(task["model"] == "HiFormer" and task["mode"] == "target_only"
                            for task in RUN.TASKS))
        self.assertEqual(RUN._BASE.ARTIFACTS, ROOT / "artifacts/rise_hiformer_target_only_extension")
        for index in range(3):
            command = RUN.training_command(index, 1)
            self.assertIn(str(ENTRY), command)
            self.assertIn("--nproc_per_node=2", command)
            self.assertNotIn(str(RUN.LEGACY_RUNNER), command)
            self.assertIn(RUN.TASKS[index]["dataset"], RUN.identifier(index, 1))
        for task_id, attempt in ((-1, 1), (3, 1), (0, 0)):
            with self.assertRaises(ValueError):
                RUN.identifier(task_id, attempt)

    def test_manifest_preserves_reference_protocol_and_audits_new_source(self):
        for index, task in enumerate(RUN.TASKS):
            with self.subTest(dataset=task["dataset"]):
                reference = RUN.reference_params(task)
                actual = RUN.effective_params(index, 1)
                self.assertEqual({key for key in reference.keys() | actual.keys()
                                  if reference.get(key) != actual.get(key)}, {"model_id"})
                manifest = RUN.review_manifest(index, 1)
                self.assertEqual(manifest["study"], "rise_hiformer_target_only_extension")
                self.assertEqual(manifest["process_local_treatment"], "target_only")
                self.assertEqual(manifest["parameter_expectations"]["active_dense"], task["expected_dense"])
                self.assertEqual(manifest["parameter_expectations"]["derivation"]["removed_site_parameters"], 62988)
                sources = manifest["source_sha256"]
                for path in (ENTRY, RUN.LEGACY_RUNNER, RUN.SBATCH,
                             ROOT / "experiments/rise_sample_analysis/scripts/targeted_controls.py"):
                    self.assertIn(str(path.relative_to(ROOT)), sources)
                self.assertEqual(manifest["command"], RUN.training_command(index, 1))

    def test_rejects_drifted_reference_protocol(self):
        dataset = RUN.TASKS[0]["dataset"]
        text = self.logs[dataset].read_text().replace('"seed": 20262028', '"seed": 20262029')
        self.logs[dataset].write_text(text)
        with self.assertRaisesRegex(ValueError, "Historical targeted protocol changed"):
            RUN.review_manifest(0, 1)

    def build_model(self, task, mode):
        params = deepcopy(self.params[task["dataset"]])
        for column in params["feature_cols"]:
            if "vocab_size" in column:
                column["vocab_size"] = min(column["vocab_size"], 8)
        params.update(gpu=-1, distributed=False, world_size=1, rank=0, local_rank=0,
                      enable_torch_compile=False, enable_bf16=False, verbose=0,
                      model_root=str(self.fixture_root), sisa_enabled=mode != "native")
        feature_map = feature_map_from_config(params)
        torch.manual_seed(20262028)
        if mode == "target_only":
            with RUN._BASE.patched_build(mode):
                model = HiFormer(feature_map, **params)
        else:
            model = HiFormer(feature_map, **params)
        return model, torch.get_rng_state().clone()

    def test_real_dense_initializations_rng_and_only_target_site_match(self):
        for task in RUN.TASKS:
            with self.subTest(dataset=task["dataset"]):
                full, full_rng = self.build_model(task, "full")
                target, target_rng = self.build_model(task, "target_only")
                self.assertTrue(torch.equal(full_rng, target_rng))
                full_state, target_state = full.state_dict(), target.state_dict()
                self.assertTrue(set(target_state) < set(full_state))
                for name, value in target_state.items():
                    torch.testing.assert_close(value, full_state[name], atol=0, rtol=0)
                removed = set(full_state) - set(target_state)
                self.assertTrue(all("mixer_layers." in name and ".sisa_score_bias." in name
                                    for name in removed))
                self.assertEqual(sum(full_state[name].numel() for name in removed), 62988)
                audit = RUN._BASE.audit_model(target, "target_only", task["expected_dense"])
                self.assertEqual(audit["enabled_site_count"], 1)
                self.assertEqual([site["enabled"] for site in audit["sites"]], [True, False, False, False])
                self.assertEqual(audit["sites"][0]["parameter_seed"], 20260822)
                del full_state, full
                gc.collect()
                native, native_rng = self.build_model(task, "native")
                self.assertTrue(torch.equal(native_rng, target_rng))
                native_state = native.state_dict()
                self.assertEqual(set(native_state), {name for name in target_state if "sisa_score_bias" not in name})
                for name, value in native_state.items():
                    torch.testing.assert_close(value, target_state[name], atol=0, rtol=0)
                del native_state, target_state, native, target
                gc.collect()

    def test_sbatch_has_exact_three_tasks_without_concurrency_cap(self):
        text = RUN.SBATCH.read_text()
        self.assertIn("#SBATCH --array=0-2\n", text)
        self.assertNotIn("--array=0-2%", text)
        self.assertIn("#SBATCH --gres=gpu:2", text)
        self.assertIn("scripts/run_hiformer_target_only_extension.py", text)


if __name__ == "__main__":
    unittest.main()
