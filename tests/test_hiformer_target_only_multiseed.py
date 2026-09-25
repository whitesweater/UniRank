"""Protocol and construction regressions for HiFormer target-only seed completion."""

from copy import deepcopy
import gc
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model_zoo.HiFormer import HiFormer
from unirank.features import feature_map_from_config
from unirank.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_hiformer_target_only_multiseed.py"
SPEC = importlib.util.spec_from_file_location("hiformer_multiseed_test", ENTRY)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class HiFormerTargetOnlyMultiseedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".hiformer-multiseed-test-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.fixture_root = Path(temporary.name)
        self.logs = {}
        for task in RUN.TASKS:
            params = load_config(str(ROOT / "config"), f"HiFormer_{task['dataset']}")
            params.update(seed=task["seed"], world_size=task["world_size"],
                          batch_size=task["batch_size"], epochs=1, max_len=100, accumulation_steps=1,
                          sisa_enabled=True, sisa_score_dim=16, sisa_lambda_init=.1,
                          sisa_score_scale=1., enable_bf16=True, enable_torch_compile=True)
            if task["seed"] == 20262029:
                params.update(dataloader_seed=2028, sisa_parameter_seed=20260823)
            else:
                params.pop("dataloader_seed", None)
                params.pop("sisa_parameter_seed", None)
            if task["seed"] == 20262027 and task["dataset"] in {"QK_Video_Action", "MerRec_Action"}:
                params["data_root"] = RUN.HISTORICAL_DATA_ROOT
                for split in ("train", "valid", "test"):
                    for suffix in ("data", "user_info", "item_info"):
                        params[f"{split}_{suffix}"] = f"{RUN.HISTORICAL_DATA_ROOT}/{task['dataset']}/{split}/{suffix}"
            path = self.fixture_root / f"{task['dataset']}_{task['seed']}.log"
            dense = task["expected_dense"] + 62988
            path.write_text("INFO Params: " + json.dumps(params) + "\n"
                            f"Number of dense parameters: {dense}.\n"
                            "Number of embedding parameters: 100.\n"
                            f"Total number of parameters: {dense + 100}.\n"
                            "******** Test evaluation ********\n")
            self.logs[task["dataset"], task["seed"]] = path
        reference_patch = patch.object(RUN, "reference_log", side_effect=lambda task: self.logs[task["dataset"], task["seed"]])
        reference_patch.start()
        self.addCleanup(reference_patch.stop)

    def test_exact_scope_seed_bundle_launch_and_gpu_arguments(self):
        datasets = ("QK_Video_Action", "MerRec_Action", "KuaiRand_Video_Action", "TencentGR_10M_Action")
        self.assertEqual([(t["dataset"], t["seed"]) for t in RUN.TASKS],
                         [(d, seed) for seed in (20262027, 20262029) for d in datasets])
        self.assertEqual([t["reference_task"] for t in RUN.TASKS],
                         ["14811_9", "14810_15", "14850_11", "543257_3", "549595_10", "549595_2", "549595_6", "549595_14"])
        for index, task in enumerate(RUN.TASKS):
            world = 4 if index < 4 else 2
            self.assertEqual(task["world_size"] * task["batch_size"], 32768)
            self.assertIn(f"--nproc_per_node={world}", RUN.training_command(index, 1))
            self.assertIn(str(ENTRY), RUN.training_command(index, 1))
            self.assertEqual(RUN.gpu_argument(task), "0,1,2,3" if world == 4 else "0,1")
            self.assertIn(f"_seed{task['seed']}_ws{world}_bs{task['batch_size']}_attempt1", RUN.identifier(index, 1))
        for task_id, attempt in ((-1, 1), (8, 1), (0, 0)):
            with self.assertRaises(ValueError):
                RUN.identifier(task_id, attempt)

    def test_historical_defaults_paths_and_only_declared_config_differences(self):
        for index, task in enumerate(RUN.TASKS):
            with self.subTest(index=index):
                raw = RUN.reference_params(task)
                actual = RUN.effective_params(index, 1)
                manifest = RUN.review_manifest(index, 1)
                self.assertEqual(actual["dataloader_seed"], 2026 if index < 4 else 2028)
                self.assertEqual(actual["sisa_parameter_seed"], 20260821 if index < 4 else 20260823)
                self.assertEqual(actual["feature_cols"], raw["feature_cols"])
                self.assertEqual(actual["train_data"], f"./datasets/{task['dataset']}/train/data")
                differences = {key for key in raw.keys() | actual.keys() if raw.get(key) != actual.get(key)}
                expected = {"model_id"}
                if index < 4:
                    expected |= {"dataloader_seed", "sisa_parameter_seed"}
                if index < 2:
                    expected |= set(RUN.DATA_PATH_KEYS)
                self.assertEqual(differences, expected)
                self.assertEqual(set(manifest["allowed_parameter_differences"]), expected)
                self.assertFalse(manifest["historical_training_source_revision_verified"])
                self.assertEqual(manifest["parameter_expectations"]["active_dense"], task["expected_dense"])
                for path in (ENTRY, RUN.LEGACY_RUNNER, RUN.SBATCH, RUN.CONTROLS):
                    self.assertIn(str(path.relative_to(ROOT)), manifest["source_sha256"])

    def test_protocol_and_default_drift_rejected(self):
        for task_id, key, replacement in ((0, "batch_size", 16384), (0, "accumulation_steps", 2),
                                          (4, "dataloader_seed", 2027), (4, "sisa_parameter_seed", 20260822)):
            path = self.logs[RUN.TASKS[task_id]["dataset"], RUN.TASKS[task_id]["seed"]]
            original = path.read_text()
            values, end = json.JSONDecoder().raw_decode(original.split("INFO Params: ")[1])
            values[key] = replacement
            path.write_text("INFO Params: " + json.dumps(values) + original.split("INFO Params: ")[1][end:])
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                RUN.review_manifest(task_id, 1)
            path.write_text(original)

    def test_prepare_freezes_source_and_config_and_rejects_overwrite(self):
        with patch.object(RUN, "ARTIFACTS", self.fixture_root / "artifacts"):
            output, original = RUN.prepare(0, 1)
            self.assertEqual(RUN.prepare(0, 1)[1], original)
            RUN.verify_frozen(output, 0, 1)
            for relative, digest in original["source_sha256"].items():
                self.assertEqual(RUN.sha256(output / "source" / relative), digest)
            config = output / "config/model_config.yaml"
            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "configuration"):
                RUN.prepare(0, 1)
            with self.assertRaisesRegex(ValueError, "configuration"):
                RUN.verify_frozen(output, 0, 1)

    def test_fresh_outputs_started_and_allocated_resources(self):
        task = RUN.TASKS[0]
        output = self.fixture_root / "artifacts"
        output.mkdir()
        RUN.ensure_fresh_outputs(output, task, "unused_multiseed_test")
        (output / "started.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            RUN.ensure_fresh_outputs(output, task, "unused_multiseed_test")
        environment = {"SLURM_JOB_ID": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_CPUS_PER_TASK": "32", "SLURM_MEM_PER_NODE": str(960 * 1024)}
        with patch.dict(os.environ, environment, clear=True), patch.object(RUN.subprocess, "check_output", return_value="GPU A\nGPU A\nGPU A\nGPU A\n"):
            self.assertEqual(len(RUN.validate_resources(task)), 4)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            with self.assertRaisesRegex(RuntimeError, "four|4"):
                RUN.validate_resources(task)
        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "4 ranks"):
                RUN.run_worker(0, 1)

    def build_model(self, task_id, mode):
        task = RUN.TASKS[task_id]
        params = deepcopy(RUN.effective_params(task_id, 1))
        for column in params["feature_cols"]:
            if "vocab_size" in column:
                column["vocab_size"] = min(column["vocab_size"], 8)
        params.update(gpu=-1, distributed=False, world_size=1, rank=0, local_rank=0,
                      enable_torch_compile=False, enable_bf16=False, verbose=0,
                      model_root=str(self.fixture_root), sisa_enabled=mode != "native")
        feature_map = feature_map_from_config(params)
        torch.manual_seed(task["seed"])
        if mode == "target_only":
            with RUN.patched_build(mode):
                model = HiFormer(feature_map, **params)
        else:
            model = HiFormer(feature_map, **params)
        return model, torch.get_rng_state().clone()

    def test_retained_parameters_global_rng_and_target_seed_all_tasks(self):
        for index, task in enumerate(RUN.TASKS):
            with self.subTest(task=index):
                full, full_rng = self.build_model(index, "full")
                target, target_rng = self.build_model(index, "target_only")
                self.assertTrue(torch.equal(full_rng, target_rng))
                full_state, target_state = full.state_dict(), target.state_dict()
                self.assertTrue(set(target_state) < set(full_state))
                for name, value in target_state.items():
                    torch.testing.assert_close(value, full_state[name], atol=0, rtol=0)
                removed = set(full_state) - set(target_state)
                self.assertTrue(all("mixer_layers." in name and ".sisa_score_bias." in name for name in removed))
                self.assertEqual(sum(full_state[name].numel() for name in removed), 62988)
                before = RUN.rng_fingerprints()
                audit = RUN.audit_model(target, "target_only", task["expected_dense"])
                self.assertEqual(before, RUN.rng_fingerprints())
                self.assertEqual([site["enabled"] for site in audit["sites"]], [True, False, False, False])
                self.assertEqual(audit["sites"][0]["parameter_seed"], 20260821 if index < 4 else 20260823)
                del full_state, full
                gc.collect()
                native, native_rng = self.build_model(index, "native")
                self.assertTrue(torch.equal(native_rng, target_rng))
                for name, value in native.state_dict().items():
                    torch.testing.assert_close(value, target_state[name], atol=0, rtol=0)
                del native, target_state, target
                gc.collect()

    def test_sbatch_accepts_eight_tasks_without_concurrency_cap(self):
        text = RUN.SBATCH.read_text()
        self.assertNotIn("--array=0-7%", text)
        self.assertIn("scripts/run_hiformer_target_only_multiseed.py", text)
        self.assertIn("#SBATCH --array=4-7", text)


if __name__ == "__main__":
    unittest.main()
