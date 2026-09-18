"""CPU tests for accepting OneTrans repeated-seed evidence without false pairing."""

import importlib.util
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_onetrans_seed_completion.py"
SPEC = importlib.util.spec_from_file_location("collect_onetrans_seed_completion", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class OneTransSeedCollectorTest(unittest.TestCase):
    def test_user_cancellation_is_preserved_when_sacct_omits_unstarted_elements(self):
        receipt = {"returncode": 0, "cancelled_job_task_ids": ["100_6", "100_7"]}
        self.assertEqual(collector.resolve_account("100_6", {}, receipt)["state"], "CANCELLED_BY_USER")
        self.assertEqual(collector.resolve_account("100_0", {}, receipt), {})
        self.assertEqual(collector.resolve_account("200_6", {}, receipt), {})
        self.assertEqual(collector.resolve_account("100_6", {}, dict(receipt, returncode=1)), {})

    def fixture(self):
        task = collector.study_task(0)
        params = dict(rank=0, local_rank=0, seed=task.seed, model_id=task.run_id,
                      world_size=4, batch_size=8192, label_col=[{"name": "click"}])
        coverage = [dict(split=split, rank=rank, blocks=1, assigned_rows=8193,
                         batches=1, processed_rows=8192)
                    for rank in range(4) for split in ("train", "valid", "test")]
        allocations = "\n".join(
            f"[Rank {rank}] blocked load balanced: blocks=1, rows=8193, batches=1"
            for _ in range(3) for rank in range(4))
        log = ("INFO Params: " + json.dumps(params) + "\n" + allocations
               + "\n******** Test evaluation ********\n"
               + "INFO [Task: click][Metrics] logloss: 0.2 - AUC: 0.8\n")
        manifest = dict(task_id=0, model="OneTrans", dataset=task.dataset, setting="sisa",
                        seed=task.seed, run_id=task.run_id, attempt=1, world_size=4,
                        per_gpu_batch=8192, global_batch=32768, dataloader_seed=2027,
                        sisa_parameter_seed=20260822, array_job_id="100", job_id="101",
                        gpu_names=["NVIDIA H100 80GB HBM3"] * 4, git_revision="a" * 40,
                        effective_params=params)
        slurm = (log + "ONETRANS_SEED_PROTOCOL " + json.dumps(manifest) + "\n"
                 + f"ONETRANS_SEED_COMPLETE run_id={task.run_id}\n")
        return dict(task=task, expected=params, expected_coverage=coverage, checkpoint=log,
                    slurm=slurm, manifest=manifest, outcome={"returncode": 0}, array_job="100",
                    account=dict(state="COMPLETED", exit_code="0:0", raw_job_id="101"))

    def test_accepts_four_gpu_protocol_and_rejects_two_gpu_manifest(self):
        data = self.fixture()
        result = collector.audit_new_run(**data)
        self.assertTrue(result["eligible"], result["issues"])
        self.assertEqual(len(result["coverage"]), 12)
        data["manifest"]["world_size"] = 2
        result = collector.audit_new_run(**data)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["metrics"], {})

    def test_incomplete_or_failed_task_never_exposes_test_metrics(self):
        for change in (dict(state="RUNNING"), dict(state="PENDING"), dict(exit_code="1:0")):
            data = self.fixture()
            data["account"].update(change)
            result = collector.audit_new_run(**data)
            self.assertFalse(result["eligible"])
            self.assertEqual(result["metrics"], {})
            self.assertIsNone(result["macro_auc"])
        data = self.fixture()
        data["outcome"] = {}
        self.assertFalse(collector.audit_new_run(**data)["eligible"])

    def test_rejects_validation_only_duplicate_nonfinite_or_disagreeing_metrics(self):
        for transform in (
            lambda text: text.replace("Test evaluation", "Validation evaluation"),
            lambda text: text + "INFO [Task: click][Metrics] logloss: 0.2 - AUC: 0.8\n",
            lambda text: text.replace("AUC: 0.8", "AUC: nan"),
            lambda text: text.replace("AUC: 0.8", "AUC: 0.9"),
        ):
            data = self.fixture()
            data["checkpoint"] = transform(data["checkpoint"])
            self.assertFalse(collector.audit_new_run(**data)["eligible"])

    def test_rejects_seed_group_id_or_coverage_drift(self):
        for old, new in (("20262028", "20262027"), ('"world_size": 4', '"world_size": 2')):
            data = self.fixture()
            data["checkpoint"] = data["checkpoint"].replace(old, new)
            self.assertFalse(collector.audit_new_run(**data)["eligible"])
        data = self.fixture()
        data["expected"]["group_id"] = "user_id"
        self.assertFalse(collector.audit_new_run(**data)["eligible"])
        data = self.fixture()
        data["slurm"] = data["slurm"].replace("rows=8193", "rows=8194", 1)
        self.assertFalse(collector.audit_new_run(**data)["eligible"])

    def test_historical_path_portability_does_not_mask_other_fields(self):
        params = dict(data_root="/old/project/datasets", train_data="/old/project/datasets/Taobao_Action/train/data",
                      group_id="user_index", model_root="./checkpoints/")
        normalized = collector.portable_params(params, "Taobao_Action")
        self.assertEqual(normalized["data_root"], "./datasets")
        self.assertEqual(normalized["train_data"], "./datasets/Taobao_Action/train/data")
        self.assertEqual(normalized["group_id"], "user_index")
        self.assertTrue(collector.config_issues(normalized, dict(normalized, group_id="user_id")))

    def test_three_seed_summary_requires_unique_complete_sisa_seeds(self):
        runs = [dict(dataset="D", model="OneTrans", setting="sisa", seed=seed, eligible=True,
                     metrics={"click": {"AUC": auc, "logloss": 1 - auc}},
                     macro_auc=auc, macro_logloss=1 - auc)
                for seed, auc in ((20262027, .7), (20262028, .8), (20262029, .9))]
        runs.append(dict(dataset="D", model="OneTrans", setting="baseline", seed=20262027,
                         eligible=True, metrics={"click": {"AUC": .6, "logloss": .4}},
                         macro_auc=.6, macro_logloss=.4))
        summary = collector.three_seed_summary(runs)
        self.assertEqual(len(summary), 2)
        self.assertAlmostEqual(summary[0]["sisa_auc_mean"], .8)
        self.assertAlmostEqual(summary[0]["sisa_auc_sample_sd"], .1)
        self.assertEqual(summary[0]["baseline_seed"], 20262027)
        runs[2]["eligible"] = False
        self.assertEqual(collector.three_seed_summary(runs), [])
        runs[2]["eligible"] = True
        self.assertEqual(collector.three_seed_summary(runs + [runs[2]]), [])

    def test_attempt_override_changes_only_run_identity(self):
        first, retry = collector.study_task(3), collector.study_task(3, 2)
        self.assertEqual(first.seed, retry.seed)
        self.assertNotEqual(first.run_id, retry.run_id)
        self.assertEqual(collector.parse_overrides(["3=2"], int), {3: 2})
        with self.assertRaises(ValueError):
            collector.parse_overrides(["10=2"], int)

    def test_accounting_expands_arrays_and_never_maps_compressed_ranges_to_tasks(self):
        output = "630948_0|630949|PENDING|0:0\n630948_1|630950|RUNNING|0:0\n"
        with patch.object(collector.subprocess, "run", return_value=SimpleNamespace(stdout=output, stderr="")) as run:
            records, _ = collector.query_accounting({"630948"})
            self.assertIn("--array", run.call_args.args[0])
            self.assertEqual(records["630948_0"]["state"], "PENDING")
            self.assertEqual(records["630948_1"]["state"], "RUNNING")
        with patch.object(collector.subprocess, "run", return_value=SimpleNamespace(
                stdout="630948_[0-9%4]|630948|PENDING|0:0\n", stderr="")):
            records, _ = collector.query_accounting({"630948"})
            self.assertNotIn("630948_0", records)


if __name__ == "__main__":
    unittest.main()
