"""CPU regression tests for three-seed target-only evidence eligibility."""

from copy import deepcopy
import importlib.util
import hashlib
import json
import shutil
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_hiformer_multiseed_test", ROOT / "scripts/collect_hiformer_target_only_multiseed.py")
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


class CollectHiFormerTargetOnlyMultiseedTest(unittest.TestCase):
    def submission(self):
        tasks = []
        for index in range(8):
            seed = (20262027, 20262029)[index // 4]
            dataset = COLLECT.DATASETS[index % 4]
            array = "900" if index < 4 else "901"
            world, batch = (4, 8192) if seed == 20262027 else (2, 16384)
            run_id = f"RISE_hiformer_target_only_{dataset}_seed{seed}_ws{world}_bs{batch}_attempt1"
            tasks.append({"task_index": index, "seed": seed, "dataset": dataset,
                          "array_id": array, "job_task_id": f"{array}_{index}", "attempt": 1,
                          "run_id": run_id,
                          "manifest": f"artifacts/rise_hiformer_target_only_multiseed/{run_id}/frozen_manifest.json",
                          "checkpoint_log": f"checkpoints/{dataset}/{run_id}.log",
                          "slurm_log": f"logs/rise-hiformer-target-only-multiseed-{array}_{index}.out"})
        return {"tasks": tasks, "arrays": [{"array_id": "900", "task_indices": list(range(4))},
                                            {"array_id": "901", "task_indices": list(range(4, 8))}]}

    def runs(self):
        return [{"dataset": dataset, "seed": seed, "eligible": True, "macro_auc": .8 + n * .01,
                 "macro_logloss": .2 - n * .01, "job_task_id": f"job{seed}_{i}",
                 "state": "COMPLETED", "evidence_origin": "reused_seed28" if n == 1 else "new_multiseed",
                 "issues": [], "pending_checks": [], "metrics": {}, "parameter_counts": {}}
                for i, dataset in enumerate(COLLECT.DATASETS) for n, seed in enumerate(COLLECT.SEEDS)]

    def references(self):
        return [{"dataset": row["dataset"], "seed": row["seed"], "eligible": True,
                 "metrics": {"fixture": {"AUC": .79, "logloss": .21}}, "job_task_id": "full"}
                for row in self.runs()]

    def test_task_scope_and_split_arrays(self):
        submission = self.submission()
        result = COLLECT.normalize_submission(submission)
        self.assertEqual(len(result), 8)
        self.assertEqual([row["world_size"] for row in result], [4] * 4 + [2] * 4)
        submission["tasks"][0]["dataset"] = COLLECT.DATASETS[1]
        with self.assertRaises(ValueError):
            COLLECT.normalize_submission(submission)

    def test_single_gap_retry_can_use_another_array(self):
        submission = self.submission()
        retry = submission["tasks"][2]
        retry.update(array_id="902", job_task_id="902_2", attempt=2)
        for field in ("run_id", "manifest", "checkpoint_log"):
            retry[field] = retry[field].replace("attempt1", "attempt2")
        retry["slurm_log"] = "logs/rise-hiformer-target-only-multiseed-902_2.out"
        submission["arrays"].append({"array_id": "902", "task_indices": [2]})
        self.assertEqual(COLLECT.normalize_submission(submission)[2]["job_task_id"], "902_2")

    def test_summary_requires_three_distinct_eligible_seeds_and_sample_sd(self):
        runs = self.runs()
        rows = COLLECT.summarize(runs, self.references(), [])
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]["target_only_auc_mean"], .81)
        self.assertAlmostEqual(rows[0]["target_only_auc_sd"], .01)
        self.assertAlmostEqual(rows[0]["delta_auc_mean"], .02)
        runs[0].update(eligible=False, macro_auc=None, macro_logloss=None)
        row = COLLECT.summarize(runs, self.references(), [])[0]
        self.assertEqual(row["completed_seeds"], 2)
        self.assertIsNone(row["target_only_auc_mean"])
        self.assertIsNone(row["target_only_auc_sd"])
        runs[0] = deepcopy(runs[1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            COLLECT.summarize(runs, self.references(), [])

    def test_pending_run_does_not_generate_comparison(self):
        run = self.runs()[0]
        run.update(eligible=False, metrics={"fixture": {"AUC": .9, "logloss": .1}})
        self.assertEqual(COLLECT.compare_run(run, self.references()[0]), [])

    def test_all_terminal_only_refers_to_eight_new_jobs(self):
        runs = self.runs()
        runs[1].update(state="UNKNOWN", eligible=False)
        flags = COLLECT.completion_flags(runs)
        self.assertTrue(flags["all_terminal"])
        self.assertFalse(flags["complete"])
        self.assertTrue(flags["needs_attention"])
        runs[0]["state"] = "RUNNING"
        self.assertFalse(COLLECT.completion_flags(runs)["all_terminal"])

    def test_report_always_retains_twelve_rows_and_separates_single_baseline(self):
        runs = self.runs()
        runs[0].update(eligible=False, macro_auc=None, macro_logloss=None)
        evidence = {"runs": runs, "comparisons": [], "historical_references": self.references(),
                    "summary": COLLECT.summarize(runs, self.references(), []),
                    "complete": False, "all_terminal": True, "needs_attention": True}
        report = COLLECT.render_report(evidence)
        self.assertIn("11/12", report)
        self.assertIn("Baseline", report)
        self.assertIn("样本标准差", report)
        self.assertIn("跨布局", report)
        self.assertIn("单次", report)
        self.assertEqual(sum("job2026" in line for line in report.splitlines()), 12)


_OLD_SPEC = importlib.util.spec_from_file_location(
    "collect_hiformer_old_fixture", ROOT / "tests/test_collect_hiformer_target_only_extension.py")
_OLD = importlib.util.module_from_spec(_OLD_SPEC)
_OLD_SPEC.loader.exec_module(_OLD)


class CollectHiFormerMultiseedEligibilityTest(unittest.TestCase):
    write = _OLD.CollectHiFormerTargetOnlyExtensionTest.write
    write_json = _OLD.CollectHiFormerTargetOnlyExtensionTest.write_json

    def setUp(self):
        _OLD.CollectHiFormerTargetOnlyExtensionTest.setUp(self)
        old_spec = dict(self.spec)
        self.spec = CollectHiFormerTargetOnlyMultiseedTest().submission()["tasks"][5]
        self.spec = COLLECT.normalize_submission({"tasks": [
            self.spec if index == 5 else row for index, row in enumerate(
                CollectHiFormerTargetOnlyMultiseedTest().submission()["tasks"])]})[5]
        self.spec["array_id"] = "901"
        self.params.update(seed=20262029, dataloader_seed=2028, sisa_parameter_seed=20260823)
        self.reference.update(raw_params=dict(self.params), params=dict(self.params), seed=20262029,
                              historical_default_additions={}, path_relocation_map={})
        old_manifest = self.root / old_spec["manifest"]
        manifest = json.loads(old_manifest.read_text())
        manifest.update(study="rise_hiformer_target_only_multiseed", task_id=5, run_id=self.spec["run_id"],
                        reference_params=self.params, normalized_reference_params=self.params,
                        effective_params=dict(self.params, model_id=self.spec["run_id"]),
                        historical_default_additions={}, path_relocation_map={},
                        seed_bundle=COLLECT.seed_bundle(self.spec), world_size=2, batch_size=16384, global_batch=32768)
        source_hashes = {}
        for relative in manifest["source_sha256"]:
            relative = relative.replace("run_hiformer_target_only_extension", "run_hiformer_target_only_multiseed").replace(
                "submit_hiformer_target_only_extension", "submit_hiformer_target_only_multiseed")
            self.write(relative, "fixture execution source\n")
            source_hashes[relative] = hashlib.sha256((self.root / relative).read_bytes()).hexdigest()
            self.write(Path(self.spec["artifact_dir"]) / "source" / relative, "fixture execution source\n")
        manifest["source_sha256"] = source_hashes
        self.write_json(self.spec["manifest"], manifest)
        artifact = Path(self.spec["artifact_dir"])
        self.write_json(artifact / "config/model_config.yaml", {"Base": {}, self.spec["run_id"]: manifest["effective_params"]})
        self.write_json(artifact / "config/dataset_config.yaml", {self.dataset: {}})
        self.write_json(artifact / "started.json", {"job_id": "902", "array_job_id": "901", "array_task_id": "5", "gpu_names": ["GPU fixture"] * 2, "world_size": 2, "seed_bundle": COLLECT.seed_bundle(self.spec)})
        self.write_json(artifact / "outcome.json", {"job_id": "902", "returncode": 0})
        self.audits = [COLLECT.expected_model_audit(self.spec, self.reference, rank) for rank in range(2)]
        for audit in self.audits:
            self.write_json(artifact / f"model_audit_rank{audit['rank']}.json", audit)
        self.log_text = self.log_text.replace(old_spec["run_id"], self.spec["run_id"]).replace("20262028", "20262029").replace("20260822", "20260823").replace('"dataloader_seed": 2027', '"dataloader_seed": 2028')
        self.slurm_text = self.slurm_text.replace(old_spec["run_id"], self.spec["run_id"]).replace("20262028", "20262029").replace("20260822", "20260823").replace('"dataloader_seed": 2027', '"dataloader_seed": 2028').replace('"task_id": 0', '"task_id": 5')
        lines = [self.log_text]
        for rank in range(2):
            worker = COLLECT.expected_worker(self.spec, rank)
            self.write_json(artifact / f"worker_rank{rank}.json", worker)
            rng = {"rank": rank, "world_size": 2, "run_id": self.spec["run_id"],
                   "seed_bundle": COLLECT.seed_bundle(self.spec), "audit_preserved_rng": True,
                   "before_audit": {key: "a" * 64 for key in ("python", "numpy", "torch_cpu", "torch_cuda_current")},
                   "after_audit": {key: "a" * 64 for key in ("python", "numpy", "torch_cpu", "torch_cuda_current")}}
            self.write_json(artifact / f"rng_audit_rank{rank}.json", rng)
            lines.extend(("RISE_TARGETED_WORKER " + json.dumps(worker),
                          "RISE_TARGETED_MODEL_AUDIT " + json.dumps(self.audits[rank]),
                          "RISE_TARGETED_RNG_AUDIT " + json.dumps(rng)))
            lines.extend(f"[Rank {rank}] blocked load balanced: blocks=2, rows=33000, batches=2" for _ in range(3))
        lines.append(f"RISE_TARGETED_COMPLETE run_id={self.spec['run_id']}")
        self.slurm_text = "\n".join(lines) + "\n"
        self.write(self.spec["checkpoint_log"], self.log_text)
        self.write(self.spec["slurm_log"], self.slurm_text)
        shutil.copyfile((self.root / old_spec["checkpoint_log"]).with_suffix(".model"),
                        (self.root / self.spec["checkpoint_log"]).with_suffix(".model"))
        self.accounts = [{"job_task_id": "901_5", "raw_job_id": "902", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:30:00", "node": "fixture"}]

    def audit(self):
        sources = COLLECT.BASE.Sources(self.root)
        sources.read(self.reference["checkpoint_log"])
        return COLLECT.audit_attempt(self.spec, sources, self.accounts, self.reference)

    def test_valid_completed_29_run_exposes_full_label_metrics(self):
        run = self.audit()
        self.assertTrue(run["eligible"], run["issues"] + run["pending_checks"])
        self.assertEqual(set(run["metrics"]), set(COLLECT.BASE.LABELS[self.dataset]))

    def test_running_even_with_test_is_hidden(self):
        self.accounts[0]["state"] = "RUNNING"
        run = self.audit()
        self.assertFalse(run["eligible"])
        self.assertEqual(run["metrics"], {})
        self.assertIsNone(run["macro_auc"])

    def test_dual_log_coverage_config_and_saved_sources_are_enforced(self):
        self.write(self.spec["slurm_log"], self.slurm_text.replace("AUC: 0.81", "AUC: 0.82", 1))
        self.assertTrue(any("Test metrics differ" in item for item in self.audit()["issues"]))
        self.write(self.spec["slurm_log"], self.slurm_text.replace("rows=33000", "rows=33001", 1))
        self.assertTrue(any("Coverage mismatch" in item for item in self.audit()["issues"]))
        self.write(self.spec["slurm_log"], self.slurm_text)
        self.write(Path(self.spec["artifact_dir"]) / "source/model_zoo/HiFormer.py", "changed snapshot")
        self.assertTrue(any("Frozen source snapshot mismatch" in item for item in self.audit()["issues"]))

    def test_four_rank_seed27_reconstruction_is_audited(self):
        previous = dict(self.spec)
        submission = CollectHiFormerTargetOnlyMultiseedTest().submission()
        self.spec = COLLECT.normalize_submission(submission)[1]
        self.params.update(seed=20262027, world_size=4, batch_size=8192,
                           dataloader_seed=2026, sisa_parameter_seed=20260821)
        raw = {key: value for key, value in self.params.items() if key not in {"dataloader_seed", "sisa_parameter_seed"}}
        self.reference.update(params=self.params, raw_params=raw, seed=20262027,
                              historical_default_additions={"dataloader_seed": 2026, "sisa_parameter_seed": 20260821},
                              coverage=[{"split": split, "rank": rank, "blocks": 2, "assigned_rows": 33000,
                                         "batches": 2, "processed_rows": 16384}
                                        for rank in range(4) for split in ("train", "valid", "test")])
        artifact = Path(self.spec["artifact_dir"])
        shutil.copytree(self.root / previous["artifact_dir"], self.root / artifact)
        manifest = json.loads((self.root / self.spec["manifest"]).read_text())
        manifest.update(task_id=1, run_id=self.spec["run_id"], world_size=4, batch_size=8192,
                        reference_params=raw, normalized_reference_params=self.params,
                        effective_params=dict(self.params, model_id=self.spec["run_id"]),
                        historical_default_additions=self.reference["historical_default_additions"],
                        allowed_parameter_differences=["dataloader_seed", "model_id", "sisa_parameter_seed"],
                        seed_bundle=COLLECT.seed_bundle(self.spec))
        self.write_json(self.spec["manifest"], manifest)
        self.write_json(artifact / "config/model_config.yaml", {"Base": {}, self.spec["run_id"]: manifest["effective_params"]})
        self.write_json(artifact / "started.json", {"job_id": "904", "array_job_id": "900", "array_task_id": "1", "gpu_names": ["GPU fixture"] * 4,
                                                    "world_size": 4, "seed_bundle": COLLECT.seed_bundle(self.spec)})
        self.write_json(artifact / "outcome.json", {"job_id": "904", "returncode": 0})
        self.log_text = ("INFO Params: " + json.dumps(manifest["effective_params"]) + "\n"
                         + "\n".join(self.log_text.splitlines()[1:]).replace(previous["run_id"], self.spec["run_id"]) + "\n")
        lines = [self.log_text]
        for rank in range(4):
            audit = COLLECT.expected_model_audit(self.spec, self.reference, rank)
            worker = COLLECT.expected_worker(self.spec, rank)
            rng = {"rank": rank, "world_size": 4, "run_id": self.spec["run_id"], "seed_bundle": COLLECT.seed_bundle(self.spec),
                   "audit_preserved_rng": True,
                   "before_audit": {name: "a" * 64 for name in ("python", "numpy", "torch_cpu", "torch_cuda_current")},
                   "after_audit": {name: "a" * 64 for name in ("python", "numpy", "torch_cpu", "torch_cuda_current")}}
            for filename, value in ((f"model_audit_rank{rank}.json", audit), (f"worker_rank{rank}.json", worker),
                                    (f"rng_audit_rank{rank}.json", rng)):
                self.write_json(artifact / filename, value)
            lines.extend(("RISE_TARGETED_WORKER " + json.dumps(worker), "RISE_TARGETED_MODEL_AUDIT " + json.dumps(audit),
                          "RISE_TARGETED_RNG_AUDIT " + json.dumps(rng)))
            lines.extend(f"[Rank {rank}] blocked load balanced: blocks=2, rows=33000, batches=2" for _ in range(3))
        lines.append(f"RISE_TARGETED_COMPLETE run_id={self.spec['run_id']}")
        self.write(self.spec["checkpoint_log"], self.log_text)
        self.write(self.spec["slurm_log"], "\n".join(lines) + "\n")
        shutil.copyfile((self.root / previous["checkpoint_log"]).with_suffix(".model"),
                        (self.root / self.spec["checkpoint_log"]).with_suffix(".model"))
        self.accounts = [{"job_task_id": "900_1", "raw_job_id": "904", "state": "COMPLETED", "exit_code": "0:0", "elapsed": "00:30:00", "node": "fixture"}]
        run = self.audit()
        self.assertTrue(run["eligible"], run["issues"] + run["pending_checks"])
        (self.root / artifact / "model_audit_rank3.json").unlink()
        self.assertFalse(self.audit()["eligible"])

    def test_missing_or_changed_rng_receipt_is_rejected(self):
        path = Path(self.spec["artifact_dir"]) / "rng_audit_rank1.json"
        rng = json.loads((self.root / path).read_text())
        rng["after_audit"]["torch_cpu"] = "b" * 64
        self.write_json(path, rng)
        run = self.audit()
        self.assertFalse(run["eligible"])
        self.assertTrue(any("changed RNG" in message for message in run["issues"]))
        (self.root / path).unlink()
        self.assertTrue(any("RNG audit artifact unavailable" in message for message in self.audit()["pending_checks"]))

    def test_parameter_seed_and_field_activation_mismatch_are_rejected(self):
        audit = deepcopy(self.audits[0])
        audit["sites"][0]["parameter_seed"] = 20260822
        self.assertTrue(COLLECT.model_audit_issues(self.spec, self.reference, audit, 0))
        audit = deepcopy(self.audits[0])
        audit["sites"][1]["enabled"] = True
        self.assertTrue(COLLECT.model_audit_issues(self.spec, self.reference, audit, 0))


if __name__ == "__main__":
    unittest.main()
