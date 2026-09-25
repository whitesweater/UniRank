"""Gate completed evidence; never promote pending/failed attempts to metrics."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import collect_new_experiments as COLLECT


class CollectNewExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = COLLECT.discover(self.root)[0]
        self.code = self.root / "run_expid.py"
        self.code.write_text("# frozen source\n")
        self.reference_params = {
            "model": "HyFormer", "dataset_id": "MerRec_Action", "seed": 20262028,
            "world_size": 2, "rank": 0, "local_rank": 0, "batch_size": 16384,
            "epochs": 1, "max_len": 100, "dataloader_seed": 2027, "sisa_parameter_seed": 20260822,
            "enable_bf16": True, "enable_torch_compile": True, "sisa_enabled": True,
            "model_id": "historical_E", "model_root": "./checkpoints/"}
        self.params = dict(self.reference_params, sisa_enabled=False, model_id=self.spec["run_id"],
                           model_root="./artifacts/rise_sample_analysis/training_checkpoints/")
        self.coverage_text = "\n".join(
            f"[Rank {rank}] blocked load balanced: blocks=1, rows={rows}, batches={batches}, est_cost=1"
            for rank in (0, 1) for rows, batches in ((40000, 2), (20000, 1), (40000, 2)))
        self.metrics = {label: {"AUC": .7, "logloss": .1} for label in COLLECT.LABELS["MerRec_Action"]}
        self.reference_log = "checkpoints/MerRec_Action/historical_E.log"
        self.write(self.reference_log, self.log(self.reference_params, total=110, dense=10))
        self.reference = {"params": self.reference_params, "checkpoint_log": self.reference_log,
                          "job_task_id": "548166_1", "metrics": self.metrics,
                          "coverage": COLLECT.HELPER.parse_coverage(self.coverage_text, 16384),
                          "parameter_counts": {"total_parameters": 110, "dense_parameters": 10, "embedding_parameters": 100},
                          "eligible": True, "issues": []}
        self.write(self.spec["checkpoint_log"], self.log(self.params))
        self.slurm = "logs/unirank-rise-matched-baseline-643809_0.out"
        self.write(self.slurm, self.log(self.params, coverage=True))
        self.checkpoint = (self.root / self.spec["checkpoint_log"]).with_suffix(".model")
        with zipfile.ZipFile(self.checkpoint, "w") as archive:
            archive.writestr("synthetic/data.pkl", b"synthetic test dictionary")
        self.manifest = {"run_id": self.spec["run_id"], "attempt": 1, "task_id": 0,
                         "effective_params": self.params, "reference_params": self.reference_params,
                         "reference_log_sha256": self.hash(self.root / self.reference_log),
                         "source_sha256": {"run_expid.py": self.hash(self.code)}}
        self.write(self.spec["manifest"], json.dumps(self.manifest))
        artifact = Path(self.spec["artifact_dir"])
        self.write(artifact / "started.json", json.dumps({"job_id": "643845", "gpus": ["NVIDIA H100 80GB HBM3"] * 2}))
        self.write(artifact / "outcome.json", json.dumps({"returncode": 0, "job_id": "643845"}))
        self.account = {"job_task_id": "643809_0", "raw_job_id": "643845", "state": "COMPLETED",
                        "exit_code": "0:0", "elapsed": "00:01:00", "node": "node1"}

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def hash(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def log(self, params, coverage=False, total=105, dense=5):
        prefix = "INFO Params: " + json.dumps(params) + "\n"
        counts = (f"Total number of parameters: {total}.\nNumber of embedding parameters: 100.\n"
                  f"Number of dense parameters: {dense}.\n")
        metrics = "\n".join(f"[Task: {label}][Metrics] logloss: 0.100000 - AUC: 0.700000"
                            for label in COLLECT.LABELS[params["dataset_id"]])
        return prefix + counts + (self.coverage_text + "\n" if coverage else "") + "******** Test evaluation ********\n" + metrics + (
            f"\nPreserved best model checkpoint: /synthetic/{params['model_id']}.model\n")

    def audit(self, account=None):
        sources = COLLECT.Sources(self.root)
        sources.read(self.reference_log)
        return COLLECT.audit_attempt(self.spec, sources, [account or self.account], self.reference)

    def test_completed_run_requires_all_gates(self):
        result = self.audit()
        self.assertTrue(result["eligible"], (result["issues"], result["pending_checks"]))
        self.assertEqual(result["metrics"], self.metrics)
        self.assertAlmostEqual(result["macro_auc"], .7)

    def test_running_or_failed_runs_never_expose_final_metrics(self):
        for state, exit_code in (("RUNNING", "0:0"), ("FAILED", "1:0")):
            result = self.audit(dict(self.account, state=state, exit_code=exit_code))
            self.assertFalse(result["eligible"])
            self.assertEqual(result["metrics"], {})
            self.assertIsNone(result["macro_auc"])

    def test_actual_raw_allocation_wins_over_duplicate_pending_placeholder(self):
        records = COLLECT.parse_accounting(
            "643809_0|643845|RUNNING|0:0|00:02:00|node1\n"
            "643809_0|643809|PENDING|0:0|00:00:00|None assigned\n"
            "643809_0.batch|643845.batch|RUNNING|0:0|00:02:00|node1\n")
        self.assertEqual(COLLECT.select_account(records, "643809_0")["state"], "RUNNING")
        self.assertEqual(COLLECT.select_account(records, "643809_0", "643845")["raw_job_id"], "643845")
        self.assertEqual(COLLECT.select_account(records, "643809_0", "other")["state"], "UNKNOWN")

    def test_source_drift_rejects_eligibility(self):
        self.code.write_text("# changed after frozen manifest\n")
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertTrue(any("source differs" in issue for issue in result["issues"]))

    def test_final_metrics_must_match_both_logs_and_be_finite(self):
        self.write(self.slurm, self.log(self.params, coverage=True).replace("AUC: 0.700000", "AUC: 0.710000", 1))
        result = self.audit()
        self.assertTrue(any("metrics disagree" in issue for issue in result["issues"]))
        self.write(self.spec["checkpoint_log"], self.log(self.params).replace("AUC: 0.700000", "AUC: nan", 1))
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertEqual(result["metrics"], {})

    def test_coverage_and_frozen_protocol_mismatches_are_not_accepted(self):
        self.write(self.slurm, self.log(self.params, coverage=True).replace("rows=40000, batches=2", "rows=40000, batches=1", 1))
        result = self.audit()
        self.assertTrue(any("Coverage mismatch" in issue for issue in result["issues"]))
        self.write(self.spec["checkpoint_log"], self.log(dict(self.params, batch_size=8192)))
        result = self.audit()
        self.assertTrue(any("batch_size" in issue for issue in result["issues"]))

    def test_missing_checkpoint_and_nonzero_outcome_prevent_metrics(self):
        self.checkpoint.unlink()
        self.write(Path(self.spec["artifact_dir"]) / "outcome.json", json.dumps({"returncode": 1, "job_id": "643845"}))
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertEqual(result["metrics"], {})
        self.assertIn("Successful process outcome unavailable", result["pending_checks"])

    def test_attempt_discovery_retains_old_and_new_attempts(self):
        run = self.spec["run_id"].replace("attempt1", "attempt2")
        (self.root / "artifacts/rise_sample_analysis/matched_baselines" / run).mkdir(parents=True)
        specs = COLLECT.discover(self.root)
        same = [spec for spec in specs if spec["logical_task"] == self.spec["logical_task"]]
        self.assertEqual([spec["attempt"] for spec in same], [1, 2])

    def test_control_requires_exact_two_worker_construction_receipts(self):
        self.spec = COLLECT.discover(self.root)[3]
        self.reference_params = dict(self.reference_params, model="RankMixer", dataset_id="KuaiRand_Video_Action")
        self.params = dict(self.reference_params, model_id=self.spec["run_id"])
        self.reference.update(params=self.reference_params, job_task_id="548166_7")
        manifest = {"run_id": self.spec["run_id"], "attempt": 1, "effective_params": self.params,
                    "control": "local", "reference_job_task": "548166_7", "job_id": "643847",
                    "array_job_id": "643822", "array_task_id": "0", "gpu_names": ["NVIDIA H100 80GB HBM3"] * 2,
                    "source_sha256": {"run_expid.py": self.hash(self.code)}}
        self.write(self.spec["manifest"], json.dumps(manifest))
        self.write(Path(self.spec["artifact_dir"]) / "outcome.json", json.dumps({"returncode": 0}))
        self.write(self.spec["checkpoint_log"], self.log(self.params, total=110, dense=10))
        checkpoint = (self.root / self.spec["checkpoint_log"]).with_suffix(".model")
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("synthetic/data.pkl", b"synthetic")
        markers = "\n".join("RISE_CONTROL_WORKER " + json.dumps({"control": "local", "rank": rank, "score_dim": 16})
                              for rank in (0, 1))
        tail = f"\nRISE_CONTROL_COMPLETE run_id={self.spec['run_id']}\n"
        self.slurm = "logs/rise-rankmixer-controls-643822_0.out"
        self.write(self.slurm, markers + "\n" + self.log(self.params, coverage=True, total=110, dense=10) + tail)
        self.account = dict(self.account, job_task_id="643822_0", raw_job_id="643847")
        complete = self.audit()
        self.assertTrue(complete["eligible"], (complete["issues"], complete["pending_checks"]))
        self.write(self.slurm, self.log(self.params, coverage=True, total=110, dense=10) + tail)
        missing = self.audit()
        self.assertFalse(missing["eligible"])
        self.assertIn("Exact two-rank control-construction receipts unavailable", missing["pending_checks"])


if __name__ == "__main__":
    unittest.main()
