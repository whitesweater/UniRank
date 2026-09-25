"""CPU regression coverage for eligibility and failure isolation in the collector."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/collect_hiformer_target_only_extension.py"
SPEC = importlib.util.spec_from_file_location("collect_hiformer_extension_test", ENTRY)
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


class CollectHiFormerTargetOnlyExtensionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = "MerRec_Action"
        run_id = f"RISE_hiformer_target_only_{self.dataset}_seed20262028_attempt1"
        self.spec = {
            "model": "HiFormer", "dataset": self.dataset, "mode": "target_only",
            "kind": "hiformer_target_only_extension", "task_index": 0, "attempt": 1,
            "run_id": run_id, "job_task_id": "900_0",
            "artifact_dir": f"artifacts/rise_hiformer_target_only_extension/{run_id}",
            "manifest": f"artifacts/rise_hiformer_target_only_extension/{run_id}/frozen_manifest.json",
            "checkpoint_log": f"checkpoints/{self.dataset}/{run_id}.log",
            "slurm_log": "logs/hiformer-target-only-900_0.out",
        }
        self.params = {
            "model": "HiFormer", "dataset_id": self.dataset, "model_id": "full",
            "seed": 20262028, "world_size": 2, "rank": 0, "local_rank": 0,
            "batch_size": 16384, "epochs": 1, "max_len": 100,
            "dataloader_seed": 2027, "sisa_parameter_seed": 20260822,
            "enable_bf16": True, "enable_torch_compile": True,
            "sisa_enabled": True, "num_layers": 3, "num_heads": 2,
            "token_dim": 256, "embedding_dim": 16, "sisa_score_dim": 16,
            "feature_cols": [{"name": [f"item{i}" for i in range(12)],
                              "source": "item", "type": "categorical"}],
        }
        coverage = [{"split": split, "rank": rank, "blocks": 2, "assigned_rows": 33000,
                     "batches": 2, "processed_rows": 32768}
                    for rank in (0, 1) for split in ("train", "valid", "test")]
        self.reference = {
            "params": self.params, "coverage": coverage, "job_task_id": "548166_2",
            "checkpoint_log": "checkpoints/MerRec_Action/full.log",
            "parameter_counts": {"dense_parameters": 1_000_000, "embedding_parameters": 2_000,
                                 "total_parameters": 1_002_000},
            "eligible": True, "issues": [],
        }
        self.write(self.reference["checkpoint_log"], "frozen full RISE reference\n")
        self.expected_counts = {"active_dense": 937012, "active_embedding": 2000,
                                "active_total": 939012, "stored_total": 939012}
        source_names = [
            "scripts/run_hiformer_target_only_extension.py",
            "scripts/submit_hiformer_target_only_extension.sbatch",
            "experiments/rise_sample_analysis/scripts/run_targeted_controls.py",
            "experiments/rise_sample_analysis/scripts/targeted_controls.py",
            "model_zoo/HiFormer.py", "unirank/pytorch/layers/attentions/sisa.py",
            "run_expid.py", "pyproject.toml", "uv.lock",
        ]
        source_hashes = {}
        for name in source_names:
            self.write(name, f"frozen fixture source: {name}\n")
            source_hashes[name] = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
        effective = dict(self.params, model_id=run_id)
        manifest = {
            "study": "rise_hiformer_target_only_extension", "task_id": 0, "attempt": 1,
            "model": "HiFormer", "dataset": self.dataset, "mode": "target_only", "run_id": run_id,
            "reference_job_task": "548166_2", "reference_log": self.reference["checkpoint_log"],
            "reference_params": self.params, "effective_params": effective,
            "reference_log_sha256": hashlib.sha256((self.root / self.reference["checkpoint_log"]).read_bytes()).hexdigest(),
            "allowed_parameter_differences": ["model_id"], "process_local_treatment": "target_only",
            "parameter_expectations": self.expected_counts, "source_sha256": source_hashes,
        }
        self.write_json(self.spec["manifest"], manifest)
        artifact = Path(self.spec["artifact_dir"])
        self.write_json(artifact / "started.json", {"job_id": "901", "array_job_id": "900",
                                                   "array_task_id": "0", "gpu_names": ["NVIDIA H100 80GB HBM3"] * 2})
        self.write_json(artifact / "outcome.json", {"job_id": "901", "returncode": 0})
        self.audits = [COLLECT.expected_model_audit(self.spec, self.reference, rank) for rank in (0, 1)]
        for audit in self.audits:
            self.write_json(artifact / f"model_audit_rank{audit['rank']}.json", audit)
        common = ["INFO Params: " + json.dumps(effective),
                  "Total number of parameters: 939012.", "Number of dense parameters: 937012.",
                  "Number of embedding parameters: 2000.", "******** Test evaluation ********"]
        common.extend(f"[Task: {label}][Metrics] logloss: 0.2 - AUC: 0.81"
                      for label in COLLECT.BASE.LABELS[self.dataset])
        common.append(f"Preserved best model checkpoint: /fixture/{run_id}.model")
        self.log_text = "\n".join(common) + "\n"
        slurm = [self.log_text]
        for rank in (0, 1):
            slurm.append("RISE_TARGETED_WORKER " + json.dumps(
                {"task_id": 0, "mode": "target_only", "rank": rank, "run_id": run_id}))
            slurm.append("RISE_TARGETED_MODEL_AUDIT " + json.dumps(self.audits[rank]))
            slurm.extend(f"[Rank {rank}] blocked load balanced: blocks=2, rows=33000, batches=2"
                         for _ in ("train", "valid", "test"))
        slurm.append(f"RISE_TARGETED_COMPLETE run_id={run_id}")
        self.slurm_text = "\n".join(slurm) + "\n"
        self.write(self.spec["checkpoint_log"], self.log_text)
        self.write(self.spec["slurm_log"], self.slurm_text)
        checkpoint = (self.root / self.spec["checkpoint_log"]).with_suffix(".model")
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("checkpoint/data.pkl", b"test metadata")
        self.accounts = [{"job_task_id": "900_0", "raw_job_id": "901", "state": "COMPLETED",
                          "exit_code": "0:0", "elapsed": "00:30:00", "node": "fixture"}]

    def write(self, path, text):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, path, value):
        self.write(path, json.dumps(value))

    def audit(self):
        sources = COLLECT.BASE.Sources(self.root)
        sources.read(self.reference["checkpoint_log"])
        return COLLECT.audit_attempt(self.spec, sources, self.accounts, self.reference)

    def test_completed_consistent_run_exposes_metrics(self):
        result = self.audit()
        self.assertTrue(result["eligible"], result["issues"] + result["pending_checks"])
        self.assertEqual(set(result["metrics"]), set(COLLECT.BASE.LABELS[self.dataset]))
        self.assertAlmostEqual(result["macro_auc"], .81)
        self.assertEqual(result["pending_checks"], [])

    def test_running_run_cannot_expose_even_complete_test_metrics(self):
        self.accounts[0]["state"] = "RUNNING"
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertEqual(result["metrics"], {})
        self.assertIsNone(result["macro_auc"])

    def test_completed_missing_label_is_rejected(self):
        self.write(self.spec["checkpoint_log"], self.log_text.replace(
            "[Task: Purchase][Metrics] logloss: 0.2 - AUC: 0.81\n", ""))
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertEqual(result["metrics"], {})
        self.assertTrue(any("label coverage" in message for message in result["issues"]))

    def test_coverage_mismatch_is_rejected(self):
        self.write(self.spec["slurm_log"], self.slurm_text.replace("rows=33000", "rows=33001", 1))
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertTrue(any("Coverage mismatch" in message for message in result["issues"]))

    def test_dual_log_metric_disagreement_is_rejected(self):
        self.write(self.spec["slurm_log"], self.slurm_text.replace("AUC: 0.81", "AUC: 0.82", 1))
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertTrue(any("Final Test metrics differ" in message for message in result["issues"]))

    def test_retained_site_seed_or_field_activation_is_rejected(self):
        wrong = deepcopy(self.audits[0])
        wrong["sites"][0]["parameter_seed"] += 1
        self.assertTrue(COLLECT.model_audit_issues(self.spec, self.reference, wrong, 0))
        wrong = deepcopy(self.audits[0])
        wrong["sites"][1]["enabled"] = True
        self.assertTrue(COLLECT.model_audit_issues(self.spec, self.reference, wrong, 0))

    def test_source_drift_and_failed_outcome_are_rejected(self):
        self.write("model_zoo/HiFormer.py", "changed execution source")
        result = self.audit()
        self.assertFalse(result["eligible"])
        self.assertTrue(any("Current source differs" in message for message in result["issues"]))
        self.write_json(Path(self.spec["artifact_dir"]) / "outcome.json", {"job_id": "901", "returncode": 1})
        self.assertTrue(any("process outcome" in message for message in self.audit()["pending_checks"]))

    def test_dataset_widths_do_not_reuse_qk_hidden_size(self):
        for fields, width, target_count in ((12, 192, 7874), (6, 96, 3938), (15, 240, 9842)):
            reference = deepcopy(self.reference)
            reference["params"]["feature_cols"][0]["name"] = [f"item{i}" for i in range(fields)]
            counts = COLLECT.parameter_expectations(reference)
            self.assertEqual(counts["target_hidden_size"], width)
            self.assertEqual(counts["target_site_parameters"], target_count)
            self.assertEqual(counts["removed_field_parameters"], 62988)

    def test_submission_has_exact_approved_scope(self):
        tasks = []
        for index, dataset in enumerate(COLLECT.DATASETS):
            run_id = f"RISE_hiformer_target_only_{dataset}_seed20262028_attempt1"
            tasks.append({"task_index": index, "attempt": 1, "dataset": dataset,
                          "job_task_id": f"900_{index}", "run_id": run_id,
                          "manifest": f"artifacts/rise_hiformer_target_only_extension/{run_id}/frozen_manifest.json",
                          "checkpoint_log": f"checkpoints/{dataset}/{run_id}.log",
                          "slurm_log": f"logs/hiformer-target-only-900_{index}.out"})
        self.assertEqual(len(COLLECT.normalize_submission({"array_id": "900", "tasks": tasks})), 3)
        tasks[0]["dataset"] = "QK_Video_Action"
        with self.assertRaisesRegex(ValueError, "exactly the three"):
            COLLECT.normalize_submission({"array_id": "900", "tasks": tasks})

    def four_dataset_evidence(self, reject_qk=False):
        datasets = ("QK_Video_Action", *COLLECT.DATASETS)
        specs = [dict(self.spec, dataset=dataset, job_task_id=f"900_{index}")
                 for index, dataset in enumerate(COLLECT.DATASETS)]
        hashes = []

        def reference_audit(model, dataset, sources):
            reference = deepcopy(self.reference)
            reference.update(dataset=dataset, job_task_id=COLLECT.REFERENCE_JOBS[dataset],
                             metrics={label: {"AUC": .80, "logloss": .2}
                                      for label in COLLECT.BASE.LABELS[dataset]})
            for key in ("checkpoint_log", "slurm_log"):
                reference[key] = f"references/{dataset}/{key}"
                sources.read(reference[key])
            return reference

        for dataset in datasets:
            for key in ("checkpoint_log", "slurm_log"):
                path = f"references/{dataset}/{key}"
                self.write(path, f"reference {dataset}/{key}")
                hashes.append(f"{path},{hashlib.sha256((self.root / path).read_bytes()).hexdigest()}")
        self.write("experiments/sisa_paper_report/archive/results/source_manifest.csv",
                   "path,sha256\n" + "\n".join(hashes))

        def audit(spec, sources, accounts, reference):
            dataset = spec["dataset"]
            eligible = not (reject_qk and dataset == "QK_Video_Action")
            auc, loss = (.79, .21) if dataset == "TencentGR_10M_Action" else (.81, .19)
            return {**spec, "job_task_id": spec.get("job_task_id", spec.get("expected_job_task")),
                    "state": "COMPLETED", "eligible": eligible,
                    "metrics": {label: {"AUC": auc, "logloss": loss}
                                for label in COLLECT.BASE.LABELS[dataset]} if eligible else {},
                    "macro_auc": auc if eligible else None, "macro_logloss": loss if eligible else None,
                    "issues": ["Rejected QK audit"] if not eligible else [], "pending_checks": []}

        with patch.object(COLLECT, "normalize_submission", return_value=specs), \
                patch.object(COLLECT.BASE, "reference_audit", side_effect=reference_audit), \
                patch.object(COLLECT, "audit_attempt", side_effect=audit), \
                patch.object(COLLECT.TARGETED, "audit_attempt", side_effect=audit) as historical_audit:
            evidence = COLLECT.collect(self.root, {}, [])
        self.assertEqual(historical_audit.call_count, 1)
        historical_spec = historical_audit.call_args.args[0]
        self.assertEqual((historical_spec["mode"], historical_spec["task_index"], historical_spec["attempt"]),
                         ("target_only", 1, 1))
        return evidence

    def test_final_report_includes_four_datasets_and_seventeen_labels(self):
        evidence = self.four_dataset_evidence()
        self.assertTrue(evidence["complete"])
        self.assertEqual([run["dataset"] for run in evidence["runs"]],
                         ["QK_Video_Action", *COLLECT.DATASETS])
        self.assertEqual(len(evidence["historical_references"]), 4)
        self.assertEqual(len(evidence["comparisons"]), 21)
        report = COLLECT.render_report(evidence)
        self.assertIn("4/4", report)
        self.assertIn("3/4 个数据集上提高 Macro-AUC", report)
        self.assertIn("3/4 个数据集上降低 Macro-LogLoss", report)
        self.assertIn("644098_1", report)
        self.assertIn("548166_10", report)
        self.assertIn("−", report)
        self.assertIn("-0.01000000", report)
        self.assertNotIn("待测试", report)
        self.assertNotIn("三数据集结论", report)
        self.assertIn("本轮新增三个", report)

    def test_existing_qk_failure_cannot_form_complete_four_dataset_report(self):
        evidence = self.four_dataset_evidence(reject_qk=True)
        self.assertFalse(evidence["complete"])
        self.assertEqual(sum(run["eligible"] for run in evidence["runs"]), 3)
        self.assertFalse(any(row["dataset"] == "QK_Video_Action" for row in evidence["comparisons"]))
        self.assertIn("3/4", COLLECT.render_report(evidence))

    def test_native_baseline_deltas_use_eligible_target_minus_baseline(self):
        path = COLLECT.STUDY / "results/native_baselines.csv"
        self.write(path, "kind,dataset,baseline_seed,baseline_job,baseline_auc,baseline_logloss\n"
                   "paper_seed27,MerRec_Action,20262027,native27,0.80,0.20\n"
                   "paper_seed27,QK_Video_Action,20262027,qk27,0.80,0.20\n")
        evidence = self.four_dataset_evidence(reject_qk=True)
        rows = evidence["native_baseline_comparisons"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset"], "MerRec_Action")
        self.assertAlmostEqual(rows[0]["delta_auc"], .01)
        self.assertAlmostEqual(rows[0]["delta_logloss"], -.01)
        self.assertIn(str(path), [source["path"] for source in evidence["sources"]])
        self.assertIn("相对主表原生 Baseline", COLLECT.render_report(evidence))

    def test_native_baseline_seed_cohorts_remain_separate(self):
        path = COLLECT.STUDY / "results/native_baselines.csv"
        self.write(path, "kind,dataset,baseline_seed,baseline_job,baseline_auc,baseline_logloss\n"
                   "paper_seed27,TencentGR_10M_Action,20262027,native27,0.77,0.23\n"
                   "matched_seed28,TencentGR_10M_Action,20262028,native28,0.78,0.22\n")
        evidence = self.four_dataset_evidence()
        rows = evidence["native_baseline_comparisons"]
        self.assertEqual([row["baseline_seed"] for row in rows], [20262027, 20262028])
        self.assertEqual([row["baseline_job"] for row in rows], ["native27", "native28"])
        self.assertAlmostEqual(rows[0]["delta_auc"], .02)
        self.assertAlmostEqual(rows[1]["delta_auc"], .01)
        report = COLLECT.render_report(evidence)
        self.assertIn("相对主表原生 Baseline（seed20262027）", report)
        self.assertIn("相对同种子原生 Baseline（seed20262028）", report)
        self.assertIn("world size 和样本覆盖不同", report)
        self.assertIn("完整 RISE AUC", report)
        self.write(path, (self.root / path).read_text().replace(
            "matched_seed28,TencentGR_10M_Action,20262028", "matched_seed28,TencentGR_10M_Action,20262027"))
        with self.assertRaisesRegex(ValueError, "cohort, seed"):
            COLLECT.native_baseline_comparisons(COLLECT.BASE.Sources(self.root), evidence["runs"])


if __name__ == "__main__":
    unittest.main()
