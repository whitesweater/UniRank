"""Evidence acceptance boundaries for the paper's B/C/D component ablations."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_sisa_component_results.py"
SPEC = importlib.util.spec_from_file_location("collect_sisa_component_results", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def metric(label="click", auc="0.8", loss="0.2"):
    return f"INFO [Task: {label}][Metrics] logloss: {loss} - AUC: {auc}\n"


class ComponentEvidenceTest(unittest.TestCase):
    def test_validation_metrics_cannot_be_used_as_test(self):
        with self.assertRaisesRegex(ValueError, "Test evaluation"):
            collector.parse_test_metrics(metric(), ["click"])
        text = metric(auc="0.99") + "******** Test evaluation ********\n" + metric()
        self.assertEqual(collector.parse_test_metrics(text, ["click"])["click"]["AUC"], 0.8)

    def test_duplicate_missing_extra_or_nonfinite_test_metrics_are_rejected(self):
        prefix = "******** Test evaluation ********\n"
        for body in (metric() + metric(), metric("other"), metric(auc="nan"),
                     metric(loss="inf"), metric(auc="1.1"), metric(loss="-0.1")):
            with self.subTest(body=body), self.assertRaises(ValueError):
                collector.parse_test_metrics(prefix + body, ["click"])
        with self.assertRaises(ValueError):
            collector.parse_test_metrics(prefix + metric() + prefix + metric(), ["click"])

    def test_frozen_protocol_rejects_seed_or_variant_change(self):
        reference = {"seed": "20262028", "rank": "0", "model_id": "old"}
        actual = {"seed": "20262028", "rank": "0", "model_id": "new",
                  "sisa_variant": "decay_only"}
        self.assertEqual(collector.protocol_issues(actual, reference, "new", "decay_only"), [])
        for key, value in (("seed", "20262027"), ("sisa_variant", "rotation_only"),
                           ("rank", "1"), ("unknown_field", "True")):
            altered = dict(actual, **{key: value})
            self.assertTrue(collector.protocol_issues(altered, reference, "new", "decay_only"))

    def test_params_must_have_one_complete_rank_zero_session(self):
        text = 'INFO Params: ' + json.dumps({"rank": "0", "seed": "20262028"})
        self.assertEqual(collector.parse_params(text)["rank"], "0")
        with self.assertRaises(ValueError):
            collector.parse_params(text + "\n" + text)
        with self.assertRaises(ValueError):
            collector.parse_params('INFO Params: {"rank": "1"}')

    def test_running_job_never_qualifies_even_if_metrics_exist(self):
        marker = "SISA_COMPONENT_COMPLETE run_id=run"
        self.assertEqual(collector.completion_issues(
            {"state": "COMPLETED", "exit_code": "0:0"}, {"returncode": 0}, marker, "run"), [])
        cases = [
            ({"state": "RUNNING", "exit_code": "0:0"}, {"returncode": 0}, marker),
            ({"state": "COMPLETED", "exit_code": "1:0"}, {"returncode": 0}, marker),
            ({"state": "COMPLETED", "exit_code": "0:0"}, {}, marker),
            ({"state": "COMPLETED", "exit_code": "0:0"}, {"returncode": 0}, marker + "wrong"),
        ]
        for account, outcome, text in cases:
            with self.subTest(account=account, outcome=outcome, text=text):
                self.assertTrue(collector.completion_issues(account, outcome, text, "run"))

    def test_coverage_is_paired_by_rank_and_rejects_mismatch(self):
        text = "\n".join(
            f"[Rank {rank}] blocked load balanced: blocks=2, rows={20 + split}, batches=2, est_cost=20"
            for split in range(3) for rank in (1, 0)
        )
        expected = [{"split": split, "rank": rank, "blocks": 2, "assigned_rows": 20 + i,
                     "batches": 2, "processed_rows": 16}
                    for rank in (0, 1) for i, split in enumerate(("train", "valid", "test"))]
        coverage = collector.parse_coverage(text, 8)
        self.assertEqual(collector.coverage_issues(coverage, expected), [])
        coverage[0]["batches"] = 3
        self.assertTrue(collector.coverage_issues(coverage, expected))
        self.assertTrue(collector.coverage_issues(coverage[:-1], expected))

    def test_manifest_rejects_wrong_seed_or_hardware(self):
        manifest = dict(experiment="M_D", setting="B", variant="content_only", task_id=0,
                        run_id="run", seed=20262028, world_size=2, per_gpu_batch=16384,
                        global_batch=32768, dataloader_seed=2027, sisa_parameter_seed=20260822,
                        array_job_id="123", job_id="124", attempt=1,
                        gpu_names=["NVIDIA H100 80GB HBM3"] * 2, git_revision="a" * 40)
        args = ("study", "M", "D", "B", 0, "run", "123", {"raw_job_id": "124"})
        self.assertEqual(collector.manifest_issues(manifest, *args), [])
        for key, value in (("seed", 20262027), ("dataloader_seed", 2026),
                           ("sisa_parameter_seed", 20260821), ("world_size", 4),
                           ("gpu_names", ["L40S"] * 2), ("setting", "C")):
            with self.subTest(key=key):
                self.assertTrue(collector.manifest_issues(dict(manifest, **{key: value}), *args))

    def test_only_four_eligible_settings_form_a_comparison(self):
        runs = [{"study": "study", "model": "M", "dataset": "D", "setting": setting,
                 "eligible": True, "macro_auc": auc, "macro_logloss": loss,
                 "metrics": {"click": {"AUC": auc, "logloss": loss}}}
                for setting, auc, loss in (("B", .7, .4), ("C", .72, .38),
                                           ("D", .71, .39), ("E", .75, .35))]
        comparisons = collector.build_comparisons(runs)
        self.assertAlmostEqual(comparisons[0]["macro"]["E-C-D+B"]["AUC"], .02)
        self.assertAlmostEqual(comparisons[0]["macro"]["E-C-D+B"]["logloss"], -.02)
        runs[2]["eligible"] = False
        self.assertEqual(collector.build_comparisons(runs), [])

    def test_collection_scrubs_metrics_when_completion_or_cross_check_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(relative, text):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)

            def write_csv(relative, rows):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            reference = {"model_id": "reference", "rank": "0", "local_rank": "0",
                         "seed": "20262028", "label_col": "[{'name': 'click'}]"}
            write("experiments/ablations/study/configs/reference_params.json", json.dumps(reference))
            coverage = [{"split": split, "rank": rank, "blocks": 1, "assigned_rows": 16384,
                         "batches": 1, "processed_rows": 16384}
                        for rank in (0, 1) for split in ("train", "valid", "test")]
            write_csv("experiments/ablations/study/configs/data_coverage.csv", coverage)
            archive = dict(job_task_id="548166_7", state="COMPLETED", exit_code="0:0", model="M",
                           dataset="D", seed="20262028", setting="sisa", world_size="2",
                           per_gpu_batch="16384", global_batch="32768", total_parameters="100",
                           dense_parameters="10", checkpoint_log="checkpoints/D/reference.log",
                           slurm_log="logs/unirank-sisa-seed20262028-ws2-548166_7.out")
            write_csv("experiments/sisa_paper_report/archive/results/run_evidence.csv", [archive])
            write_csv("experiments/sisa_paper_report/archive/results/label_results.csv",
                      [dict(model="M", dataset="D", label="click", sisa_20262028_auc="0.8",
                            sisa_20262028_logloss="0.2")])
            coverage_text = "\n".join(
                f"[Rank {rank}] blocked load balanced: blocks=1, rows=16384, batches=1"
                for _ in range(3) for rank in (0, 1))

            def log(params):
                return ("INFO Params: " + json.dumps(params) + "\n"
                        "INFO Total number of parameters: 100.\n"
                        "INFO Number of dense parameters: 10.\n" + coverage_text + "\n"
                        "******** Test evaluation ********\n" + metric())

            write(archive["checkpoint_log"], log(reference))
            write(archive["slurm_log"], log(reference)
                  + "SISA_SINGLE_SEED_HARDWARE gpu_count=2 gpu_name=NVIDIA H100 80GB HBM3 host=test\n"
                  + "SISA_SINGLE_SEED_COMPLETE task_id=7 run_id=reference protocol=ws2_bs16384_acc1\n")
            accounts = {}
            for index, setting in enumerate("BCD"):
                run_id = f"SISA_components_M_D_{setting}_seed20262028_attempt1"
                params = dict(reference, model_id=run_id, sisa_variant=collector.VARIANTS[setting])
                manifest = dict(experiment="M_D", setting=setting, variant=collector.VARIANTS[setting],
                                task_id=index, run_id=run_id, seed=20262028, world_size=2,
                                per_gpu_batch=16384, global_batch=32768, dataloader_seed=2027,
                                sisa_parameter_seed=20260822, array_job_id="123", job_id=str(124 + index),
                                attempt=1, gpu_names=["NVIDIA H100 80GB HBM3"] * 2, git_revision="a" * 40)
                write(f"artifacts/study/{run_id}/manifest.json", json.dumps(manifest))
                write(f"artifacts/study/{run_id}/outcome.json", '{"returncode": 0}')
                write(f"checkpoints/D/{run_id}.log", log(params))
                write(f"logs/unirank-sisa-components-bcd-123_{index}.out", log(params)
                      + "SISA_COMPONENT_PROTOCOL " + json.dumps(manifest) + "\n"
                      + f"SISA_COMPONENT_COMPLETE run_id={run_id}\n")
                accounts[f"123_{index}"] = dict(state="COMPLETED", exit_code="0:0", raw_job_id=str(124 + index))
            with patch.object(collector, "STUDIES", (("study", "M", "D", "123", "7", "components"),)):
                runs, _ = collector.collect(root, accounts)
                self.assertTrue(all(run["eligible"] for run in runs))
                accounts["123_0"]["state"] = "RUNNING"
                runs, _ = collector.collect(root, accounts)
                self.assertFalse(runs[0]["eligible"])
                self.assertEqual(runs[0]["metrics"], {})
                self.assertIsNone(runs[0]["macro_auc"])
                accounts["123_0"]["state"] = "COMPLETED"
                path = root / "checkpoints/D/SISA_components_M_D_B_seed20262028_attempt1.log"
                path.write_text(path.read_text().replace("AUC: 0.8", "AUC: 0.9"))
                runs, _ = collector.collect(root, accounts)
                self.assertFalse(runs[0]["eligible"])
                self.assertEqual(runs[0]["metrics"], {})
                self.assertIn("Final Test metrics differ between checkpoint and Slurm logs", runs[0]["issues"])


if __name__ == "__main__":
    unittest.main()
