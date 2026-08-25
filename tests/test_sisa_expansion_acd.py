from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_sisa_expansion_baselines import (
    baseline_task,
    compare_baseline,
)
from scripts.collect_sisa_expansion_acd_results import (
    ERROR_PATTERN,
    expansion_task,
    resolve_slurm_log,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ACDExpansionProtocolTest(unittest.TestCase):
    def test_baseline_task_mapping(self):
        self.assertEqual(
            baseline_task(0),
            baseline_task(0).__class__(0, "OneTrans", "TencentGR_10M_Action"),
        )
        self.assertEqual(baseline_task(8).model, "UniMixer")
        self.assertEqual(baseline_task(18).model, "HyFormer")
        self.assertEqual(baseline_task(28).model, "UltraHSTU")
        self.assertEqual(baseline_task(36).dataset, "MerRec_Action")
        with self.assertRaises(ValueError):
            baseline_task(1)

    def test_acd_launcher_requests_four_gpus_and_bounded_concurrency(self):
        source = (
            REPOSITORY_ROOT / "scripts" / "submit_sisa_expansion_acd.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=acd_u", source)
        self.assertIn("#SBATCH --gres=gpu:4", source)
        self.assertIn("#SBATCH --array=0-37%4", source)
        self.assertIn("SLURM_SUBMIT_DIR", source)
        self.assertIn("ACD_EXPANSION_HARDWARE gpu_count=4", source)
        self.assertIn("SISA_ACD_EXPANSION_COMPLETE", source)

    def test_final_results_use_the_canonical_experiment_archive(self):
        collector = (
            REPOSITORY_ROOT / "scripts" / "collect_sisa_expansion_acd_results.py"
        ).read_text(encoding="utf-8")
        audit = (
            REPOSITORY_ROOT / "scripts" / "audit_sisa_expansion_baselines.py"
        ).read_text(encoding="utf-8")
        finalizer = (
            REPOSITORY_ROOT / "scripts" / "finalize_sisa_expansion_acd.sbatch"
        ).read_text(encoding="utf-8")
        canonical_results = "experiments/sisa_expansion_acd/results"

        self.assertIn(canonical_results, collector)
        self.assertIn(canonical_results, audit)
        self.assertIn(canonical_results, finalizer)

    def test_experiment_archive_has_reports_results_and_ablation_template(self):
        required_paths = (
            "experiments/README.md",
            "experiments/sisa_native_strict/report.md",
            "experiments/sisa_expansion_acd/report.md",
            "experiments/sisa_expansion_acd/planning.md",
            "experiments/sisa_expansion_acd/results/runs.csv",
            "experiments/sisa_expansion_acd/results/metrics.csv",
            "experiments/sisa_expansion_acd/results/paired_summary.csv",
            "experiments/sisa_expansion_acd/results/baseline_audit.csv",
            "experiments/sisa_expansion_acd/results/summary.md",
            "experiments/ablations/README.md",
            "experiments/templates/ablation/README.md",
        )
        missing = [
            path for path in required_paths if not (REPOSITORY_ROOT / path).is_file()
        ]
        self.assertEqual(missing, [])

    def test_full_expansion_task_mapping(self):
        self.assertEqual(expansion_task(0).setting, "baseline")
        self.assertEqual(expansion_task(1).setting, "sisa")
        self.assertEqual(expansion_task(8).model, "UniMixer")
        self.assertEqual(expansion_task(18).model, "HyFormer")
        self.assertEqual(expansion_task(28).model, "UltraHSTU")
        self.assertEqual(expansion_task(37).dataset, "MerRec_Action")
        with self.assertRaises(ValueError):
            expansion_task(38)

    def test_retry_log_discovery_accepts_a_different_job_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            log_dir = repository / "logs"
            log_dir.mkdir()
            retry_log = log_dir / "unirank-sisa-ultrahstu-eager-545515_29.out"
            retry_log.write_text("complete\n", encoding="utf-8")

            self.assertEqual(
                resolve_slurm_log(repository, 545515, 29),
                retry_log,
            )

    def test_retry_log_discovery_rejects_ambiguous_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            log_dir = repository / "logs"
            log_dir.mkdir()
            (log_dir / "retry-a-545515_29.out").touch()
            (log_dir / "retry-b-545515_29.out").touch()

            with self.assertRaises(RuntimeError):
                resolve_slurm_log(repository, 545515, 29)

    def test_protocol_heartbeat_setting_is_not_an_nccl_error(self):
        protocol = "nccl_heartbeat_timeout_sec=3600"
        self.assertFalse(ERROR_PATTERN.search(protocol))
        self.assertTrue(ERROR_PATTERN.search("NCCL watchdog timed out"))

    def test_baseline_audit_enforces_auc_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            official = (
                repository
                / "benchmark"
                / "TencentGR_10M_Action"
                / "OneTrans_TencentGR_10M_Action.log"
            )
            local = (
                repository
                / "checkpoints"
                / "TencentGR_10M_Action"
                / "SISA_expansion_OneTrans_TencentGR_10M_Action_baseline_seed20262027.log"
            )
            official.parent.mkdir(parents=True)
            local.parent.mkdir(parents=True)
            official.write_text(
                "******** Test evaluation ********\n"
                "[Task: click][Metrics] logloss: 0.1 - AUC: 0.800000\n",
                encoding="utf-8",
            )
            local.write_text(
                "******** Test evaluation ********\n"
                "[Task: click][Metrics] logloss: 0.1 - AUC: 0.809000\n",
                encoding="utf-8",
            )

            rows, errors = compare_baseline(repository, baseline_task(0), 0.01)
            self.assertFalse(errors)
            self.assertTrue(rows[0]["within_tolerance"])

            local.write_text(
                "******** Test evaluation ********\n"
                "[Task: click][Metrics] logloss: 0.1 - AUC: 0.811000\n",
                encoding="utf-8",
            )
            _, errors = compare_baseline(repository, baseline_task(0), 0.01)
            self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
