from __future__ import annotations

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import collect_sisa_single_seed_results as collector
from scripts.sisa_single_seed_tasks import EXPERIMENTS, single_seed_task


class SingleSeedResultCollectorTest(unittest.TestCase):
    def test_run_overrides_select_protocol_job_and_attempt(self):
        overrides = collector.parse_run_overrides(
            [
                "0=600001:ws4_bs8192_acc1_oom_fallback:2",
                "7=600002:ws2_bs16384_acc1:3",
            ]
        )

        self.assertEqual(overrides[0].array_job, 600001)
        self.assertEqual(overrides[0].protocol, collector.OOM_FALLBACK_PROTOCOL)
        self.assertEqual(overrides[0].attempt, 2)
        self.assertEqual(overrides[7].protocol, collector.PRIMARY_PROTOCOL)

    def test_invalid_or_duplicate_run_overrides_are_rejected(self):
        invalid_sets = (
            ["bad"],
            ["16=1:ws2_bs16384_acc1:1"],
            ["0=0:ws2_bs16384_acc1:1"],
            ["0=1:unknown:1"],
            ["0=1:ws2_bs16384_acc1:0"],
            [
                "0=1:ws2_bs16384_acc1:1",
                "0=2:ws4_bs8192_acc1_oom_fallback:2",
            ],
        )
        for values in invalid_sets:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    collector.parse_run_overrides(values)

    def test_test_metric_parser_ignores_validation_section(self):
        checkpoint_text = """
[Task: Like][Metrics] AUC: 0.100000 - logloss: 0.900000
******** Test evaluation ********
[Task: Like][Metrics] logloss: 0.200000 - AUC: 0.800000
"""

        self.assertEqual(
            collector.parse_test_metrics(checkpoint_text),
            {"Like": {"logloss": 0.2, "AUC": 0.8}},
        )

    def test_error_classification_prioritizes_oom_and_preserves_traceback(self):
        errors = collector.classify_errors(
            "Traceback (most recent call last):\nCUDA out of memory",
            "",
            "OUT_OF_MEMORY",
        )

        self.assertEqual(errors, ["oom", "traceback"])

    def test_nccl_timeout_environment_variable_is_not_an_error(self):
        self.assertEqual(
            collector.classify_errors(
                "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600",
                "",
                "RUNNING",
            ),
            [],
        )

    def test_scheduler_outcome_reads_sacct_state_and_exit_code(self):
        completed = subprocess.CompletedProcess(
            args=["sacct"],
            returncode=0,
            stdout="COMPLETED|0:0|\n",
            stderr="",
        )
        with patch.object(subprocess, "run", return_value=completed):
            outcome = collector.scheduler_outcome("600100_0")

        self.assertEqual(outcome, ("COMPLETED", "0:0", "sacct"))

    def test_full_matrix_writes_three_outputs_and_incomplete_exits_one(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            self._write_complete_matrix(repository, main_job=600100)
            scheduler = ("COMPLETED", "0:0", "sacct")
            with patch.object(collector, "scheduler_outcome", return_value=scheduler):
                complete_status = collector.main(
                    [
                        "--repository",
                        str(repository),
                        "--main-job",
                        "600100",
                    ]
                )

            self.assertEqual(complete_status, 0)
            output_directory = repository / collector.OUTPUT_DIRECTORY
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"runs.csv", "metrics.csv", "summary.md"},
            )
            with (output_directory / "runs.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                run_rows = list(csv.DictReader(stream))
            with (output_directory / "metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                metric_rows = list(csv.DictReader(stream))
            self.assertEqual(len(run_rows), 16)
            self.assertTrue(all(row["complete"] == "True" for row in run_rows))
            self.assertEqual(len(metric_rows), 16)
            self.assertIn(
                "Complete tasks: **16/16**",
                (output_directory / "summary.md").read_text(encoding="utf-8"),
            )

            task = single_seed_task(0)
            assignment = collector.RunAssignment(
                task_id=0,
                array_job=600100,
                protocol=collector.PRIMARY_PROTOCOL,
                attempt=1,
            )
            checkpoint_log = (
                repository
                / "checkpoints"
                / task.dataset
                / f"{collector.run_id(task, assignment)}.log"
            )
            checkpoint_log.unlink()
            with patch.object(collector, "scheduler_outcome", return_value=scheduler):
                incomplete_status = collector.main(
                    [
                        "--repository",
                        str(repository),
                        "--main-job",
                        "600100",
                    ]
                )
            self.assertEqual(incomplete_status, 1)
            with (output_directory / "runs.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                incomplete_rows = list(csv.DictReader(stream))
            self.assertEqual(incomplete_rows[0]["complete"], "False")
            self.assertIn(
                "checkpoint_log_missing",
                incomplete_rows[0]["incomplete_reasons"],
            )

    def test_custom_seed_uses_isolated_ids_logs_and_result_directory(self):
        seed = 20262029
        dataloader_seed = 2028
        sisa_parameter_seed = 20260823
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            self._write_complete_matrix(
                repository,
                main_job=600200,
                seed=seed,
                dataloader_seed=dataloader_seed,
                sisa_parameter_seed=sisa_parameter_seed,
            )
            scheduler = ("COMPLETED", "0:0", "sacct")
            with patch.object(collector, "scheduler_outcome", return_value=scheduler):
                status = collector.main(
                    [
                        "--repository",
                        str(repository),
                        "--main-job",
                        "600200",
                        "--seed",
                        str(seed),
                        "--dataloader-seed",
                        str(dataloader_seed),
                        "--sisa-parameter-seed",
                        str(sisa_parameter_seed),
                    ]
                )

            self.assertEqual(status, 0)
            output_directory = (
                repository / f"experiments/sisa_single_seed{seed}/results"
            )
            self.assertTrue((output_directory / "summary.md").is_file())
            self.assertFalse((repository / collector.OUTPUT_DIRECTORY).exists())
            with (output_directory / "runs.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["seed"] for row in rows}, {str(seed)})
            self.assertTrue(
                all(f"seed{seed}" in row["checkpoint_log"] for row in rows)
            )

    def test_custom_seed_cannot_target_a_different_seed_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            protected_directory = (
                repository / "experiments/sisa_single_seed20262028/results"
            )
            protected_directory.mkdir(parents=True)
            sentinel = protected_directory / "summary.md"
            sentinel.write_text("protected\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as context:
                collector.main(
                    [
                        "--repository",
                        str(repository),
                        "--main-job",
                        "600300",
                        "--seed",
                        "20262029",
                        "--dataloader-seed",
                        "2028",
                        "--sisa-parameter-seed",
                        "20260823",
                        "--output-directory",
                        "experiments/sisa_single_seed20262028/results",
                    ]
                )

            self.assertEqual(context.exception.code, 2)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "protected\n")

    def _write_complete_matrix(
        self,
        repository: Path,
        main_job: int,
        *,
        seed: int = collector.SEED,
        dataloader_seed: int = collector.DATALOADER_SEED,
        sisa_parameter_seed: int = collector.SISA_PARAMETER_SEED,
    ) -> None:
        self.assertEqual(len(EXPERIMENTS), 16)
        (repository / "logs").mkdir(parents=True)
        for task_id in range(len(EXPERIMENTS)):
            task = single_seed_task(task_id)
            assignment = collector.RunAssignment(
                task_id=task_id,
                array_job=main_job,
                protocol=collector.PRIMARY_PROTOCOL,
                attempt=1,
            )
            expected_run_id = collector.run_id(task, assignment, seed=seed)
            slurm_log = (
                repository
                / "logs"
                / f"unirank-sisa-seed{seed}-ws2-{main_job}_{task_id}.out"
            )
            slurm_log.write_text(
                "\n".join(
                    (
                        "SISA_SINGLE_SEED_HARDWARE gpu_count=2 "
                        "gpu_name=NVIDIA H100 80GB HBM3 host=node01",
                        "SISA_SINGLE_SEED_PROTOCOL "
                        "protocol=ws2_bs16384_acc1 world_size=2 "
                        "per_gpu_batch=16384 global_batch=32768 "
                        f"accumulation_steps=1 epochs=1 seed={seed} "
                        f"dataloader_seed={dataloader_seed} "
                        f"sisa_parameter_seed={sisa_parameter_seed} "
                        "bf16=true",
                        f"task_id={task_id} job_id={main_job}_{task_id} "
                        f"experiment={task.experiment} setting=sisa "
                        f"study=sisa_single_seed{seed} "
                        f"run_id={expected_run_id} attempt=1 telemetry=x",
                        f"SISA_SINGLE_SEED_COMPLETE task_id={task_id} "
                        f"run_id={expected_run_id} protocol=ws2_bs16384_acc1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            checkpoint_directory = repository / "checkpoints" / task.dataset
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            (checkpoint_directory / f"{expected_run_id}.log").write_text(
                "\n".join(
                    (
                        '"batch_size": "16384",',
                        '"accumulation_steps": "1",',
                        '"epochs": "1",',
                        f'"seed": "{seed}",',
                        f'"dataloader_seed": "{dataloader_seed}",',
                        f'"sisa_parameter_seed": "{sisa_parameter_seed}",',
                        '"world_size": "2",',
                        f'"model": "{task.model}",',
                        f'"dataset_id": "{task.dataset}",',
                        f'"model_id": "{expected_run_id}",',
                        '"max_len": "100",',
                        '"enable_bf16": "True",',
                        '"sisa_enabled": "True",',
                        '"sisa_score_dim": "16",',
                        '"sisa_lambda_init": "0.1",',
                        '"sisa_score_scale": "1.0",',
                        "DDP blocked training: True",
                        "******** Test evaluation ********",
                        "[Task: Click][Metrics] logloss: 0.200000 - AUC: 0.800000",
                    )
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    unittest.main()
