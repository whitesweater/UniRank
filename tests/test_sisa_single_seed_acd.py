from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from run_expid import apply_runtime_overrides
from scripts.sisa_single_seed_tasks import (
    DATASETS,
    EXPERIMENTS,
    MODELS,
    single_seed_task,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_LAUNCHER = (
    REPOSITORY_ROOT / "scripts" / "submit_sisa_single_seed_acd.sbatch"
)
OOM_LAUNCHER = (
    REPOSITORY_ROOT / "scripts" / "submit_sisa_single_seed_oom4_acd.sbatch"
)
TASK_RUNNER = REPOSITORY_ROOT / "scripts" / "run_sisa_single_seed_task.sh"


class RuntimeOverrideTest(unittest.TestCase):
    def test_seed_bundle_and_batch_override_yaml_values(self):
        params = {"seed": 20262027, "batch_size": 8192}
        args = {
            "seed": 20262028,
            "batch_size": 16384,
            "dataloader_seed": 2027,
            "sisa_parameter_seed": 20260822,
        }

        apply_runtime_overrides(params, args, world_size=2)

        self.assertEqual(params["seed"], 20262028)
        self.assertEqual(params["batch_size"], 16384)
        self.assertEqual(params["dataloader_seed"], 2027)
        self.assertEqual(params["sisa_parameter_seed"], 20260822)

    def test_none_overrides_preserve_yaml_values(self):
        params = {"seed": 20262027, "batch_size": 8192}
        args = {
            "seed": None,
            "batch_size": None,
            "dataloader_seed": None,
            "sisa_parameter_seed": None,
        }

        apply_runtime_overrides(params, args, world_size=4)

        self.assertEqual(params, {"seed": 20262027, "batch_size": 8192})

    def test_invalid_runtime_overrides_are_rejected(self):
        invalid_cases = (
            ({"seed": -1}, 2),
            ({"seed": 2 ** 32 - 1}, 2),
            ({"batch_size": 0}, 2),
            ({"dataloader_seed": -1}, 2),
            ({"sisa_parameter_seed": -1}, 2),
        )
        for overrides, world_size in invalid_cases:
            args = {
                "seed": None,
                "batch_size": None,
                "dataloader_seed": None,
                "sisa_parameter_seed": None,
                **overrides,
            }
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    apply_runtime_overrides({}, args, world_size=world_size)


class SingleSeedLauncherTest(unittest.TestCase):
    def test_task_matrix_is_exactly_four_by_four_sisa_cells(self):
        self.assertEqual(len(EXPERIMENTS), 16)
        self.assertEqual(len(set(EXPERIMENTS)), 16)
        self.assertEqual(
            set(EXPERIMENTS),
            {f"{model}_{dataset}" for model in MODELS for dataset in DATASETS},
        )
        self.assertEqual(single_seed_task(0).experiment, "Zenith_MerRec_Action")
        with self.assertRaises(ValueError):
            single_seed_task(16)

    def test_primary_launcher_requests_two_h100_slots_per_task(self):
        source = PRIMARY_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:2", source)
        self.assertIn("#SBATCH --cpus-per-task=16", source)
        self.assertIn("#SBATCH --mem=480G", source)
        self.assertIn("#SBATCH --array=0-15%8", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertIn("SISA_WORLD_SIZE=2", source)
        self.assertIn("SISA_BATCH_SIZE=16384", source)
        self.assertIn('SISA_SEED="${SISA_SEED:-20262028}"', source)
        self.assertIn(
            'SISA_STUDY="${SISA_STUDY:-sisa_single_seed${SISA_SEED}}"',
            source,
        )

    def test_oom_launcher_requires_explicit_manual_task_selection(self):
        source = OOM_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:4", source)
        self.assertIn("#SBATCH --cpus-per-task=32", source)
        self.assertIn("#SBATCH --mem=960G", source)
        self.assertNotIn("#SBATCH --array=", source)
        self.assertIn("SISA_WORLD_SIZE=4", source)
        self.assertIn("SISA_BATCH_SIZE=8192", source)

    def test_runner_preserves_protocol_and_merrec_guards(self):
        source = TASK_RUNNER.read_text(encoding="utf-8")
        required = (
            "--nproc_per_node=\"$SISA_WORLD_SIZE\"",
            "--seed \"$seed\"",
            "--batch-size \"$SISA_BATCH_SIZE\"",
            "--dataloader-seed \"$dataloader_seed\"",
            "--sisa-parameter-seed \"$sisa_parameter_seed\"",
            "--sisa-enabled",
            "--sparse-optimizer-foreach false",
            "--sparse-adagrad-chunk-size 16777216",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "SISA_SINGLE_SEED_COMPLETE",
            'artifacts/$study/telemetry',
            "unsafe study identifier",
            "study/seed mismatch",
        )
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, source)

    def test_new_shell_scripts_have_valid_syntax(self):
        for script in (PRIMARY_LAUNCHER, OOM_LAUNCHER, TASK_RUNNER):
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    ["bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
