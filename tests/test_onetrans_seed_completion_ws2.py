from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from unirank.utils import load_config
from scripts import run_onetrans_seed_completion as four_gpu
from scripts import run_onetrans_seed_completion_ws2 as two_gpu


class OneTransTwoGpuProtocolTest(unittest.TestCase):
    def test_only_eight_retained_sisa_tasks_and_distinct_outputs(self):
        for task_id in four_gpu.ACTIVE_TASK_IDS:
            task = two_gpu.study_task(task_id)
            original = four_gpu.study_task(task_id)
            self.assertEqual((task.dataset, task.seed, task.setting),
                             (original.dataset, original.seed, "sisa"))
            self.assertIn("_ws2_bs16384_", task.run_id)
            self.assertNotEqual(task.run_id, original.run_id)
        for task_id in (-1, 6, 7, 10):
            with self.assertRaises(ValueError):
                two_gpu.study_task(task_id)

    def test_effective_protocol_changes_only_world_batch_and_output_name(self):
        for task_id in four_gpu.ACTIVE_TASK_IDS:
            task = two_gpu.study_task(task_id)
            params = load_config(str(four_gpu.CONFIG_DIR), task.experiment)
            two_gpu.validate_protocol(task, params)
            old = four_gpu.expected_params(four_gpu.study_task(task_id))
            new = two_gpu.effective_params(task, params)
            self.assertEqual({key for key in old.keys() | new.keys() if old.get(key) != new.get(key)},
                             {"world_size", "batch_size", "model_id"})
            self.assertEqual(new["world_size"] * new["batch_size"], 32768)
            with self.assertRaises(ValueError):
                two_gpu.validate_protocol(task, {**params, "epochs": 2})

    def test_command_uses_two_gpus_without_changing_optimizer_or_seeds(self):
        task = two_gpu.study_task(8)
        command = two_gpu.training_command(task)
        self.assertIn("--nproc_per_node=2", command)
        self.assertEqual(command[command.index("--gpu") + 1], "0,1")
        self.assertEqual(command[command.index("--batch-size") + 1], "16384")
        self.assertEqual(command[command.index("--seed") + 1], "20262028")
        self.assertIn("--sisa-enabled", command)
        self.assertNotIn("--sparse-optimizer-foreach", command)
        self.assertNotIn("--sparse-adagrad-chunk-size", command)

    def test_existing_output_is_never_overwritten(self):
        task = two_gpu.study_task(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            two_gpu.ensure_fresh_outputs(task, root)
            for path in (root / "checkpoints" / task.dataset / f"{task.run_id}.log",
                         root / "checkpoints" / task.dataset / f"{task.run_id}.model",
                         root / "artifacts" / two_gpu.STUDY / task.run_id):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                with self.assertRaises(FileExistsError):
                    two_gpu.ensure_fresh_outputs(task, root)
                two_gpu.ensure_fresh_outputs(two_gpu.study_task(0, attempt=2), root)
                path.unlink()


if __name__ == "__main__":
    unittest.main()
