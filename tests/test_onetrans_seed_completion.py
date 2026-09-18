from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from unirank.utils import load_config
from scripts.run_onetrans_seed_completion import (
    CONFIG_DIR, DATASETS, effective_params, ensure_fresh_outputs,
    expected_params, study_task, training_command, validate_protocol,
)


class OneTransSeedCompletionTest(unittest.TestCase):
    def test_cancelled_taobao_ids_cannot_be_launched_but_keep_audit_identity(self):
        from scripts.run_onetrans_seed_completion import ACTIVE_TASK_IDS, validate_task_active

        self.assertEqual(ACTIVE_TASK_IDS, (0, 1, 2, 3, 4, 5, 8, 9))
        for task_id in range(10):
            task = study_task(task_id)
            if task_id in (6, 7):
                self.assertEqual(task.dataset, "Taobao_Action")
                with self.assertRaises(ValueError):
                    validate_task_active(task)
            else:
                validate_task_active(task)

    def test_exactly_ten_missing_sisa_runs_without_baselines_or_seed27(self):
        tasks = [study_task(i) for i in range(10)]
        self.assertEqual({(t.dataset, t.seed) for t in tasks},
                         {(d, s) for d in DATASETS for s in (20262028, 20262029)})
        self.assertEqual({t.setting for t in tasks}, {"sisa"})
        self.assertEqual({t.model for t in tasks}, {"OneTrans"})
        self.assertEqual(len({t.run_id for t in tasks}), 10)
        for task_id, attempt in [(-1, 1), (10, 1), (0, 0)]:
            with self.assertRaises(ValueError):
                study_task(task_id, attempt)

    def test_command_retains_four_rank_budget_and_full_sisa(self):
        for task_id in range(10):
            task = study_task(task_id)
            command = training_command(task)
            self.assertIn("--nproc_per_node=4", command)
            for key, value in {
                "--seed": str(task.seed), "--batch-size": "8192",
                "--dataloader-seed": str(task.seed - 20260001),
                "--sisa-parameter-seed": str(task.seed - 1206),
                "--gpu": "0,1,2,3", "--sisa-score-dim": "16",
            }.items():
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertIn("--sisa-enabled", command)
            self.assertNotIn("--sisa-variant", command)
            self.assertNotIn("--sparse-optimizer-foreach", command)
            self.assertNotIn("--sparse-adagrad-chunk-size", command)

    def test_runtime_params_match_historical_protocol_except_named_seed_changes(self):
        for task_id in range(10):
            task = study_task(task_id)
            params = load_config(str(CONFIG_DIR), task.experiment)
            validate_protocol(task, params)
            self.assertEqual(effective_params(task, params), expected_params(task))
            if task.dataset == "TencentGR_10M_Action":
                self.assertEqual(params["group_id"], "user_id")
            for key, value in [("max_len", 50), ("epochs", 2), ("batch_size", 16384),
                               ("label_col", []), ("group_id", "invalid")]:
                altered = copy.deepcopy(params)
                altered[key] = value
                with self.subTest(task=task_id, field=key), self.assertRaises(ValueError):
                    validate_protocol(task, altered)

    def test_overwrite_guard_protects_logs_checkpoints_and_manifest(self):
        task = study_task(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ensure_fresh_outputs(task, root)
            for path in [root / "checkpoints" / task.dataset / f"{task.run_id}.log",
                         root / "checkpoints" / task.dataset / f"{task.run_id}.model",
                         root / "artifacts/onetrans_seed_completion" / task.run_id]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                with self.assertRaises(FileExistsError):
                    ensure_fresh_outputs(task, root)
                ensure_fresh_outputs(study_task(0, attempt=2), root)
                path.unlink()


if __name__ == "__main__":
    unittest.main()
