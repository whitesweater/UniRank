from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from unirank.pytorch.models.rank_model import BaseModel


class CheckpointRetentionTest(unittest.TestCase):
    def test_first_non_best_checkpoint_creates_archive_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            feature_map = SimpleNamespace(dataset_id="ToyDataset")
            model = BaseModel(
                feature_map,
                model_id="fresh_run",
                gpu=-1,
                model_root=str(model_root),
                verbose=0,
                metrics=["AUC"],
                reduce_lr_on_plateau=False,
                early_stop_patience=10,
                save_best_only=True,
            )
            Path(model.model_dir).mkdir(parents=True, exist_ok=True)

            snapshot = {"value": ""}

            def save_snapshot(checkpoint):
                Path(checkpoint).write_text(snapshot["value"], encoding="utf-8")

            model.save_weights = save_snapshot
            model._best_metric = float("-inf")
            model._stopping_steps = 0
            model._stop_training = False

            model._epoch_index = 0
            model._total_steps = 10
            snapshot["value"] = "best"
            model.checkpoint_and_earlystop({"AUC": 0.70})

            model._epoch_index = 1
            model._total_steps = 20
            snapshot["value"] = "non-best"
            model.checkpoint_and_earlystop({"AUC": 0.60})

            archived = list(
                (Path(model.model_dir) / "archive" / model.model_id).rglob("*.model")
            )
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(encoding="utf-8"), "non-best")

    def test_non_best_checkpoints_are_soft_deleted_into_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            feature_map = SimpleNamespace(dataset_id="ToyDataset")
            model = BaseModel(
                feature_map,
                model_id="retention_test",
                gpu=-1,
                model_root=str(model_root),
                verbose=0,
                metrics=["AUC"],
                reduce_lr_on_plateau=False,
                early_stop_patience=10,
                save_best_only=True,
            )
            Path(model.model_dir).mkdir(parents=True, exist_ok=True)
            Path(model.checkpoint).write_text("previous-run-best", encoding="utf-8")

            snapshot = {"value": ""}

            def save_snapshot(checkpoint):
                destination = Path(checkpoint)
                destination.write_text(snapshot["value"], encoding="utf-8")

            model.save_weights = save_snapshot
            model._best_metric = float("-inf")
            model._stopping_steps = 0
            model._stop_training = False

            evaluations = (
                (0, 10, 0.70, "epoch-1"),
                (1, 20, 0.60, "epoch-2"),
                (2, 30, 0.80, "epoch-3"),
            )
            for epoch_index, total_steps, auc, value in evaluations:
                model._epoch_index = epoch_index
                model._total_steps = total_steps
                snapshot["value"] = value
                model.checkpoint_and_earlystop({"AUC": auc})

            self.assertEqual(
                Path(model.checkpoint).read_text(encoding="utf-8"),
                "epoch-3",
            )
            archived = sorted(
                (Path(model.model_dir) / "archive" / model.model_id).rglob("*.model")
            )
            self.assertEqual(len(archived), 3)
            self.assertEqual(
                {path.read_text(encoding="utf-8") for path in archived},
                {"previous-run-best", "epoch-1", "epoch-2"},
            )


if __name__ == "__main__":
    unittest.main()
