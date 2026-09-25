"""CPU checks for the independent HiFormer export wrapper."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import export_hiformer_predictions as WRAPPER


class HiFormerPredictionExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous = (WRAPPER.engine.DATASET, WRAPPER.engine.LABELS)
        self.addCleanup(self.restore_engine)

    def restore_engine(self):
        WRAPPER.engine.DATASET, WRAPPER.engine.LABELS = self.previous

    def fixture(self, dataset_id):
        labels = WRAPPER.DATASETS[dataset_id]["labels"]
        root = self.root / dataset_id
        for name in ("data", "user_info", "item_info"):
            (root / name).mkdir(parents=True)
        (root / "meta_data.json").write_text(json.dumps({"action_vocab": {"exposure": 1, "click": 2}}))
        for part, count in [(0, 5), (1, 3)]:
            frame = pd.DataFrame({"user_index": [1] * count, "item_index": [3] * count,
                                  "seq_len": np.arange(count), "user_id": [101 + part] * count})
            for label in labels:
                frame[label] = np.arange(count) % 2
            frame.to_parquet(root / "data" / f"part-{part:05d}.parquet")
            pd.DataFrame({"user_index": [1], "full_item_seq": [[1, 0, 2, 3, 1]],
                          "full_action_seq": [[2, 0, 1, 2, 2]]}).to_parquet(
                              root / "user_info" / f"part-{part:05d}.parquet")
            pd.DataFrame({"item_index": [1, 2, 3], "item_id": [901, 902, 903]}).to_parquet(
                root / "item_info" / f"part-{part:05d}.parquet")
        return {"data_root": str(self.root), "dataset_id": dataset_id,
                "test_data": str(root / "data"), "test_user_info": str(root / "user_info"),
                "test_item_info": str(root / "item_info"), "batch_size": 2, "max_len": 3,
                "dataloader_seed": 2027, "block_cache_size": 2, "embedding_dim": 4,
                "group_id": "user_id", "feature_cols": [
                    {"name": name, "type": "meta"} for name in ("user_index", "item_index", "seq_len")
                ] + [{"name": "user_id", "type": "categorical", "vocab_size": 1000},
                     {"name": "item_id", "type": "categorical", "source": "item", "vocab_size": 1000},
                     {"name": "action", "type": "categorical", "source": "action", "vocab_size": 4}],
                "label_col": [{"name": label} for label in labels]}

    def test_both_reference_logs_have_exact_dataset_labels_and_models(self):
        for dataset_id, spec in WRAPPER.DATASETS.items():
            overrides = WRAPPER.configure_engine(dataset_id)
            row = WRAPPER.reference_record(dataset_id)
            params, metrics = WRAPPER.engine.parse_log(WRAPPER.ROOT / row["checkpoint_log"])
            self.assertEqual(params["model"], "HiFormer")
            self.assertEqual(set(metrics), set(spec["labels"]))
            self.assertEqual(overrides["DATASET"]["effective"], dataset_id)
            self.assertEqual(tuple(overrides["LABELS"]["effective"]), spec["labels"])

    def test_collator_label_state_survives_spawn_without_parent_engine_globals(self):
        dataset_id = "TencentGR_10M_Action"
        params = self.fixture(dataset_id)
        WRAPPER.configure_engine(dataset_id)
        _, loader = WRAPPER.build_loader(params, rank=0, world_size=1, num_workers=1)
        batches = list(loader)
        self.assertEqual(len(batches), 3)
        for batch, meta in batches:
            self.assertIn("is_click", meta)
            self.assertIn("is_conversion", meta)
            self.assertNotIn("Like", meta)
            np.testing.assert_array_equal(meta["item_id"], [903, 903])
        # Shut down persistent workers so the temporary fixture can be removed.
        loader._iterator._shutdown_workers()

    def test_unequal_rank_totals_preserve_source_assignment(self):
        params = self.fixture("QK_Video_Action")
        WRAPPER.configure_engine("QK_Video_Action")
        dataset, loader = WRAPPER.build_loader(params, rank=0, world_size=2, num_workers=0)
        allocation = WRAPPER.coverage_by_rank(dataset, 2)
        self.assertEqual([sum(rank.values()) for rank in allocation], [4, 2])
        modified = dict(WRAPPER.DATASETS["QK_Video_Action"], formal_rows=6)
        with patch.dict(WRAPPER.DATASETS, {"QK_Video_Action": modified}):
            self.assertEqual(WRAPPER.validate_coverage("QK_Video_Action", allocation, True), {0: 4, 1: 2})
            with self.assertRaises(ValueError):
                WRAPPER.validate_coverage("QK_Video_Action", [{0: 4}, {0: 2}], True)
        self.assertEqual(sum(len(meta["row_offset"]) for _, meta in loader), 4)

    def test_schema_and_metrics_use_correct_labels_for_each_dataset(self):
        for dataset_id, spec in WRAPPER.DATASETS.items():
            params = self.fixture(dataset_id)
            WRAPPER.configure_engine(dataset_id)
            _, loader = WRAPPER.build_loader(params, rank=0, world_size=1, num_workers=0)
            _, meta = next(iter(loader))
            directory = self.root / f"predictions-{dataset_id}"
            directory.mkdir()
            receipt = WRAPPER.engine.write_shard(directory, meta, {
                label: np.array([0.2, 0.8], dtype=np.float32) for label in spec["labels"]})
            path = directory / receipt["file"]
            table = pq.read_table(path)
            self.assertEqual(table["dataset"].to_pylist(), [dataset_id, dataset_id])
            self.assertEqual(set(name for name in table.column_names if name.endswith("_pred")),
                             {f"{label}_pred" for label in spec["labels"]})
            metrics = WRAPPER.engine.global_metrics([path])
            self.assertEqual(set(metrics), set(spec["labels"]))
            self.assertTrue(all(value["AUC"] == 1 for value in metrics.values()))

    def test_both_sources_frozen_and_baseline_paths_are_dataset_specific(self):
        hashes = WRAPPER.source_hashes()
        self.assertIn("experiments/rise_sample_analysis/scripts/export_hiformer_predictions.py", hashes)
        self.assertIn("experiments/rise_sample_analysis/scripts/export_predictions.py", hashes)
        WRAPPER.configure_engine("QK_Video_Action")
        allowed = WRAPPER.ROOT / "artifacts/rise_sample_analysis/training_checkpoints/QK_Video_Action/native.log"
        self.assertEqual(WRAPPER.engine.validate_extra_log(allowed), allowed)
        with self.assertRaises(ValueError):
            WRAPPER.engine.validate_extra_log(WRAPPER.ROOT / "checkpoints/MerRec_Action/native.log")

    def test_hiformer_restore_is_strict(self):
        class TinyHiFormer(torch.nn.Linear):
            def __init__(self, feature_map, **kwargs):
                super().__init__(1, 1)
                self._torch_compile_enabled = True

        params = self.fixture("QK_Video_Action")
        module = importlib.import_module("model_zoo.HiFormer")
        checkpoint = self.root / "tiny.model"
        torch.save({"weight": torch.zeros(1, 1)}, checkpoint)
        with patch.object(module, "HiFormer", TinyHiFormer):
            with self.assertRaises(RuntimeError):
                WRAPPER.load_model(params, checkpoint, -1)
            torch.save({"weight": torch.ones(1, 1), "bias": torch.zeros(1)}, checkpoint)
            model = WRAPPER.load_model(params, checkpoint, -1)
            self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
