"""CPU proof of serial logical-rank batching and eval-only execution."""

from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import export_serial_predictions as SERIAL
import test_prediction_export as FIXTURE


class SerialPredictionExportTest(unittest.TestCase):
    def setUp(self):
        self.fixture = FIXTURE.PredictionExportTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.params = dict(self.fixture.params, test_data=str(self.root / "data"),
                           test_user_info=str(self.root / "user_info"), test_item_info=str(self.root / "item_info"),
                           batch_size=2, max_len=3, block_cache_size=2, dataloader_seed=2027, seed=20262028)

    def assert_inputs_equal(self, left, right):
        for left_part, right_part in zip(left, right):
            if isinstance(left_part, dict):
                self.assertEqual(set(left_part), set(right_part))
                for key in left_part:
                    torch.testing.assert_close(left_part[key], right_part[key], atol=0, rtol=0)
            elif isinstance(left_part, (list, tuple)):
                for a, b in zip(left_part, right_part):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
            else:
                torch.testing.assert_close(left_part, right_part, atol=0, rtol=0)

    def test_serial_union_is_unique_and_identical_to_two_rank_production_batches(self):
        sample_ids = set()
        rank_counts = []
        for rank in (0, 1):
            dataset, serial_loader = SERIAL.loader_for_rank(self.params, rank, 0)
            original = self.fixture.dataset(SERIAL.engine.BlockedParquetBatchDataset, rank, 2)
            collator = self.fixture.collator(original).original
            reference_batches = [collator(payload) for payload in original]
            serial_batches = list(serial_loader)
            self.assertEqual(len(serial_batches), len(reference_batches))
            for (serial_input, meta), reference in zip(serial_batches, reference_batches):
                self.assert_inputs_equal(serial_input, reference)
                for key in zip(meta["part_id"].tolist(), meta["row_offset"].tolist()):
                    self.assertNotIn(key, sample_ids)
                    sample_ids.add(key)
            rank_counts.append(sum(len(meta["row_offset"]) for _, meta in serial_batches))
        self.assertEqual(rank_counts, [4, 2])
        self.assertEqual(sample_ids, {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)})

    def test_serial_variant_loads_once_and_commits_both_logical_ranks(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.distributed = False
                self._ddp_model = None
                self.eval()

            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                return {f"{label}_pred": batch[0][label].float().reshape(-1, 1) * 0.6 + 0.2
                        for label in SERIAL.engine.LABELS}

        dataset, _ = SERIAL.loader_for_rank(self.params, 0, 0)
        allocation = SERIAL.logical_coverage(dataset)
        output = self.root / "serial"
        output.mkdir()
        with patch.object(SERIAL.engine, "load_model", return_value=TinyModel()) as restore:
            invariants = SERIAL.serial_variant(self.params, self.root / "unused.model", output, allocation,
                                               "frozen-data", workers=0, limit_batches=None, resume=False)
        self.assertEqual(restore.call_count, 1)
        self.assertEqual(restore.call_args.args[0]["world_size"], 1)
        self.assertFalse(restore.call_args.args[0]["distributed"])
        self.assertTrue(invariants["all_modules_eval"])
        ranks = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
        self.assertEqual([rank["rows"] for rank in ranks], [4, 2])
        self.assertEqual(sum(rank["rows"] for rank in ranks), 6)
        with patch.object(SERIAL.engine, "load_model", side_effect=AssertionError("Resume should not reload")):
            resumed = SERIAL.serial_variant(self.params, self.root / "unused.model", output, allocation,
                                            "frozen-data", workers=0, limit_batches=None, resume=True)
        self.assertFalse(resumed["model_loaded_in_this_attempt"])

    def test_real_tiny_hyformer_eval_is_rng_and_rank_independent(self):
        from model_zoo.HyFormer import HyFormer
        feature_map = SERIAL.engine.feature_map_from_config(self.params)
        model = HyFormer(feature_map, model_id="cpu_eval_proof", task="binary_classification", gpu=-1,
                         num_tasks=5, num_layers=1, num_heads=2, num_ns_token=2, num_global_token=1,
                         embedding_dim=4, token_dim=8, tower_hidden_units=[8], net_dropout=0.4,
                         model_root=str(self.root), verbose=0, metrics=["AUC", "logloss"],
                         loss=["binary_crossentropy"] * 5, enable_torch_compile=False, enable_bf16=False,
                         sisa_enabled=True, sisa_score_dim=4, sisa_parameter_seed=20260822)
        model.eval()
        SERIAL.assert_eval_invariants(model)
        _, loader = SERIAL.loader_for_rank(self.params, 0, 0)
        batch, _ = next(iter(loader))
        with torch.no_grad(), patch.multiple(
                torch.distributed, all_reduce=unittest.mock.DEFAULT, all_gather=unittest.mock.DEFAULT,
                broadcast=unittest.mock.DEFAULT, barrier=unittest.mock.DEFAULT) as collectives:
            for operation in collectives.values():
                operation.side_effect = AssertionError("Forward unexpectedly communicated")
            torch.manual_seed(17)
            before = torch.get_rng_state().clone()
            first = model.forward(batch)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            model.rank, model.world_size, model.distributed = 1, 2, True
            torch.manual_seed(91)
            second = model.forward(batch)
        for label in SERIAL.engine.LABELS:
            torch.testing.assert_close(first[f"{label}_pred"], second[f"{label}_pred"], atol=0, rtol=0)

    def test_manifest_sources_include_frozen_engine_and_serial_wrapper(self):
        hashes = SERIAL.source_hashes()
        self.assertIn("experiments/rise_sample_analysis/scripts/export_predictions.py", hashes)
        self.assertIn("experiments/rise_sample_analysis/scripts/export_serial_predictions.py", hashes)
        self.assertEqual(SERIAL.merge_coverage([{0: 4}, {1: 2}]), {0: 4, 1: 2})
        with self.assertRaises(ValueError):
            SERIAL.merge_coverage([{0: 4}, {0: 2}])


if __name__ == "__main__":
    unittest.main()
