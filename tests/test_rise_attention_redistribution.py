"""CPU regression checks for the fixed-checkpoint attention diagnostic."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_rise_attention_redistribution as diagnostic


class RiseAttentionRedistributionTest(unittest.TestCase):
    def setUp(self):
        self.original_engine = (diagnostic.engine.DATASET, diagnostic.engine.LABELS)
        self.addCleanup(self.restore_engine)

    def restore_engine(self):
        diagnostic.engine.DATASET, diagnostic.engine.LABELS = self.original_engine

    def probabilities(self, values=(0.25, 0.75)):
        return {label: np.array(values, dtype=np.float32) for label in diagnostic.LABELS}

    def test_selection_is_sorted_unique_deterministic_and_independent_of_random_state(self):
        selected = diagnostic.selected_offsets(3, 64, 12, 20260923)
        np.random.seed(1)
        np.random.random(200)
        torch.manual_seed(123)
        np.testing.assert_array_equal(selected, diagnostic.selected_offsets(3, 64, 12, 20260923))
        self.assertEqual(len(np.unique(selected)), 12)
        self.assertTrue(np.all(np.diff(selected) > 0))
        self.assertTrue(np.all((selected >= 0) & (selected < 64)))
        self.assertTrue(set(diagnostic.selected_offsets(3, 64, 6, 20260923)) <= set(selected))
        self.assertFalse(np.array_equal(selected, diagnostic.selected_offsets(4, 64, 12, 20260923)))

    def test_selection_rejects_invalid_counts_and_returns_full_window(self):
        for window, count in ((0, 0), (8, 0), (8, -1), (8, 9)):
            with self.subTest(window=window, count=count), self.assertRaises(ValueError):
                diagnostic.selected_offsets(0, window, count, 5)
        np.testing.assert_array_equal(diagnostic.selected_offsets(0, 8, 8, 5), np.arange(8))

    def test_reconstruction_obeys_explicit_and_default_scale(self):
        query = torch.tensor([[[[2.0, 0.0, 0.0, 0.0]]]])
        key = torch.tensor([[[[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
        delta = torch.zeros(1, 1, 1, 2)
        mask = torch.ones_like(delta, dtype=torch.bool)
        expected = torch.softmax(torch.tensor([2.0, 0.0]), 0).numpy()
        for scale in (None, 2.0):
            actual = diagnostic.reconstruct_attention(query, key, delta, mask, scale)
            np.testing.assert_allclose(actual["native_scores"][0, 0], [2.0, 0.0])
            np.testing.assert_allclose(actual["weight_before"][0, 0], expected, atol=1e-7)
            self.assertEqual(actual["tv"].item(), 0.0)

    def test_float_mask_matches_boolean_and_excludes_large_padding_correction(self):
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[1.0, 0.0], [100.0, 0.0], [0.0, 1.0]]]])
        delta = torch.tensor([[[[0.0, 1e6, 1.0]]]])
        mask = torch.tensor([[[[1.0, 0.0, 1.0]]]])
        actual = diagnostic.reconstruct_attention(query, key, delta, mask, 1.0)
        boolean = diagnostic.reconstruct_attention(query, key, delta, mask.bool(), 1.0)
        for name in actual:
            np.testing.assert_array_equal(actual[name], boolean[name])
        self.assertEqual(actual["weight_before"][0, 0, 1], 0.0)
        self.assertEqual(actual["weight_after"][0, 0, 1], 0.0)
        np.testing.assert_allclose(actual["weight_after"][0, 0], [0.5, 0.0, 0.5])
        self.assertEqual(actual["omega"].item(), 1.0)

    def test_empty_history_stays_undefined_and_does_not_poison_nonempty_row(self):
        query, key = torch.zeros(2, 1, 1, 2), torch.zeros(2, 1, 3, 2)
        delta = torch.zeros(2, 1, 1, 3)
        mask = torch.tensor([[[[0, 0, 0]]], [[[0, 1, 1]]]], dtype=torch.float32)
        actual = diagnostic.reconstruct_attention(query, key, delta, mask, 1.0)
        for name in ("weight_before", "weight_after", "tv", "omega"):
            self.assertTrue(np.isnan(actual[name][0]).all())
            self.assertTrue(np.isfinite(actual[name][1]).all())
        np.testing.assert_array_equal(actual["weight_after"][1, 0], [0.0, 0.5, 0.5])

    def test_positive_delta_can_reduce_weight_and_common_shift_preserves_weights(self):
        query, key = torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 2, 1)
        mask = torch.ones(1, 1, 1, 2)
        actual = diagnostic.reconstruct_attention(query, key, torch.tensor([[[[1.0, 2.0]]]]), mask, 1.0)
        self.assertGreater(actual["delta"][0, 0, 0], 0)
        self.assertLess(actual["weight_after"][0, 0, 0], actual["weight_before"][0, 0, 0])
        constant = diagnostic.reconstruct_attention(query, key, torch.full_like(mask, 7.0), mask, 1.0)
        self.assertEqual(constant["tv"].item(), 0.0)
        self.assertEqual(constant["omega"].item(), 0.0)

    def test_delta_uses_actual_query_dtype_cast_before_fp32_reconstruction(self):
        query = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
        key = torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16)
        delta = torch.tensor([[[[1.003, 1.011]]]], dtype=torch.float32)
        mask = torch.ones_like(delta)
        actual = diagnostic.reconstruct_attention(query, key, delta, mask, 1.0)
        rounded = delta.to(torch.bfloat16).float()
        self.assertFalse(torch.equal(delta, rounded))
        np.testing.assert_array_equal(actual["delta"], rounded.squeeze(-2).numpy())
        np.testing.assert_allclose(actual["weight_after"], rounded.softmax(-1).squeeze(-2).numpy())

    def test_reconstruction_rejects_nonbinary_masks_invalid_scale_and_nonfinite_scores(self):
        query, key = torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 2, 1)
        delta, mask = torch.zeros(1, 1, 1, 2), torch.ones(1, 1, 1, 2)
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                diagnostic.reconstruct_attention(query, key, delta, mask, scale)
        with self.assertRaises(ValueError):
            diagnostic.reconstruct_attention(query, key, delta, mask * 0.5, 1.0)
        with self.assertRaises(ValueError):
            diagnostic.reconstruct_attention(query, key, delta + float("nan"), mask, 1.0)

    def test_strict_prediction_gate_rejects_one_ulp_change(self):
        reference, actual = self.probabilities(), self.probabilities()
        self.assertTrue(diagnostic.prediction_comparison(actual, reference)["passed"])
        actual[diagnostic.LABELS[0]][0] = np.nextafter(actual[diagnostic.LABELS[0]][0], np.float32(1))
        strict = diagnostic.prediction_comparison(actual, reference)
        self.assertFalse(strict["passed"])
        self.assertFalse(strict["bitwise_equal"])
        self.assertGreater(strict["labels"][diagnostic.LABELS[0]]["max_abs_error"], 0)
        historical = diagnostic.prediction_comparison(actual, reference, atol=5e-4, rtol=5e-3)
        self.assertTrue(historical["passed"])
        self.assertFalse(historical["bitwise_equal"])

    def test_prediction_gate_rejects_shape_nonfinite_empty_and_invalid_tolerance(self):
        for bad in (np.array([[0.25, 0.75]]), np.array([np.nan, 0.75]),
                    np.array([0.25, np.inf]), np.array([])):
            reference, actual = self.probabilities(), self.probabilities()
            actual[diagnostic.LABELS[0]] = bad
            with self.subTest(values=bad), self.assertRaises(ValueError):
                diagnostic.prediction_comparison(actual, reference)
        reference = self.probabilities()
        reference[diagnostic.LABELS[1]][0] = np.nan
        with self.assertRaises(ValueError):
            diagnostic.prediction_comparison(self.probabilities(), reference)
        for atol, rtol in ((-1, 0), (0, -1), (np.nan, 0), (0, np.inf)):
            with self.assertRaises(ValueError):
                diagnostic.prediction_comparison(self.probabilities(), self.probabilities(), atol=atol, rtol=rtol)

    def test_historical_manifest_shard_and_identity_are_required(self):
        diagnostic.wrapper.configure_engine(diagnostic.DATASET)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "E"
            directory.mkdir()
            meta = {"part_id": np.zeros(3, dtype=np.int32), "row_offset": np.arange(3),
                    "user_id": np.array([11, 12, 13]), "item_id": np.array([21, 22, 23]),
                    "seq_len": np.array([1, 2, 3]), "effective_length": np.array([1, 2, 3]),
                    "is_click": np.array([0, 1, 1], dtype=np.uint8),
                    "is_conversion": np.array([0, 0, 1], dtype=np.uint8)}
            values = self.probabilities((0.1, 0.2, 0.3))
            diagnostic.engine.write_shard(directory, meta, values)
            (directory / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "frozen",
                "model": "HiFormer", "dataset": diagnostic.DATASET, "variant": "E"}))
            complete = {"metric_check_passed": True, "smoke_only": False}
            (directory / "complete.json").write_text(json.dumps(complete))
            indices = np.array([2, 0])
            samples = [{"meta": {key: value[indices] for key, value in meta.items()}}]
            actual = diagnostic.historical_predictions(root, samples, "frozen")
            for label in diagnostic.LABELS:
                np.testing.assert_array_equal(actual[label], values[label][indices])
            with self.assertRaises(ValueError):
                diagnostic.historical_predictions(root, samples, "different")
            samples[0]["meta"]["item_id"][0] += 1
            with self.assertRaises(ValueError):
                diagnostic.historical_predictions(root, samples, "frozen")
            complete["smoke_only"] = True
            (directory / "complete.json").write_text(json.dumps(complete))
            with self.assertRaises(ValueError):
                diagnostic.historical_predictions(root, samples, "frozen")

    def test_sampling_uses_production_collator_without_labels_in_selection(self):
        diagnostic.wrapper.configure_engine(diagnostic.DATASET)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / diagnostic.DATASET
            for name in ("data", "user_info", "item_info"):
                (data_root / name).mkdir(parents=True)
            (data_root / "meta_data.json").write_text(json.dumps({
                "action_vocab": {"click": 1, "conversion": 2, "exposure": 3}}))
            for part in range(8):
                pd.DataFrame({"user_index": [0] * 4, "item_index": [1, 2, 3, 4],
                    "seq_len": [0, 1, 2, 3], "user_id": [part + 10] * 4,
                    "is_click": [0, 1, 0, 1], "is_conversion": [0, 0, 0, 1]}).to_parquet(
                        data_root / "data" / f"part-{part:05d}.parquet")
                pd.DataFrame({"user_index": [0], "full_item_seq": [[1, 2, 3, 4]],
                    "full_action_seq": [[3, 1, 3, 2]]}).to_parquet(
                        data_root / "user_info" / f"part-{part:05d}.parquet")
                pd.DataFrame({"item_index": [1, 2, 3, 4], "item_id": [101, 102, 103, 104]}).to_parquet(
                    data_root / "item_info" / f"part-{part:05d}.parquet")
            params = {"data_root": str(root), "dataset_id": diagnostic.DATASET,
                "test_data": str(data_root / "data"), "test_user_info": str(data_root / "user_info"),
                "test_item_info": str(data_root / "item_info"), "batch_size": 4, "max_len": 3,
                "dataloader_seed": 2027, "block_cache_size": 1, "embedding_dim": 4,
                "group_id": "user_id", "feature_cols": [
                    {"name": name, "type": "meta"} for name in ("user_index", "item_index", "seq_len")
                ] + [{"name": "user_id", "type": "categorical", "vocab_size": 1000},
                     {"name": "item_id", "type": "categorical", "source": "item", "vocab_size": 1000},
                     {"name": "action", "type": "categorical", "source": "action", "vocab_size": 4}],
                "label_col": [{"name": label} for label in diagnostic.LABELS]}
            spec = dict(diagnostic.wrapper.DATASETS[diagnostic.DATASET], formal_rows=32)
            with patch.dict(diagnostic.wrapper.DATASETS, {diagnostic.DATASET: spec}):
                samples, _, files = diagnostic.collect_samples(params, 2, 20260923, root)
                for sample in samples:
                    meta, batch = sample["meta"], sample["batch"]
                    offsets = diagnostic.selected_offsets(int(meta["part_id"][0]), 4, 2, 20260923)
                    self.assertEqual(batch[2].shape[0], params["batch_size"])
                    np.testing.assert_array_equal(sample["selected_indices"], offsets)
                    np.testing.assert_array_equal(meta["row_offset"], offsets)
                    np.testing.assert_array_equal(meta["item_id"], offsets + 101)
                    np.testing.assert_array_equal(meta["effective_length"], offsets)
                    np.testing.assert_array_equal(batch[1]["action"][:, -1].numpy(), np.zeros(4))
                    for row, length in zip(offsets, offsets):
                        expected_history = np.zeros(3, dtype=np.int64)
                        if length:
                            expected_history[-length:] = np.arange(101, 101 + length)
                        np.testing.assert_array_equal(batch[1]["item_id"][row, :-1].numpy(), expected_history)
                for path in sorted((data_root / "data").glob("*.parquet")):
                    frame = pd.read_parquet(path)
                    frame["is_click"] = 1 - frame["is_click"]
                    frame["is_conversion"] = 1 - frame["is_conversion"]
                    frame.to_parquet(path)
                altered, _, _ = diagnostic.collect_samples(params, 2, 20260923, root)
            self.assertEqual(len(files), 24)
            for left, right in zip(samples, altered):
                np.testing.assert_array_equal(left["meta"]["row_offset"], right["meta"]["row_offset"])
                np.testing.assert_array_equal(left["meta"]["item_id"], right["meta"]["item_id"])

    def test_capture_selects_rows_only_after_complete_forward(self):
        model = torch.nn.Module()
        model.attention_layers = torch.nn.Module()
        model.attention_layers.dot_attention = diagnostic.ScaledDotProductAttention()
        query = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        key = torch.tensor([1.0, 0.0]).view(1, 1, 2, 1).expand(4, -1, -1, -1)
        value = key.clone()
        delta = torch.arange(8, dtype=torch.float32).view(4, 1, 1, 2)
        mask = torch.ones(4, 1, 1, 2)
        attention = model.attention_layers.dot_attention
        expected, _ = attention(query, key, value, scale=1.0, mask=mask, score_bias=delta)
        capture = diagnostic.Capture(model)
        self.addCleanup(capture.close)
        with patch.object(diagnostic, "reconstruct_attention", wraps=diagnostic.reconstruct_attention) as reconstruct:
            actual, _ = attention(query, key, value, scale=1.0, mask=mask, score_bias=delta)
            self.assertFalse(reconstruct.called)
            self.assertIs(capture.entries[0][0][0], query)
            self.assertEqual(capture.entries[0][0][0].shape[0], 4)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            indices = np.array([3, 1], dtype=np.int64)
            selected = capture.consume(indices)
            self.assertEqual(reconstruct.call_count, 1)
            self.assertEqual(reconstruct.call_args.args[0].shape[0], 2)
        direct = diagnostic.reconstruct_attention(query[indices], key[indices], delta[indices], mask[indices], 1.0)
        for name in selected:
            np.testing.assert_array_equal(selected[name], direct[name])
        self.assertEqual(len(capture.entries), 0)

    def test_cache_persists_full_batch_and_selected_indices(self):
        sample = {"batch": (torch.arange(4),), "selected_indices": np.array([1, 3], dtype=np.int64),
                  "meta": {"part_id": np.array([0, 0]), "row_offset": np.array([1, 3])}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.pt"
            diagnostic.save_sample_cache(path, [sample])
            restored = torch.load(path, map_location="cpu", weights_only=True)[0]
        self.assertEqual(restored["batch"][0].shape[0], 4)
        torch.testing.assert_close(restored["selected_indices"], torch.tensor([1, 3]))
        torch.testing.assert_close(restored["meta"]["row_offset"], restored["selected_indices"])

    def test_capture_route_recompiles_cached_plain_graph_and_preserves_predictions(self):
        class TargetAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dot_attention = diagnostic.ScaledDotProductAttention()

            def forward(self, query, key, value, mask, delta):
                return self.dot_attention(query, key, value, scale=1.0,
                                          mask=mask, score_bias=delta)[0]

        class CompiledModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_layers = torch.compile(TargetAttention(), backend="eager")
                self.eval()

            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                values = self.attention_layers(*batch).reshape(-1)
                return {f"{label}_pred": values for label in diagnostic.LABELS}

        query = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        key = torch.tensor([1.0, 0.0]).view(1, 1, 2, 1).expand(4, -1, -1, -1)
        mask = torch.ones(4, 1, 1, 2)
        delta = torch.arange(8, dtype=torch.float32).view(4, 1, 1, 2)
        indices = np.array([3, 1], dtype=np.int64)
        samples = [{"batch": (query, key, key.clone(), mask, delta),
                    "selected_indices": indices}]
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic.wrapper, "load_model",
                             side_effect=lambda *_: CompiledModel()):
            plain, _ = diagnostic.run_route({"seed": 20262028}, Path("unused"),
                                             samples, False, Path(temporary))
            captured, attention = diagnostic.run_route({"seed": 20262028}, Path("unused"),
                                                        samples, True, Path(temporary))
        self.assertTrue(diagnostic.prediction_comparison(captured, plain)["bitwise_equal"])
        self.assertEqual(len(attention), 1)
        expected = diagnostic.reconstruct_attention(query[indices], key[indices],
                                                     delta[indices], mask[indices], 1.0)
        for name in expected:
            np.testing.assert_array_equal(attention[0][name], expected[name])

    def test_route_predicts_full_batch_before_selecting_rows(self):
        class BatchSensitiveModel:
            def __init__(self):
                self.observed_sizes = []

            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                size = len(batch[0])
                self.observed_sizes.append(size)
                return {f"{label}_pred": (batch[0] + size).float() for label in diagnostic.LABELS}

        model = BatchSensitiveModel()
        samples = [{"batch": (torch.arange(4),), "selected_indices": np.array([1, 3])}]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(diagnostic.wrapper, "load_model", return_value=model), \
                patch.object(torch.cuda, "empty_cache"):
            predictions, captured = diagnostic.run_route({"seed": 20262028}, Path("unused"),
                                                          samples, False, Path(temporary))
        self.assertEqual(model.observed_sizes, [4])
        self.assertEqual(captured, [])
        for values in predictions.values():
            np.testing.assert_array_equal(values, [5.0, 7.0])


if __name__ == "__main__":
    unittest.main()
