from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from observe_mechanism import (
    LABELS, MechanismObserver, control_statistics, deterministic_indices,
    eager_checkpoint_state, load_sample_cache, prediction_gate, save_sample_cache,
    selection_statistics, subset_batch,
)
from benchmark_inference import elapsed_summary, validate_variant_order
from model_zoo.HyFormer import HyFormerBlock
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig, SISAScoreBias


class RISEMechanismTest(unittest.TestCase):
    def test_tv_bound_is_sharp_for_two_keys_and_reversal_is_detected(self):
        a = 1.7
        query = torch.tensor([[[[a]]]])
        key = torch.tensor([[[[1.0], [0.0]]]])
        delta = torch.tensor([[[[-a, a]]]])
        stats = selection_statistics(query, key, delta, scale=1.0)
        self.assertAlmostEqual(stats["omega_mean"].item(), 2 * a, places=6)
        self.assertAlmostEqual(stats["tv_mean"].item(), math.tanh(a / 2), places=6)
        self.assertAlmostEqual(stats["tv_mean"].item(), stats["tv_bound_mean"].item(), places=6)
        self.assertEqual(stats["top_key_changed_fraction"].item(), 1.0)

    def test_constant_correction_preserves_selection_and_mask_cannot_reopen(self):
        query, key = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 5, 4)
        constant = torch.full((2, 2, 3, 5), 4.0)
        stats = selection_statistics(query, key, constant)
        torch.testing.assert_close(stats["omega_mean"], torch.zeros(2))
        self.assertLess(stats["tv_max"].max().item(), 2e-7)
        self.assertEqual(stats["top_key_changed_fraction"].max().item(), 0)
        mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
        mask[..., 0] = True
        constant[..., 1:] = 1e5
        masked = selection_statistics(query, key, constant, mask=mask)
        self.assertEqual(masked["tv_max"].max().item(), 0)
        self.assertEqual(masked["omega_mean"].max().item(), 0)
        self.assertEqual(masked["allowed_pairs"].tolist(), [6, 6])

    def test_native_padding_keys_are_preserved_and_invalid_queries_excluded(self):
        query = torch.zeros(1, 1, 2, 1)
        key = torch.zeros(1, 1, 4, 1)
        stats = selection_statistics(query, key, None,
            query_valid=torch.tensor([[False, True]]),
            key_valid=torch.tensor([[False, False, True, True]]))
        self.assertEqual(stats["valid_attention_rows"].item(), 1)
        self.assertEqual(stats["allowed_pairs"].item(), 4)
        self.assertEqual(stats["native_padding_mass_mean"].item(), 0.5)
        self.assertEqual(stats["corrected_padding_mass_mean"].item(), 0.5)

    def test_all_masked_rows_are_excluded_without_nan_in_maxima(self):
        query, key = torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 3, 2)
        stats = selection_statistics(query, key, None, mask=torch.zeros(1, 1, 1, 3, dtype=torch.bool))
        self.assertEqual(stats["valid_attention_rows"].item(), 0)
        self.assertTrue(torch.isnan(stats["tv_mean"]).all())
        self.assertEqual(stats["tv_max"].item(), 0)

    def test_b_has_identity_am_pa_and_extreme_e_reports_clipping(self):
        torch.manual_seed(52)
        hidden = torch.randn(2, 6, 8)
        valid = torch.ones(2, 6, dtype=torch.bool)
        content = SISAScoreBias(8, 2, 4, variant="content_only")
        stats, _, _ = control_statistics(content, hidden, valid, 4)
        torch.testing.assert_close(stats["am_amplitude_mean_valid_pairs"], torch.ones(2))
        for name in ("frozen_am_delta_effect_abs_mean", "frozen_pa_delta_effect_abs_mean",
                     "phase_increment_abs_mean", "query_clipping_fraction", "key_clipping_fraction"):
            self.assertEqual(stats[name].abs().max().item(), 0)
        full = SISAScoreBias(8, 2, 4, decay_reference="query", exp_clamp=0.1)
        with torch.no_grad():
            full.decay_weight.zero_()
            full.decay_projection_bias.fill_(3.0)
        stats, _, _ = control_statistics(full, hidden, valid, 4)
        self.assertGreater(stats["query_clipping_fraction"].min().item(), 0)
        self.assertGreater(stats["key_clipping_fraction"].min().item(), 0)

    def test_observer_captures_all_sites_without_changing_forward(self):
        torch.manual_seed(18)
        config = SISAAttentionConfig(enabled=True, score_dim=4, decay_reference="sequence_end")
        model = HyFormerBlock(8, 2, 3, 4, 2, sequence_sisa_config=config,
                               query_sisa_config=replace(config, decay_reference="query")).eval()
        inputs = (torch.randn(2, 5, 8), torch.randn(2, 2, 8), torch.randn(2, 4, 8),
                  torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32))
        with torch.no_grad():
            before = model(*inputs)
            observer = MechanismObserver(model)
            observer.meta = {"part_id": np.array([0, 0]), "row_offset": np.array([2, 3]),
                             "effective_length": np.array([3, 5])}
            observer.variant = "E"
            after = model(*inputs)
            self.assertEqual(len(observer.rows), 12)
            self.assertEqual(len({row["site"] for row in observer.rows}), 6)
            self.assertFalse(observer.pending)
            self.assertTrue(all(not row["native_mask_present"] for row in observer.rows))
            self.assertTrue(all(row["tv_bound_violation_max"] < 1e-6 for row in observer.rows))
            observer.close()
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(len(observer.handles), 0)

    def test_checkpoint_mapping_is_exact_and_rejects_collisions_missing_and_dtype(self):
        expected = {"block.weight": torch.zeros(2, 3), "tower.0.bias": torch.zeros(1)}
        source = {"block._orig_mod.weight": torch.ones(2, 3),
                  "tower.0._orig_mod.bias": torch.ones(1)}
        mapped, mapping = eager_checkpoint_state(source, expected)
        self.assertEqual(set(mapped), set(expected))
        self.assertEqual(mapping["block._orig_mod.weight"], "block.weight")
        for bad in (dict(source, **{"block.weight": torch.zeros(2, 3)}),
                    {"block._orig_mod.weight": torch.ones(2, 3)},
                    dict(source, **{"block._orig_mod.weight": torch.ones(2, 3, dtype=torch.float64)})):
            with self.assertRaises(ValueError):
                eager_checkpoint_state(bad, expected)

    def test_sample_selection_and_cache_preserve_row_identity_and_shapes(self):
        meta = {"part_id": np.zeros(8, dtype=np.int32), "row_offset": np.arange(8),
                "user_id": np.arange(8) + 100}
        indices = deterministic_indices(meta, 3, 2026)
        np.testing.assert_array_equal(indices, deterministic_indices(meta, 3, 2026))
        altered = dict(meta, user_id=np.zeros(8))
        np.testing.assert_array_equal(indices, deterministic_indices(altered, 3, 2026))
        batch = ({"user_id": torch.arange(8)}, {"item_id": torch.arange(40).view(8, 5)},
                 torch.ones(8, 4), [torch.ones(8, 4)])
        subset = subset_batch(batch, indices)
        torch.testing.assert_close(subset[1]["item_id"], batch[1]["item_id"][indices])
        samples = [{"batch": subset, "meta": {k: v[indices] for k, v in meta.items()}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.pt"
            save_sample_cache(path, samples)
            restored = load_sample_cache(path)
        np.testing.assert_array_equal(restored[0]["meta"]["row_offset"], indices)
        torch.testing.assert_close(restored[0]["batch"][2], subset[2])

    def test_prediction_gate_refuses_large_differences_and_nonfinite_values(self):
        reference = {label: np.array([0.1, 0.3]) for label in LABELS}
        actual = {label: value.copy() for label, value in reference.items()}
        self.assertTrue(prediction_gate(actual, reference, 1e-4, 1e-3)["passed"])
        actual["Like"][0] += 0.1
        self.assertFalse(prediction_gate(actual, reference, 1e-4, 1e-3)["passed"])
        actual["Like"][0] = np.nan
        with self.assertRaises(ValueError):
            prediction_gate(actual, reference, 1e-4, 1e-3)

    def test_timing_summary_does_not_label_block_quantiles_request_tail_latency(self):
        summary = elapsed_summary([1, 2, 3, 4])
        self.assertEqual(summary["median_block_mean_ms"], 2.5)
        self.assertTrue(summary["tail_latency_not_measured"])
        for values in ([], [0], [float("nan")]):
            with self.assertRaises(ValueError):
                elapsed_summary(values)

    def test_variant_order_requires_exact_available_models_without_fake_native(self):
        self.assertEqual(validate_variant_order(["E", "B"], False), ["E", "B"])
        self.assertEqual(validate_variant_order(["A", "B", "E"], True), ["A", "B", "E"])
        self.assertEqual(validate_variant_order(None, True), ["B", "E", "A"])
        for order, has_native in ((["A", "B", "E"], False), (["B", "B", "E"], False),
                                  (["B", "E"], True), (["E"], False)):
            with self.assertRaises(ValueError):
                validate_variant_order(order, has_native)


if __name__ == "__main__":
    unittest.main()
