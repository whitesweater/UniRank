"""CPU checks for fixed-checkpoint semantic attention and local sensitivity."""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_rise_semantic_attention as diagnostic


class RiseSemanticAttentionTest(unittest.TestCase):
    def test_local_off_replaces_only_bias_and_retains_tensor_mask_qkv(self):
        args = tuple(torch.randn(2, 1, n, 3) for n in (1, 4, 4))
        mask = torch.tensor([[[[1, 0, 1, 1]]]]).expand(2, -1, -1, -1)
        delta = torch.randn(2, 1, 1, 4, dtype=torch.bfloat16)
        kwargs = {"scale": 3 ** .5, "mask": mask, "score_bias": delta}
        replaced_args, replaced = diagnostic.zero_score_bias(args, kwargs)
        self.assertIs(replaced_args, args)
        self.assertIs(replaced["mask"], mask)
        self.assertIs(kwargs["score_bias"], delta)
        self.assertEqual(replaced["score_bias"].dtype, delta.dtype)
        self.assertEqual(replaced["score_bias"].shape, delta.shape)
        self.assertTrue(torch.equal(replaced["score_bias"], torch.zeros_like(delta)))
        self.assertEqual(replaced["scale"], kwargs["scale"])
        with self.assertRaises(ValueError):
            diagnostic.zero_score_bias(args, dict(kwargs, score_bias=None))

    def test_native_mask_and_empty_rows_remain_explicit(self):
        q, k = torch.zeros(2, 1, 1, 2), torch.zeros(2, 1, 3, 2)
        delta = torch.tensor([[[[0., 1000., 1.]]]]).expand(2, -1, -1, -1)
        mask = torch.tensor([[[[1, 0, 1]]], [[[0, 0, 0]]]])
        raw = diagnostic.reconstruct_attention(q, k, delta, mask, 2 ** .5)
        self.assertEqual(raw["weight_after"][0, 0, 1], 0)
        self.assertTrue(np.isnan(raw["weight_after"][1]).all())
        self.assertTrue(np.isnan(raw["tv"][1]).all())

    def test_selection_matches_frozen_sha_protocol(self):
        chosen = diagnostic.selected_offsets(2, 16384, 64, 20260923)
        np.testing.assert_array_equal(chosen, diagnostic.selected_offsets(2, 16384, 64, 20260923))
        self.assertEqual(len(chosen), 64)
        self.assertTrue(np.all(np.diff(chosen) > 0))

    def test_bce_gain_has_correct_sign_and_validates_probabilities(self):
        y = np.array([1, 0, 1])
        full = diagnostic.binary_cross_entropy(y, np.array([.8, .2, .4]))
        off = diagnostic.binary_cross_entropy(y, np.array([.6, .4, .4]))
        self.assertTrue(np.all((off - full)[:2] > 0))
        self.assertEqual((off - full)[2], 0)
        self.assertTrue(np.isfinite(diagnostic.binary_cross_entropy(y, np.array([0., 1., 1.]))).all())
        for bad in (np.array([.1, -.1, .2]), np.array([.1, np.nan, .2])):
            with self.assertRaises(ValueError):
                diagnostic.binary_cross_entropy(y, bad)

    def test_all_six_labels_required_and_one_ulp_rejected(self):
        actual = {name: np.array([.2, .8], dtype=np.float32) for name in diagnostic.LABELS}
        reference = {name: value.copy() for name, value in actual.items()}
        self.assertTrue(diagnostic.prediction_comparison(actual, reference)["bitwise_equal"])
        actual["long_view"][0] = np.nextafter(actual["long_view"][0], np.float32(1))
        self.assertFalse(diagnostic.prediction_comparison(actual, reference)["bitwise_equal"])
        del actual["long_view"]
        with self.assertRaises(ValueError):
            diagnostic.prediction_comparison(actual, reference)

    def test_route_uses_full_batch_and_full_prediction_hashes(self):
        class BatchSensitive:
            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                self.size = len(batch[0])
                return {f"{label}_pred": (batch[0] + self.size).float()
                        for label in diagnostic.LABELS}

        model = BatchSensitive()
        samples = [{"batch": (torch.arange(4),), "selected_indices": np.array([1, 3])}]
        with tempfile.TemporaryDirectory() as temporary, patch.object(diagnostic, "load_model", return_value=model):
            pred, raw, info = diagnostic.run_route({"seed": 3}, Path("unused"), samples, "plain", Path(temporary))
        self.assertEqual(model.size, 4)
        self.assertEqual(raw, [])
        self.assertEqual(info["full_forward_rows"], 4)
        self.assertEqual(set(info["full_prediction_sha256"]), set(diagnostic.LABELS))
        for value in pred.values():
            np.testing.assert_array_equal(value, [5, 7])

    def test_real_compiled_routes_preserve_capture_and_change_only_selected_site(self):
        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dot_attention = diagnostic.ScaledDotProductAttention()

            def forward(self, q, k, v, mask, delta):
                return self.dot_attention(q, k, v, scale=1., mask=mask, score_bias=delta)[0]

        class CompiledModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_layers = torch.compile(Target(), backend="inductor")
                self.other_site = Target()
                self.eval()

            def _get_amp_context(self):
                return nullcontext()

            def forward(self, batch):
                values = self.attention_layers(*batch).reshape(-1)
                other = self.other_site(*batch).reshape(-1)
                return {f"{label}_pred": (values + other) / 2 for label in diagnostic.LABELS}

        q = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        k = torch.tensor([1., 0.]).view(1, 1, 2, 1).expand(4, -1, -1, -1)
        mask = torch.ones(4, 1, 1, 2)
        delta = torch.tensor([0., 2.]).view(1, 1, 1, 2).expand(4, -1, -1, -1)
        indices = np.array([3, 1])
        sample = {"batch": (q, k, k.clone(), mask, delta), "selected_indices": indices}
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        with tempfile.TemporaryDirectory() as temporary, patch.object(diagnostic, "load_model", side_effect=lambda *_: CompiledModel()):
            plain, _, pinfo = diagnostic.run_route({"seed": 3}, Path("unused"), [sample], "plain", Path(temporary))
            captured, raw, cinfo = diagnostic.run_route({"seed": 3}, Path("unused"), [sample], "capture", Path(temporary))
            off, _, oinfo = diagnostic.run_route({"seed": 3}, Path("unused"), [sample], "local_off", Path(temporary))
        self.assertTrue(diagnostic.prediction_comparison(plain, captured)["bitwise_equal"])
        self.assertEqual(pinfo["full_prediction_sha256"], cinfo["full_prediction_sha256"])
        self.assertEqual(oinfo["intervened_site"], diagnostic.SITE)
        self.assertEqual(cinfo["selected_qkv_signatures_by_batch"], oinfo["selected_qkv_signatures_by_batch"])
        expected_raw = diagnostic.reconstruct_attention(q[indices], k[indices], delta[indices], mask[indices], 1.)
        for key in expected_raw:
            np.testing.assert_array_equal(raw[0][key], expected_raw[key])
        native = torch.softmax(q[:, :, :, 0] * k[:, :, :, 0], -1)[:, :, 0].reshape(-1)
        corrected = torch.softmax(q[:, :, :, 0] * k[:, :, :, 0] + delta[:, :, 0], -1)[:, :, 0].reshape(-1)
        np.testing.assert_allclose(off[diagnostic.LABELS[0]], ((native + corrected) / 2).numpy()[indices], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
