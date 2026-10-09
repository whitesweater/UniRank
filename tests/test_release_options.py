import unittest

import torch

from run_expid import apply_sisa_overrides
from unirank.pytorch.layers import ScaledDotProductAttention


class ReleaseOptionsTest(unittest.TestCase):
    def test_default_keeps_native_backbone_and_yaml_can_enable(self):
        params = {"model": "HiFormer"}
        apply_sisa_overrides(params, {})
        self.assertFalse(params["sisa_enabled"])
        params = {"model": "HiFormer", "sisa_enabled": True}
        apply_sisa_overrides(params, {"sisa_enabled": None})
        self.assertTrue(params["sisa_enabled"])

    def test_explicit_disable_and_separate_run_identity(self):
        params = {"model": "RankMixer", "sisa_enabled": True, "model_id": "original"}
        apply_sisa_overrides(params, {"sisa_enabled": False, "run_id": "native_run"})
        self.assertFalse(params["sisa_enabled"])
        self.assertEqual(params["model_id"], "native_run")

    def test_unsupported_backbone_and_invalid_paper_variant_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "supported"):
            apply_sisa_overrides({"model": "UniMixer"}, {"sisa_enabled": True})
        with self.assertRaises(ValueError):
            apply_sisa_overrides({"model": "HiFormer"}, {"sisa_enabled": True, "sisa_variant": "typo"})
        with self.assertRaises(ValueError):
            apply_sisa_overrides({"model": "HiFormer"}, {"sisa_enabled": True, "sisa_score_dim": 3})

    def test_all_masked_keys_remain_zero_with_additive_bias(self):
        attention = ScaledDotProductAttention().eval()
        q, k, v = (torch.randn(2, 2, 3, 4) for _ in range(3))
        mask = torch.zeros(2, 2, 3, 3, dtype=torch.bool)
        output, _ = attention(q, k, v, mask=mask, score_bias=torch.ones(2, 2, 3, 3))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_causal_mask_cannot_be_reopened_by_large_future_bias(self):
        attention = ScaledDotProductAttention().eval()
        q = k = torch.zeros(1, 1, 3, 3)
        v = torch.eye(3)[None, None]
        bias = torch.full((1, 1, 3, 3), 10000.0).triu(1)
        output, _ = attention(q, k, v, is_causal=True, score_bias=bias)
        torch.testing.assert_close(output.triu(1), torch.zeros_like(output), rtol=0, atol=0)
        torch.testing.assert_close(output.sum(-1), torch.ones(1, 1, 3))
