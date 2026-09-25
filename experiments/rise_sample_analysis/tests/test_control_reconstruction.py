from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_control_reconstruction import reconstruction_gate, reconstruction_statistics
from unirank.pytorch.layers.attentions.sisa import SISAScoreBias


class ControlReconstructionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(823)
        self.query, self.key = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 5, 4)
        self.delta = torch.randn(2, 2, 3, 5)

    def test_identical_deltas_pass_every_position_and_selection(self):
        result = reconstruction_statistics(self.query, self.key, self.delta, self.delta.clone())
        self.assertTrue(result["delta_bitwise_equal_per_example"].all())
        self.assertEqual(result["delta_failed_positions_all"].sum().item(), 0)
        self.assertEqual(result["reconstruction_tv_max"].max().item(), 0)
        table = pd.DataFrame({key: value.cpu().numpy() for key, value in result.items()})
        self.assertTrue(reconstruction_gate(table)["passed"])

    def test_constant_error_can_preserve_selection_but_still_fail_delta_gate(self):
        result = reconstruction_statistics(self.query, self.key, self.delta, self.delta + 0.1)
        self.assertGreater(result["delta_failed_positions_all"].sum().item(), 0)
        self.assertLess(result["reconstruction_tv_max"].max().item(), 2e-7)
        table = pd.DataFrame({key: value.cpu().numpy() for key, value in result.items()})
        gate = reconstruction_gate(table)
        self.assertTrue(gate["all_selection_rows_passed"])
        self.assertFalse(gate["passed"])

    def test_masked_position_error_is_still_counted_in_pointwise_audit(self):
        mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
        mask[..., -1] = False
        rebuilt = self.delta.clone()
        rebuilt[..., -1] += 10
        result = reconstruction_statistics(self.query, self.key, self.delta, rebuilt, mask=mask)
        self.assertEqual(result["delta_failed_positions_used"].sum().item(), 0)
        self.assertEqual(result["delta_failed_positions_all"].sum().item(), 12)
        self.assertEqual(result["reconstruction_tv_max"].max().item(), 0)

    def test_recomputing_same_sisa_on_same_hidden_is_an_exact_cpu_control(self):
        module = SISAScoreBias(8, 2, 4, decay_reference="query")
        hidden, mask = torch.randn(2, 6, 8), torch.ones(2, 6, dtype=torch.bool)
        with torch.no_grad():
            actual = module(hidden, mask, 4, query_slice=slice(-1, None), key_slice=slice(0, -1))
            rebuilt = SISAScoreBias.forward(module, hidden, mask, 4,
                query_slice=slice(-1, None), key_slice=slice(0, -1))
        query, key = torch.randn(2, 2, 1, 4), torch.randn(2, 2, 5, 4)
        result = reconstruction_statistics(query, key, actual, rebuilt)
        self.assertEqual(result["delta_failed_positions_all"].sum().item(), 0)
        self.assertTrue(result["delta_bitwise_equal_per_example"].all())

    def test_nonfinite_reconstruction_is_rejected(self):
        rebuilt = self.delta.clone()
        rebuilt[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            reconstruction_statistics(self.query, self.key, self.delta, rebuilt)


if __name__ == "__main__":
    unittest.main()
