from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capture_compiled_control_states import CompiledControlStateCapture
from controls import patched_build
from model_zoo.HyFormer import HyFormerBlock
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig
from unirank.pytorch.layers.attentions.target_attention import MultiHeadTargetAttention


class CompiledControlStatesTest(unittest.TestCase):
    def test_six_site_states_capture_twice_preserves_cpu_compiled_output_and_restores_methods(self):
        torch.manual_seed(929)
        config = SISAAttentionConfig(enabled=True, score_dim=4, decay_reference="sequence_end")
        model = HyFormerBlock(8, 2, 3, 4, 2, sequence_sisa_config=config,
                             query_sisa_config=replace(config, decay_reference="query")).eval()
        compiled = torch.compile(model, backend="eager")
        inputs = (torch.randn(2, 5, 8), torch.randn(2, 2, 8), torch.randn(2, 4, 8),
                  torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32))
        meta = {"part_id": np.array([0, 0]), "row_offset": np.array([10, 20])}
        with torch.no_grad():
            original = compiled(*inputs)
            capture = CompiledControlStateCapture(compiled)
            modules = list(capture.modules.values())
            torch._dynamo.reset()
            try:
                for _ in range(2):
                    capture.reset()
                    output = compiled(*inputs)
                    for a, b in zip(original, output):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(len(capture.collect_states()), 6)
                    self.assertEqual(len(capture.control_entries), 12)
                    rows = capture.observation_rows(meta, "E")
                    self.assertEqual(len(rows), 12)
                    self.assertTrue(all(not row["control_state_recomputed_from_hidden"] for row in rows))
            finally:
                capture.close()
        self.assertTrue(all("_final_channels" not in module.__dict__ for module in modules))

    def test_compiled_local_target_site_is_not_duplicated_and_zero_gauge_is_captured(self):
        torch.manual_seed(930)
        config = SISAAttentionConfig(enabled=True, score_dim=4, decay_reference="query")
        with patched_build("local"):
            model = MultiHeadTargetAttention(input_dim=8, attention_dim=8, sisa_config=config).eval()
        compiled = torch.compile(model, backend="eager")
        inputs = (torch.randn(2, 8), torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool))
        with torch.no_grad():
            original = compiled(*inputs)
            capture = CompiledControlStateCapture(compiled, expected_sites=1)
            torch._dynamo.reset()
            try:
                for _ in range(2):
                    capture.reset()
                    output = compiled(*inputs)
                    torch.testing.assert_close(output, original, rtol=0, atol=0)
                    states = capture.collect_states()
                    self.assertEqual(len(states), 1)
                    site = next(iter(states.values()))
                    self.assertEqual(site["query"][0][3].abs().max().item(), 0)
                    self.assertEqual(site["key"][0][3].abs().max().item(), 0)
            finally:
                capture.close()


if __name__ == "__main__":
    unittest.main()
