from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_mechanism_precision import CompiledTensorCapture
from model_zoo.HyFormer import HyFormerBlock
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig


class RISECompiledCaptureTest(unittest.TestCase):
    def test_tensor_only_hooks_capture_six_sites_without_changing_cpu_compiled_output(self):
        torch.manual_seed(2323)
        config = SISAAttentionConfig(enabled=True, score_dim=4, decay_reference="sequence_end")
        model = HyFormerBlock(8, 2, 3, 4, 2, sequence_sisa_config=config,
                             query_sisa_config=replace(config, decay_reference="query")).eval()
        compiled = torch.compile(model, backend="eager")
        inputs = (torch.randn(2, 5, 8), torch.randn(2, 2, 8), torch.randn(2, 4, 8),
                  torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32))
        with torch.no_grad():
            before = compiled(*inputs)
            capture = CompiledTensorCapture(compiled)
            torch._dynamo.reset()
            try:
                for _ in range(2):
                    capture.reset()
                    after = compiled(*inputs)
                    self.assertEqual(len(capture.collect()), 6)
                    for left, right in zip(before, after):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    rows = capture.selection_rows({"part_id": np.array([0, 0]),
                        "row_offset": np.array([4, 6])}, "E")
                    self.assertEqual(len(rows), 12)
                    self.assertTrue(all(row["tv_bound_violation_max"] < 1e-6 for row in rows))
            finally:
                capture.close()
        self.assertEqual(capture.entries, [])


if __name__ == "__main__":
    unittest.main()
