from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from permutation_math import make_permutation, gather_history, combined_mapping, align_bias, joint_raw_batch
from unirank.pytorch.layers.attentions.sisa import SISAScoreBias


class PermutationTest(unittest.TestCase):
    def test_irregular_padding_candidate_and_roundtrip(self):
        mask = np.array([[1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]], bool)
        for mode in ("identity", "off", "reverse", "random"):
            mapping = torch.tensor(make_permutation(mask, mode), dtype=torch.long)
            full = combined_mapping(mapping)
            hidden = torch.arange(3 * 6 * 4).reshape(3, 6, 4)
            permuted = gather_history(hidden, full)
            self.assertTrue(torch.equal(permuted[:, -1], hidden[:, -1]))
            self.assertTrue(torch.equal(gather_history(permuted, full.argsort(1)), hidden))
            self.assertTrue(torch.equal(permuted[:, :-1][~torch.tensor(mask)], hidden[:, :-1][~torch.tensor(mask)]))

    def test_content_only_bias_is_equivariant_after_alignment(self):
        torch.manual_seed(9)
        hidden = torch.randn(4, 7, 12)
        mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 1]], dtype=torch.bool)
        module = SISAScoreBias(12, 2, 4, variant="content_only")
        reference = module(hidden, mask, 6, query_slice=slice(-1, None), key_slice=slice(0, -1))
        for mode in ("reverse", "random"):
            mapping = torch.tensor(make_permutation(mask[:, :-1], mode), dtype=torch.long)
            full = combined_mapping(mapping)
            value = module(gather_history(hidden, full), gather_history(mask, full), 6,
                           query_slice=slice(-1, None), key_slice=slice(0, -1))
            torch.testing.assert_close(align_bias(value, mapping), reference, atol=1e-6, rtol=1e-6)

    def test_joint_raw_record_permutation_preserves_labels_and_attributes(self):
        mask = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0]])
        batch = [{"label": torch.tensor([0, 1]), "user": torch.tensor([5, 6])},
                 {"item": torch.arange(10).reshape(2, 5), "action": torch.arange(10).reshape(2, 5) + 20},
                 mask, [mask.clone(), mask.clone()]]
        mapping = torch.tensor(make_permutation(mask, "reverse"), dtype=torch.long)
        changed = joint_raw_batch(batch, mapping)
        restored = joint_raw_batch(changed, mapping.argsort(1))
        for name in batch[1]:
            self.assertTrue(torch.equal(restored[1][name], batch[1][name]))
        self.assertTrue(torch.equal(changed[0]["label"], batch[0]["label"]))
        tuple_result = joint_raw_batch(tuple(batch), mapping)
        self.assertIsInstance(tuple_result, tuple)
        for name in batch[1]:
            self.assertTrue(torch.equal(tuple_result[1][name], changed[1][name]))

    def test_random_is_reproducible_and_seed_specific(self):
        mask = np.ones((3, 12), bool)
        self.assertTrue(np.array_equal(make_permutation(mask, "random", 3), make_permutation(mask, "random", 3)))
        self.assertFalse(np.array_equal(make_permutation(mask, "random", 3), make_permutation(mask, "random", 4)))


if __name__ == "__main__":
    unittest.main()
