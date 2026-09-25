from copy import deepcopy
import gc
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_site_replication as run
from unirank.features import feature_map_from_config
from unirank.pytorch.layers.attentions.sisa import SISAAttentionConfig
import model_zoo


class CrossModelSiteReplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_frozen_protocol_and_expected_counts(self):
        expected = [11164164, 7505412, 11167446, 7568400, 11227152, 7568400]
        for task in range(6):
            model, mode = run.TASKS[task]
            reference = run.reference_params((model, run.DATASET))
            actual = run.params_for(task, 1)
            changes = {k for k in reference if reference[k] != actual[k]}
            self.assertEqual(changes, {"model_id", "sisa_enabled"} if mode == "baseline" else {"model_id"})
            self.assertEqual(run.expected_counts(task)["dense"], expected[task])

    def test_patch_restores_and_rejects_enabled_native_branch(self):
        original = SISAAttentionConfig.build
        with run.placement("baseline"):
            with self.assertRaises(ValueError):
                SISAAttentionConfig(enabled=True).build(8, 2)
        self.assertIs(SISAAttentionConfig.build, original)
        with self.assertRaises(ValueError):
            with run.placement("not_a_site"):
                pass

    def test_real_backbones_preserve_initialization_rng_sites_and_dense_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ["Zenith", "HyFormer"]:
                base = deepcopy(run.reference_params((name, run.DATASET)))
                for feature in base["feature_cols"]:
                    if "vocab_size" in feature:
                        feature["vocab_size"] = 8
                base.update(gpu=-1, distributed=False, world_size=1, rank=0, local_rank=0,
                            enable_torch_compile=False, enable_bf16=False, model_root=directory)
                cls = getattr(model_zoo, name)
                torch.manual_seed(20262028)
                full = cls(feature_map_from_config(base), **base)
                full_state = {key: value.detach().clone() for key, value in full.state_dict().items()}
                full_rng = torch.random.get_rng_state().clone()
                del full
                gc.collect()
                for task, (model, mode) in enumerate(run.TASKS):
                    if model != name:
                        continue
                    params = dict(base, sisa_enabled=mode != "baseline", model_id=f"small_{task}")
                    torch.manual_seed(20262028)
                    with run.placement(mode):
                        candidate = cls(feature_map_from_config(params), **params)
                    audit = run.audit_model(candidate, task)
                    expected_sites = 0 if mode == "baseline" else 1 if mode == "history_only" else 3
                    self.assertEqual(audit["active_sites"], expected_sites)
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), full_rng))
                    for key, value in candidate.state_dict().items():
                        self.assertIn(key, full_state)
                        torch.testing.assert_close(value, full_state[key], rtol=0, atol=0)
                    del candidate
                    gc.collect()
                del full_state

    def test_native_checkpoint_has_no_auxiliary_parameters(self):
        for task in [0, 1]:
            self.assertFalse(run.params_for(task, 1)["sisa_enabled"])
        for task in [2, 3, 4, 5]:
            self.assertTrue(run.params_for(task, 1)["sisa_enabled"])


if __name__ == "__main__":
    unittest.main()
