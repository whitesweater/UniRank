from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from run_expid import apply_runtime_overrides
from unirank.pytorch.layers import SISAAttentionConfig, SISAScoreBias


class SISAComponentAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(79)
        self.hidden = torch.randn(2, 5, 8, requires_grad=True)
        self.mask = torch.tensor([[False, True, True, True, True], [False] * 5])

    def build(self, variant, reference="query"):
        return SISAScoreBias(8, 2, score_dim=4, variant=variant,
                             decay_reference=reference)

    def test_content_only_equals_masked_low_rank_dot_product(self):
        module = self.build("content_only")
        b = F.linear(self.hidden, module.b_weight).view(2, 5, 2, 4).transpose(1, 2)
        c = F.linear(self.hidden, module.c_weight).view(2, 5, 2, 4).transpose(1, 2)
        expected = c @ b.transpose(-1, -2)
        expected *= module.lambdas(dtype=expected.dtype, device=expected.device)[None, :, None, None]
        valid = self.mask[:, None, :, None] & self.mask[:, None, None, :]
        expected = expected.masked_fill(~valid, 0)
        torch.testing.assert_close(module(self.hidden, self.mask, 4), expected)

    def test_single_components_equal_mathematically_disabled_full_branch(self):
        for variant in ("decay_only", "rotation_only", "content_only"):
            for reference in ("query", "sequence_end"):
                with self.subTest(variant=variant, reference=reference):
                    module = self.build(variant, reference)
                    full = self.build("full", reference)
                    with torch.no_grad():
                        if variant != "decay_only":
                            full.decay_weight.zero_()
                            full.decay_projection_bias.fill_(-1000)
                        if variant != "rotation_only":
                            full.phase_weight.zero_()
                    torch.testing.assert_close(
                        module(self.hidden, self.mask, 4), full(self.hidden, self.mask, 4)
                    )

    def test_active_gradients_and_shared_initialization_are_preserved(self):
        full = self.build("full")
        rng_state = torch.random.get_rng_state().clone()
        for variant in ("content_only", "decay_only", "rotation_only"):
            with self.subTest(variant=variant):
                module = self.build(variant)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
                for name, value in module.state_dict().items():
                    self.assertTrue(torch.equal(value, full.state_dict()[name]), name)
                module(self.hidden, self.mask, 4).square().sum().backward()
                for name, parameter in module.named_parameters():
                    active = not (
                        (name.startswith("decay_") and variant != "decay_only")
                        or (name == "phase_weight" and variant != "rotation_only")
                    )
                    self.assertEqual(parameter.requires_grad, active, name)
                    if active:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
                    else:
                        self.assertIsNone(parameter.grad, name)

    def test_config_propagates_variant_and_rejects_typos(self):
        config = SISAAttentionConfig.from_params(
            {"sisa_enabled": True, "sisa_variant": "decay_only"},
            decay_reference="query",
        )
        self.assertEqual(config.for_site(2).build(8, 2).variant, "decay_only")
        with self.assertRaises(ValueError):
            self.build("decay")

    def test_runtime_override_requires_enabled_sisa(self):
        params = {}
        apply_runtime_overrides(params, {"sisa_enabled": True, "sisa_variant": "content_only"})
        self.assertEqual(params["sisa_variant"], "content_only")
        with self.assertRaises(ValueError):
            apply_runtime_overrides({}, {"sisa_variant": "decay_only"})
        with self.assertRaises(ValueError):
            apply_runtime_overrides({}, {"sisa_enabled": True, "sisa_variant": "typo"})


class ComponentStudyProtocolTest(unittest.TestCase):
    def test_third_study_has_three_hiformer_taac_tasks_without_merrec_overrides(self):
        from scripts.run_sisa_component_ablation import (
            runtime_environment, study_task, training_command,
        )

        study = "sisa_components_hiformer_taac_bcd"
        for task_id, variant in enumerate(("content_only", "decay_only", "rotation_only")):
            task = study_task(task_id, study=study)
            self.assertEqual(task.experiment, "HiFormer_TencentGR_10M_Action")
            self.assertEqual(task.study, study)
            self.assertEqual(task.variant, variant)
            command = training_command(task)
            self.assertIn("--nproc_per_node=2", command)
            self.assertEqual(command[command.index("--seed") + 1], "20262028")
            self.assertEqual(command[command.index("--batch-size") + 1], "16384")
            self.assertEqual(command[command.index("--dataloader-seed") + 1], "2027")
            self.assertEqual(command[command.index("--sisa-parameter-seed") + 1], "20260822")
            self.assertNotIn("--sparse-optimizer-foreach", command)
            self.assertNotIn("--sparse-adagrad-chunk-size", command)
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", runtime_environment(task, {}))
            self.assertNotEqual(task.run_id, study_task(task_id).run_id)
            self.assertNotEqual(task.run_id, study_task(task_id, study="sisa_components_hyformer_merrec_bcd").run_id)
        with self.assertRaises(ValueError):
            study_task(3, study=study)

    def test_new_study_has_three_distinct_hyformer_merrec_tasks(self):
        from scripts.run_sisa_component_ablation import study_task, training_command

        study = "sisa_components_hyformer_merrec_bcd"
        for task_id, variant in enumerate(("content_only", "decay_only", "rotation_only")):
            task = study_task(task_id, study=study)
            self.assertEqual(task.experiment, "HyFormer_MerRec_Action")
            self.assertEqual(task.study, study)
            self.assertEqual(task.variant, variant)
            self.assertNotEqual(task.run_id, study_task(task_id).run_id)
            command = training_command(task)
            self.assertEqual(command[command.index("--sparse-optimizer-foreach") + 1], "false")
            self.assertEqual(command[command.index("--sparse-adagrad-chunk-size") + 1], "16777216")
            self.assertIn("--nproc_per_node=2", command)
            self.assertEqual(command[command.index("--seed") + 1], "20262028")
        with self.assertRaises(ValueError):
            study_task(3, study=study)
        with self.assertRaises(ValueError):
            study_task(0, study="unknown_study")

    def test_new_study_keeps_merrec_optimizer_and_allocator_protocol(self):
        from scripts.run_sisa_component_ablation import (
            effective_params, runtime_environment, study_task,
        )
        task = study_task(0, study="sisa_components_hyformer_merrec_bcd")
        params = effective_params(task, {})
        self.assertIs(params["sparse_optimizer_foreach"], False)
        self.assertEqual(params["sparse_adagrad_chunk_size"], 16777216)
        self.assertEqual(runtime_environment(task, {})["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", runtime_environment(study_task(0), {}))

    def test_exactly_three_bcd_tasks_and_no_extra_training_controls(self):
        from scripts.run_sisa_component_ablation import study_task, training_command

        tasks = [study_task(i) for i in range(3)]
        self.assertEqual({t.experiment for t in tasks}, {
            "RankMixer_KuaiRand_Video_Action",
        })
        self.assertEqual([t.setting for t in tasks], ["B", "C", "D"])
        for task in tasks:
            command = training_command(task)
            self.assertIn("--nproc_per_node=2", command)
            self.assertEqual(command[command.index("--batch-size") + 1], "16384")
            self.assertEqual(command[command.index("--seed") + 1], "20262028")
            self.assertEqual(command[command.index("--dataloader-seed") + 1], "2027")
            self.assertEqual(command[command.index("--sisa-parameter-seed") + 1], "20260822")
            self.assertEqual(command[command.index("--sisa-variant") + 1], task.variant)
        for invalid in (-1, 3):
            with self.assertRaises(ValueError):
                study_task(invalid)

    def test_protocol_preflight_rejects_changed_training_or_labels(self):
        from scripts.run_sisa_component_ablation import STUDY_EXPERIMENTS, study_task, validate_protocol
        from unirank.utils import load_config

        for study in STUDY_EXPERIMENTS:
            task = study_task(0, study=study)
            params = load_config("config", task.experiment)
            validate_protocol(task, params)
            for changed in ({"batch_size": 16384}, {"label_col": []}, {"shuffle": False}):
                with self.subTest(study=study, changed=changed):
                    with self.assertRaises(ValueError):
                        validate_protocol(task, {**params, **changed})

    def test_effective_historical_diff_contains_only_variant_and_run_name(self):
        import json
        from scripts.run_sisa_component_ablation import (
            STUDY_EXPERIMENTS, effective_params, parse_logged_value, study_directory, study_task,
        )
        from unirank.utils import load_config

        for study in STUDY_EXPERIMENTS:
            directory = study_directory(study_task(0, study=study))
            reference = {
                key: parse_logged_value(value)
                for key, value in json.loads((directory / "configs/reference_params.json").read_text()).items()
            }
            for task_id in range(3):
                task = study_task(task_id, study=study)
                actual = effective_params(task, load_config("config", task.experiment))
                changes = {key for key in reference.keys() | actual.keys()
                           if reference.get(key) != actual.get(key)}
                self.assertEqual(changes, {"model_id", "sisa_variant"})


if __name__ == "__main__":
    unittest.main()
