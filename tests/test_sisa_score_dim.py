from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from model_zoo.HyFormer import HyFormerBlock
from scripts import run_sisa_score_dim as runner
from scripts import collect_sisa_score_dim_results as collector
from unirank.utils import load_config
from unirank.pytorch.layers import SISAAttentionConfig


class SISAScoreDimensionProtocolTest(unittest.TestCase):
    def test_r64_four_gpu_retry_keeps_global_batch_and_original_execution(self):
        task = runner.study_task(2, attempt=3)
        self.assertFalse(runner.uses_activation_checkpointing(task))
        self.assertEqual(runner.task_world_size(task), 4)
        self.assertEqual(runner.task_batch_size(task), 8192)
        params = load_config(runner.config_directory(task), runner.EXPERIMENT)
        runner.validate_protocol(task, params)
        actual = runner.effective_params(task, params)
        reference = dict(runner.reference_params(), sisa_variant="full")
        self.assertEqual({key for key in actual.keys() | reference.keys()
                          if actual.get(key) != reference.get(key)},
                         {"model_id", "sisa_score_dim", "world_size", "batch_size"})
        self.assertEqual(actual["world_size"] * actual["batch_size"] * actual["accumulation_steps"], 32768)
        command = runner.training_command(task)
        self.assertEqual(command[3], "run_expid.py")
        self.assertIn("--nproc_per_node=4", command)
        self.assertEqual(command[command.index("--gpu") + 1], "0,1,2,3")
        self.assertEqual(command[command.index("--batch-size") + 1], "8192")
        self.assertNotIn("--activation-memory-budget", command)

    def test_r64_checkpointing_preserves_outputs_and_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(20262028)
        direct = HyFormerBlock(
            input_dim=256, num_heads=2, num_layers=3, num_ns_token=4, num_global_token=2,
            sequence_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64,
                parameter_seed=20260822, decay_reference="sequence_end"),
            query_sisa_config=SISAAttentionConfig(enabled=True, score_dim=64,
                parameter_seed=20260822, decay_reference="query"))
        recomputed = copy.deepcopy(direct)
        for use_bf16 in (False, True):
            with self.subTest(bf16=use_bf16):
                direct.zero_grad(set_to_none=True)
                recomputed.zero_grad(set_to_none=True)
                inputs = (torch.randn(2, 6, 256, requires_grad=True),
                          torch.randn(2, 2, 256, requires_grad=True),
                          torch.randn(2, 4, 256, requires_grad=True),
                          torch.tensor([[False, True, True, True, True, True], [True] * 6]))
                cloned = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                               for value in inputs)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
                    output = direct(*inputs)
                    checked = checkpoint(recomputed, *cloned, use_reentrant=False)
                    for actual, expected in zip(checked, output):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    loss = output[1].float().square().mean()
                    checked_loss = checked[1].float().square().mean()
                loss.backward()
                checked_loss.backward()
                for actual, expected in zip(recomputed.parameters(), direct.parameters()):
                    self.assertEqual(actual.grad is None, expected.grad is None)
                    if actual.grad is not None:
                        torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)
                for actual, expected in zip(cloned, inputs):
                    if actual.requires_grad:
                        torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0)

    def test_r64_retry_changes_only_activation_storage_in_addition_to_dimension(self):
        task = runner.study_task(2, attempt=2)
        self.assertTrue(runner.uses_activation_checkpointing(task))
        params = load_config(str(runner.config_directory(task)), runner.EXPERIMENT)
        runner.validate_protocol(task, params)
        expected = dict(runner.reference_params(), sisa_variant="full")
        actual = runner.effective_params(task, params)
        changed = {key for key in expected.keys() | actual.keys()
                   if expected.get(key) != actual.get(key)}
        self.assertEqual(changed, {"model_id", "sisa_score_dim", "gradient_checkpointing"})
        self.assertIs(actual["gradient_checkpointing"], True)
        self.assertEqual(actual["batch_size"], 16384)
        self.assertEqual(actual["world_size"], 2)
        self.assertEqual(actual["accumulation_steps"], 1)
        command = runner.training_command(task)
        self.assertEqual(command[command.index("--config") + 1], str(runner.config_directory(task)))
        with self.assertRaises(ValueError):
            runner.validate_protocol(task, dict(params, batch_size=4096))
        original = runner.study_task(2)
        self.assertFalse(runner.uses_activation_checkpointing(original))
        self.assertTrue(runner.protocol_issues(dict(actual, model_id=original.run_id), original))

    def test_only_eight_thirty_two_and_sixty_four_are_training_tasks(self):
        tasks = [runner.study_task(index) for index in range(3)]
        self.assertEqual([task.score_dim for task in tasks], [8, 32, 64])
        self.assertEqual(len({task.run_id for task in tasks}), 3)
        for task in tasks:
            self.assertEqual(task.experiment, "HyFormer_MerRec_Action")
            command = runner.training_command(task)
            self.assertIn("--nproc_per_node=2", command)
            for flag, value in (("--sisa-score-dim", str(task.score_dim)),
                                ("--sisa-variant", "full"), ("--seed", "20262028"),
                                ("--batch-size", "16384"), ("--dataloader-seed", "2027"),
                                ("--sisa-parameter-seed", "20260822"),
                                ("--sparse-optimizer-foreach", "false"),
                                ("--sparse-adagrad-chunk-size", "16777216")):
                self.assertEqual(command[command.index(flag) + 1], value)
        for invalid in (-1, 3, 4):
            with self.assertRaises(ValueError):
                runner.study_task(invalid)
        with self.assertRaises(ValueError):
            runner.study_task(0, attempt=0)
        self.assertNotEqual(tasks[0].run_id, runner.study_task(0, attempt=2).run_id)

    def test_frozen_protocol_changes_only_dimension_and_output_name(self):
        reference = runner.reference_params()
        params = load_config("config", runner.EXPERIMENT)
        for index in range(3):
            task = runner.study_task(index)
            runner.validate_protocol(task, params)
            actual = runner.effective_params(task, params)
            normalized = dict(reference, sisa_variant="full")
            changed = {key for key in actual.keys() | normalized.keys()
                       if actual.get(key) != normalized.get(key)}
            self.assertEqual(changed, {"model_id", "sisa_score_dim"})
            for overrides in ({"seed": 123}, {"batch_size": 8192}, {"num_heads": 4},
                              {"sisa_variant": "content_only"}, {"label_col": []},
                              {"sisa_score_scale": .5}, {"unexpected": 1}):
                self.assertTrue(runner.protocol_issues(dict(actual, **overrides), task))
            for overrides in ({"batch_size": 16384}, {"max_len": 50}, {"shuffle": False}):
                with self.assertRaises(ValueError):
                    runner.validate_protocol(task, dict(params, **overrides))

    def test_existing_logs_models_or_artifacts_cannot_be_overwritten(self):
        task = runner.study_task(0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.check_outputs(task, root)
            candidates = [root / "checkpoints/MerRec_Action" / (task.run_id + suffix)
                          for suffix in (".log", ".model")]
            candidates.append(root / "artifacts/sisa_score_dim" / task.run_id)
            for path in candidates:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                with self.assertRaises(FileExistsError):
                    runner.check_outputs(task, root)
                path.unlink()


class SISAScoreDimensionEvidenceTest(unittest.TestCase):
    def test_interleaved_progress_utf8_keeps_raw_hash_and_ascii_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b"progress: \xe2\x96\r\n" + self.log.encode()
            (root / "slurm.out").write_bytes(raw)
            sources = collector.RuntimeSources(root)
            text = sources.read(Path("slurm.out"))
            self.assertEqual(sources.records["slurm.out"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertGreater(sources.records["slurm.out"]["utf8_replacement_characters"], 0)
            self.assertEqual(collector.parse_test_metrics(text, self.labels),
                             collector.parse_test_metrics(self.log, self.labels))
            (root / "manifest.json").write_bytes(b'{"field": "\xe2\x96"}')
            with self.assertRaises(UnicodeDecodeError):
                sources.read(Path("manifest.json"))

    def test_four_gpu_retry_requires_four_rank_coverage_and_matching_resources(self):
        task = runner.study_task(2, attempt=3)
        params = runner.effective_params(task, load_config(runner.config_directory(task), runner.EXPERIMENT))
        manifest = runner.task_manifest(task, ["GPU"] * 4, "a" * 40,
                                       {"SLURM_JOB_ID": "101", "SLURM_ARRAY_JOB_ID": "100"})
        manifest.update(source_sha256=self.sources, environment={"torch": "2.8.0+cu126"})
        log = self.log.replace(json.dumps(self.params), json.dumps(params))
        log = log.replace("810042365", "810472445").replace("7770909", "8200989")
        coverage = [dict(split=split, rank=rank, blocks=1, assigned_rows=8192,
                         batches=1, processed_rows=8192)
                    for rank in range(4) for split in ("train", "valid", "test")]
        slurm = (log + "\n".join(
            f"[Rank {rank}] blocked load balanced: blocks=1, rows=8192, batches=1"
            for _ in range(3) for rank in range(4)) + "\n"
            + "SISA_SCORE_DIM_PROTOCOL " + json.dumps(manifest) + "\n"
            + f"SISA_SCORE_DIM_COMPLETE run_id={task.run_id}\n")
        result = self.audit(task=task, manifest=manifest, checkpoint=log, slurm=slurm,
                            expected_coverage=coverage)
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["eligible"])
        self.assertEqual(result["comparison_scope"], "dimension_with_four_gpu_layout")
        self.assertEqual(result["global_batch"], 32768)
        self.assertFalse(self.audit(task=task, manifest=manifest, checkpoint=log, slurm=slurm)["eligible"])
        self.assertFalse(self.audit(task=task, manifest=dict(manifest, gpu_names=["GPU"] * 2),
                                    checkpoint=log, slurm=slurm, expected_coverage=coverage)["eligible"])

    def test_r64_checkpoint_retry_is_audited_and_explicitly_labeled(self):
        task = runner.study_task(2, attempt=2)
        params = runner.effective_params(task, load_config(str(runner.config_directory(task)), runner.EXPERIMENT))
        manifest = runner.task_manifest(task, ["GPU"] * 2, "a" * 40,
                                        {"SLURM_JOB_ID": "101", "SLURM_ARRAY_JOB_ID": "100"})
        manifest["source_sha256"] = self.sources
        log = self.log.replace(json.dumps(self.params), json.dumps(params))
        log = log.replace("810042365", "810472445").replace("7770909", "8200989")
        slurm = self.slurm.replace(self.log, log).replace(json.dumps(self.manifest), json.dumps(manifest))
        slurm = slurm.replace(self.task.run_id, task.run_id)
        result = self.audit(task=task, manifest=manifest, checkpoint=log, slurm=slurm)
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["eligible"])
        self.assertIs(result["gradient_checkpointing"], True)
        self.assertEqual(result["comparison_scope"], "dimension_with_activation_checkpointing")
        invalid = self.audit(task=task, manifest=manifest,
                             checkpoint=log.replace('"gradient_checkpointing": true',
                                                    '"gradient_checkpointing": false'), slurm=slurm)
        self.assertFalse(invalid["eligible"])

    def test_expanded_pending_array_placeholder_cannot_replace_running_record(self):
        running = "100_1|102|RUNNING|0:0\n100_1.batch|102.batch|RUNNING|0:0\n"
        pending = "100_1|100|PENDING|0:0\n100_2|100|PENDING|0:0\n"
        for text in (running + pending, pending + running):
            records = collector.parse_accounting(text)
            self.assertEqual(records["100_1"], {
                "raw_job_id": "102", "state": "RUNNING", "exit_code": "0:0"})
            self.assertEqual(records["100_2"]["state"], "PENDING")
            self.assertNotIn("100_1.batch", records)
        completed = "100_1|102|COMPLETED|0:0\n"
        records = collector.parse_accounting(completed + pending)
        self.assertEqual(records["100_1"]["state"], "COMPLETED")

    def setUp(self):
        self.task = runner.study_task(0)
        self.params = runner.effective_params(self.task, load_config("config", runner.EXPERIMENT))
        self.labels = [entry["name"] for entry in self.params["label_col"]]
        self.coverage = [{"split": split, "rank": rank, "blocks": 1, "assigned_rows": 16384,
                          "batches": 1, "processed_rows": 16384}
                         for rank in (0, 1) for split in ("train", "valid", "test")]
        self.manifest = runner.task_manifest(self.task, ["GPU"] * 2, "a" * 40,
                                            {"SLURM_JOB_ID": "101", "SLURM_ARRAY_JOB_ID": "100"})
        self.sources = {"unirank/pytorch/layers/attentions/sisa.py": "a" * 64}
        self.manifest["source_sha256"] = self.sources
        self.account = {"state": "COMPLETED", "exit_code": "0:0", "raw_job_id": "101"}
        self.outcome = {"returncode": 0}
        self.log = ("INFO Params: " + json.dumps(self.params) + "\n"
                    "INFO Total number of parameters: 810042365.\n"
                    "INFO Number of dense parameters: 7770909.\n"
                    "******** Test evaluation ********\n" + "".join(
                        f"[Task: {label}][Metrics] logloss: 0.1 - AUC: 0.8\n"
                        for label in self.labels))
        self.slurm = (self.log + "\n".join(
            f"[Rank {rank}] blocked load balanced: blocks=1, rows=16384, batches=1"
            for _ in range(3) for rank in (0, 1)) + "\n"
            + "SISA_SCORE_DIM_PROTOCOL " + json.dumps(self.manifest) + "\n"
            + f"SISA_SCORE_DIM_COMPLETE run_id={self.task.run_id}\n")

    def audit(self, **overrides):
        args = dict(task=self.task, array_id="100", account=self.account,
                    manifest=self.manifest, outcome=self.outcome, checkpoint=self.log,
                    slurm=self.slurm, expected_coverage=self.coverage,
                    expected_sources=self.sources)
        args.update(overrides)
        return collector.audit_run(**args)

    def test_complete_matching_logs_are_eligible(self):
        result = self.audit()
        self.assertEqual(result["issues"], [])
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["macro_auc"], .8)
        self.assertEqual(set(result["metrics"]), set(self.labels))

    def test_incomplete_mismatched_or_invalid_results_have_no_metrics(self):
        cases = [
            {"account": dict(self.account, state="RUNNING")},
            {"account": dict(self.account, exit_code="1:0")},
            {"outcome": {}},
            {"manifest": dict(self.manifest, score_dim=32)},
            {"manifest": dict(self.manifest, array_job_id="999")},
            {"manifest": dict(self.manifest, source_sha256={})},
            {"slurm": self.slurm.replace("SISA_SCORE_DIM_COMPLETE", "WRONG_COMPLETE")},
            {"slurm": self.slurm.replace("batches=1", "batches=2")},
            {"slurm": self.slurm.replace("AUC: 0.8", "AUC: 0.9")},
            {"checkpoint": self.log.replace('"sisa_variant": "full"',
                                              '"sisa_variant": "decay_only"')},
            {"checkpoint": self.log.replace("Test evaluation", "Validation evaluation")},
            {"checkpoint": self.log.replace("810042365", "810103805")},
            {"checkpoint": self.log.replace("AUC: 0.8", "AUC: nan")},
        ]
        for case in cases:
            with self.subTest(case=list(case)):
                result = self.audit(**case)
                self.assertFalse(result["eligible"])
                self.assertTrue(result["issues"])
                self.assertEqual(result["metrics"], {})
                self.assertIsNone(result["macro_auc"])
                self.assertIsNone(result["macro_logloss"])

    def test_deltas_require_eligible_predeclared_reference(self):
        reference = dict(self.audit(), score_dim=16, macro_auc=.79, macro_logloss=.11)
        result = collector.comparisons([reference, dict(self.audit(), score_dim=8)])
        self.assertAlmostEqual(result[0]["delta_auc"], .01)
        self.assertAlmostEqual(result[0]["delta_logloss"], -.01)
        reference["eligible"] = False
        self.assertEqual(collector.comparisons([reference, dict(self.audit(), score_dim=8)]), [])


if __name__ == "__main__":
    unittest.main()
