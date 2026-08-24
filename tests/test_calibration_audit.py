from __future__ import annotations

import unittest

from scripts.audit_onetrans_taobao_calibration import (
    REFERENCE_AUC,
    build_report,
    extract_test_auc,
    validate_protocol,
)


class CalibrationAuditTest(unittest.TestCase):
    def setUp(self):
        self.slurm_text = (
            "STRICT_PROTOCOL world_size=4 per_gpu_batch=8192 global_batch=32768 "
            "accumulation_steps=1 epochs=1 seed=20262027 bf16=true\n"
            "setting=baseline\n"
            "SISA_NATIVE_STRICT_COMPLETE run_id=test\n"
        )
        self.checkpoint_text = "\n".join(
            (
                '"batch_size": "8192"',
                '"accumulation_steps": "1"',
                '"epochs": "1"',
                '"seed": "20262027"',
                '"max_len": "100"',
                '"world_size": "4"',
                '"enable_bf16": "True"',
                "Start training: 540 local batches/epoch",
                "DDP blocked training: True",
                "******** Test evaluation ********",
                "[Task: is_click][Metrics] logloss: 0.1 - AUC: 0.629531",
                "[Task: cart][Metrics] logloss: 0.1 - AUC: 0.745796",
                "[Task: fav][Metrics] logloss: 0.1 - AUC: 0.783280",
                "[Task: buy][Metrics] logloss: 0.1 - AUC: 0.767286",
            )
        )

    def test_valid_protocol_and_metrics_pass(self):
        self.assertEqual(validate_protocol(self.slurm_text, self.checkpoint_text), [])
        self.assertEqual(extract_test_auc(self.checkpoint_text), REFERENCE_AUC)

    def test_reference_tolerance_is_diagnostic_not_structural(self):
        metrics = dict(REFERENCE_AUC)
        metrics["buy"] += 0.02
        report, within_all = build_report(metrics, tolerance=0.01)
        self.assertFalse(within_all)
        self.assertIn("review", report)

    def test_missing_world_size_is_a_structural_failure(self):
        checkpoint_text = self.checkpoint_text.replace('"world_size": "4"', "")
        errors = validate_protocol(self.slurm_text, checkpoint_text)
        self.assertTrue(any("world_size" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
