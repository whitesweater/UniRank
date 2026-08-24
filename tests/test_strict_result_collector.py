from __future__ import annotations

import unittest

from scripts.collect_sisa_native_strict_results import (
    L40_TASKS,
    RTX_TASKS,
    assignment,
    effective_slurm_outcome,
    logical_task,
    parse_test_metrics,
    parse_job_overrides,
    parse_gpu_type_overrides,
)


class StrictResultCollectorTest(unittest.TestCase):
    def test_assignments_cover_matrix_once_and_keep_pairs_on_same_gpu(self):
        assigned = {4} | set(RTX_TASKS) | set(L40_TASKS)
        self.assertEqual(assigned, set(range(32)))
        self.assertFalse(({4} | set(RTX_TASKS)) & set(L40_TASKS))
        for task_id in range(0, 32, 2):
            baseline = assignment(task_id, 14806, 14810, 14811)[1]
            sisa = assignment(task_id + 1, 14806, 14810, 14811)[1]
            self.assertEqual(baseline, sisa)

    def test_logical_task_mapping(self):
        self.assertEqual(logical_task(0).model, "OneTrans")
        self.assertEqual(logical_task(4).dataset, "Taobao_Action")
        self.assertEqual(logical_task(4).setting, "baseline")
        self.assertEqual(logical_task(31).model, "Zenith")
        self.assertEqual(logical_task(31).dataset, "MerRec_Action")
        self.assertEqual(logical_task(31).setting, "sisa")

    def test_retry_job_overrides_are_parsed(self):
        self.assertEqual(
            parse_job_overrides(["30=14900", "31=14900"]),
            {30: 14900, 31: 14900},
        )
        with self.assertRaises(ValueError):
            parse_job_overrides(["32=14900"])

    def test_retry_gpu_type_overrides_are_parsed(self):
        self.assertEqual(
            parse_gpu_type_overrides(["30=l40s", "31=l40s"]),
            {30: "l40s", 31: "l40s"},
        )
        with self.assertRaises(ValueError):
            parse_gpu_type_overrides(["30=h20"])

    def test_parser_uses_only_final_test_section(self):
        text = "\n".join(
            (
                "******** Test evaluation ********",
                "[Task: click][Metrics] AUC: 0.1 - logloss: 0.9",
                "******** Test evaluation ********",
                "[Task: click][Metrics] logloss: 0.2 - AUC: 0.8",
                "[Task: buy][Metrics] logloss: 0.3 - AUC: 0.7",
            )
        )
        self.assertEqual(
            parse_test_metrics(text),
            {
                "click": {"logloss": 0.2, "AUC": 0.8},
                "buy": {"logloss": 0.3, "AUC": 0.7},
            },
        )

    def test_completion_marker_survives_purged_slurm_record(self):
        self.assertEqual(
            effective_slurm_outcome("UNKNOWN", "UNKNOWN", True),
            (
                "COMPLETED_MARKER",
                "0:0 (success marker)",
                "success_marker",
            ),
        )

    def test_live_slurm_outcome_is_not_overridden(self):
        self.assertEqual(
            effective_slurm_outcome("RUNNING", "0:0", True),
            ("RUNNING", "0:0", "scontrol"),
        )


if __name__ == "__main__":
    unittest.main()
