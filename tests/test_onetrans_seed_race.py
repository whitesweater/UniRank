from __future__ import annotations

import unittest
from unittest import mock

from scripts import monitor_onetrans_seed_race as monitor
from scripts.monitor_onetrans_seed_race import select_winner


class OneTransSeedRaceTest(unittest.TestCase):
    def test_poll_cancels_only_the_matched_loser(self):
        plan = {"arrays": {"ws4": "100", "ws2": "200"}}
        records = {f"{array}_{task}": {"state": "PENDING", "start": "Unknown"}
                   for array in ("100", "200") for task in monitor.ALLOWED_TASKS}
        records["200_0"] = {"state": "RUNNING", "start": "2026-09-18T16:00:00"}
        state = {}
        with mock.patch.object(monitor, "observations", return_value=records), \
             mock.patch.object(monitor, "atomic_json"), \
             mock.patch.object(monitor.subprocess, "run") as cancel:
            cancel.return_value = mock.Mock(returncode=0, stderr="")
            self.assertFalse(monitor.poll(plan, state))
        self.assertEqual(cancel.call_args.args[0], ["scancel", "100_0"])
        self.assertEqual(cancel.call_count, 1)
        self.assertEqual(set(state["decisions"]), {"0"})
        self.assertEqual(state["decisions"]["0"]["winner_protocol"], "ws2")

    def test_pending_pairs_never_cancel_anything(self):
        plan = {"arrays": {"ws4": "100", "ws2": "200"}}
        records = {f"{array}_{task}": {"state": "PENDING", "start": "2026-09-19T12:00:00"}
                   for array in ("100", "200") for task in monitor.ALLOWED_TASKS}
        with mock.patch.object(monitor, "observations", return_value=records), \
             mock.patch.object(monitor, "atomic_json"), \
             mock.patch.object(monitor.subprocess, "run") as cancel:
            self.assertFalse(monitor.poll(plan, {}))
            cancel.assert_not_called()

    def test_registered_plan_cannot_include_taobao_or_unrelated_tasks(self):
        import json

        plan = {"policy": "first_started_wins", "arrays": {"ws4": "100", "ws2": "200"},
                "task_ids": list(monitor.ALLOWED_TASKS)}
        with mock.patch.object(monitor.Path, "read_text", return_value=json.dumps(plan)):
            self.assertEqual(monitor.load_plan(), plan)
        plan["task_ids"].append(6)
        with mock.patch.object(monitor.Path, "read_text", return_value=json.dumps(plan)):
            with self.assertRaises(ValueError):
                monitor.load_plan()

    def test_pending_start_predictions_do_not_select_a_winner(self):
        jobs = {"ws4": {"state": "PENDING", "start": "2026-09-19T12:00:00"},
                "ws2": {"state": "PENDING", "start": "2026-09-18T18:00:00"}}
        self.assertIsNone(select_winner(jobs))

    def test_running_job_wins_even_if_other_pending_prediction_is_earlier(self):
        jobs = {"ws4": {"state": "PENDING", "start": "2026-09-18T15:00:00"},
                "ws2": {"state": "RUNNING", "start": "2026-09-18T16:00:00"}}
        self.assertEqual(select_winner(jobs), "ws2")

    def test_actual_start_order_and_deterministic_tie(self):
        jobs = {"ws4": {"state": "RUNNING", "start": "2026-09-18T16:00:01"},
                "ws2": {"state": "RUNNING", "start": "2026-09-18T16:00:00"}}
        self.assertEqual(select_winner(jobs), "ws2")
        jobs["ws4"]["start"] = jobs["ws2"]["start"]
        self.assertEqual(select_winner(jobs), "ws4")

    def test_completed_or_failed_started_job_preserves_start_order(self):
        jobs = {"ws4": {"state": "COMPLETED", "start": "2026-09-18T12:00:00"},
                "ws2": {"state": "RUNNING", "start": "2026-09-18T16:00:00"}}
        self.assertEqual(select_winner(jobs), "ws4")
        jobs["ws4"]["state"] = "FAILED"
        self.assertEqual(select_winner(jobs), "ws4")

    def test_missing_start_is_not_used_as_evidence(self):
        jobs = {"ws4": {"state": "CANCELLED", "start": "Unknown"},
                "ws2": {"state": "PENDING", "start": "2026-09-18T18:00:00"}}
        self.assertIsNone(select_winner(jobs))


if __name__ == "__main__":
    unittest.main()
