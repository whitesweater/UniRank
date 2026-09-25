import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_targeted_controls as collector


class TargetedCollectorTest(unittest.TestCase):
    def test_wrong_site_and_seed_are_rejected(self):
        spec = dict(collector.TASKS[1], run_id="test")
        correct = collector.expected_model_audit(spec, 0)
        self.assertEqual(collector.model_audit_issues(spec, correct, 0), [])
        changed = copy.deepcopy(correct)
        changed["sites"][0]["parameter_seed"] += 1
        self.assertTrue(collector.model_audit_issues(spec, changed, 0))
        changed = copy.deepcopy(correct)
        changed["sites"][1]["enabled"] = True
        self.assertTrue(collector.model_audit_issues(spec, changed, 0))

    def test_wrong_rank_and_capacity_are_rejected(self):
        spec = dict(collector.TASKS[0], run_id="test")
        data = collector.expected_model_audit(spec, 0)
        self.assertTrue(collector.model_audit_issues(spec, data, 1))
        data["active_dense_parameters"] += 1
        self.assertTrue(collector.model_audit_issues(spec, data, 0))

    def test_failed_later_attempt_does_not_replace_eligible_run(self):
        runs = [dict(logical_task="x", attempt=i, eligible=(i == 2), selected=False) for i in [1, 2, 3]]
        collector.select_attempts(runs)
        self.assertEqual([r["attempt"] for r in runs if r["selected"]], [2])
        self.assertEqual(len(runs), 3)

    def test_interleaved_audits_and_progress_are_not_discarded(self):
        text = 'AUDIT {"rank":0}AUDIT {"rank":1}\n'
        self.assertEqual(collector.embedded_json(text, "AUDIT"), [{"rank": 0}, {"rank": 1}])
        progress = collector.progress_snapshot('Rank 0 | Epoch 1: 0%| | 0/100 [x]\rRank 0 | Epoch 1: 100%|x| 100/100 [x]')
        self.assertTrue(progress["0"]["advancing_within_snapshot"])

    def test_discovery_preserves_attempt_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            task = collector.TASKS[2]
            run_id = f"RISE_targeted_{task['model']}_{task['dataset']}_{task['mode']}_seed20262028_attempt2"
            (Path(directory) / "artifacts/rise_targeted_controls" / run_id).mkdir(parents=True)
            found = [r for r in collector.discover(directory) if r["task_index"] == 2]
            self.assertEqual([r["attempt"] for r in found], [1, 2])


if __name__ == "__main__":
    unittest.main()
