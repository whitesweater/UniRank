from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import compare_sisa_single_seed_results as comparison


def metric(
    model: str,
    dataset: str,
    label: str,
    auc: float,
    logloss: float,
    **extra: str,
) -> dict[str, str]:
    return {
        "model": model,
        "dataset": dataset,
        "label": label,
        "setting": "sisa",
        "AUC": str(auc),
        "logloss": str(logloss),
        **extra,
    }


class SisaSingleSeedComparisonTest(unittest.TestCase):
    def test_summary_accepts_same_protocol_seed_pair_note(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            comparison.write_summary(
                path,
                {
                    "label_count": 1,
                    "cell_count": 1,
                    "mean_delta_auc_label_weighted": 0.0,
                    "mean_delta_auc_cell_macro": 0.0,
                    "auc_improved_labels": 0,
                    "positive_auc_cells": 0,
                    "mean_delta_logloss_label_weighted": 0.0,
                    "logloss_improved_labels": 0,
                    "lower_logloss_cells": 0,
                    "auc_within_0_005": 1,
                    "same_best_model_tasks": 1,
                    "ranking_tasks": 1,
                },
                [],
                [],
                comparison_note="same-protocol seed pair",
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("same-protocol seed pair", text)
            self.assertNotIn("GPU topology", text)

    def test_seed_specific_paths_reject_cross_seed_inputs_and_outputs(self):
        metrics, output = comparison.seed_study_paths(20262029)
        self.assertEqual(
            metrics,
            Path("experiments/sisa_single_seed20262029/results/metrics.csv"),
        )
        self.assertEqual(
            output,
            Path("experiments/sisa_single_seed20262029/comparison"),
        )
        with self.assertRaisesRegex(ValueError, "seed-specific path"):
            comparison.seed_study_paths(
                20262029,
                output_dir=Path("experiments/sisa_single_seed20262028/comparison"),
            )

    def test_previous_selection_uses_strict_and_expansion_without_overlap(self):
        strict = [
            metric(
                "HiFormer",
                "QK_Video_Action",
                "click",
                0.8,
                0.2,
                gpu_type="l40s",
            ),
            metric(
                "HyFormer",
                "QK_Video_Action",
                "click",
                0.7,
                0.3,
                gpu_type="l40s",
            ),
        ]
        expansion = [
            metric(
                "HyFormer",
                "QK_Video_Action",
                "click",
                0.81,
                0.19,
                gpu_name="NVIDIA H100 80GB HBM3",
            ),
            metric(
                "HiFormer",
                "TencentGR_10M_Action",
                "is_click",
                0.82,
                0.18,
                gpu_name="NVIDIA H100 80GB HBM3",
            ),
        ]

        selected = comparison.select_previous_metrics(strict, expansion)

        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {row["previous_source"] for row in selected},
            {"sisa_native_strict", "sisa_expansion_acd"},
        )
        self.assertNotIn(
            ("HyFormer", "QK_Video_Action", "click"),
            {
                comparison.metric_key(row)
                for row in selected
                if row["previous_source"] == "sisa_native_strict"
            },
        )

    def test_comparison_aligns_keys_and_uses_metric_direction(self):
        previous = [
            {
                **metric("HiFormer", "QK_Video_Action", "click", 0.80, 0.20),
                "previous_source": "sisa_native_strict",
                "previous_protocol": "old",
                "previous_gpu": "l40s",
            }
        ]
        new = [
            {
                **metric("HiFormer", "QK_Video_Action", "click", 0.81, 0.19),
                "protocol": "ws2_bs16384_acc1",
                "seed": "20262028",
            }
        ]

        rows = comparison.compare_metrics(new, previous)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["delta_auc"], 0.01)
        self.assertAlmostEqual(rows[0]["delta_logloss"], -0.01)
        self.assertEqual(rows[0]["auc_direction"], "improved")
        self.assertEqual(rows[0]["logloss_direction"], "improved")
        self.assertEqual(rows[0]["joint_direction"], "both_improved")

    def test_duplicate_or_misaligned_keys_are_rejected(self):
        row = metric("HiFormer", "QK_Video_Action", "click", 0.80, 0.20)
        with self.assertRaisesRegex(ValueError, "duplicate new metric key"):
            comparison.compare_metrics([row, dict(row)], [])

        previous = {
            **metric("HiFormer", "QK_Video_Action", "follow", 0.80, 0.20),
            "previous_source": "sisa_native_strict",
            "previous_protocol": "old",
            "previous_gpu": "l40s",
        }
        with self.assertRaisesRegex(ValueError, "metric keys do not align"):
            comparison.compare_metrics([row], [previous])


if __name__ == "__main__":
    unittest.main()
