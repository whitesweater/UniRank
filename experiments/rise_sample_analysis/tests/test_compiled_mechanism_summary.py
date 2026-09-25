from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_compiled_mechanism import (
    IDS, LABELS, METRICS, SITES, length_groups, parse_accounting,
    validate_observations, validate_prediction_equality,
)


class CompiledMechanismSummaryTest(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({"part_id": [2, 3], "row_offset": [413, 928],
            "user_id": [123, 234], "item_id": [9, 10], "seq_len": [30, 150],
            "effective_length": [30, 100], **{label: [0, 1] for label in LABELS}})

    def observations(self):
        rows = []
        for site in SITES:
            for _, source in self.meta.iterrows():
                count = 2 * source.effective_length if ".seq_layers." in site else 4
                values = {metric: 0.0 for metric in METRICS}
                values.update(valid_attention_rows=count, allowed_pairs=count * 100,
                              omega_mean=1, omega_max=2, tv_mean=.1, tv_max=.2, tv_bound_mean=.2,
                              native_logit_sd_mean=1)
                rows.append({**source.to_dict(), **values, "site": site, "variant": "B",
                    "capture_route": "compiled_tensor_capture", "native_mask_present": False,
                    "native_q_dtype": "torch.bfloat16", "native_k_dtype": "torch.bfloat16", "delta_dtype": "torch.bfloat16"})
        return pd.DataFrame(rows)

    def test_complete_actual_capture_schema_is_accepted(self):
        result = validate_observations(self.observations(), self.meta, "B")
        self.assertEqual(result["rows"], 12)
        self.assertEqual(result["sites"], 6)

    def test_missing_site_duplicate_and_changed_metadata_are_rejected(self):
        rows = self.observations()
        for bad in (rows.iloc[:-1], pd.concat([rows.iloc[:-1], rows.iloc[:1]], ignore_index=True)):
            with self.assertRaises(ValueError):
                validate_observations(bad, self.meta, "B")
        rows.loc[0, "item_id"] += 1
        with self.assertRaises(ValueError):
            validate_observations(rows, self.meta, "B")

    def test_eager_route_and_extra_control_statistics_cannot_enter_selection_report(self):
        rows = self.observations()
        rows.loc[0, "capture_route"] = "eager"
        with self.assertRaises(ValueError):
            validate_observations(rows, self.meta, "B")
        rows = self.observations().assign(query_clipping_fraction=0.0)
        with self.assertRaises(ValueError):
            validate_observations(rows, self.meta, "B")

    def test_bound_certificate_and_native_query_count_are_checked(self):
        for column, value in (("tv_mean", .3), ("top_order_certificate_violation_fraction", .01),
                              ("valid_attention_rows", 1)):
            rows = self.observations()
            rows.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_observations(rows, self.meta, "B")

    def test_passed_json_does_not_override_one_ulp_prediction_change(self):
        rows, old = self.meta.copy(), self.meta.copy()
        for label in LABELS:
            values = np.array([.1, .3], dtype=np.float32)
            rows[f"{label}_compiled_plain"] = values
            rows[f"{label}_compiled_capture"] = values
            old[f"{label}_compiled"] = values
        entry = {"passed": True, "bitwise_equal": True, "atol": 5e-4, "rtol": 5e-3,
                 "labels": {label: {"passed": True, "rows": 2, "failed_rows": 0, "max_abs": 0, "mean_abs": 0}
                            for label in LABELS}}
        gates = {name: entry for name in ("compiled_capture_vs_compiled_plain", "compiled_plain_vs_failed_run_compiled")}
        validate_prediction_equality(rows, old, self.meta, gates, "B")
        rows.loc[0, "Like_compiled_capture"] = np.nextafter(np.float32(.1), np.float32(1))
        with self.assertRaises(ValueError):
            validate_prediction_equality(rows, old, self.meta, gates, "B")

    def test_length_boundaries_and_failed_job_provenance_are_explicit(self):
        np.testing.assert_array_equal(length_groups([0, 1, 20, 21, 50, 51, 99, 100]),
                                      ["0", "1-20", "1-20", "21-50", "21-50", "51-99", "51-99", "100"])
        with self.assertRaises(ValueError):
            length_groups([101])
        record = "643911|FAILED|11:0|00:01:33|ACD1-63\n643949|COMPLETED|0:0|00:01:14|ACD1-63\n"
        self.assertEqual(parse_accounting(record)["643911"]["state"], "FAILED")
        with self.assertRaises(ValueError):
            parse_accounting(record.replace("FAILED|11:0", "COMPLETED|0:0"))


if __name__ == "__main__":
    unittest.main()
