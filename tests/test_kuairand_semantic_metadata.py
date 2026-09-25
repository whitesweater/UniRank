import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_metadata", ROOT / "scripts/prepare_kuairand_semantic_metadata.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class KuaiRandSemanticMetadataTest(unittest.TestCase):
    def test_complete_quoted_multiline_and_truncated_record(self):
        data = ('final_video_id,caption,show_cover_text,duration\n'
                '4,"篮球,第一节\n第二节",封面,23000\n'
                '5,"incomplete\n').encode()
        rows, issues = MODULE.parse_fragment(data)
        self.assertEqual([row["final_video_id"] for row in rows], [4])
        self.assertEqual(rows[0]["caption"], "篮球,第一节\n第二节")
        self.assertEqual(issues[0]["kind"], "csv_incomplete_or_invalid")

    def test_reject_incomplete_physical_tail(self):
        rows, _ = MODULE.parse_fragment(b"1,complete,,1000\n2,partial,,20", start=4)
        self.assertEqual(rows, [])

    def test_missing_and_malformed_duration_are_distinct(self):
        rows, _ = MODULE.parse_fragment("0,标题,,\n1,标题,,错误\n2,标题,,9000\n".encode())
        self.assertEqual([r["duration_issue"] for r in rows], ["missing", "invalid_numeric", None])
        self.assertEqual(rows[2]["item_id"], 3)

    def test_no_duplicate_or_out_of_order_id(self):
        with self.assertRaisesRegex(ValueError, "Nonmonotonic"):
            MODULE.parse_fragment(b"2,caption,,1000\n1,caption,,2000\n")


if __name__ == "__main__":
    unittest.main()
