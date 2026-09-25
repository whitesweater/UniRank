#!/usr/bin/env python3
"""Fetch official KuaiRand caption annotations by ID using verified HTTP ranges.

Annotations are for interpretation only; this program never writes model inputs.
All server fragments and headers are retained with hashes. Incomplete or malformed
CSV records are rejected, and an exact requested final_video_id is required.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import time


URL = "https://zenodo.org/records/18159199/files/kuairand_video_captions.csv"
TOTAL_BYTES = 3222932768
MAX_VIDEO_ID = 32038724
FIELDS = ["final_video_id", "caption", "show_cover_text", "duration"]
DEFAULT_ROOT = Path("artifacts/rise_attention_semantic/metadata")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_label(value):
    labels = ["0", "0-3s", "3-7s", "7-15s", "15-30s", "30-60s", "60s+"]
    for upper, label in zip([0, 3000, 7000, 15000, 30000, 60000, math.inf], labels):
        if value <= upper:
            return label
    raise ValueError(value)


def annotate_row(row):
    result = dict(zip(FIELDS, row))
    result["final_video_id"] = int(row[0])
    result["item_id"] = int(row[0]) + 1
    result["caption_nonempty"] = bool(row[1].strip() or row[2].strip())
    result["duration_valid"] = False
    result["duration_ms"] = None
    result["duration_issue"] = "missing" if not row[3].strip() else "invalid_numeric"
    try:
        duration = float(row[3])
        if math.isfinite(duration) and duration >= 0:
            result.update(duration_valid=True, duration_ms=duration, duration_issue=None)
    except ValueError:
        pass
    return result


def parse_fragment(payload, start=0):
    """Strictly parse complete records, retaining exact original field values.

    For nonzero ranges the leading partial physical line is discarded. A numeric
    ID at the start of a later line provides a candidate record boundary. Strict
    CSV parsing then supports quoted multiline records; sequence order and schema
    are checked. The last physical line is always removed unless it ends in LF.
    A later overlapping fetch is required for any row used in the final output.
    """
    errors = []
    if start:
        first_newline = payload.find(b"\n")
        if first_newline < 0:
            return [], [{"kind": "no_complete_record"}]
        payload = payload[first_newline + 1:]
        match = re.search(rb"(?m)^\d+,", payload)
        if match is None:
            return [], [{"kind": "no_record_boundary"}]
        payload = payload[match.start():]
    last_newline = payload.rfind(b"\n")
    if last_newline < 0:
        return [], [{"kind": "no_complete_record"}]
    payload = payload[:last_newline + 1]
    decoded = payload.decode("utf-8", errors="strict")
    reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
    result = []
    last_id = -1
    try:
        for row in reader:
            if row == FIELDS:
                continue
            if len(row) != 4 or not row[0].isdigit():
                errors.append({"kind": "schema", "physical_line": reader.line_num})
                continue
            identifier = int(row[0])
            if not 0 <= identifier <= MAX_VIDEO_ID or identifier <= last_id:
                raise ValueError("Nonmonotonic or out-of-range video ID in CSV fragment")
            last_id = identifier
            result.append(annotate_row(row))
    except csv.Error as exc:
        errors.append({"kind": "csv_incomplete_or_invalid", "message": str(exc),
                       "physical_line": reader.line_num})
    return result, errors


class CaptionRanges:
    def __init__(self, root=DEFAULT_ROOT, block_bytes=131072):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "ranges"
        self.cache.mkdir(exist_ok=True)
        self.block_bytes = block_bytes
        self.receipts = []
        self.rows = {}
        self.row_sources = {}
        self.brackets = [(0, 0), (MAX_VIDEO_ID + 1, TOTAL_BYTES)]
        self.identity = self.check_source_identity()
        for receipt_path in sorted(self.cache.glob("*.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("source_last_modified") != self.identity["last_modified"]:
                continue
            data_path = receipt_path.with_suffix(".csv.part")
            headers_path = receipt_path.with_suffix(".headers")
            if (sha256(data_path) != receipt["sha256"] or
                    sha256(headers_path) != receipt["headers_sha256"]):
                raise RuntimeError("Cached metadata fragment hash mismatch")
            rows, _ = parse_fragment(data_path.read_bytes(), receipt["start"])
            if rows:
                self.brackets.extend([(rows[0]["final_video_id"], receipt["start"]),
                                      (rows[-1]["final_video_id"], receipt["end"])])
                for row in rows:
                    identifier = row["final_video_id"]
                    if identifier in self.rows and any(
                            self.rows[identifier][field] != row[field] for field in FIELDS):
                        raise RuntimeError("Cached overlapping CSV fragments disagree")
                    self.rows[identifier] = row
                    self.row_sources[identifier] = dict(receipt, path=str(data_path))

    def check_source_identity(self):
        result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                                 "--head", "--max-time", "30", URL],
                                check=True, capture_output=True, text=True)
        headers = result.stdout
        modified = re.findall(r"(?im)^last-modified:\s*(.+?)\r?$", headers)
        lengths = re.findall(r"(?im)^content-length:\s*(\d+)\r?$", headers)
        if not modified or not lengths or int(lengths[-1]) != TOTAL_BYTES:
            raise RuntimeError("Missing Last-Modified or unexpected source byte length")
        identity = {"url": URL, "last_modified": modified[-1].strip(),
                    "total_bytes": int(lengths[-1]), "tls_verification": True,
                    "official_md5_not_locally_verified": "9fd4e7587ffd0c8dc289d013858fb229"}
        path = self.root / "source_identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise RuntimeError("Official source identity changed between retrievals")
        path.write_text(json.dumps(identity, indent=2) + "\n")
        (self.root / "source_identity.headers").write_text(headers)
        return identity

    def fetch(self, start, length=None):
        start = max(0, min(int(start), TOTAL_BYTES - 1))
        end = min(TOTAL_BYTES - 1, start + (length or self.block_bytes) - 1)
        stem = self.cache / f"{start:010d}-{end:010d}"
        data_path = stem.with_suffix(".csv.part")
        headers_path = stem.with_suffix(".headers")
        receipt_path = stem.with_suffix(".json")
        cached = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if cached and cached.get("source_last_modified") == self.identity["last_modified"]:
            receipt = cached
            if (sha256(data_path) != receipt["sha256"] or
                    sha256(headers_path) != receipt["headers_sha256"]):
                raise RuntimeError("Cached response hash mismatch")
        else:
            temporary = stem.with_suffix(".downloading")
            began = time.monotonic()
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                            "--retry", "1", "--retry-delay", "2", "--max-time", "45",
                            "--header", "If-Unmodified-Since: " + self.identity["last_modified"],
                            "--range", f"{start}-{end}", "--dump-header", str(headers_path),
                            "--output", str(temporary), URL], check=True)
            headers = headers_path.read_text()
            matches = re.findall(r"(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)", headers)
            if not matches or tuple(map(int, matches[-1])) != (start, end, TOTAL_BYTES):
                raise RuntimeError("Server did not return the exact requested HTTP range")
            if temporary.stat().st_size != end - start + 1:
                raise RuntimeError("HTTP range body length mismatch")
            temporary.replace(data_path)
            receipt = {"url": URL, "start": start, "end": end,
                       "total_bytes": TOTAL_BYTES, "sha256": sha256(data_path),
                       "headers_sha256": sha256(headers_path), "tls_verification": True,
                       "source_last_modified": self.identity["last_modified"],
                       "request_if_unmodified_since": self.identity["last_modified"],
                       "elapsed_seconds": time.monotonic() - began}
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        rows, issues = parse_fragment(data_path.read_bytes(), start)
        receipt = dict(receipt, path=str(data_path), parsed_records=len(rows), csv_issues=issues)
        self.receipts.append(receipt)
        if rows:
            # Approximate endpoints guide interpolation only. Exact ID matching
            # remains mandatory and is independent of these approximations.
            self.brackets.extend([(rows[0]["final_video_id"], start),
                                  (rows[-1]["final_video_id"], end)])
            for row in rows:
                identifier = row["final_video_id"]
                if identifier in self.rows and any(
                        self.rows[identifier][field] != row[field] for field in FIELDS):
                    raise RuntimeError("Overlapping fragments disagree on an exact CSV record")
                self.rows[identifier] = row
                self.row_sources[identifier] = receipt
        return rows

    def lookup(self, identifier):
        identifier = int(identifier)
        if not 0 <= identifier <= MAX_VIDEO_ID:
            raise ValueError(f"Invalid final_video_id {identifier}")
        attempts = []
        for _ in range(18):
            if identifier in self.rows:
                candidate = self.rows[identifier]
                # Independent overlapping range, shifted 4 KiB from the fetch
                # location, must recover all four identical original fields.
                source = self.row_sources[identifier]
                check_start = max(0, source["start"] - 4096)
                check = self.fetch(check_start, source["end"] - check_start + 4097)
                verification = next((r for r in check if r["final_video_id"] == identifier), None)
                if verification is None or any(candidate[k] != verification[k] for k in FIELDS):
                    raise RuntimeError(f"Overlapping exact-record validation failed for {identifier}")
                return dict(candidate, exact_id_verified=True, overlap_verified=True,
                            lookup_ranges=attempts, source_sha256=source["sha256"],
                            source_range_path=source["path"],
                            verification_range_path=self.receipts[-1]["path"],
                            verification_sha256=self.receipts[-1]["sha256"])
            bounds = sorted(set(self.brackets))
            lower = max((pair for pair in bounds if pair[0] <= identifier), default=(0, 0))
            upper = min((pair for pair in bounds if pair[0] > identifier),
                        default=(MAX_VIDEO_ID + 1, TOTAL_BYTES))
            fraction = (identifier - lower[0]) / max(1, upper[0] - lower[0])
            center = lower[1] + fraction * (upper[1] - lower[1])
            start = max(0, int(center) - self.block_bytes // 2)
            if start in attempts:
                start = max(0, start - self.block_bytes)
            attempts.append(start)
            self.fetch(start)
        return {"final_video_id": identifier, "item_id": identifier + 1,
                "exact_id_verified": False, "reason": "not_found_within_request_budget",
                "lookup_ranges": attempts}


def prefix_audit(root, item_file):
    """Audit a bounded official prefix against unchanged processed item metadata."""
    from collections import Counter
    import pyarrow.parquet as pq

    root = Path(root)
    prefix_paths = [root / "captions_head.csv", root / "captions_range_probe.part"]
    payload = b"".join(path.read_bytes() for path in prefix_paths)
    rows, issues = parse_fragment(payload)
    # Empty durations are missing metadata; malformed numeric values are distinct.
    invalid = Counter(row["duration_issue"] for row in rows if not row["duration_valid"])
    by_item = {row["item_id"]: row for row in rows if row["duration_valid"]}
    table = pq.read_table(item_file, columns=["item_id", "duration_bucket"]).to_pandas()
    matched = table[table.item_id.isin(by_item)]
    counts = Counter((duration_label(by_item[int(row.item_id)]["duration_ms"]),
                      int(row.duration_bucket)) for row in matched.itertuples())
    mapping = {}
    for (label, code), count in counts.items():
        if label in mapping and mapping[label] != code:
            raise RuntimeError(f"Duration metadata does not agree: {label}")
        mapping[label] = code
    controls = []
    for shift in [-1, 1]:
        shifted = table[table.item_id.isin({i - shift for i in by_item})]
        total = 0
        agree = 0
        for row in shifted.itertuples():
            label = duration_label(by_item[int(row.item_id) + shift]["duration_ms"])
            if label in mapping:
                total += 1
                agree += mapping[label] == int(row.duration_bucket)
        controls.append({"offset_from_correct_join": shift, "n": total,
                         "agree": agree, "agreement": agree / total if total else None})
    receipt = {
        "status": "passed" if len(matched) and len(mapping) >= 5 else "insufficient",
        "mapping": "encoded item_id = official final_video_id + 1; zero is padding",
        "official_id_range": [0, MAX_VIDEO_ID],
        "official_id_space_source": "https://kuairand.com/figs/three-version.png",
        "official_metadata_source": URL,
        "processed_source": str(item_file),
        "processed_source_sha256": sha256(item_file),
        "preprocessing_source": "data/KuaiRand/preprocess_Kuairand_seq_action.py",
        "preprocessing_sha256": sha256("data/KuaiRand/preprocess_Kuairand_seq_action.py"),
        "prefix_files": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                         for p in prefix_paths],
        "records": len(rows), "min_id": rows[0]["final_video_id"],
        "max_id": rows[-1]["final_video_id"], "csv_issues": issues,
        "invalid_duration": dict(invalid), "valid_duration_records": len(by_item),
        "matched_records": len(matched), "duration_agreement": 1.0,
        "observed_duration_label_to_code": mapping,
        "duration_joint_counts": [{"label": key[0], "encoded_code": key[1], "n": count}
                                  for key, count in counts.items()],
        "shifted_join_controls": controls,
        "scope": "Bounded prefix and one existing processed test item table; no model input changed.",
        "limitations": ["Absent duration code 5 is not inferred from this audit.",
                        "Official annotations can have empty or malformed fields; retain their raw values."]}
    (root / "mapping_validation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    pq.write_table(__import__("pyarrow").Table.from_pylist(rows), root / "caption_prefix.parquet")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit-prefix", action="store_true")
    parser.add_argument("--item-file", type=Path,
                        default=Path("datasets/KuaiRand_Video_Action/test/item_info/part-00000.parquet"))
    parser.add_argument("--item-ids", type=int, nargs="*")
    parser.add_argument("--ids-json", type=Path)
    parser.add_argument("--expected-buckets-csv", type=Path,
                        help="CSV with encoded_item_id,duration_bucket for exact displayed tokens")
    args = parser.parse_args()
    if args.audit_prefix:
        receipt = prefix_audit(args.root, args.item_file)
        print(json.dumps({k: receipt[k] for k in ["status", "records", "matched_records",
                                                "duration_agreement", "invalid_duration"]}))
    ids = args.item_ids or []
    request_sources = []
    if args.ids_json:
        payload = args.ids_json.read_bytes()
        ids.extend(json.loads(payload))
        request_sources.append({"path": str(args.ids_json), "sha256": hashlib.sha256(payload).hexdigest()})
    if args.expected_buckets_csv:
        request_sources.append({"path": str(args.expected_buckets_csv),
                                "sha256": sha256(args.expected_buckets_csv)})
    if ids:
        source = CaptionRanges(args.root)
        expected = {}
        if args.expected_buckets_csv:
            with args.expected_buckets_csv.open(newline="") as handle:
                for request in csv.DictReader(handle):
                    item_id = int(request.get("encoded_item_id") or request["item_id"])
                    code = int(request["duration_bucket"])
                    if item_id in expected and expected[item_id] != code:
                        raise ValueError(f"Conflicting processed duration buckets for item {item_id}")
                    expected[item_id] = code
        selected_path = args.root / "selected_captions.json"
        selected = json.loads(selected_path.read_text()) if selected_path.exists() else []
        selected_by_id = {row["item_id"]: row for row in selected}
        mapping_path = args.root / "mapping_validation.json"
        mapping_passed = (mapping_path.exists() and
                          json.loads(mapping_path.read_text())["status"] == "passed")
        label_codes = (json.loads(mapping_path.read_text())["observed_duration_label_to_code"]
                       if mapping_passed else {})
        for item_id in sorted(set(ids)):
            row = source.lookup(item_id - 1)
            row["encoded_item_id"] = item_id
            row["cover_text"] = row.get("show_cover_text", "")
            row["mapping_verified"] = bool(mapping_passed and row.get("exact_id_verified")
                                           and row.get("overlap_verified"))
            row["semantic_text_valid"] = any(
                sum(char.isalnum() for char in text) >= 4
                for text in (row.get("caption", ""), row["cover_text"]))
            row["expected_duration_bucket"] = expected.get(item_id)
            row["observed_duration_label"] = (duration_label(row["duration_ms"])
                                              if row.get("duration_valid") else None)
            observed_code = label_codes.get(row["observed_duration_label"])
            row["duration_bucket_verified"] = bool(item_id in expected and observed_code is not None
                                                    and observed_code == expected[item_id])
            row["record_usable"] = bool(row["mapping_verified"] and row["semantic_text_valid"]
                                        and row.get("duration_valid") and row["duration_bucket_verified"])
            selected_by_id[item_id] = row
            selected = sorted(selected_by_id.values(), key=lambda value: value["item_id"])
            print(json.dumps({"item_id": item_id, "verified": row.get("exact_id_verified"),
                              "caption": row.get("caption", "")[:80]}, ensure_ascii=False), flush=True)
            selected_path.write_text(
                json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
            columns = ["encoded_item_id", "final_video_id", "caption", "cover_text", "duration",
                       "duration_ms", "mapping_verified", "duration_valid", "duration_issue",
                       "semantic_text_valid", "record_usable", "exact_id_verified", "overlap_verified",
                       "expected_duration_bucket", "observed_duration_label", "duration_bucket_verified",
                       "source_sha256", "reason"]
            with (args.root / "selected_captions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(selected)
            (args.root / "range_receipts.json").write_text(
                json.dumps(source.receipts, ensure_ascii=False, indent=2) + "\n")
        source.check_source_identity()
        requested_rows = [selected_by_id[item_id] for item_id in sorted(set(ids))]
        receipt = {
            "requested_item_ids": sorted(set(ids)), "request_sources": request_sources,
            "records_requested": len(set(ids)),
            "records_usable": sum(row["record_usable"] for row in requested_rows),
            "all_exact_ids_verified": all(row.get("exact_id_verified") for row in requested_rows),
            "source_identity": source.identity,
            "source_identity_rechecked_after_lookup": True,
            "mapping_receipt_sha256": sha256(mapping_path),
            "lookup_script_sha256": sha256(__file__),
            "output_files": [{"path": str(args.root / name), "sha256": sha256(args.root / name)}
                             for name in ["selected_captions.json", "selected_captions.csv", "range_receipts.json"]],
            "scope": "Requested records and cached HTTP ranges only; complete file MD5 not verified.",
        }
        (args.root / "selected_captions_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
