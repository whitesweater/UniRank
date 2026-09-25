"""Fail before training if the reviewed source/configuration snapshot changed."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
snapshot = json.loads(Path(__file__).with_name("preflight.json").read_text())
differences = [name for name, digest in snapshot["source_sha256"].items()
               if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
if differences:
    raise SystemExit(f"LOW_SEED_RETRY_SOURCE_CHANGED: {differences}")
print(f"LOW_SEED_RETRY_FROZEN_OK revision={snapshot['git_revision']}", flush=True)
