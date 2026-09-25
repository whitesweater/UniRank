"""Run each study test module in a fresh process to isolate compiler state.

A combined unittest process previously segfaulted in PyTorch compiler tests.
No test is skipped: every module must return zero, and full logs are retained.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts/rise_sample_analysis/validation" / f"isolated_tests_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONFAULTHANDLER="1")
    results = []
    for path in sorted((STUDY / "tests").glob("test_*.py")):
        command = [sys.executable, "-X", "faulthandler", "-m", "unittest", "discover", "-s", str(STUDY / "tests"), "-p", path.name, "-v"]
        with (output / f"{path.stem}.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        content = (output / f"{path.stem}.log").read_text()
        count = re.findall(r"Ran (\d+) tests? in", content)
        record = {"module": path.name, "returncode": result.returncode, "tests": int(count[-1]) if count else None}
        results.append(record)
        print(json.dumps(record), flush=True)
    receipt = {"created_utc": stamp, "process_isolation": True, "modules": results,
               "all_passed": all(r["returncode"] == 0 and r["tests"] is not None for r in results),
               "total_tests": sum(r["tests"] or 0 for r in results),
               "combined_process_failure_retained": "/tmp/rise_final_study_tests.txt",
               "scope": "All study modules run; isolation is required for compiler-state tests. Does not claim combined-process suite passes."}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (STUDY / "results/isolated_test_receipt.json").write_text(json.dumps({**receipt, "log_directory": str(output.relative_to(ROOT))}, indent=2) + "\n")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
