# SPDX-License-Identifier: Apache-2.0
"""Publish the megakernel step bench's last result row as the cdk job result (for cdk dash).

    python scripts/vllm/benchmark/mk_bench_publish.py KERNEL [CASE]

Reads the newest results.jsonl under $P3_OUT_ROOT/<job id>/ (test_p4_a2_mk_bench.py's run
directory), takes the last row (of CASE when given), and stores
{"kernel", "case", "variant", "metrics": {"tokens", "step_ms", "tok_per_s", ...}} with
result-updater (fetched from gs://cloud-devkit/tools when the image lacks it).
"""

import glob
import json
import os
import shutil
import subprocess
import sys


def main():
    kernel = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p3_bench_common as common  # the bench's run directory: <P3_OUT_ROOT>/<job id>/..

    root = os.environ.get("P3_OUT_ROOT") or common.DEFAULT_OUT_ROOT
    files = sorted(glob.glob(os.path.join(root, common.job_id(), "*", "results.jsonl")),
                   key=os.path.getmtime)
    rows = []
    for f in files:
        with open(f) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if "loop_ms" in r and (case is None or r.get("case") == case)]
    if not rows:
        print("mk_bench_publish: no result row", flush=True)
        return 1
    r = rows[-1]
    calls = r["rows"] if isinstance(r["rows"], list) else [r["rows"]]
    tokens = int(sum(calls))
    out = {
        "kernel": kernel,
        "case": r["case"],
        "variant": r["variant"],
        "environment_id": r.get("environment_id"),
        "metrics": {
            "tokens": tokens,
            "step_ms": r["loop_ms"],
            "tok_per_s": round(tokens / r["loop_ms"] * 1000.0, 1),
            "calls": len(calls),
            "single_ms_min": min(r["single_ms"]),
            "hidden_finite": r.get("hidden_finite"),
        },
    }
    if r.get("cores_agree") is not None:
        out["metrics"]["cores_agree"] = r["cores_agree"]
    print("==> result " + json.dumps(out), flush=True)
    upd = shutil.which("result-updater")
    if upd is None:
        subprocess.run(["gcloud", "storage", "cp", "gs://cloud-devkit/tools/result-updater",
                        "/tmp/result-updater"], check=False, capture_output=True)
        if os.path.exists("/tmp/result-updater"):
            os.chmod("/tmp/result-updater", 0o755)
            upd = "/tmp/result-updater"
    if upd is None:
        print("mk_bench_publish: no result-updater; result not published", flush=True)
        return 1
    return subprocess.run([upd, json.dumps(out)]).returncode


if __name__ == "__main__":
    sys.exit(main())
