"""Verify captured artifact hashes, native instruction counts and serving stats."""

import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == record["sha256"], name
        if "uncompressed_sha256" in record:
            with gzip.open(root / name, "rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            assert actual == record["uncompressed_sha256"], name

    summary = json.loads((root / "summary.json").read_text())
    counts = summary["native_counts"]
    for role, column in [("baseline", 0), ("candidate", 2)]:
        body = (root / f"model-routing-{role}-body.llo").read_text()
        assert len(body.splitlines()) == counts["instruction_records"][column]
        for opcode, expected in counts.items():
            if opcode in {"columns", "instruction_records"}:
                continue
            actual = len(re.findall(r"\b" + re.escape(opcode) + r"\b", body))
            assert actual == expected[column], (role, opcode, actual,
                                                expected[column])

    samples = defaultdict(list)
    with (root / "serving-request-latencies.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            samples[(row["stage"], int(row["pair"]),
                     row["role"])].append(float(row["e2e_latency_ms"]))
    assert sum(map(len, samples.values())) == 780
    for stage, expected in summary["stages"].items():
        baseline, candidate = [], []
        for pair in range(expected["pairs"]):
            for role, target in [("baseline", baseline),
                                 ("candidate", candidate)]:
                values = samples[(stage, pair, role)]
                assert len(values) == 30
                target.append(np.mean(values))
        baseline, candidate = np.array(baseline), np.array(candidate)
        np.testing.assert_allclose(baseline.mean(),
                                   expected["baseline_ms"],
                                   rtol=0,
                                   atol=1e-9)
        np.testing.assert_allclose(candidate.mean(),
                                   expected["candidate_ms"],
                                   rtol=0,
                                   atol=1e-9)
        ratios = candidate / baseline
        indices = np.random.default_rng(1729).integers(0,
                                                       len(ratios),
                                                       size=(10000,
                                                             len(ratios)))
        ci = np.quantile(ratios[indices].mean(axis=1), [0.025, 0.975])
        np.testing.assert_allclose(ci,
                                   expected["mean_paired_ratio_ci95"],
                                   rtol=0,
                                   atol=1e-12)
        reduction = (1 - candidate.mean() / baseline.mean()) * 100
        print(
            f"{stage}: {baseline.mean():.6f} -> {candidate.mean():.6f} ms; "
            f"{reduction:.6f}% reduction; mean paired ratio CI95 {ci.tolist()}"
        )
    print(
        f"Verified {len(manifest)} captured artifacts, full decompressed LLO hashes, "
        "native callee instruction counts and 780 timed requests.")


if __name__ == "__main__":
    main()
