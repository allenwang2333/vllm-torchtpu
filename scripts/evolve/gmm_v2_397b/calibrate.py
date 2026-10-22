"""Judge the gmm_v2 397b scorer's calibration jobs (faults.py arms) and measure its noise.

    python calibrate.py <job1.json> [<job2.json> ...] [--configs a,b] [--tolerance T]

Each file is one rep's scorer stdout from a job carrying golden (control),
golden_aa, wrong_kernel and slow_kernel. Per job and config it requires:
golden correct within the tolerance and its op name equal to the served trace's
where the config names one; golden_aa with golden's checksum and dispatched
identity (the A/A ratio is the noise sample); wrong_kernel above the tolerance;
slow_kernel correct and slower than golden by more than the noise floor. The
noise floor is the widest |golden/golden_aa - 1| over every job and config.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path


def arms_of(out: dict) -> dict:
    return {a["name"]: a for a in out.get("arms") or []}


def traced_configs() -> set[str]:
    """bench.py's configs that name a served-trace op name, read without importing it."""
    tree = ast.parse((Path(__file__).parent / "bench.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CONFIGS":
            return {n for n, c in ast.literal_eval(node.value).items() if c.get("trace_signature")}
    raise SystemExit("bench.py has no CONFIGS literal")


def weighted_score(arm: dict, ref: dict, weights: dict[str, float]) -> float:
    """The campaign's geomean: prod (ref/arm)^w over the weighted configs, w normalised."""
    total = sum(weights.values())
    return math.prod((ref[c]["latency_us"] / arm[c]["latency_us"]) ** (w / total)
                     for c, w in weights.items())


def score_noise(outs: list[dict], weights: dict[str, float]) -> list[tuple[str, float]]:
    """golden_aa's weighted score against golden's reading in every job, its own and the others.

    The campaign scores a job without golden against golden's readings from
    earlier jobs, so the cross-job pairs are the noise a candidate is judged in.
    """
    out = []
    for i, oi in enumerate(outs, 1):
        aa = arms_of(oi)["golden_aa"]["configs"]
        for j, oj in enumerate(outs, 1):
            out.append((f"aa{i}/golden{j}", weighted_score(aa, arms_of(oj)["golden"]["configs"], weights)))
    return out


def judge(outs: list[dict], configs: list[str], tolerance: float) -> tuple[bool, list[str], float]:
    bad, lines, spreads = [], [], []
    traced = traced_configs()
    lines.append(f"{'job':<4}{'config':<18}{'golden us':>10}{'g err':>8}{'A/A':>8}"
                 f"{'wrong err':>11}{'slow ratio':>11}")
    slow_ratios = []
    for j, out in enumerate(outs, 1):
        arms = arms_of(out)
        missing = [n for n in ("golden", "golden_aa", "wrong_kernel", "slow_kernel") if n not in arms]
        if missing:
            bad.append(f"job {j}: no arm {', '.join(missing)}")
            continue
        g, aa, w, s = (arms[n]["configs"] for n in ("golden", "golden_aa", "wrong_kernel", "slow_kernel"))
        for c in configs:
            if not all(c in x for x in (g, aa, w, s)):
                bad.append(f"job {j} {c}: an arm has no result")
                continue
            gu = g[c]["latency_us"]
            aa_r, slow_r = gu / aa[c]["latency_us"], gu / s[c]["latency_us"]
            spreads.append(abs(aa_r - 1))
            slow_ratios.append((j, c, slow_r))
            lines.append(f"{j:<4}{c:<18}{gu:>10.2f}{g[c]['worst_error']:>8.4f}{aa_r:>8.4f}"
                         f"{w[c]['worst_error']:>11.3e}{slow_r:>11.4f}")
            if g[c]["worst_error"] > tolerance or g[c].get("declined"):
                bad.append(f"job {j} {c}: golden fails the tolerance {tolerance:g}")
            if c in traced and g[c].get("metrics", {}).get("signature_matches_trace") != 1.0:
                bad.append(f"job {j} {c}: golden's op name is not the served trace's")
            if aa[c]["checksum"] != g[c]["checksum"] or aa[c]["dispatched"] != g[c]["dispatched"]:
                bad.append(f"job {j} {c}: golden_aa does not reproduce golden's output or identity")
            if w[c]["worst_error"] <= tolerance:
                bad.append(f"job {j} {c}: wrong_kernel passes at {w[c]['worst_error']:.3e}")
            if s[c]["worst_error"] > tolerance:
                bad.append(f"job {j} {c}: slow_kernel fails correctness")
    floor = max(spreads) if spreads else float("nan")
    for j, c, r in slow_ratios:
        if r > 1 - floor:
            bad.append(f"job {j} {c}: slow_kernel at {r:.4f} is inside the noise floor {floor:.4f}")
    lines.append(f"noise floor (widest A/A spread over {len(spreads)} samples): {floor:.4f}")
    return not bad, lines + [f"FAIL: {b}" for b in bad], floor


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("outputs", nargs="+", type=Path)
    p.add_argument("--configs", default="decode_gate_up,decode_down,prefill_gate_up,prefill_down")
    p.add_argument("--tolerance", type=float, required=True)
    p.add_argument("--weights", default="", help="config=w,... : also report the weighted A/A score noise")
    a = p.parse_args(argv)
    outs = [json.loads(f.read_text()) for f in a.outputs]
    ok, lines, _ = judge(outs, a.configs.split(","), a.tolerance)
    print("\n".join(lines))
    if a.weights:
        weights = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))}
        pairs = score_noise(outs, weights)
        print("weighted A/A scores: " + ", ".join(f"{n} {v:.4f}" for n, v in pairs))
        print(f"score noise floor (widest |score-1|): {max(abs(v - 1) for _, v in pairs):.4f}")
    print("calibration PASSED" if ok else "calibration FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
