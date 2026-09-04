#!/usr/bin/env python3
"""Emit one arm's ladder result in the reviewer's contract shape.

Reads every rep's own `vllm bench serve` result JSON -- never a summary, never a
mean of three -- and reports the median of the per-rep medians with the spread
beside it. A number without a spread is how this campaign once quoted a 3% win
that was noise.

Usage: k3r5_arm_json.py <arm> <arm results dir> <out.json> [k=v ...]
  k=v are merged into the top level (branch, sha, weights, gate_env,
  compilation_seconds, cache_cleared, sanity_verdict, notes).
"""
import glob, json, os, statistics, sys

# #365, same ladder, published. Deltas are reported against BOTH this and the
# build's own `base` arm: #365 is another branch and image, and c=8's prompt
# count moved 16 -> 32.
REF_TTFT = {1: 995.2, 2: 1449.2, 4: 3287.4, 8: 4508.3, 16: 9154.8, 32: 6121.8}
REF_TPOT = {1: 18.954, 2: 19.405, 4: 21.128, 8: 28.420, 16: 47.083, 32: 82.705}


def spread(xs):
    return (max(xs) - min(xs)) / statistics.median(xs) * 100 if len(xs) > 1 else 0.0


def main():
    arm, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = dict(kv.split("=", 1) for kv in sys.argv[4:] if "=" in kv)

    reps, profiled = {}, []
    for f in glob.glob(os.path.join(root, "**", "isl*_osl*_c*.json"), recursive=True):
        if ".warmup" in f:
            continue
        # The profiled capture cell is a named artifact, not a rep: it runs with
        # --profile and a different prompt count, and averaging it into the
        # medians would quietly move the number the arm is judged on.
        if os.sep + "profile_c1" + os.sep in f:
            profiled.append(f)
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        reps.setdefault(int(d["max_concurrency"]), []).append(d)

    cells, nprompts = {}, {}
    for c, ds in sorted(reps.items()):
        ttft = [d["median_ttft_ms"] for d in ds]
        tpot = [d["median_tpot_ms"] for d in ds]
        cell = {
            "reps": len(ds),
            "ttft_ms": round(statistics.median(ttft), 3),
            "tpot_ms": round(statistics.median(tpot), 4),
            "out_tok_s": round(statistics.median(d["output_throughput"] for d in ds), 2),
            "total_tok_s": round(statistics.median(
                d.get("total_token_throughput", 0.0) for d in ds), 2),
            "ttft_spread_pct": round(spread(ttft), 3),
            "tpot_spread_pct": round(spread(tpot), 3),
            "completed": ds[0].get("completed"),
        }
        if c in REF_TTFT:
            cell["d_ttft_pct_vs_365"] = round((cell["ttft_ms"] / REF_TTFT[c] - 1) * 100, 2)
            cell["d_tpot_pct_vs_365"] = round((cell["tpot_ms"] / REF_TPOT[c] - 1) * 100, 2)
        cells[str(c)] = cell
        nprompts[str(c)] = ds[0].get("num_prompts")

    # Deltas against this build's own base arm, when it exists next to us.
    base_path = os.path.join(os.path.dirname(out), "base.json")
    if arm != "base" and os.path.exists(base_path):
        try:
            base = json.load(open(base_path))["cells"]
            for c, cell in cells.items():
                if c in base:
                    cell["d_ttft_pct_vs_base"] = round(
                        (cell["ttft_ms"] / base[c]["ttft_ms"] - 1) * 100, 2)
                    cell["d_tpot_pct_vs_base"] = round(
                        (cell["tpot_ms"] / base[c]["tpot_ms"] - 1) * 100, 2)
        except Exception as e:
            print(f"[arm_json] could not read base arm: {e}", file=sys.stderr)

    doc = {
        "arm": arm,
        "branch": extra.get("branch", "geyuhao/k3r5-ladder"),
        "sha": extra.get("sha", ""),
        "weights": extra.get("weights", "real"),
        "num_prompts": nprompts,
        "gate": {
            "env": extra.get("gate_env", ""),
            "compilation_seconds": float(extra["compilation_seconds"])
            if extra.get("compilation_seconds", "").replace(".", "", 1).isdigit() else None,
            "cache_cleared": extra.get("cache_cleared", "true") == "true",
        },
        "cells": cells,
        "sanity": {"verdict": extra.get("sanity_verdict", "NOT_RUN"),
                   "worst_distinct_ratio": float(extra["worst_distinct_ratio"])
                   if "worst_distinct_ratio" in extra else None},
        "profile_capture_cells": sorted(os.path.relpath(f, root) for f in profiled),
        "notes": extra.get("notes", ""),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"[arm_json] wrote {out}: {len(cells)} cells "
          f"({', '.join('c%s tpot %.3f' % (c, v['tpot_ms']) for c, v in cells.items())})")


if __name__ == "__main__":
    main()
