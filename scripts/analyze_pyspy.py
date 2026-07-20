#!/usr/bin/env python3
"""Aggregate py-spy speedscope profiles into host-time buckets per DP worker.

Buckets each weighted sample (by the deepest matching frame) into:
  BARRIER   - CPU-gloo DP all_reduce / distributed_c10d (the lockstep barrier)
  WAIT_DEV  - device sync / wait_for_copy / synchronize (blocked on TPU result)
  FORWARD   - model forward / executable dispatch / logits
  SAMPLE    - sampling / async copy launch
  OTHER     - everything else (prepare_inputs, python, update_states, ...)

We attribute each sample to the *most specific* bucket by scanning its stack
from leaf to root for signature substrings. Prints per-worker and mean %.

Usage: analyze_pyspy.py <dir_with_speedscope_json | file...>
"""
import glob
import json
import os
import sys

# Order matters: first match (leaf->root) wins per sample.
SIGS = [
    ("BARRIER", ("all_reduce", "distributed_c10d", "processgroupgloo",
                 "_dp_coordinated_step", "pg_wrapper", "->allreduce",
                 "work.wait", "reducescatter", "allgather_into", "_allgather")),
    ("WAIT_DEV", ("wait_for_copy", "synchronize", "_modify_prev_results",
                  "mark_step", "_xla_sync", "wait_device_ops", "torch_tpu_sync",
                  "block_until_ready", "transfertohost", "to_cpu", ".item")),
    ("SAMPLE", ("sample_tokens", "sampler", "sample(", "async_copy",
                "asynctpucopy", "_async_output")),
    ("FORWARD", ("forward_model", "execute_model", "compute_selected_logits",
                 "forward(", "decoder", "attention", "moe", "mlp",
                 "compiledmodel", "__call__", "compute_logits")),
]


def bucket_of(stack_names):
    """stack_names: leaf-first list of frame 'name' strings (lowercased)."""
    for leaf in stack_names:  # leaf -> root
        for label, subs in SIGS:
            for s in subs:
                if s in leaf:
                    return label
    return "OTHER"


def load_speedscope(path):
    with open(path) as f:
        doc = json.load(f)
    shared = doc.get("shared", {})
    frames = shared.get("frames", [])
    fnames = []
    for fr in frames:
        nm = (fr.get("name") or "")
        fl = (fr.get("file") or "")
        fnames.append((nm + " " + fl).lower())
    return doc, fnames


def analyze_profile(path):
    doc, fnames = load_speedscope(path)
    profiles = doc.get("profiles", [])
    # py-spy 'sampled' profiles: each has samples=[[frameidx,...],...] leaf-last,
    # and weights=[w,...]. Thread name in profile['name'].
    buckets = {}
    total = 0.0
    # Pick the busiest thread (max total weight) = worker main loop.
    best = None
    for pf in profiles:
        if pf.get("type") != "sampled":
            continue
        w = sum(pf.get("weights", []))
        if best is None or w > best[0]:
            best = (w, pf)
    if best is None:
        return None
    pf = best[1]
    samples = pf.get("samples", [])
    weights = pf.get("weights", [])
    for stack, wt in zip(samples, weights):
        # speedscope samples are leaf-LAST; reverse to leaf-first.
        names = [fnames[i] for i in reversed(stack) if 0 <= i < len(fnames)]
        b = bucket_of(names)
        buckets[b] = buckets.get(b, 0.0) + wt
        total += wt
    return buckets, total, pf.get("name", "?")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.speedscope.json")))
        else:
            files.append(a)
    if not files:
        print("no speedscope json found")
        sys.exit(2)

    order = ["BARRIER", "WAIT_DEV", "FORWARD", "SAMPLE", "OTHER"]
    agg = {k: 0.0 for k in order}
    aggtot = 0.0
    print(f"{'worker':<28}" + "".join(f"{k:>10}" for k in order))
    rows = []
    for fp in files:
        res = analyze_profile(fp)
        if not res:
            print(f"{os.path.basename(fp):<28}  (no samples)")
            continue
        buckets, total, thr = res
        rows.append((fp, buckets, total))
        for k in order:
            agg[k] += buckets.get(k, 0.0)
        aggtot += total
        pcts = "".join(f"{100*buckets.get(k,0.0)/total:9.1f}%" for k in order)
        print(f"{os.path.basename(fp)[:28]:<28}{pcts}")
    if aggtot:
        print("-" * (28 + 10 * len(order)))
        pcts = "".join(f"{100*agg[k]/aggtot:9.1f}%" for k in order)
        print(f"{'MEAN (weighted)':<28}{pcts}")
        print(f"\nBARRIER mean = {100*agg['BARRIER']/aggtot:.1f}%  "
              f"(prior claim: ~50-58% prefill)")


if __name__ == "__main__":
    main()
