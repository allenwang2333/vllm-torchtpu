#!/usr/bin/env python3
"""Attribute a TPU xplane profile by HLO op category.

Usage: python3 scripts/analyze_profile.py <profile_dir_or_xplane.pb>

Buckets HLO self-time into: MXU matmul (dot/convolution), collectives
(all-gather / reduce-scatter / all-reduce / all-to-all / collective-permute),
Pallas/custom kernels (custom-call -> GDN, RPA, quantized matmul), elementwise
(fusion), layout (copy/transpose/bitcast), and other. Prints the split plus the
top ops, so we can see the DP collective tax vs GMM vs GDN and whether
collectives are exposed.
"""
import glob
import json
import os
import sys


def find_xspace(path):
    if os.path.isfile(path):
        return [path]
    pats = ["**/*.xplane.pb", "**/*.xspace.pb", "**/*.pb"]
    hits = []
    for p in pats:
        hits += glob.glob(os.path.join(path, p), recursive=True)
    # prefer xplane
    hits = sorted(set(hits), key=lambda f: (0 if "xplane" in f else 1, f))
    return hits


CATEGORY = [
    ("matmul(MXU)", ("convolution", "dot", "fusion.*conv", "%convolution")),
    ("collective", ("all-gather", "all-reduce", "reduce-scatter", "all-to-all",
                    "collective-permute", "collective", "async")),
    ("custom/Pallas", ("custom-call", "custom_call")),
    ("elementwise(fusion)", ("fusion", "loop", "reduce", "map")),
    ("layout(copy)", ("copy", "transpose", "bitcast", "reshape", "data formatting")),
]


def classify(category, name):
    s = (str(category) + " " + str(name)).lower()
    # collectives first (they can appear as fusions)
    for key in ("all-gather", "all-reduce", "reduce-scatter", "all-to-all",
                "collective-permute", "collective"):
        if key in s:
            return "collective"
    if "custom-call" in s or "custom_call" in s:
        return "custom/Pallas"
    if "convolution" in s or s.strip().startswith("dot") or " dot" in s or "%dot" in s or "matmul" in s:
        return "matmul(MXU)"
    if "copy" in s or "transpose" in s or "bitcast" in s or "data formatting" in s:
        return "layout(copy)"
    if "fusion" in s or "reduce" in s or "loop" in s or "map" in s or "elementwise" in s:
        return "elementwise(fusion)"
    return "other"


def rows_from_json(obj):
    """hlo_stats/framework_op_stats return a JSON table; normalize to dicts."""
    if isinstance(obj, (bytes, str)):
        obj = json.loads(obj)
    # gviz-like: {cols:[{id/label}], rows:[{c:[{v:..}]}]}
    if isinstance(obj, dict) and "rows" in obj and "cols" in obj:
        cols = [c.get("id") or c.get("label") for c in obj["cols"]]
        out = []
        for r in obj["rows"]:
            vals = [c.get("v") for c in r.get("c", [])]
            out.append(dict(zip(cols, vals)))
        return out
    if isinstance(obj, list):
        return obj
    # some tools: {"data": [...]} or {"opStats": [...]}
    for k in ("data", "opStats", "hloStats", "ops"):
        if isinstance(obj, dict) and k in obj:
            return obj[k]
    return []


def pick(d, *names):
    for n in names:
        for k in d:
            if k and k.lower() == n.lower():
                return d[k]
    # fuzzy
    for n in names:
        for k in d:
            if k and n.lower() in k.lower():
                return d[k]
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    xs = find_xspace(sys.argv[1])
    if not xs:
        print("No xplane/xspace .pb found under", sys.argv[1])
        sys.exit(2)
    print("xspace files:", *[os.path.relpath(x) for x in xs], sep="\n  ")
    from xprof.convert import raw_to_tool_data as r

    tool = "hlo_stats"
    try:
        data = r.xspace_to_tool_data(xs, tool, {})
    except Exception as e:
        print(f"hlo_stats failed ({e}); trying framework_op_stats")
        tool = "framework_op_stats"
        data = r.xspace_to_tool_data(xs, tool, {})
    # xspace_to_tool_data returns (body_bytes, mimetype_str)
    if isinstance(data, tuple):
        data = data[0]
    rows = rows_from_json(data)
    if not rows:
        print("No rows parsed. Raw head:", str(data)[:500])
        sys.exit(3)
    print(f"[{tool}] {len(rows)} rows; sample keys: {list(rows[0].keys())}")

    buckets = {}
    total = 0.0
    top = []
    for d in rows:
        cat = pick(d, "category", "hlo_category", "Category")
        name = pick(d, "hlo_op_name", "op_name", "name", "Operation", "hlo_op_expression")
        t = pick(d, "total_self_time", "self_time", "total_self_time_ps",
                 "self_time_ps", "TotalSelfTimeInNs", "total_self_time_in_us",
                 "occurrences_total_self_time_us")
        try:
            t = float(t)
        except (TypeError, ValueError):
            continue
        b = classify(cat, name)
        buckets[b] = buckets.get(b, 0.0) + t
        total += t
        top.append((t, b, str(cat), str(name)[:60]))

    print(f"\n=== HLO self-time by category (total={total:.3g}) ===")
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {b:22s} {v:14.3g}  {100*v/total:5.1f}%")
    print("\n=== top 25 ops by self-time ===")
    for t, b, cat, name in sorted(top, reverse=True)[:25]:
        print(f"  {100*t/total:5.1f}%  {b:20s} {cat:18s} {name}")


if __name__ == "__main__":
    main()
