#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-op time breakdown of a vllm_torchtpu torch-profiler capture (.xplane.pb).

    python3 scripts/vllm/benchmark/xplane_ops.py <dir or file> [--steps N] [--top 80]

For each capture file: the planes and lines it holds, then for the first TPU
device plane the "XLA Ops" events aggregated by op name (total ms, count, ms per
step) and by category (hlo_category stat, else a name heuristic), plus the "XLA
Modules" (programs) aggregated the same way. --steps divides totals to per-step.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
try:
    from vllm_torchtpu.tools import xplane_pb2
except Exception:  # standalone: proto module next to this file
    import xplane_pb2  # type: ignore


def _stat_val(s):
    w = s.WhichOneof("value")
    return getattr(s, w) if w else None


def _ev_stats(plane, ev_or_md):
    out = {}
    for s in ev_or_md.stats:
        md = plane.stat_metadata.get(s.metadata_id)
        name = md.name if md else str(s.metadata_id)
        v = _stat_val(s)
        if s.WhichOneof("value") == "ref_value":
            r = plane.stat_metadata.get(v)
            v = r.name if r else v
        out[name] = v
    return out


def summarize(path, steps, top, dump_names):
    xs = xplane_pb2.XSpace()
    with open(path, "rb") as f:
        xs.ParseFromString(f.read())
    res = {"file": path, "planes": []}
    dev = None
    for p in xs.planes:
        lines = []
        for ln in p.lines:
            tot = sum(e.duration_ps for e in ln.events) / 1e9
            lines.append((ln.name or ln.display_name, len(ln.events), round(tot, 3)))
        res["planes"].append({"name": p.name, "lines": lines})
        if dev is None and re.match(r"^/device:TPU:\d+$", p.name):
            dev = p
    if dev is None:
        return res
    res["device_plane"] = dev.name
    by_line = {}
    for ln in dev.lines:
        by_line.setdefault(ln.name or ln.display_name, []).append(ln)
    out = {}
    for lname in ("XLA Modules", "XLA Ops", "Steps", "XLA TraceMe", "SparseCore"):
        lns = by_line.get(lname)
        if not lns:
            continue
        agg = collections.defaultdict(lambda: [0.0, 0])
        cat = collections.defaultdict(lambda: [0.0, 0])
        base = collections.defaultdict(lambda: [0.0, 0])
        prog = collections.defaultdict(lambda: [0.0, 0])
        span = [None, None]
        for ln in lns:
            for e in ln.events:
                md = dev.event_metadata.get(e.metadata_id)
                name = (md.display_name or md.name) if md else str(e.metadata_id)
                ms = e.duration_ps / 1e9
                agg[name][0] += ms
                agg[name][1] += 1
                t0 = ln.timestamp_ns * 1e3 + e.offset_ps
                t1 = t0 + e.duration_ps
                span[0] = t0 if span[0] is None else min(span[0], t0)
                span[1] = t1 if span[1] is None else max(span[1], t1)
                if lname == "XLA Ops":
                    st = _ev_stats(dev, md) if md else {}
                    c = st.get("hlo_category") or _guess_cat(name)
                    cat[c][0] += ms
                    cat[c][1] += 1
                    b = re.sub(r"[._]\d+$", "", name)
                    b = re.sub(r"\.\d+$", "", b)
                    base[(c, b)][0] += ms
                    base[(c, b)][1] += 1
                    pid = str(st.get("program_id", "?"))
                    prog[pid][0] += ms
                    prog[pid][1] += 1
        tot = sum(v[0] for v in agg.values())
        rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
        out[lname] = {
            "total_ms": round(tot, 3),
            "span_ms": round((span[1] - span[0]) / 1e9, 3) if span[0] is not None else 0,
            "n_names": len(agg),
            "top": [(n, round(v[0], 3), v[1], round(v[0] / steps, 4)) for n, v in rows[:top]],
        }
        if base:
            out[lname]["by_base"] = sorted(
                ((f"{c} | {b}", round(v[0], 3), v[1], round(v[0] / steps, 4)) for (c, b), v in base.items()),
                key=lambda r: -r[1],
            )[:top]
            out[lname]["by_program"] = sorted(
                ((pid, round(v[0], 3), v[1], round(v[0] / steps, 4)) for pid, v in prog.items()),
                key=lambda r: -r[1],
            )[:20]
        if cat:
            out[lname]["by_category"] = sorted(
                ((c, round(v[0], 3), v[1], round(v[0] / steps, 4)) for c, v in cat.items()),
                key=lambda r: -r[1],
            )
    res["lines"] = out
    if dump_names:
        # metadata for the top op names: every stat the profiler attached
        ops = by_line.get("XLA Ops", [])
        names = {r[0] for r in out.get("XLA Ops", {}).get("top", [])[:dump_names]}
        seen = {}
        for ln in ops:
            for e in ln.events:
                md = dev.event_metadata.get(e.metadata_id)
                if not md:
                    continue
                name = md.display_name or md.name
                if name in names and name not in seen:
                    st = _ev_stats(dev, md)
                    st.update({"ev." + k: v for k, v in _ev_stats(dev, e).items()})
                    seen[name] = {k: (str(v)[:400]) for k, v in st.items()}
        res["op_meta"] = seen
    return res


def _guess_cat(name):
    n = name.lower()
    for key, c in (
        ("all-reduce", "all-reduce"), ("all-gather", "all-gather"),
        ("reduce-scatter", "reduce-scatter"), ("all-to-all", "all-to-all"),
        ("collective-permute", "collective-permute"), ("custom-call", "custom-call"),
        ("convolution", "convolution"), ("dot", "dot"), ("fusion", "fusion"),
        ("copy", "copy"), ("dynamic-update-slice", "dus"), ("gather", "gather"),
        ("scatter", "scatter"), ("sort", "sort"),
    ):
        if key in n:
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--steps", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--dump-names", type=int, default=40)
    ap.add_argument("--max-files", type=int, default=2)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    files = [a.path] if os.path.isfile(a.path) else sorted(
        glob.glob(os.path.join(a.path, "**", "*.xplane.pb"), recursive=True))
    print(f"XPLANE files: {len(files)}")
    for f in files:
        print("  ", f, os.path.getsize(f))
    # rank 0 first when the names carry it
    files.sort(key=lambda f: (0 if re.search(r"rank0[^0-9]", os.path.basename(f)) else 1, f))
    allres = []
    for f in files[: a.max_files]:
        r = summarize(f, a.steps, a.top, a.dump_names)
        allres.append(r)
        print(f"===== {f}")
        for p in r["planes"]:
            print(f"  plane {p['name']}: " + "; ".join(f"{n}({c}, {t}ms)" for n, c, t in p["lines"][:12]))
        for lname, d in r.get("lines", {}).items():
            print(f"  --- {r['device_plane']} / {lname}: total {d['total_ms']} ms, span "
                  f"{d['span_ms']} ms, {d['n_names']} names (per step = /{a.steps})")
            for row in d.get("by_category", []):
                print(f"      CAT {row[0]:<28} {row[1]:>10.3f} ms  n={row[2]:<6} /step {row[3]:.4f}")
            for row in d.get("by_program", []):
                print(f"      PROG {row[0]:<24} {row[1]:>10.3f} ms  n={row[2]:<6} /step {row[3]:.4f}")
            for row in d.get("by_base", []):
                print(f"      BASE {row[1]:>10.3f} ms  n={row[2]:<6} /step {row[3]:>8.4f}  {row[0][:140]}")
            for row in d["top"]:
                print(f"      {row[1]:>10.3f} ms  n={row[2]:<6} /step {row[3]:>8.4f}  {row[0][:150]}")
        for name, st in r.get("op_meta", {}).items():
            print(f"  META {name[:80]}: {json.dumps(st)[:900]}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(allres, f)


if __name__ == "__main__":
    main()
