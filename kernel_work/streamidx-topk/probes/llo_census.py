"""R1c -- static opcode census across the three `compute`-mode arms.

Reproduces the three tables in `R1c.md` from the Mosaic dump written by
`probe_llo_capture.py` (job `j-a78a5081-7f73-42ad-b483`):

  §1   `tpu.relayout` per Mosaic pass, and grouped by signature
  §2.2 the opcode census at `post-apply-vector-layout-simplify` (-0013)
  §2.1 the op delta by stage, incl. `post-finalize-llo` -- the one to quote
  §2.3 `tpu.load` sublane masks -- the actual payload-width test
  §2   the `tpu.transpose` shapes

It bears on: is the 973.9 ns/block that R1b measured between the incumbent and
Option A attributable to

  (T) the eliminated lane transpose, or
  (W) the halved payload width?

**The three arms cannot settle that. A FOURTH ARM CAN, and it has been run** --
pass `--arms inc,a_reshape,w_reshape` against job `j-a0566857-c6f2-4667-9eed`'s
dump and this script prints the DISCRIMINATOR table at the bottom. Result: on
every instruction and code-size metric `w_reshape` lands on `a_reshape`
(position <= +0.032), and on every byte and footprint metric it lands on `inc`
(+1.008). (T) on instructions; (W) owns the bytes. See `R1c.md` section 7 --
including the part section 7 is careful about, which is that a BUNDLE share is
not a share of the measured 973.9 ns.

Two corrections that got the three-arm version here, both from the
`/critical-review` of R1c's first draft:

* **Vector-load COUNT is not a test for (W).** The counts are near-equal (9,296
  vs 9,368) but the incumbent's loads carry `sublanes [true,true,...]` against
  Option A's `[true,false,...]`: 1024 B vs 512 B per load, 1.984x the bytes in
  the same instruction count. This script prints that mask breakdown now. (W)
  costs no extra vector ops and ~2x the VMEM read bytes; a static census cannot
  price the bytes.
* **The three arms are collinear** in scale placement and record width, so
  neither hypothesis can be isolated from them at all. `R1c.md` section 4.1
  specified the missing fourth arm; section 7 reports it.

What the census DOES show: the entire instruction-count delta is on the scale
path. Quote the `post-finalize-llo` delta (+4,926 ops), not the Mosaic-MLIR one
(+45,983) -- the MLIR at `post-apply-vector-layout-simplify` is ~9x larger and
is dominated by IR that never executes.

EVERYTHING THIS PRINTS IS STATIC COMPILE-TIME ANALYSIS -- opcode counts, not
measured time. Label it as such wherever it is quoted and never send it to
`/log-experiment`.

The LLO half of R1c (bundles, spills, VMEM, XLU occupancy) is not computed
here; it comes from `xprof-cli get_llo_fit_summary --program=tpu_custom_call.1
--bypass_cache=True` against each arm's `jf/` subdirectory.

Usage:
    python3 llo_census.py --root <synced>/llo [--arms inc,a_reshape,a_concat]
    python3 llo_census.py --root <synced>/llo --arms inc,a_reshape,w_reshape
"""

import argparse
import collections
import os
import re
import sys

# An op OCCURRENCE, not a mere token: either `%res = dialect.op` or a
# statement-leading `dialect.op`. Matching bare `dialect.op` anywhere on the
# line -- the obvious thing -- also counts memref layout ATTRIBUTES such as
# `tpu.tiled` and `tpu.memory_space`, which are not ops and which inflate every
# arm's total by ~20k. R1c's numbers come from this regex.
OP = re.compile(r'(?:=\s*|^\s*)([a-z_][a-z_0-9]*\.[a-z_][a-z_0-9]*)\b')

# The ops that discriminate (T) from (W); always shown even when rare.
ALWAYS = ("tpu.transpose", "tpu.relayout", "tpu.load", "tpu.matmul",
          "tpu.dynamic_rotate", "tpu.roll_vectors", "tpu.unroll_vectors")

PASSES = ("post-infer-vector-layout", "post-relayout-insertion",
          "post-apply-vector-layout")


def kernel_episode(mosaic_dir):
    """The arm's kernel compilation episode.

    An arm that ran first in the job also has earlier SparseCore/setup
    episodes in the same directory -- a run-order artifact, not a layout
    property. The kernel episode is the LAST one by id.
    """
    eps = collections.defaultdict(list)
    for fn in sorted(os.listdir(mosaic_dir)):
        m = re.match(r'(\d+)-\d+-mosaic-dump-', fn)
        if m:
            eps[m.group(1)].append(fn)
    if not eps:
        return None, []
    ep = sorted(eps)[-1]
    return ep, eps[ep]


def pass_file(mosaic_dir, files, label):
    """Resolve a pass label to its dump file.

    Substring matching with `hits[-1]` silently resolves
    `post-apply-vector-layout` to `...-simplify`, which is a DIFFERENT pass
    with a delta 15 % smaller. R1c's first draft mislabelled its main table
    that way. Prefer the file whose label is EXACTLY `label`.
    """
    exact = [f for f in files
             if re.match(r'\d+-\d+-mosaic-dump-\w+-' + re.escape(label)
                         + r'\.txt$', f)]
    if exact:
        return os.path.join(mosaic_dir, exact[-1])
    hits = [f for f in files if label in f]
    return os.path.join(mosaic_dir, hits[-1]) if hits else None


# One VMEM sublane slot is 128 lanes x 4 B, whatever the element type: an f32
# row is 128x4 B directly, and a (4,128)(4,1)-tiled i8 row packs 4 logical rows
# into the same 512 B. So a masked load of `nsub` sublanes moves nsub*512 B
# regardless of dtype.
#
# [rev] An earlier version of this file computed `nsub * 128 * (4 if "x4x" in
# mref else 1)`. That happened to be right for the packed-i8 PAYLOAD memrefs --
# which is where the whole (W) argument lives -- and 4x LOW for every f32 load
# and for `memref<1x32x128xi8>`, which lacks the literal "x4x". The error was
# common-mode across arms (all three issue the identical 36+36+4 small loads),
# so every discriminator POSITION was unaffected; the absolute per-block totals
# were low by 233,472 B each and the inc/a all-loads ratio read 1.97x instead
# of 1.92x. Caught in review of R1c section 7.
SUBLANE_BYTES = 128 * 4


def load_masks(path):
    """`tpu.load` grouped by sublane mask and memref -- the (W) test.

    The COUNT of loads is not a test for payload width; the MASK is. A
    2-sublane masked load moves twice the bytes of a 1-sublane one in the same
    instruction.
    """
    c = collections.Counter()
    pat = re.compile(r'tpu\.load\b.*?sublanes\s*\[([^\]]*)\].*?:\s*(memref<[^,]*)')
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                mask = [t.strip() for t in m.group(1).split(",")]
                c[(sum(t == "true" for t in mask), m.group(2))] += 1
    return c


def count_ops(path):
    c = collections.Counter()
    with open(path, errors="replace") as fh:
        for line in fh:
            for m in OP.finditer(line):
                c[m.group(1)] += 1
    return c


def relayout_signatures(path):
    """`tpu.relayout` grouped by (in_layout, out_layout, vector type)."""
    c = collections.Counter()
    pat = re.compile(r'tpu\.relayout\s+[^:]*:\s*(vector<[^>]*>)')
    lay = re.compile(r'(in_layout|out_layout)\s*=\s*\[?(#tpu\.vpad<"[^"]*">)')
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            ls = dict(lay.findall(line))
            c[(m.group(1), ls.get("in_layout", "?"),
               ls.get("out_layout", "?"))] += 1
    return c


def table(rows, headers):
    w = [max(len(str(r[i])) for r in ([headers] + rows))
         for i in range(len(headers))]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the synced llo/ directory")
    ap.add_argument("--arms", default="inc,a_reshape,a_concat")
    args = ap.parse_args()
    arms = args.arms.split(",")

    print("=" * 78)
    print("STATIC COMPILE-TIME OPCODE CENSUS -- not measured time")
    print("=" * 78)

    files = {}
    for a in arms:
        md = os.path.join(args.root, a, "mosaic")
        if not os.path.isdir(md):
            print(f"[{a}] NO MOSAIC DIR at {md}")
            continue
        ep, fs = kernel_episode(md)
        print(f"[{a}] kernel episode {ep}, {len(fs)} files "
              f"({len(os.listdir(md))} in dir)")
        files[a] = (md, fs)
    present = [a for a in arms if a in files]
    if not present:
        return 2

    # --- R1c section 1: relayout per pass -----------------------------------
    print("\n" + "=" * 78)
    print("tpu.relayout by pass. NOTE: 0 at post-infer-vector-layout is NOT")
    print("evidence of anything -- relayout-insertion is the NEXT pass, so")
    print("this check cannot fail. The 4 tpu.transpose in inc ARE author-")
    print("written; they are present in pass -0001 (original). What the")
    print("compiler adds is the relayout that feeds them.")
    print("=" * 78)
    rows = []
    for label in PASSES:
        r = [label]
        for a in present:
            md, fs = files[a]
            p = pass_file(md, fs, label)
            r.append(count_ops(p).get("tpu.relayout", 0) if p else "-")
        rows.append(r)
    table(rows, ["pass"] + present)

    print("\n" + "=" * 78)
    print("tpu.relayout signatures at post-relayout-insertion")
    print("The PAYLOAD relayout is identical across arms => the width costs")
    print("no extra vector OPS. It still costs 2x the BYTES: see the load-")
    print("mask table below. The SCALE relayout is the only structural diff.")
    print("=" * 78)
    sig = {}
    for a in present:
        md, fs = files[a]
        p = pass_file(md, fs, "post-relayout-insertion")
        sig[a] = relayout_signatures(p) if p else collections.Counter()
    keys = sorted(set().union(*[set(s) for s in sig.values()]),
                  key=lambda k: -max(sig[a][k] for a in present))
    for k in keys:
        print(f"  {k[0]}")
        print(f"    in  {k[1]}")
        print(f"    out {k[2]}")
        print("    " + "  ".join(f"{a}={sig[a][k]}" for a in present))

    # --- R1c section 2.1: expanded-IR census --------------------------------
    print("\n" + "=" * 78)
    print("opcode census at post-apply-vector-layout-SIMPLIFY (-0013)")
    print("structural reading only -- NOT a cost model: ~9x the final-LLO")
    print("delta below, dominated by IR that never executes")
    print("=" * 78)
    cnt = {}
    for a in present:
        md, fs = files[a]
        p = pass_file(md, fs, "post-apply-vector-layout-simplify")
        cnt[a] = count_ops(p) if p else collections.Counter()
    base = present[1] if len(present) > 1 else present[0]
    keys = sorted(set().union(*[set(c) for c in cnt.values()]),
                  key=lambda k: -max(cnt[a][k] for a in present))
    rows, tot = [], [0] * len(present)
    for k in keys:
        v = [cnt[a][k] for a in present]
        tot = [tot[i] + v[i] for i in range(len(v))]
        if max(v) >= 100 or k in ALWAYS:
            rows.append([k] + v + [f"{v[0]-v[1]:+d}" if len(v) > 1 else ""])
    rows.append(["TOTAL OPS"] + tot
                + [f"{tot[0]-tot[1]:+d}" if len(tot) > 1 else ""])
    table(rows, ["op"] + present + [f"{present[0]}-{base}"])

    # --- the FINAL-LLO delta: the number to quote ---------------------------
    # The Mosaic MLIR above is ~9x larger than the code that runs. Quote this.
    print("\n" + "=" * 78)
    print("op deltas by stage -- QUOTE post-finalize-llo, not the MLIR passes")
    print("=" * 78)
    rows = []
    for label in ("post-apply-vector-layout",
                  "post-apply-vector-layout-simplify",
                  "post-finalize-llo"):
        r, per = [label], {}
        for a in present:
            md, fs = files[a]
            p = pass_file(md, fs, label)
            per[a] = count_ops(p) if p else collections.Counter()
            r.append(sum(per[a].values()))
        r.append(f"{r[1]-r[2]:+d}" if len(present) > 1 else "")
        rows.append(r)
        if label == "post-finalize-llo" and len(present) > 1:
            fam = {}
            for k in ("vcombine", "slane", "vld", "vst", "matmul"):
                fam[k] = [sum(n for op, n in per[a].items() if k in op)
                          for a in present]
            for k, v in fam.items():
                if max(v):
                    rows.append([f"    {k}*"] + v
                                + [f"{v[0]-v[1]:+d}" if len(v) > 1 else ""])
    table(rows, ["stage"] + present + [f"{present[0]}-{base}"])

    # --- the (W) test: load MASKS, not load counts --------------------------
    print("\n" + "=" * 78)
    print("tpu.load by sublane mask -- the actual (W) test")
    print("(equal COUNTS with unequal MASKS = equal instructions, unequal bytes)")
    print("=" * 78)
    for a in present:
        md, fs = files[a]
        p = pass_file(md, fs, "post-apply-vector-layout-simplify")
        m = load_masks(p) if p else collections.Counter()
        print(f"  [{a}]")
        tot_b = sum(n * nsub * SUBLANE_BYTES for (nsub, _), n in m.items())
        for (nsub, mref), n in m.most_common(4):
            b = n * nsub * SUBLANE_BYTES
            print(f"     {n:6d} loads x {nsub} sublane(s)  {mref}>  "
                  f"~{b:,} B")
        print(f"     approx VMEM read bytes per grid block: {tot_b:,} "
              f"(sum over ALL masks, not just the 4 shown)")

    # --- transpose shapes ---------------------------------------------------
    print("\n" + "=" * 78)
    print("tpu.transpose shapes at post-apply-vector-layout-simplify")
    print("=" * 78)
    for a in present:
        md, fs = files[a]
        p = pass_file(md, fs, "post-apply-vector-layout-simplify")
        c = collections.Counter()
        if p:
            for line in open(p, errors="replace"):
                m = re.search(r'(tpu\.transpose.*)', line)
                if m:
                    c[re.sub(r'%[A-Za-z0-9_]+', '%v', m.group(1))] += 1
        print(f"  [{a}] {sum(c.values())} transposes")
        for k, v in c.most_common():
            print(f"     {v} x {k}")

    # --- the discriminator --------------------------------------------------
    if {"inc", "a_reshape", "w_reshape"} <= set(present):
        discriminate(files, cnt)
    return 0


def discriminate(files, cnt):
    """Where does `w_reshape` land on each metric?

    `w` = {inc's 256-lane record, A's lane-major scale rows}. For any metric M
    with M(inc) != M(a_reshape), M(w) says which factor M follows:

        M(w) ~ M(a_reshape)  ->  M follows the SCALE PLACEMENT   (hypothesis T)
        M(w) ~ M(inc)        ->  M follows the RECORD WIDTH      (hypothesis W)

    Reported as a position in [0, 1] from a_reshape to inc. This is STATIC
    compile-time analysis; none of it is time, and it cannot price bytes.
    """
    print("\n" + "=" * 78)
    print("DISCRIMINATOR -- w = {inc's 256-lane record, A's scale rows}")
    print("  0.00 = w matches a_reshape (metric follows SCALE PLACEMENT, T)")
    print("  1.00 = w matches inc       (metric follows RECORD WIDTH, W)")
    print("=" * 78)

    vals = {}
    for k in ("tpu.transpose", "tpu.sublane_shuffle", "tpu.gather",
              "tpu.load"):
        vals[k] = {a: cnt[a][k] for a in cnt}
    vals["TOTAL ops (-0013)"] = {a: sum(cnt[a].values()) for a in cnt}

    lb = {}
    for a in ("inc", "a_reshape", "w_reshape"):
        md, fs = files[a]
        p = pass_file(md, fs, "post-apply-vector-layout-simplify")
        m = load_masks(p) if p else collections.Counter()
        lb[a] = sum(n * nsub * SUBLANE_BYTES for (nsub, _), n in m.items())
    vals["VMEM read bytes"] = lb

    rows = []
    for k, v in vals.items():
        i, a, w = v["inc"], v["a_reshape"], v["w_reshape"]
        pos = "n/a (inc == a_re)" if i == a else f"{(w - a) / (i - a):+.3f}"
        rows.append([k, i, a, w, pos])
    table(rows, ["metric", "inc", "a_reshape", "w_reshape", "w position"])
    print("\n  `w` carries one extra 256-lane row per page (257 vs 256), so its")
    print("  load COUNT sits ~0.4% above inc's by construction. The per-load")
    print("  sublane MASK, not the count, is what carries the width.")
    print("  XLU occupancy, bundles, spills and VMEM are NOT here: they come")
    print("  from `xprof-cli get_llo_fit_summary` against each arm's jf/ dir.")


if __name__ == "__main__":
    sys.exit(main())
