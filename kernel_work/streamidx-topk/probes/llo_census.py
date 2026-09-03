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

**It cannot settle that, and an earlier version of this docstring claimed a
test that does not work.** Two corrections, both from the `/critical-review` of
R1c's first draft:

* **Vector-load COUNT is not a test for (W).** The counts are near-equal (9,296
  vs 9,368) but the incumbent's loads carry `sublanes [true,true,...]` against
  Option A's `[true,false,...]`: 1024 B vs 512 B per load, 1.985x the bytes in
  the same instruction count. This script prints that mask breakdown now. (W)
  costs no extra vector ops and 2x the VMEM read bytes; a static census cannot
  price the bytes.
* **The three arms are collinear** in scale placement and record width, so
  neither hypothesis can be isolated from them at all. See `R1c.md` §4.1 for
  the missing fourth arm.

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
        tot_b = 0
        for (nsub, mref), n in m.most_common(4):
            lanes = 128
            packed = 4 if "x4x" in mref else 1
            b = n * nsub * lanes * packed
            tot_b += b
            print(f"     {n:6d} loads x {nsub} sublane(s)  {mref}>  "
                  f"~{b:,} B")
        print(f"     approx VMEM read bytes per grid block: {tot_b:,}")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
