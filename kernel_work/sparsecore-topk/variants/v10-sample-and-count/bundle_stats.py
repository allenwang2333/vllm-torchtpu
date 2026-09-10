"""Bundle-level statics for the v10 arms, from host-only AOT dumps.

    python bundle_stats.py

Reads `aot/<arm>/*call-start_*_bundles.txt` -- the scheduled SparseCore VLIW
bundles XLA emits with no TPU attached -- and reports, per arm:

  * total TEC (tile-execute-core) bundles, the sequencer's instruction count;
  * the opcode histogram;
  * every basic block that walks the row, classified by what it contains:
      `vst.idx.add.s32.msk`  = plsc.addupdate_scatter   -> a HISTOGRAM pass
      `vmpcnt.ones.xlane`    = all_reduce_population_count -> a COUNT pass
      `vadd.scan.msk.s32`    = plsc.cumsum              -> a bucket sweep
      `stream.linear.*`      = store_compressed / gather -> the EMIT pass

Why this file exists: the *marginal cost of one extra threshold in a count
pass* is the single unmeasured term that decides this node, and probes/15 only
measured a one-threshold pass.  Bundles per walked vector are a static,
TPU-free lower bound on that marginal, and they also test the transfer
assumption -- whether the probe's 0.191x count/scatter ratio is a property of
the mechanism or of the probe's own toy kernel.

CAUTION carried from a sibling build: XLA's `*-memory-usage-report.txt` and
`after_codegen.txt` are BLIND to Mosaic's in-kernel SparseCore VMEM (they
report 0 for kernels that provably allocate tens of KiB).  Nothing here reads
them, and no VMEM claim in README.md rests on them.
"""

import collections
import pathlib
import re

HERE = pathlib.Path(__file__).parent
ARMS = ("base", "sc2", "sc4", "sc16")

BUNDLE = re.compile(r"^0x[0-9a-f]+:\s*\{")
LABEL = re.compile(r"^([A-Za-z_.][\w.]*):\s*$")
OPCODE = re.compile(r"=\s+([a-z][a-z0-9_.]*)")

SCATTER = "vst.idx.add.s32.msk"
POPCNT = "vmpcnt.ones.xlane"
SCAN = "vadd.scan.msk.s32"


def parse(path):
    """-> (scs_blocks, tec_blocks); each a list of (label, [bundle_text])."""
    sections, cur, blocks, bundle = {}, None, None, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line in ("scs", "tec"):
            cur = line
            blocks = sections.setdefault(cur, [])
            continue
        if blocks is None:
            continue
        m = LABEL.match(line)
        if m:
            blocks.append((m.group(1), []))
            bundle = None
            continue
        if BUNDLE.match(line):
            if not blocks:
                blocks.append(("<entry>", []))
            bundle = [line]
            blocks[-1][1].append(bundle)
        elif bundle is not None:
            bundle.append(line)
    return sections.get("scs", []), sections.get("tec", [])


def ops(block):
    h = collections.Counter()
    for bundle in block:
        for ln in bundle:
            h.update(OPCODE.findall(ln))
    return h


def kind(h):
    if h[SCATTER]:
        return "HIST-scatter"
    if h["stream.linear.gather"] or h["stream.linear.scatter"]:
        return "EMIT-stream"
    if h[SCAN]:
        return "SWEEP-scan"
    if h[POPCNT]:
        return "COUNT-popcnt"
    return ""


def kernel_dump(arm):
    d = HERE / "aot" / arm
    got = [p for p in d.glob("*call-start_*_bundles.txt")
           if p.name.startswith("sc_topk_direct")]
    if len(got) != 1:
        raise SystemExit(f"{arm}: expected 1 kernel bundle dump, got {got}")
    return got[0]


BR = re.compile(r"sbr\.rel\s+(?:@!?p\d+\s+)?([\w.]+)-([\w.]+)")


def innermost_loops(tec):
    """Natural loops, found from the SYMBOLIC branch targets the dump prints.

    `sbr.rel .LBB2_34-.Ltmp25` names both ends, so a back edge is a target that
    is at or before its own block.  A 2-block region is an innermost loop body,
    which is the only granularity at which "bundles per walked vector" means
    anything -- everything larger has an epilogue folded into it.
    """
    pos = {lab: i for i, (lab, _) in enumerate(tec)}
    out, seen = [], set()
    for i, (lab, bs) in enumerate(tec):
        text = "\n".join("\n".join(x) for x in bs)
        for tgt, _src in BR.findall(text):
            j = pos.get(tgt)
            if j is None or j > i or (j, i) in seen:
                continue
            seen.add((j, i))
            h = collections.Counter()
            n = 0
            for _l, blk in tec[j:i + 1]:
                n += len(blk)
                h.update(ops(blk))
            out.append((tgt, lab, i - j + 1, n, h))
    return out


def main():
    per_arm = {}
    for arm in ARMS:
        p = kernel_dump(arm)
        scs, tec = parse(p)
        per_arm[arm] = (p, scs, tec)

    print("=" * 78)
    print("TOTAL BUNDLES (static instruction count, not a runtime)")
    print("=" * 78)
    print(f"{'arm':<6}{'scs':>8}{'tec':>8}{'tec d(base)':>14}   dump")
    b_tec = sum(len(b) for _, b in per_arm["base"][2])
    for arm in ARMS:
        p, scs, tec = per_arm[arm]
        s, t = sum(len(b) for _, b in scs), sum(len(b) for _, b in tec)
        print(f"{arm:<6}{s:>8}{t:>8}{t - b_tec:>+14}   {p.name[:44]}")

    print()
    print("=" * 78)
    print("OPCODE HISTOGRAM, TEC section (top 24 by base count + all movers)")
    print("=" * 78)
    hists = {}
    for arm in ARMS:
        h = collections.Counter()
        for _, blk in per_arm[arm][2]:
            h.update(ops(blk))
        hists[arm] = h
    keys = [k for k, _ in hists["base"].most_common(24)]
    keys += [k for k in sorted(set().union(*[set(h) for h in hists.values()]))
             if k not in keys
             and len({hists[a][k] for a in ARMS}) > 1]
    print(f"{'opcode':<26}" + "".join(f"{a:>9}" for a in ARMS))
    for k in keys:
        row = [hists[a][k] for a in ARMS]
        flag = "  <-- moves" if len(set(row)) > 1 else ""
        print(f"{k:<26}" + "".join(f"{v:>9}" for v in row) + flag)

    print()
    print("=" * 78)
    print("ROW-WALKING BLOCKS (>= 8 bundles and carrying a walk primitive)")
    print("=" * 78)
    for arm in ARMS:
        print(f"--- {arm}")
        tot = collections.Counter()
        for label, blk in per_arm[arm][2]:
            h = ops(blk)
            kd = kind(h)
            if not kd or len(blk) < 8:
                continue
            tot[kd] += len(blk)
            print(f"  {label:<14}{len(blk):>5} bundles  {kd:<13}"
                  f" scat={h[SCATTER]:<3} pcnt={h[POPCNT]:<3}"
                  f" scan={h[SCAN]:<3} vld={h['vld']:<3}"
                  f" cmp={h['vgt.s32'] + h['vlt.s32'] + h['veq.s32']:<3}")
        print(f"  {'TOTAL':<14}" + "  ".join(f"{k}={v}"
                                             for k, v in sorted(tot.items())))

    print()
    print("=" * 78)
    print("INNERMOST LOOP BODIES (2 blocks) -- bundles per ITERATION.")
    print("The count pass's `unroll` sets vectors/iteration: 4 for arity<=4, "
          "2 for arity 16.")
    print("=" * 78)
    for arm in ARMS:
        print(f"--- {arm}")
        for tgt, lab, nblk, n, h in innermost_loops(per_arm[arm][2]):
            if nblk != 2 or n < 8:
                continue
            kd = kind(h)
            if not kd:
                continue
            print(f"  [{tgt} .. {lab}]".ljust(34)
                  + f"{n:>4} bundles  {kd:<13}"
                  f" scat={h[SCATTER]:<3} pcnt={h[POPCNT]:<3}"
                  f" scan={h[SCAN]:<3} vld={h['vld']:<3}")

    print()
    print("=" * 78)
    print("THE MARGINAL COST OF ONE EXTRA THRESHOLD")
    print("=" * 78)
    print("""\
  The count pass's innermost body, at a FIXED unroll of 4 (4 vectors/iter):
      arity 2  (1 threshold)   11 bundles
      arity 4  (3 thresholds)  30 bundles
  b(m) = a + c*m  =>  c = 9.5 bundles/threshold, a = 1.5 bundles of walk.
  So one extra threshold costs 9.5/11 = 0.864 of a WHOLE one-threshold pass.
  arity 16 runs at unroll 2: 60 bundles / 2 vectors x 15 thresholds = 120 per
  4 vectors, against a linear 144 -- sublinear, but it SPILLS (sld 0/2/5/52 and
  sst 1/2/4/29 across base/sc2/sc4/sc16), which is why.

  For contrast, in the same kernel:
      incumbent hist0 inner body        9 bundles / 4 vectors = 2.25
      incumbent refinement inner body  14 bundles / 4 vectors = 3.50
      v10 count inner body (1 thr)     11 bundles / 4 vectors = 2.75
  A count pass ISSUES 1.22x the L0 histogram's bundles.  probes/15 timed it at
  0.191x.  Both can be true -- the scatter stalls on tilespmem, it does not
  issue -- but it means the 0.191x is a memory effect that does NOT transfer to
  an extra threshold, which is pure issue.  That is the term that decides v10.""")


main()
