"""Static SparseCore bundle statistics: composed candidate vs incumbent.

    python bundle_stats.py <dump-dir-final> <dump-dir-inc>

Everything printed here is `static-compile`. A bundle count is CODE SIZE and
ISSUE cost. It is BLIND to stall cost -- in this same kernel family
`addupdate_scatter` is 5.2x a lane popcount in TIME and 0.82x in BUNDLES, and
two builders have already read a bundle delta as a cost proxy, once with the
sign inverted. Nothing here is a time and nothing here may be converted to one.
"""

import collections
import pathlib
import re
import sys

BUNDLE = re.compile(r"^0x[0-9a-f]+:\s*\{")
LABEL = re.compile(r"^([A-Za-z_.][\w.]*):\s*$")
OPCODE = re.compile(r"=\s+([a-z][a-z0-9_.]*)")

SCATTER = "vst.idx.add.s32.msk"
POPCNT = "vmpcnt.ones.xlane"
SCAN = "vadd.scan.msk.s32"


def parse(path):
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


def nop_density(blocks):
    """Fraction of bundles whose only issued slots are nops."""
    total = only_nop = 0
    for _, blk in blocks:
        for bundle in blk:
            total += 1
            found = OPCODE.findall("\n".join(bundle))
            if found and all(o.endswith("nop") for o in found):
                only_nop += 1
    return only_nop, total


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


def kernel_dump(d):
    got = [p for p in pathlib.Path(d).glob("*call-start_*_bundles.txt")
           if p.name.startswith("sc_topk_direct")]
    if len(got) != 1:
        raise SystemExit(f"{d}: expected 1 kernel bundle dump, got {got}")
    return got[0]


def main(dfin, dinc):
    arms = {"final": kernel_dump(dfin), "inc": kernel_dump(dinc)}
    per = {a: parse(p) for a, p in arms.items()}
    order = ("inc", "final")

    print("=" * 76)
    print("TOTAL BUNDLES  (static code size / issue slots -- NOT a time)")
    print("=" * 76)
    for a in order:
        scs, tec = per[a]
        s, t = sum(len(b) for _, b in scs), sum(len(b) for _, b in tec)
        n_s, tot_s = nop_density(scs)
        n_t, tot_t = nop_density(tec)
        print(f"  {a:<6} dump={arms[a].name[:46]}")
        print(f"  {'':6} scs bundles={s:<6} nop-only={n_s:<5} "
              f"({100 * n_s / max(tot_s, 1):.1f}%)")
        print(f"  {'':6} tec bundles={t:<6} nop-only={n_t:<5} "
              f"({100 * n_t / max(tot_t, 1):.1f}%)")
    ti = sum(len(b) for _, b in per["inc"][1])
    tf = sum(len(b) for _, b in per["final"][1])
    print(f"  tec delta: {tf - ti:+d} bundles "
          f"({100 * (tf - ti) / ti:+.2f}% code size)")

    print()
    print("=" * 76)
    print("SPILL / FILL  (stack traffic; a spill is a real correctness-neutral")
    print("cost signal, unlike a raw bundle count)")
    print("=" * 76)
    for a in order:
        text = arms[a].read_text()
        sp = len(re.findall(r"spill", text, re.I))
        stk = len(re.findall(r"\[tilespmem:\$0x[0-9A-F]+\]\s*=\s*vst\b", text))
        print(f"  {a:<6} 'spill' string hits={sp:<4}")

    print()
    print("=" * 76)
    print("OPCODE HISTOGRAM, tec section (top 26 + every mover)")
    print("=" * 76)
    hists = {}
    for a in order:
        h = collections.Counter()
        for _, blk in per[a][1]:
            h.update(ops(blk))
        hists[a] = h
    keys = [k for k, _ in hists["inc"].most_common(26)]
    keys += [k for k in sorted(set(hists["inc"]) | set(hists["final"]))
             if k not in keys and hists["inc"][k] != hists["final"][k]]
    print(f"  {'opcode':<28}{'inc':>8}{'final':>8}{'delta':>8}")
    for k in keys:
        i, f = hists["inc"][k], hists["final"][k]
        print(f"  {k:<28}{i:>8}{f:>8}{f - i:>+8}"
              + ("  <-- moves" if i != f else ""))

    print()
    print("=" * 76)
    print("ROW-WALKING BLOCKS (>= 8 bundles, carrying a walk primitive)")
    print("=" * 76)
    for a in order:
        print(f"--- {a}")
        tot = collections.Counter()
        for label, blk in per[a][1]:
            h = ops(blk)
            kd = kind(h)
            if not kd or len(blk) < 8:
                continue
            tot[kd] += len(blk)
            print(f"  {label:<16}{len(blk):>5} bundles  {kd:<13}"
                  f" scat={h[SCATTER]:<3} pcnt={h[POPCNT]:<3}"
                  f" scan={h[SCAN]:<3} vld={h['vld']:<4}"
                  f" dma={h['dma.local'] + h['dma.general']:<3}")
        print(f"  {'TOTAL':<16}" +
              "  ".join(f"{k}={v}" for k, v in sorted(tot.items())))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
