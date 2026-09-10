"""Diff two SparseCore bundle dumps: per-section bundle counts + opcode histogram.

Usage: bdiff.py <base_bundles.txt> <v3_bundles.txt>

A bundle is one `0x<hex>: {` ... `}` group -- one VLIW issue slot, i.e. one
cycle of the sequencer.  Slots inside a bundle are separated by `;`.  Section
labels are the bare `name:` lines.  Counting bundles answers "did the scheduler
actually have to issue more cycles"; the opcode histogram answers "which ops",
and an identical histogram also rules out spills (they would show as extra
sld/sst against smem).
"""
import collections
import re
import sys

BUNDLE = re.compile(r"^0x[0-9a-f]+:\s*\{")
LABEL = re.compile(r"^([A-Za-z_][A-Za-z0-9_.$]*):\s*$")
OPC = re.compile(r"=\s*@?[!a-z0-9]*\s*([a-z][a-z0-9_.]*)")


def parse(path):
    sec, cur = collections.OrderedDict(), "<preamble>"
    sec[cur] = {"bundles": 0, "ops": collections.Counter()}
    for line in open(path):
        m = LABEL.match(line.strip())
        if m:
            cur = m.group(1)
            sec.setdefault(cur, {"bundles": 0, "ops": collections.Counter()})
            continue
        if BUNDLE.match(line.strip()):
            sec[cur]["bundles"] += 1
        for slot in line.split(";"):
            if "=" not in slot:
                continue
            body = slot.split("=", 1)[1].strip()
            mo = re.match(r"@?!?p?\d*\s*([a-z][a-z0-9_.]*)", body)
            if mo:
                sec[cur]["ops"][mo.group(1)] += 1
    return sec


a, b = parse(sys.argv[1]), parse(sys.argv[2])
names = list(a) + [n for n in b if n not in a]

print(f"{'section':52s} {'base':>7s} {'v3':>7s} {'delta':>7s}")
ta = tb = 0
for n in names:
    x = a.get(n, {"bundles": 0})["bundles"]
    y = b.get(n, {"bundles": 0})["bundles"]
    ta += x
    tb += y
    if x or y:
        flag = "" if x == y else "   <-- CHANGED"
        print(f"  {n[:50]:50s} {x:7d} {y:7d} {y-x:+7d}{flag}")
print(f"  {'TOTAL':50s} {ta:7d} {tb:7d} {tb-ta:+7d}")

oa = sum((a[n]["ops"] for n in a), collections.Counter())
ob = sum((b[n]["ops"] for n in b), collections.Counter())
print(f"\n{'opcode':22s} {'base':>7s} {'v3':>7s} {'delta':>7s}   (whole program)")
for k in sorted(set(oa) | set(ob), key=lambda k: -(ob[k] - oa[k])):
    if oa[k] != ob[k]:
        print(f"  {k:20s} {oa[k]:7d} {ob[k]:7d} {ob[k]-oa[k]:+7d}")
print(f"  {'-- identical --':20s} "
      f"{sum(v for k, v in oa.items() if oa[k] == ob[k]):7d} ops, "
      f"{sum(1 for k in set(oa)|set(ob) if oa[k]==ob[k]):d} opcodes")
print(f"  {'TOTAL slots':20s} {sum(oa.values()):7d} {sum(ob.values()):7d} "
      f"{sum(ob.values())-sum(oa.values()):+7d}")

# Spill tell: scalar/vector loads and stores against smem/spill slots.
for tag, pat in (("smem spill sld/sst", r"s(ld|st)\s+.*smem"), ):
    ca = sum(len(re.findall(pat, l)) for l in open(sys.argv[1]))
    cb = sum(len(re.findall(pat, l)) for l in open(sys.argv[2]))
    print(f"\n  {tag:24s} base={ca:6d}  v3={cb:6d}  delta={cb-ca:+d}")
