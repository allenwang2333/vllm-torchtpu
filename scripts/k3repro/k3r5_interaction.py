#!/usr/bin/env python3
"""Interaction term of the F2 x F3'' 2x2, from the arm JSONs.

"They are substitutes, not addends" has been argued from mechanism many times
and measured none. With base / f2 / f3pp / f2f3pp in one build it is arithmetic:

    additive        d(joint) ~ d(f2) + d(f3pp)
    substitutes     d(joint) ~ max(d(f2), d(f3pp))
    interaction     d(joint) - [d(f2) + d(f3pp)]      (positive = sub-additive)

Usage: k3r5_interaction.py <ladder json dir> [conc]
"""
import glob, json, os, sys

d = sys.argv[1]
conc = sys.argv[2] if len(sys.argv) > 2 else "1"
arms = {}
for f in glob.glob(os.path.join(d, "*.json")):
    doc = json.load(open(f))
    name = doc["arm"].split("#")[0]
    cell = doc.get("cells", {}).get(conc)
    if cell:
        arms.setdefault(name, []).append(cell["tpot_ms"])

if "base" not in arms:
    sys.exit("no base arm: every delta here is meaningless without it")
import statistics
med = {k: statistics.median(v) for k, v in arms.items()}
b = med["base"]
print(f"c={conc}  base {b:.4f} ms over {len(arms['base'])} cell(s)")
delta = {}
for k in ("f2", "hierep", "f3pp", "f2f3pp"):
    if k in med:
        delta[k] = (med[k] / b - 1) * 100
        print(f"  {k:8} {med[k]:8.4f} ms   {delta[k]:+7.2f}%   ({len(arms[k])} cell(s))")
f2 = delta.get("f2", delta.get("hierep"))
f3 = delta.get("f3pp")
jt = delta.get("f2f3pp")
if None not in (f2, f3, jt):
    print(f"\n  sum of parts   {f2 + f3:+7.2f}%")
    print(f"  max of parts   {min(f2, f3):+7.2f}%      (more negative = bigger single win)")
    print(f"  measured joint {jt:+7.2f}%")
    print(f"  INTERACTION    {jt - (f2 + f3):+7.2f} pp   (positive = sub-additive = substitutes)")
    near_sum = abs(jt - (f2 + f3))
    near_max = abs(jt - min(f2, f3))
    print("  reading:", "ADDITIVE" if near_sum < near_max else "SUBSTITUTES",
          f"(|joint-sum|={near_sum:.2f} vs |joint-max|={near_max:.2f} pp)")
