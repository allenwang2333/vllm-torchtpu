"""Localize a bundle-dump diff to basic blocks, and to the scatter sites.

Usage: bloc.py <base_bundles.txt> <v3_bundles.txt>

Splits `execute0_lowered` into basic blocks at branch boundaries, fingerprints
each block with PCs and branch immediates stripped, and reports which blocks are
common to both arms and which are unique.  Then tags every block by the marker
ops it contains, so a changed block can be named (SCATTER = addupdate_scatter,
STORE.MSK = store_compressed, PCNT = all_reduce_population_count).
"""
import collections
import hashlib
import re
import sys

BUNDLE = re.compile(r"^0x([0-9a-f]+):\s*\{")
LABEL = re.compile(r"^([A-Za-z_][A-Za-z0-9_.$]*):\s*$")
MARK = (("SCATTER", "vst.idx.add"), ("STORE.MSK", "vst.msk"),
        ("PCNT", "vmpcnt.ones.xlane"), ("PUSH", "vpush"), ("LD", "vld"),
        ("DMA", "dma"), ("SCAN", "vscan"))


def blocks(path, want="execute0_lowered"):
    """Yield (pc, text) per basic block: a run of bundles ended by a branch."""
    cur, on, out, buf, pc0 = None, False, [], [], None
    ends_branch = False
    body = []
    for line in open(path):
        m = LABEL.match(line.strip())
        if m:
            on = (m.group(1) == want)
            continue
        if on:
            body.append(line.rstrip("\n"))
    for line in body:
        m = BUNDLE.match(line.strip())
        if m and buf and ends_branch:
            out.append((pc0, "\n".join(buf)))
            buf, pc0 = [], None
        if m:
            if pc0 is None:
                pc0 = int(m.group(1), 16)
            ends_branch = False
        buf.append(line)
        if re.search(r"\bs(br|bri)\.", line):
            ends_branch = True
    if buf:
        out.append((pc0, "\n".join(buf)))
    return out


def norm(t, regs=False):
    t = re.sub(r"^0x[0-9a-f]+:", "", t, flags=re.M)      # PCs
    t = re.sub(r"sbr\.rel\s+\$0x[0-9A-Fa-f]+", "sbr.rel $I", t)
    if regs:                                             # ignore regalloc
        t = re.sub(r"\b([svpm])\d+\b", r"\1R", t)
        t = re.sub(r"smem:\$0x[0-9A-Fa-f]+", "smem:$A", t)
    return hashlib.md5(t.encode()).hexdigest()[:10]

def opseq(t):
    """Opcode sequence only: what the scheduler issued, ignoring operands."""
    ops = []
    for line in t.splitlines():
        for slot in line.split(";"):
            if "=" in slot:
                mo = re.match(r"@?!?p?\d*\s*([a-z][a-z0-9_.]*)",
                              slot.split("=", 1)[1].strip())
                if mo:
                    ops.append(mo.group(1))
    return hashlib.md5("|".join(ops).encode()).hexdigest()[:10]


def tag(t):
    return "+".join(n for n, p in MARK if p in t) or "-"


A, B = blocks(sys.argv[1]), blocks(sys.argv[2])
fa = collections.Counter(norm(t) for _, t in A)
fb = collections.Counter(norm(t) for _, t in B)
print(f"blocks: base={len(A)}  v3={len(B)}")
print(f"distinct fingerprints: base={len(fa)}  v3={len(fb)}  "
      f"shared={len(set(fa) & set(fb))}")

nb_a = sum(t.count("0x") for _, t in A)


def bundles_of(t):
    return len([l for l in t.splitlines() if BUNDLE.match(l.strip())])


for lbl, fn in (("exact", lambda t: norm(t)),
                ("reg-normalized", lambda t: norm(t, True)),
                ("opcode-sequence", opseq)):
    ga = collections.Counter(fn(t) for _, t in A)
    gb = collections.Counter(fn(t) for _, t in B)
    sh = set(ga) & set(gb)
    nb = sum(bundles_of(t) for _, t in A if fn(t) in sh)
    print(f"  {lbl:16s} shared fingerprints={len(sh):3d}  "
          f"base bundles in a shared block={nb}")

shared = set(fa) & set(fb)
sh_bund = sum(bundles_of(t) for _, t in A if norm(t) in shared)
tot_a = sum(bundles_of(t) for _, t in A)
tot_b = sum(bundles_of(t) for _, t in B)
print(f"bundles: base={tot_a}  v3={tot_b}  "
      f"identical-block bundles={sh_bund} "
      f"({100.0*sh_bund/max(tot_a,1):.1f}% of base)")

for name, blks, other in (("BASE-ONLY", A, fb), ("V3-ONLY", B, fa)):
    print(f"\n{name} blocks (fingerprint absent from the other arm):")
    print(f"  {'pc':>7s} {'bund':>5s} {'fp':>11s}  markers")
    for pc, t in blks:
        f = norm(t)
        if f not in other:
            print(f"  {pc:7d} {bundles_of(t):5d} {f:>11s}  {tag(t)}")

# Scatter localization: program-order position of every scatter instruction.
print("\nSCATTER (vst.idx.add = addupdate_scatter) sites, in program order:")
for name, blks in (("base", A), ("v3", B)):
    sites = [(pc, t.count("vst.idx.add"), norm(t)) for pc, t in blks
             if "vst.idx.add" in t]
    tot = sum(c for _, c, _ in sites)
    print(f"  {name:5s}: {len(sites)} blocks, {tot} instructions -> " +
          ", ".join(f"pc{pc}:{c}x[{f}]" for pc, c, f in sites))
