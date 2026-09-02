"""Step-3 probe R3, reader half: opcode census + `vrot.slane` attribution off
an LLO text dump.

Given a `--xla_jf_dump_to` directory it picks the named program's
`*-41-post-DCE-3.txt` -- the last DCE before register allocation, so every
value still has a unique SSA name and a dependence chain can be walked
exactly -- and reports:

  1. the opcode census (the thing P7b §B3.4 quotes as "the opcode histogram
     of the body"),
  2. every `vrot.slane` classified by the *producer* of its rotated operand
     and by the *consumer* of its result, walked through the pure shuffle ops
     (`vcombine`, `vsel`, `vmov`) that sit between real arithmetic,
  3. what each `vpop.f32.mrb[*]` MXU result-buffer pop flows into -- the
     direct test of whether the rotates are the MXU drain.

EVERYTHING HERE IS STATIC COMPILE-TIME ANALYSIS.  Not a timing.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

DEFRE = re.compile(r'^\s*(%[\w.]+)\s*=\s*([\w.\[\]$<>-]+)(.*)$')
PASS = "41-post-DCE-3"

# opcode -> coarse class.  Order matters: longest/most specific first.
_CLASSES = ("vpop", "vmatprep", "vmatpush", "vmatmul", "vmatres", "vld",
            "vmax", "vmul", "vadd", "vrot", "vsel", "vcombine", "vst",
            "vxpose", "vperm", "vbcast", "vcmp", "vcmask", "vmand", "vsub",
            "vshrl", "vshll", "vand", "vunpack", "vstv", "vlaneseq")
# ops that only move data around and are transparent for attribution
_SHUFFLE = ("vcombine", "vsel", "vmov")


def klass(op):
    for t in _CLASSES:
        if op.startswith(t):
            return t
    return op.split(".")[0]


class Prog:
    def __init__(self, path):
        self.defs, self.rest, self.order = {}, {}, []
        for ln in open(path):
            m = DEFRE.match(ln)
            if not m:
                continue
            n, o, r = m.group(1), m.group(2), m.group(3)
            self.defs[n] = o
            self.rest[n] = r
            self.order.append(n)
        self.uses = collections.defaultdict(list)
        for n in self.order:
            for o in re.findall(r'%[\w.]+', self.rest[n]):
                self.uses[o].append(n)

    def back(self, n, depth=0, seen=None):
        if seen is None:
            seen = set()
        if n in seen or depth > 80:
            return set()
        seen.add(n)
        op = self.defs.get(n)
        if op is None:
            return {"ARG/CONST"}
        k = klass(op)
        if k in _SHUFFLE:
            s = set()
            for o in re.findall(r'%[\w.]+', self.rest[n]):
                s |= self.back(o, depth + 1, seen)
            return s or {k}
        return {k}

    def fwd(self, n, depth=0, seen=None):
        if seen is None:
            seen = set()
        if n in seen or depth > 80:
            return set()
        seen.add(n)
        out = set()
        for u in self.uses.get(n, []):
            k = klass(self.defs[u])
            if k in _SHUFFLE:
                out |= (self.fwd(u, depth + 1, seen) or {k})
            else:
                out.add(k)
        return out

    def census(self):
        return collections.Counter(klass(self.defs[n]) for n in self.order)

    def rot_attrib(self):
        cat, ex = collections.Counter(), collections.defaultdict(list)
        for n in self.order:
            if not self.defs[n].startswith("vrot.slane"):
                continue
            ops = re.findall(r'%[\w.]+', self.rest[n])
            b = "|".join(sorted(self.back(ops[0]))) if ops else "imm-only"
            f = "|".join(sorted(self.fwd(n))) or "dead"
            cat[(b, f)] += 1
            if len(ex[(b, f)]) < 3:
                ex[(b, f)].append(n)
        return cat, ex

    def pop_fanout(self):
        c = collections.Counter()
        for n in self.order:
            if self.defs[n].startswith("vpop"):
                c["|".join(sorted(self.fwd(n))) or "dead"] += 1
        return c


def find_pass_file(dump_dir, program_substr):
    pats = glob.glob(os.path.join(dump_dir, f"*{PASS}.txt"))
    hits = [p for p in pats if program_substr in os.path.basename(p)]
    if not hits:
        hits = pats
    if not hits:
        return None
    # largest = the real kernel body, not a trivial fusion
    return max(hits, key=os.path.getsize)


def report(label, path):
    p = Prog(path)
    print(f"\n{'='*78}\n### {label}\nfile: {os.path.basename(path)}\n{'='*78}")
    cen = p.census()
    print("-- opcode census (static, compile-time) --")
    for k, v in cen.most_common(20):
        print(f"{v:8d}  {k}")
    cat, ex = p.rot_attrib()
    tot = sum(cat.values())
    print(f"\n-- vrot.slane attribution ({tot} total) --")
    print(f"{'producer of rotated operand':34} -> {'consumer of result':22}"
          f" {'count':>7}   examples")
    for (b, f), v in cat.most_common():
        print(f"{b:34} -> {f:22} {v:7d}   {' '.join(ex[(b, f)])}")
    print("\n-- where each MXU result-buffer pop (vpop.f32.mrb) flows --")
    for k, v in p.pop_fanout().most_common():
        print(f"{v:8d}  vpop -> {k}")
    return {
        "label": label,
        "file": os.path.basename(path),
        "census": dict(cen),
        "rot_attrib": {f"{b} -> {f}": v for (b, f), v in cat.items()},
        "vrot_slane_total": tot,
        "pop_fanout": dict(p.pop_fanout()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    metavar="LABEL=DIR",
                    help="repeatable; DIR is the `jf/` subdir of a dump")
    ap.add_argument("--program", default="",
                    help="substring of the program name to select")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = []
    for spec in args.dump:
        label, _, d = spec.partition("=")
        f = find_pass_file(d, args.program or label)
        if f is None:
            print(f"\n### {label}: NO {PASS} FILE under {d} -- the dump did "
                  f"not take.  Empty reads identically to a kernel with "
                  f"nothing to report; this is a capture failure.")
            out.append({"label": label, "file": None})
            continue
        out.append(report(label, f))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("\nwrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
