"""Reader for `dump_llo_c.py`: opcode census, rotate immediates, spills,
scheduled bundles -- for each arm, off the LLO text dump.

Three files per arm, each answering a different question and each chosen
because a spill or a rotate is not visible in the other:

  `…-41-post-DCE-3.txt`   the last DCE before register allocation.  Every value
                          still has a unique SSA name, so this is where the
                          opcode census and R3's rotate attribution are read.
                          There are NO spills here -- they do not exist until
                          RA.
  `…-53-post-ra.txt`      after register allocation.  This is the only place
                          spill stores, fill loads and spill slots exist, and
                          it is where `vrot.slane` immediates are countable.
  `…-70-schedule-analysis_final_bundles.txt`  the scheduled VLIW bundle count,
                          which is the one static statistic that can net a
                          removed VALU op against an added spill, because it
                          accounts for issue slots.

STATIC COMPILE-TIME ANALYSIS.  Not a timing.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

DEFRE = re.compile(r'^\s*(%[\w.]+)\s*=\s*([\w.\[\]$<>-]+)(.*)$')
_CLASSES = ("vpop", "vmatprep", "vmatpush", "vmatmul", "vmatres", "vld",
            "vmax", "vmul", "vadd", "vrot", "vsel", "vcombine", "vst",
            "vxpose", "vperm", "vbcast", "vcmp", "vcmask", "vmand", "vsub",
            "vshrl", "vshll", "vand", "vunpack", "vpack", "vstv", "vlaneseq")


def klass(op):
    for t in _CLASSES:
        if op.startswith(t):
            return t
    return op.split(".")[0]


def census(path):
    c = collections.Counter()
    total = 0
    for ln in open(path, errors="replace"):
        m = DEFRE.match(ln)
        if not m:
            continue
        c[klass(m.group(2))] += 1
        total += 1
    c["__all_defs__"] = total
    return c


ROT_IMM = re.compile(r'vrot\.slane\s+(%[\w.]+)\s*,\s*(\S+)')
SPILL = re.compile(r'#(allocation\d+_spill)')


def rotate_immediates(path):
    """Histogram of `vrot.slane` amounts.  A binary 8-sublane reduction tree
    uses only {1,2,4}; a 7-way single-sublane gather uses {1..7}; a
    register-amount rotate is relayout, not reduction."""
    h = collections.Counter()
    for ln in open(path, errors="replace"):
        for _, amt in ROT_IMM.findall(ln):
            amt = amt.rstrip(";,")
            h["reg" if amt.startswith("%") else amt] += 1
    return h


def spills(path):
    """Spill stores, fill loads and distinct spill slots, from post-ra.

    Counted per *instruction occurrence* rather than per line, because
    post-ra text packs several ops onto a bundle line.
    """
    stores = fills = 0
    slots = set()
    for ln in open(path, errors="replace"):
        if "_spill" not in ln:
            continue
        for name in SPILL.findall(ln):
            slots.add(name)
        # split the bundle line into instruction-ish chunks
        for chunk in re.split(r';;|\n', ln):
            if "_spill" not in chunk:
                continue
            if re.search(r'\bvst\b', chunk):
                stores += 1
            if re.search(r'\bvld\b', chunk):
                fills += 1
    return dict(spill_stores=stores, fill_loads=fills, spill_slots=len(slots))


# The header reads `total scheduled bundles:    11600` -- the count comes
# AFTER the word, which an earlier `(\d+)\s*bundles` pattern matched zero times
# and reported as a silent 0 in the summary table.
BUNDLE_HDR = re.compile(r'total\s+scheduled\s+bundles\s*:\s*(\d[\d,]*)', re.I)


def bundles(path):
    head = []
    with open(path, errors="replace") as fh:
        for i, ln in enumerate(fh):
            if i > 40:
                break
            head.append(ln.rstrip())
    txt = "\n".join(head)
    m = BUNDLE_HDR.search(txt)
    n = int(m.group(1).replace(",", "")) if m else None
    return n, head[:12]


def pick(dump_dir, program_substr, pass_substr):
    pats = glob.glob(os.path.join(dump_dir, "**", f"*{pass_substr}*"),
                     recursive=True)
    pats = [p for p in pats if p.endswith(".txt")]
    hits = [p for p in pats if program_substr in os.path.basename(p)]
    if not hits:
        hits = pats
    if not hits:
        return None
    return max(hits, key=os.path.getsize)


def report(label, d, program):
    out = {"label": label, "dir": d, "program_filter": program}
    dce = pick(d, program, "41-post-DCE-3")
    ra = pick(d, program, "53-post-ra")
    sch = pick(d, program, "70-schedule-analysis")
    print(f"\n{'='*78}\n### {label}\n{'='*78}")
    for tag, f in (("41-post-DCE-3", dce), ("53-post-ra", ra),
                   ("70-schedule-analysis", sch)):
        print(f"  {tag}: {os.path.basename(f) if f else 'MISSING'}")
    if dce is None:
        print("  NO DUMP -- the capture did not take.  An empty read is "
              "indistinguishable from a kernel with nothing to report.")
        out["error"] = "no dump"
        return out
    out["file_dce"] = os.path.basename(dce)
    cen = census(dce)
    out["census"] = dict(cen)
    print("\n-- opcode census, 41-post-DCE-3 (static) --")
    for k, v in cen.most_common(24):
        print(f"{v:9d}  {k}")
    if ra:
        out["file_ra"] = os.path.basename(ra)
        h = rotate_immediates(ra)
        out["rotate_immediates"] = dict(h)
        out["vrot_slane_post_ra"] = sum(h.values())
        print("\n-- vrot.slane immediates, 53-post-ra --")
        if h:
            for k, v in sorted(h.items(),
                               key=lambda kv: (kv[0] == "reg", kv[0])):
                print(f"{v:9d}  rotate by {k}")
        else:
            print("        0  (no vrot.slane at all)")
        print(f"{sum(h.values()):9d}  TOTAL vrot.slane")
        sp = spills(ra)
        out.update(sp)
        print("\n-- register spilling, 53-post-ra --")
        for k, v in sp.items():
            print(f"{v:9d}  {k}")
    if sch:
        out["file_sched"] = os.path.basename(sch)
        n, head = bundles(sch)
        out["scheduled_bundles"] = n
        print("\n-- scheduled bundles, 70-schedule-analysis --")
        for ln in head:
            print("   ", ln)
        print(f"  parsed: {n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True,
                    metavar="LABEL=DIR")
    ap.add_argument("--program", default="StreamIdxTC")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    # per-arm program name, so head_minor's dump cannot be read as
    # head_major's.  `-hm` / `-hmhbm` come from `stage_a_c.make_stage_a_fn`.
    prog = {
        "incumbent": "StreamIdxTC-m-",
        "head_minor": "StreamIdxTC-m-",
        "head_major": "StreamIdxTC-hm-m-",
        "head_major_hbm": "StreamIdxTC-hmhbm-m-",
    }

    res = []
    for spec in args.dump:
        label, _, d = spec.partition("=")
        res.append(report(label, d, prog.get(label, args.program)))

    print("\n" + "=" * 78)
    print("### SUMMARY (static compile-time analysis, NOT a measurement)")
    print("=" * 78)
    hdr = ("arm", "vrot.slane", "vadd", "vmul", "vmax", "vmatmul",
           "spill st", "fill ld", "slots", "bundles")
    print(f"{hdr[0]:16}" + "".join(f"{h:>11}" for h in hdr[1:]))
    for r in res:
        c = r.get("census", {})
        print(f"{r['label']:16}" + "".join(f"{str(v):>11}" for v in (
            r.get("vrot_slane_post_ra", c.get("vrot", "-")),
            c.get("vadd", "-"),
            c.get("vmul", "-"),
            c.get("vmax", "-"),
            c.get("vmatmul", "-"),
            r.get("spill_stores", "-"),
            r.get("fill_loads", "-"),
            r.get("spill_slots", "-"),
            r.get("scheduled_bundles", "-"),
        )))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print("\nwrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
