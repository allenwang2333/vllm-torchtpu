"""Diff per-step top-k logprobs, ours vs the JAX reference.

Why this and not activation diffing: logprobs come from the REAL generating
path. No dumper, no re-execution, no warmup/prefill step offset, no hook. Every
activation-level instrument in this investigation was distorted by at least one
of those; this one cannot be.

Reads `[STEP_LOGPROBS] step=N {tok: Logprob(logprob=..., rank=..,
decoded_token='..'), ...}` from both logs and reports, per step, whether the
argmax matches, how each side's top-1/top-2 margin compares, and -- the useful
one -- our logprob for the token the reference actually picked.

Read the `ours@ref` column, not the flip step. The flip is only where
accumulated error first moves an argmax; the divergence exists before it.
"""
import os
import re
import sys

LP = re.compile(r"\[STEP_LOGPROBS\] step=(\d+) (\{.*\})\s*$")
ENTRY = re.compile(
    r"(\d+): Logprob\(logprob=([-\d.einf]+), rank=(\d+), "
    r"decoded_token=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\)")


def parse(path):
    steps = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = LP.search(line.rstrip("\n"))
            if not m:
                continue
            step = int(m.group(1))
            if step in steps:  # keep the first occurrence (rank 0 / first seq)
                continue
            entries = []
            for tok, lp, rank, dec in ENTRY.findall(m.group(2)):
                try:
                    val = float(lp)
                except ValueError:
                    val = float("-inf")
                entries.append((int(tok), val, int(rank), dec))
            entries.sort(key=lambda e: e[2])  # by rank
            if entries:
                steps[step] = entries
    return steps


def main(ours_path, ref_path):
    o, r = parse(ours_path), parse(ref_path)
    if not o:
        sys.exit(f"FATAL: no [STEP_LOGPROBS] in {ours_path}")
    if not r:
        sys.exit(f"FATAL: no [STEP_LOGPROBS] in {ref_path}")
    common = sorted(set(o) & set(r))
    print(f"ours steps={len(o)} ref steps={len(r)} common={len(common)}\n")
    print(f"{'step':>4} {'ours top1':>16} {'ref top1':>16} {'match':>6} "
          f"{'ours gap':>9} {'ref gap':>9} {'ours@ref':>9}")
    print("-" * 80)
    first_flip = None
    for s in common:
        oe, re_ = o[s], r[s]
        ot, rt = oe[0], re_[0]
        match = ot[0] == rt[0]
        ogap = (oe[0][1] - oe[1][1]) if len(oe) > 1 else float("nan")
        rgap = (re_[0][1] - re_[1][1]) if len(re_) > 1 else float("nan")
        # Cross-tree error: OUR logprob for the token the REFERENCE picked.
        # This is the column that matters. The flip step is only a threshold
        # crossing -- the numeric difference exists from step 0 and merely
        # takes until the flip to move an argmax -- so the flip location says
        # nothing about where the error starts. This measure works at every
        # step *before* any flip, while both trees still agree on the token.
        #   smooth decay  -> error accumulating over generation
        #   sharp cliff   -> check the step index against structural
        #                    boundaries (sliding window 128, CSA ratio 4,
        #                    page/packing 64); a boundary-correlated cliff is
        #                    the most localizing signal available here.
        # nan means the reference's pick fell outside our top-k, i.e. our
        # disagreement is already larger than the captured window.
        ours_at_ref = next((e[1] for e in oe if e[0] == rt[0]), float("nan"))
        if not match and first_flip is None:
            first_flip = s
        flag = "" if match else "   <== FLIP"
        # print every step up to the flip, then a few after
        if first_flip is None or s <= first_flip + 5:
            print(f"{s:>4} {ot[3][:14]:>16} {rt[3][:14]:>16} "
                  f"{str(match):>6} {ogap:>9.3f} {rgap:>9.3f} "
                  f"{ours_at_ref:>9.3f}{flag}")
    print()
    if first_flip is None:
        print("NO ARGMAX FLIP in the common steps -- the two trees agree on "
              "every generated token here.")
    else:
        print(f"FIRST ARGMAX FLIP: step {first_flip}")
        print(f"  ours top-5: {[(e[3], round(e[1], 3)) for e in o[first_flip]]}")
        print(f"  ref  top-5: {[(e[3], round(e[1], 3)) for e in r[first_flip]]}")
        print()
        print("Reading the flip -- the two directions are NOT symmetric:")
        print("  ref gap LARGE and we disagree: our distribution differs by at "
              "least that margin. A real lower bound on the error.")
        print("  ref gap SMALL: uninformative. The model was uncertain there, "
              "so any perturbation flips it -- this does NOT imply the error "
              "is small.")
        print("  The flip is a threshold crossing, not an origin. Read the "
              "ours@ref column across ALL steps instead.")


if __name__ == "__main__":
    SP = os.environ.get("DSV4_LOG_DIR", ".")
    main(sys.argv[1] if len(sys.argv) > 1 else f"{SP}/runAI_ours_real.log",
         sys.argv[2] if len(sys.argv) > 2 else f"{SP}/runAJ_ref_real.log")
