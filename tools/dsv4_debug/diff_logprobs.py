"""Diff per-step top-k logprobs, ours vs the JAX reference, across a prompt pool.

Why logprobs and not activation dumps: these come from the real generating
path. No dumper, no hooks, no warmup/prefill step offset. Every activation-level
instrument in this bring-up has been distorted by at least one of those.

Why a pool and not one prompt: a single prompt can sit on a knife edge. On the
XLA/LLVM prompt the two trees split at step 13, where our own top-1/top-2
margin was 1.25 -- close to a coin flip, so it measures luck as much as
correctness. Across several prompts, and weighting the ones where the
*reference* was confident, you get a signal worth acting on.

Reads both logs, groups by prompt (`[SWEEP] ===== prompt: LABEL =====`), and
per prompt reports the first token where the argmax flips plus how sure each
side was there.

`ours@ref` is our logprob for the token the reference actually chose. Unlike
the flip location -- which is just where accumulated error first moved an
argmax -- it is meaningful at every step, including before any flip.

Usage:
    python diff_logprobs.py <ours.log> <ref.log>
"""
import os
import re
import sys

SWEEP = re.compile(r"\[SWEEP\] =====\s*prompt:\s*(\S+)")
LP = re.compile(r"\[STEP_LOGPROBS\] step=(\d+) (\{.*\})\s*$")
ENTRY = re.compile(
    r"(\d+): Logprob\(logprob=([-\d.einf]+), rank=(\d+), "
    r"decoded_token=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\)")

# A flip only means something if the reference was sure. Below this margin both
# models found the step ambiguous, so the flip says little.
CONFIDENT = 5.0


def parse(path):
    """-> {prompt_label: {step: [(token_id, logprob, rank, decoded), ...]}}"""
    out, cur = {}, "(unlabeled)"
    with open(path, errors="replace") as f:
        for line in f:
            m = SWEEP.search(line)
            if m:
                cur = m.group(1)
                out.setdefault(cur, {})
                continue
            m = LP.search(line.rstrip("\n"))
            if not m:
                continue
            step = int(m.group(1))
            steps = out.setdefault(cur, {})
            if step in steps:  # first occurrence = rank 0 / first sequence
                continue
            entries = []
            for tok, lp, rank, dec in ENTRY.findall(m.group(2)):
                try:
                    val = float(lp)
                except ValueError:
                    val = float("-inf")
                entries.append((int(tok), val, int(rank), dec))
            entries.sort(key=lambda e: e[2])
            if entries:
                steps[step] = entries
    return {k: v for k, v in out.items() if v}


def first_flip(o_steps, r_steps):
    common = sorted(set(o_steps) & set(r_steps))
    for s in common:
        oe, re_ = o_steps[s], r_steps[s]
        if oe[0][0] == re_[0][0]:
            continue
        rgap = (re_[0][1] - re_[1][1]) if len(re_) > 1 else float("nan")
        ours_at_ref = next((e[1] for e in oe if e[0] == re_[0][0]),
                           float("nan"))
        return dict(step=s, ours=oe[0][3], ref=re_[0][3], rgap=rgap,
                    ours_at_ref=ours_at_ref, n=len(common))
    return dict(step=None, n=len(common))


def main(ours_path, ref_path):
    o, r = parse(ours_path), parse(ref_path)
    if not o:
        sys.exit(f"FATAL: no [STEP_LOGPROBS] in {ours_path}")
    if not r:
        sys.exit(f"FATAL: no [STEP_LOGPROBS] in {ref_path}")
    prompts = sorted(set(o) & set(r))
    if not prompts:
        sys.exit(f"FATAL: no prompt labels in common.\n  ours: {sorted(o)}\n"
                 f"  ref:  {sorted(r)}")

    print(f"{len(prompts)} prompts in common\n")
    print(f"{'prompt':<26} {'steps':>5} {'flip':>5} {'ours':>13} "
          f"{'ref':>13} {'ref gap':>8} {'ours@ref':>9}")
    print("-" * 92)

    rows = []
    for p in prompts:
        a = first_flip(o[p], r[p])
        if a["step"] is None:
            print(f"{p[:25]:<26} {a['n']:>5} {'--':>5}   agrees throughout")
        else:
            print(f"{p[:25]:<26} {a['n']:>5} {a['step']:>5} "
                  f"{a['ours'][:11]:>13} {a['ref'][:11]:>13} "
                  f"{a['rgap']:>8.2f} {a['ours_at_ref']:>9.2f}")
        rows.append((p, a["step"], a.get("rgap", float("nan"))))

    strong = [(p, s, g) for p, s, g in rows
              if s is not None and g == g and g >= CONFIDENT]
    weak = [(p, s, g) for p, s, g in rows
            if s is not None and g == g and g < CONFIDENT]
    clean = [p for p, s, _ in rows if s is None]

    print()
    print(f"agree all the way through   : {len(clean)}")
    print(f"flip where ref was UNSURE   : {len(weak)}"
          f"   (ref gap < {CONFIDENT}; could be luck)")
    print(f"flip where ref was CONFIDENT: {len(strong)}"
          f"   (ref gap >= {CONFIDENT}; real divergence)")

    if strong:
        print("\nLocalize with these -- the reference was sure and we still "
              "disagreed:")
        for p, s, g in sorted(strong, key=lambda t: -t[2]):
            print(f"    {p:<28} step {s:<4} ref gap {g:.2f}")
    elif weak:
        print("\nNo confident divergence. Every flip landed on a step the "
              "reference also found ambiguous,\nwhich fits small numeric error "
              "rather than a structural bug.")
    elif clean:
        print("\nNo divergence at all on this pool. Either the bug needs "
              "longer generation than this,\nor it is prompt-specific -- check "
              "the generated text before concluding anything.")


if __name__ == "__main__":
    SP = os.environ.get("DSV4_LOG_DIR", ".")
    main(sys.argv[1] if len(sys.argv) > 1 else f"{SP}/pool_ours.log",
         sys.argv[2] if len(sys.argv) > 2 else f"{SP}/pool_ref.log")
