#!/usr/bin/env python3
"""Side-by-side device-time breakdown of one profiled phase across two arms.

  python3 scripts/k3/arm_phase_diff.py results/<control> results/<treatment> decode_only

Reads the caches written by hlo_cache.py. Two things it does that a naive sum
does not:

* **Splits by Program id.** A `prefill_only` capture also contains the decode
  program -- in the control arm that was 291 ms of decode sitting inside a
  352 ms prefill capture. Summing the file conflates them.
* **Buckets by what the op is**, not by HLO category, because the question here
  is "what did moving the KDA state into the attention pool cost", and that
  work is spread across custom-calls, fusions and copies that share categories
  with everything else.

Occurrence counts are the sanity check on the comparison: the MLA ops fire once
per full-attention layer per step (24 for K3) and the KDA ops once per linear
layer per step (69), so occ/24 and occ/69 give the number of steps captured and
the two arms must agree on it before any delta means anything.
"""
import collections
import json
import re
import sys

T = "Total self time (us)"
OCC = "#Occurrences"
CAT = "HLO op category"
NAME = "HLO op name"

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
               "collective-permute", "collective-permute-start",
               "collective-permute-done")

# Ordered: first match wins.
BUCKETS = [
    ("MLA attention", lambda c, n: n.startswith("MLA-")),
    ("KDA pooled state I/O", lambda c, n: "pooled" in n or "region" in n),
    ("KDA recurrent kernel", lambda c, n: "kda" in n),
    ("MoE grouped matmul", lambda c, n: n.startswith("gmm_v")),
    ("MoE expert gather", lambda c, n: "ragged_gather" in n),
    ("MoE routing sort", lambda c, n: c == "sort"),
    ("collectives", lambda c, n: c in COLLECTIVES),
    ("KV/state cache writes", lambda c, n: "dynamic-update-slice" in n),
    ("convolution fusion", lambda c, n: c == "convolution fusion"),
    ("data formatting", lambda c, n: c in ("data formatting", "copy", "slice",
                                           "pad")),
]


def bucket(cat, name):
    for label, pred in BUCKETS:
        if pred(cat, name):
            return label
    return "other"


def load(arm, phase):
    """Rows for the program this phase is actually about.

    A `prefill_only` capture also contains the decode program, and picking the
    largest program by device time silently picks decode when the profiled
    request decodes more than it prefills -- which is how this first reported a
    2-step control against a 13-step treatment. So for the prefill phase,
    exclude every program that also appears in the same arm's decode capture.
    """
    with open(f"{arm}/hlo-{phase}.json") as fh:
        rows = json.load(fh)["rows"]
    exclude = set()
    if phase == "prefill_only":
        try:
            with open(f"{arm}/hlo-decode_only.json") as fh:
                exclude = {str(r["Program id"]) for r in json.load(fh)["rows"]}
        except FileNotFoundError:
            print(f"warning: no decode cache for {arm}; cannot separate the "
                  f"decode program out of the prefill capture")
    by_prog = collections.defaultdict(float)
    for r in rows:
        pid = str(r["Program id"])
        if pid not in exclude:
            by_prog[pid] += float(r[T] or 0)
    if not by_prog:
        raise SystemExit(f"{arm}/{phase}: every program also appears in the "
                         f"decode capture; nothing left to attribute")
    prog = max(by_prog, key=by_prog.get)
    keep = [r for r in rows if str(r["Program id"]) == prog]
    return prog, keep


def summarise(rows):
    out = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        name = re.sub(r"\.\d+$", "", str(r[NAME] or "?"))
        b = bucket(str(r[CAT]), name)
        out[b][0] += float(r[T] or 0)
        out[b][1] += int(r[OCC] or 0)
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    ctl_dir, trt_dir, phase = sys.argv[1], sys.argv[2], sys.argv[3]
    ctl_prog, ctl_rows = load(ctl_dir, phase)
    trt_prog, trt_rows = load(trt_dir, phase)
    c, t = summarise(ctl_rows), summarise(trt_rows)
    ctot = sum(v[0] for v in c.values())
    ttot = sum(v[0] for v in t.values())

    def steps(rows, prefix, layers):
        # One variant only. XLA compiles the MLA kernel three times (decode,
        # mixed, batched-decode) behind a `conditional` and the profiler
        # records an occurrence for each, so counting every name starting
        # "MLA-" triples the step count.
        n = sum(int(r[OCC] or 0) for r in rows
                if str(r[NAME] or "").startswith(prefix))
        return n / layers if layers else 0

    print(f"phase={phase}")
    print(f"control   {ctl_dir:<28} program {ctl_prog}  {ctot/1e3:8.2f} ms")
    print(f"treatment {trt_dir:<28} program {trt_prog}  {ttot/1e3:8.2f} ms")
    print(f"\ncaptured steps: MLA-d/24 layers -> "
          f"control {steps(ctl_rows, 'MLA-d-', 24):.1f}, "
          f"treatment {steps(trt_rows, 'MLA-d-', 24):.1f};  "
          f"KDA/69 layers -> control {steps(ctl_rows, 'kda_', 69):.1f}, "
          f"treatment {steps(trt_rows, 'kda_', 69):.1f}")
    print("per-step totals are only comparable if those agree\n")

    labels = sorted(set(c) | set(t), key=lambda k: -(t.get(k, [0])[0]
                                                     + c.get(k, [0])[0]))
    print(f"{'bucket':<24}{'control ms':>12}{'treat ms':>11}{'delta ms':>11}"
          f"{'delta %':>10}{'ctl occ':>10}{'trt occ':>10}")
    print("-" * 88)
    for label in labels:
        cv, co = c.get(label, [0.0, 0])
        tv, to = t.get(label, [0.0, 0])
        d = (tv - cv) / 1e3
        pct = f"{100 * (tv / cv - 1):+.1f}%" if cv else ("new" if tv else "-")
        print(f"{label:<24}{cv/1e3:12.3f}{tv/1e3:11.3f}{d:+11.3f}{pct:>10}"
              f"{co:10d}{to:10d}")
    print("-" * 88)
    print(f"{'TOTAL':<24}{ctot/1e3:12.3f}{ttot/1e3:11.3f}"
          f"{(ttot - ctot)/1e3:+11.3f}"
          f"{100 * (ttot / ctot - 1):+9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
