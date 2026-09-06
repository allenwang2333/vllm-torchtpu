"""Run the SHARED correctness instrument against the O1 candidate.

  python3 run_correctness.py                       # all cases, all switches on
  python3 run_correctness.py --arm copy-off        # the copy-fidelity control
  python3 run_correctness.py --case dec-2x8k
  python3 run_correctness.py --negative-control    # re-validate the gate on O1

This file exists so that `../instruments/correctness.py` never has to be
touched. The instrument imports the shipping `streamindex_topk` at module scope
and calls it by that name; rebinding the module attribute from outside is
running the instrument, not editing it. The thresholds, the reference, the
cases and the negative control all remain the instrument's, unchanged, which is
the only way the result means anything: a variant that changes the test has not
passed, it has moved the gate.
"""
import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, "instruments"))

import correctness  # noqa: E402  the shared, read-only gate

import api  # noqa: E402

# Every arm the microbenchmark times, so any of them can be gated. `copy-off`
# is the one worth running even though it is expected to be identical to the
# incumbent: if the copy is not faithful, no other arm's number means anything.
ARMS = {
    "o1": dict(fix_growth=True, skip_empty_launches=True,
               elide_score_prefill=True),
    "copy-off": dict(fix_growth=False, skip_empty_launches=False,
                     elide_score_prefill=False),
    "pick": dict(fix_growth=True, skip_empty_launches=False,
                 elide_score_prefill=False),
    "cond": dict(fix_growth=False, skip_empty_launches=True,
                 elide_score_prefill=False),
    "fill": dict(fix_growth=False, skip_empty_launches=False,
                 elide_score_prefill=True),
}


def main():
    argv, arm = [], "o1"
    it = iter(sys.argv[1:])
    for tok in it:
        if tok == "--arm":
            arm = next(it)
        elif tok.startswith("--arm="):
            arm = tok.split("=", 1)[1]
        else:
            argv.append(tok)
    if arm not in ARMS:
        raise SystemExit(f"unknown --arm {arm!r}; have {sorted(ARMS)}")

    print(f"=== O1 correctness, arm={arm} {ARMS[arm]} ===", flush=True)
    print(f"    gate: {correctness.__file__} "
          f"(TOL_DEFICIT={correctness.TOL_DEFICIT}, "
          f"TOL_CHURN={correctness.TOL_CHURN})", flush=True)

    # The substitution. `correctness.run_case` calls `streamindex_topk(...)` by
    # module attribute, so this is the whole hook -- no import machinery, no
    # copy of the instrument, and the tolerances above are printed from the
    # instrument's own module so the log shows what was actually enforced.
    correctness.streamindex_topk = functools.partial(api.streamindex_topk,
                                                     **ARMS[arm])

    sys.argv = [sys.argv[0]] + argv
    correctness.main()   # exits non-zero on any failure


if __name__ == "__main__":
    main()
