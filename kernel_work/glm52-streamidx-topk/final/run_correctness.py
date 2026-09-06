"""Run the SHARED correctness gate against the composed candidate, plus a
symmetric-difference check of the selected index SET against the incumbent.

  python3 run_correctness.py --arm final --negative-control
  python3 run_correctness.py --arm final --symdiff-only

`instruments/correctness.py` is imported and never edited or copied. The swap is
a module-attribute rebind: the gate does
`from ...streamindex_topk import streamindex_topk` at module level and calls the
resulting global inside `run_case`, so rebinding `correctness.streamindex_topk`
redirects it onto this candidate. Nothing in the gate -- cases, reference,
metrics, damage arms, thresholds -- moves. Thresholds live in the gate and
appear nowhere in this directory.

Two things are checked, and they are not the same thing:

  the gate       deficit and churn against a recomputed float reference, which
                 is what a top-k kernel can be held to: ties at the cut are
                 allowed to break differently.
  symdiff        the selected index set against THE INCUMBENT's, exactly, per
                 token row. Every option composed here is supposed to change
                 how the top-k is found and not what it is, so the expected
                 answer is 0 at every shape. A non-zero symdiff on a passing
                 gate is a FINDING to be explained -- which composition step
                 moved the set, and whether the movement is confined to score
                 ties -- and not an argument about where a threshold should
                 sit.

The incumbent arm is the shipping module, imported. It is loaded into a
separate module object rather than patched, because `streamindex_topk` is a
`jax.jit` object: sharing one in a process where two arms run would let arm 2
execute the executable arm 1 compiled.
"""
import argparse
import dataclasses
import functools
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import api  # noqa: E402
import pallas_mosaic_tpu as policy  # noqa: E402

# The shipping op and the shipping selector, imported untouched.
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk as inc_streamindex)

_ALL_ON = policy.Config()
_ALL_OFF = policy.Config(emit_keys=False, pack_waves=False,
                         scan_count_hist=False, fix_growth=False,
                         skip_empty_launches=False, elide_score_prefill=False,
                         wide_kv_tile=False)


def _off(**kw):
    return dataclasses.replace(_ALL_ON, **kw)


# The arms. `inc` is the shipping op; every other arm is this candidate with
# one mechanism removed, so a failure attributes to the mechanism named.
ARMS = {
    "inc": None,
    "inc_copy": _ALL_OFF,
    "final": _ALL_ON,
    "no-o8": _off(emit_keys=False),
    "no-o3": _off(pack_waves=False),
    "no-o7": _off(scan_count_hist=False),
    "no-o1": _off(fix_growth=False, skip_empty_launches=False,
                  elide_score_prefill=False),
    "no-o2": _off(wide_kv_tile=False),
}

# The shipping selector's source, as it stood when this candidate was composed.
# `inc` is only a baseline if it is the code that ships; a silent upgrade of
# the checkout underneath would turn every paired delta in the job into a
# comparison against something else.
_INC_SHA = "3d058de21236651b526eb56041bbead607e0b35da6e9e2d76164c4637b534806"


def check_incumbent_sha() -> str:
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sct
    with open(_sct.__file__, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    print(f"  incumbent sparsecore_topk.py sha256 {got}"
          f" {'== expected' if got == _INC_SHA else '!= EXPECTED ' + _INC_SHA}",
          flush=True)
    return got


def arm_callable(name):
    if name == "inc":
        return inc_streamindex
    return functools.partial(api.streamindex_topk, cfg=ARMS[name])


def symdiff_vs_incumbent(names, arm):
    """Per-token symmetric difference of the selected index sets, arm vs inc.

    Runs both arms over the shared gate's own cases and inputs, so the
    comparison is on the geometry the gate covers and not on a shape invented
    here.
    """
    import correctness as gate
    fn_a, fn_inc = arm_callable(arm), arm_callable("inc")
    kw = dict(k=gate.K, compression_ratio=gate.CR,
              num_kv_pages_per_block=gate.BKV_P,
              num_queries_per_block=gate.BQ_SZ,
              vmem_limit_bytes=gate.VMEM_LIMIT,
              decode_req_batch_size=gate.DECODE_REQ_BATCH_SIZE,
              enable_early_exit=True)
    print(f"\n-- index-set symmetric difference, {arm} vs inc --", flush=True)
    total = 0
    for name in names:
        args, m = gate.build_case(gate.CASES[name])
        a = np.asarray(fn_a(*args, **kw))
        b = np.asarray(fn_inc(*args, **kw))
        n_live = int(m["cu"][m["R"]])
        sd = rows = 0
        for t in range(n_live):
            sa = set(a[t][a[t] >= 0].tolist())
            sb = set(b[t][b[t] >= 0].tolist())
            d = len(sa ^ sb)
            sd += d
            rows += bool(d)
        total += sd
        print(f"  {name:14} symdiff {sd:8d} over {n_live} live rows"
              f"  ({rows} rows differ)", flush=True)
    verdict = ("IDENTICAL SETS" if total == 0 else
               "SETS MOVED -- explain which composition step, do not retune")
    print(f"  total symdiff {total}: {verdict}", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="final", choices=sorted(ARMS))
    ap.add_argument("--case", default="")
    ap.add_argument("--negative-control", action="store_true")
    ap.add_argument("--symdiff-only", action="store_true")
    ap.add_argument("--no-symdiff", action="store_true")
    a, rest = ap.parse_known_args()

    import correctness  # the shared gate, imported not copied

    print(f"== correctness, arm = {a.arm} ==", flush=True)
    if ARMS[a.arm] is not None:
        print(f"  cfg = {ARMS[a.arm]}", flush=True)
    check_incumbent_sha()

    names = [a.case] if a.case else list(correctness.CASES)

    sd = 0
    if not a.no_symdiff and a.arm != "inc":
        sd = symdiff_vs_incumbent(names, a.arm)
    if a.symdiff_only:
        sys.exit(0)

    correctness.streamindex_topk = arm_callable(a.arm)
    argv = [sys.argv[0]]
    if a.case:
        argv += ["--case", a.case]
    if a.negative_control:
        argv += ["--negative-control"]
    sys.argv = argv + rest
    try:
        correctness.main()   # exits non-zero on any failure
    except SystemExit as e:
        code = e.code or 0
        # The symdiff is reported, and it is advisory to the gate rather than
        # part of it: the gate is the shared instrument and this file does not
        # get to tighten it. A non-zero symdiff on a passing gate is printed
        # loudly above and must be explained in the report.
        if sd:
            print(f"\nNOTE: gate exit {code} with symdiff {sd} != 0 -- the "
                  f"selected set moved; see the block above.", flush=True)
        raise


if __name__ == "__main__":
    main()
