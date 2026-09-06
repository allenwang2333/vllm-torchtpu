"""Run ../instruments/correctness.py, unmodified, against one O9 arm.

The instrument is loaded as a module and its `streamindex_topk` symbol rebound
to the arm. Nothing in instruments/ is edited, copied or re-tolerance'd: the
cases, the f32 reference, TOL_DEFICIT and TOL_CHURN are the shared ones, so a
pass here is a pass on the gate every other variant is judged by. No tolerance
is defined in this directory.

    python3 run_correctness.py --arm c4d2 --negative-control

Ablation arms (`dma_denom > 1`) are refused: their output is deliberately wrong
and gating them would be theatre.
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arms  # noqa: E402

INSTRUMENTS = os.environ.get(
    "INST_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "instruments"))


def load_instrument(name):
    path = os.path.join(INSTRUMENTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"instr_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(arms.CONFIGS))
    ap.add_argument("--case", default="")
    ap.add_argument("--negative-control", action="store_true")
    a = ap.parse_args()

    if a.arm in arms.ABLATION_ARMS:
        raise SystemExit(f"{a.arm} is an ablation arm -- its output is "
                         f"deliberately wrong and it is not a candidate")

    corr = load_instrument("correctness")
    fn = arms.full_op(a.arm, {})
    prev = arms._sit.sparsecore_topk
    arms._sit.sparsecore_topk = arms.selector(a.arm)
    corr.streamindex_topk = fn

    cfg = arms.CONFIGS[a.arm]
    print(f"=== arm {a.arm}  cfg={cfg if cfg is not None else 'INCUMBENT (src)'}",
          flush=True)
    print(f"    gate: {os.path.realpath(corr.__file__)}  "
          f"TOL_DEFICIT={corr.TOL_DEFICIT} TOL_CHURN={corr.TOL_CHURN}",
          flush=True)

    names = [a.case] if a.case else list(corr.CASES)
    allf, clean = [], []
    try:
        for n in names:
            allf += corr.run_case(n, stats=clean)
        n_pass = len(names) - len(set(f.split()[0] for f in allf))
        print(f"\n{n_pass}/{len(names)} cases passed, {len(allf)} assertion "
              f"failures", flush=True)
        if a.negative_control:
            allf += corr.negative_control(names, clean)
    finally:
        arms._sit.sparsecore_topk = prev

    print(f"\nARM {a.arm}: {'PASS' if not allf else 'FAIL'}", flush=True)
    sys.exit(1 if allf else 0)


if __name__ == "__main__":
    main()
