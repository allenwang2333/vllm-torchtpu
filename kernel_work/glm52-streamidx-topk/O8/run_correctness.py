"""Run the SHARED correctness gate with O8's selector swapped in.

  python3 run_correctness.py --negative-control

`instruments/correctness.py` is imported and never edited or copied. The swap is
a module-attribute rebind: `correctness.py` does
`from ...streamindex_topk import streamindex_topk` at module level and calls the
resulting global inside `run_case`, so rebinding `correctness.streamindex_topk`
redirects the gate onto a clone of the shipping op whose `sparsecore_topk` is
O8's. Nothing in the gate -- cases, reference, thresholds, damage arms -- moves.

The clone is a fresh module object rather than a patch of the shipping module in
place, because `streamindex_topk` is itself a `jax.jit` object: patched in place
the O8 run would share a pjit cache entry with any incumbent run in the same
process and could silently execute the incumbent's executable.

Tolerances live in the gate (`TOL_DEFICIT`, `TOL_CHURN`) and appear nowhere in
this directory.
"""
import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

from api import flat_topk, inc_copy_topk, inc_topk, o8_topk  # noqa: E402

_SIT = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"
_ARMS = dict(inc=inc_topk, inc_copy=inc_copy_topk, flat=flat_topk, o8=o8_topk)


def clone_streamindex(alias, selector):
    spec = importlib.util.find_spec(_SIT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    if selector is not None:
        mod.sparsecore_topk = selector
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="o8", choices=sorted(_ARMS))
    ap.add_argument("--case", default="")
    ap.add_argument("--negative-control", action="store_true")
    a, rest = ap.parse_known_args()

    import correctness  # the shared gate, imported not copied

    sel = None if a.arm == "inc" else _ARMS[a.arm]
    correctness.streamindex_topk = clone_streamindex(
        f"sit_{a.arm}", sel).streamindex_topk
    print(f"== correctness gate, selector arm = {a.arm} ==", flush=True)

    argv = [sys.argv[0]]
    if a.case:
        argv += ["--case", a.case]
    if a.negative_control:
        argv += ["--negative-control"]
    sys.argv = argv + rest
    correctness.main()   # exits non-zero on any failure


if __name__ == "__main__":
    main()
