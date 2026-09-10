"""Cross-module gate: variant vs the PINNED incumbent, on identical inputs.

Why this file exists, and why the frozen 68 do not replace it:

**Nothing in the frozen suite ever compares a variant against the incumbent.**
Every id there is either the module against a numpy oracle (`_mismatch`) or the
module against *itself* (`test_emit_pairs_is_bit_identical_to_the_incumbent`
toggles `_reference_stage2_gather` **within the loaded module**, and its
docstring's "incumbent" means the reference-gather formulation, not the shipped
incumbent file). So a green 68 establishes *exactness*; it establishes
**nothing about bit-identity to the incumbent**, which is the gate three of the
four variants are quoted against.

This file loads the pinned incumbent and the variant as two distinct modules in
one process, runs them on byte-identical inputs, and reports **both** verdicts
per cell:

    multiset  -- `_mismatch(...) is None`: is the variant an exact top-k?
                 REQUIRED of every variant. A red here is a correctness failure.
    bit       -- `np.array_equal(incumbent_idx, variant_idx)`: same slots, same
                 order. Required only where the variant claims it.

`v3` is the reason the two are separated: it is exact with a different slot
order, so `bit=False, multiset=True` is a **pass** for `v3` and a **failure**
for `v2`, `v11`, `v12`.

Helpers are IMPORTED from the frozen test file, never retyped -- the score
generators in particular. A paraphrased generator is what produced this
exercise's one retracted finding.

Also carries the two added `all_negative` ids at the P=1 production shapes,
which the frozen suite reaches only at P>1 (`TWO_STAGE_SHAPES`), and reports
`n_negative_winners` so the case's non-vacuity is a measured number rather than
an assertion about a mechanism.

    python xgate.py --name v03 --variant <path> --variant-md5 <md5> \
        --incumbent <path> --incumbent-md5 <md5> --suite <test file> [--json P]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np

# Production shapes at --ctx 9216 (the two `P=1` rows the frozen file's own
# comment names as what production runs), plus one `P>1` row so a variant that
# is P=1-only is *measured* as such rather than assumed.
SHAPES = [(1024, 10240, 2048, 1, "p1-b1024-n10240"),
          (64, 10240, 2048, 1, "p1-b64-n10240"),
          (16, 10240, 2048, 2, "p2-b16-n10240")]
KINDS = ["strict_desc", "random", "heavy_ties", "all_negative", "all_neg_inf"]


def load(path: str, want_md5: str, name: str):
    p = pathlib.Path(path)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    print(f"  {name:11s} {p}  md5={got}", end="")
    if got != want_md5:
        print(f"  MD5_GATE_FAIL want={want_md5}")
        sys.exit(2)
    print("  MD5_GATE_OK")
    spec = importlib.util.spec_from_file_location(f"xgate_{name}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--variant-md5", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--incumbent-md5", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--suite-md5", default="b3241859e471fc87fa6e7a50e4ffec4b")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    print(f"=== XGATE {a.name}: variant vs PINNED incumbent ===")
    suite = load(a.suite, a.suite_md5, "suite")
    inc = load(a.incumbent, a.incumbent_md5, "incumbent")
    var = load(a.variant, a.variant_md5, "variant")
    assert inc is not var
    _scores, _mismatch = suite._scores, suite._mismatch
    print(f"  helpers imported from the frozen suite, not retyped: "
          f"_scores={_scores.__module__} _mismatch={_mismatch.__module__}")

    import jax
    import jax.numpy as jnp

    def run(mod, scores, k, rl, **kw):
        return np.asarray(jax.block_until_ready(mod.sparsecore_topk(
            jnp.asarray(scores), k, jnp.asarray(rl), **kw)))

    rows, fails = [], []
    for b, n, k, want_p, sid in SHAPES:
        got_p = inc._pick_partition(b, n, k)
        assert got_p == want_p, f"{sid}: P={got_p}, expected {want_p}"
        assert var._pick_partition(b, n, k) == want_p, (
            f"{sid}: the variant resolves a different P -- shapes are not "
            "comparable and no cell below would mean anything")
        for kind in KINDS:
            rng = np.random.default_rng(suite.SEED)
            scores = _scores(kind, b, n, rng)
            # Full rows, and the ragged case, which is where a padding
            # convention differs without being wrong.
            edges = sorted({min(max(e, 0), n) for e in
                            (0, 1, suite.LANES, k, n // want_p - 1,
                             n // want_p, n // want_p + 1, n - 1, n)})
            for lname, rl in (("full", np.full(b, n, np.int32)),
                              ("ragged", np.array(
                                  [edges[i % len(edges)] for i in range(b)],
                                  np.int32))):
                i_idx = run(inc, scores, k, rl)
                v_idx = run(var, scores, k, rl)
                bit = bool(np.array_equal(i_idx, v_idx))
                i_bad = _mismatch(scores, i_idx, k, rl)
                v_bad = _mismatch(scores, v_idx, k, rl)
                live = v_idx[v_idx >= 0]
                neg = int((np.take_along_axis(
                    scores, np.clip(v_idx, 0, n - 1), axis=1)[v_idx >= 0] < 0
                    ).sum()) if live.size else 0
                r = {"shape": sid, "kind": kind, "lengths": lname,
                     "bit_equal": bit, "incumbent_exact": i_bad is None,
                     "variant_exact": v_bad is None,
                     "n_live": int(live.size),
                     "n_negative_winners": neg,
                     "variant_reason": v_bad, "incumbent_reason": i_bad}
                rows.append(r)
                flag = ("OK " if v_bad is None else "RED")
                print(f"  {flag} {sid:16s} {kind:12s} {lname:6s} "
                      f"bit={str(bit):5s} exact={v_bad is None} "
                      f"live={live.size:>8d} neg_winners={neg:>8d}"
                      + (f"  <-- {v_bad}" if v_bad else ""))
                if v_bad is not None:
                    fails.append(r)
                if i_bad is not None:
                    print(f"      !! the INCUMBENT is also red here: {i_bad} "
                          "-- this is a harness or image finding, not a "
                          "variant finding")

    # A gate that cannot fail is not a gate: corrupt one index and require both
    # verdicts to reject it.
    print("\n  --- negative control (must go RED) ---")
    b, n, k = 64, 10240, 2048
    rng = np.random.default_rng(suite.SEED)
    scores = _scores("random", b, n, rng)
    rl = np.full(b, n, np.int32)
    good = run(var, scores, k, rl)
    bad = good.copy()
    bad[0, 0] = int(np.argmin(scores[0]))          # a real but losing index
    ctl_bit = bool(np.array_equal(good, bad))
    ctl_ms = _mismatch(scores, bad, k, rl)
    print(f"  bit_equal={ctl_bit} (want False)   multiset={ctl_ms!r} "
          "(want a reason, not None)")
    control_ok = (not ctl_bit) and (ctl_ms is not None)
    print(f"  NEGATIVE_CONTROL_{'OK' if control_ok else 'FAIL'}")

    n_bit = sum(r["bit_equal"] for r in rows)
    print(f"\n=== {a.name}: {len(rows)} cells, {len(rows) - len(fails)} exact, "
          f"{n_bit} bit-identical to the incumbent ===")
    if not fails:
        print(f"XGATE_MULTISET_OK   {a.name} is an exact top-k in every cell")
    else:
        print(f"XGATE_MULTISET_FAIL {a.name}: {len(fails)} cells")
    print(f"XGATE_BIT {'IDENTICAL' if n_bit == len(rows) else 'DIFFERS'}  "
          f"{n_bit}/{len(rows)}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"name": a.name, "variant_md5": a.variant_md5,
             "incumbent_md5": a.incumbent_md5,
             "negative_control_ok": control_ok,
             "n_cells": len(rows), "n_bit_identical": n_bit,
             "n_multiset_fail": len(fails), "rows": rows}, indent=1))
    return 0 if (not fails and control_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
