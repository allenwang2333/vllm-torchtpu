"""Option C measurement 4: correctness.

Two gates, in order.

GATE 1 -- the existing suite, `tests/kernels/test_dsv4_streamindex_topk.py`,
**unmodified**.  The test file does

    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

at its own import time, so binding the variant onto the *source module's*
attribute before pytest collects the file makes the suite exercise Option C
without a character of the suite changing.  The suite is the incumbent's own
acceptance bar, including `test_streamindex_topk_quantized`, which asserts
`assert_array_equal` on the sorted index set -- i.e. it demands the exact same
top-k membership, not an approximation.

    --mode pytest --arm incumbent        control; must reproduce 13 passed
    --mode pytest --arm head_major       the gate

GATE 2 -- only meaningful if gate 1 fails, and only if it fails on
tie-adjacent rows.  Reordering a 32-term float sum changes the last bits of
every score, so two positions whose true scores differ by less than that can
swap places in the top-k.  That is a tie-flip, not a defect.  Gate 2
distinguishes the two by scoring in **float64** -- where the reordering is
below the noise floor -- and asking two questions of each arm:

  * index-set recall against the f64 oracle's top-k, per row, minimum over rows
  * for every miss, the f64 score gap between the position the arm chose and
    the position the oracle chose.  A defect moves a position that is not near
    the boundary; a tie-flip moves one that is.

Recall alone cannot tell them apart, which is why the gap is reported too.

Neither gate is a timing.  Nothing here goes near `/log-experiment`.
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _p(*rest):
    print("[C-correct]", *rest, flush=True)


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------


def bind_layout(sxc, arm: str):
    """A jitted entry point with `q_layout` bound, as a real `PjitFunction`.

    It has to be a `PjitFunction` and not a `functools.partial` around one.
    `test_dsv4_streamindex_topk.py:208` calls `jax.eval_shape(streamindex_topk,
    ...)`; `eval_shape` recognises a jitted callable and honours its
    `static_argnames`, but any other callable it wraps in a *second* `jit` that
    does not know which arguments are static.  The outer jit then turns `k=512`
    and friends into tracers, and the inner jit refuses to hash them:

        ValueError: Non-hashable static arguments are not supported ...
        TypeError: unhashable type: DynamicJaxprTracer

    That failure is a property of the wrapper, not of the kernel -- it fired
    identically on `head_minor`, which is the incumbent's own code path.  So it
    proved nothing about Option C and had to be fixed before gate 1 could
    discriminate anything.
    """
    import jax

    inner = getattr(sxc.streamindex_topk, "__wrapped__", None)
    if inner is None:
        raise RuntimeError(
            "streamindex_topk has no __wrapped__; cannot rebuild the jit "
            "without silently changing which arguments are static")
    static = tuple(n for n in sxc.STREAMINDEX_TOPK_STATIC_ARGNAMES
                   if n != "q_layout")
    return jax.jit(functools.partial(inner, q_layout=arm),
                   static_argnames=static)


def gate1(arm: str, test_file: str, extra_args) -> int:
    import pytest

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sx
    import streamindex_topk_c as sxc

    if arm == "incumbent":
        _p("control: the unmodified src kernel, no patch applied")
    else:
        sx.streamindex_topk = bind_layout(sxc, arm)
        _p(f"patched vllm_torchtpu...streamindex_topk.streamindex_topk -> "
           f"jit(streamindex_topk_c(q_layout={arm!r})) "
           f"type={type(sx.streamindex_topk).__name__}")
        # The patch has to land before pytest imports the test module, or the
        # suite's `from ... import streamindex_topk` binds the incumbent and
        # the whole gate silently tests the wrong kernel.
        modname = "tests.kernels.test_dsv4_streamindex_topk"
        assert modname not in sys.modules, (
            "test module already imported -- the patch would be a no-op")

    _p("pytest", test_file)
    rc = pytest.main(["-q", "-p", "no:randomly", test_file, *extra_args])
    _p(f"arm={arm} pytest exit={int(rc)}")
    return int(rc)


# ---------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------

G2 = dict(T_seq=8, S_seq=8192, comp_ratio=4, page_size=16, H_I=32, H_KV=1,
          D=128, k=512, bkv_p=8, bq_sz=8)


def _build_g2(cfg, seed=0):
    """Inputs plus the exact f64 dequantized keys the hardware will see."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    T, S, cr, ps = cfg["T_seq"], cfg["S_seq"], cfg["comp_ratio"], cfg[
        "page_size"]
    H, D, width = cfg["H_I"], cfg["D"], 256
    S_valid = S // cr
    num_pages = (S_valid + ps - 1) // ps

    q_f32 = rng.standard_normal((T, H, D)).astype(np.float32)
    q = jnp.asarray(q_f32).astype(jnp.float8_e4m3fn)
    # the oracle must score exactly what the kernel scores, so it reads the
    # fp8 values back rather than the f32 they were rounded from
    q_exact = np.asarray(jnp.asarray(q, jnp.float32)).astype(np.float64)

    w_f32 = rng.uniform(0.5, 1.5, size=(T, H)).astype(np.float32)
    w = jnp.asarray(w_f32).astype(jnp.bfloat16)
    w_exact = np.asarray(jnp.asarray(w, jnp.float32)).astype(np.float64)

    kv_f32 = rng.standard_normal((num_pages * ps, D)).astype(np.float32)
    # block fp8 + ue8m0 scale, one scale per D=128 row, vectorised
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = np.clip(np.abs(kv_f32).max(axis=-1, keepdims=True), 1e-4, None)
    scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
    kq = jnp.asarray(kv_f32 / scale).astype(jnp.float8_e4m3fn)
    kq_bytes = np.asarray(jax.lax.bitcast_convert_type(kq, jnp.uint8))
    scale_e8 = jnp.asarray(scale[:, 0]).astype(jnp.float8_e8m0fnu)
    scale_byte = np.asarray(jax.lax.bitcast_convert_type(scale_e8, jnp.uint8))
    kv_exact = (np.asarray(jnp.asarray(kq, jnp.float32)).astype(np.float64) *
                np.asarray(jnp.asarray(scale_e8, jnp.float32)).astype(
                    np.float64)[:, None])

    record = np.zeros((num_pages * ps, width), np.uint8)
    record[:, :D] = kq_bytes
    record[:, D] = scale_byte
    cache = record.reshape(num_pages, ps // 4, 4, width)

    return dict(
        q=q, w=w, cache_kv=jnp.asarray(cache),
        seq_lens=jnp.asarray([S], jnp.int32),
        page_indices=jnp.arange(num_pages, dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, T], jnp.int32),
        distribution=jnp.asarray([0, 0, 1], jnp.int32),
        q_exact=q_exact, w_exact=w_exact, kv_exact=kv_exact[:S_valid],
        S_valid=S_valid, num_pages=num_pages)


def _oracle(b, cfg):
    """float64 scores, with the kernel's visibility mask applied.

    The reassociation Option C performs is ~1e-16 here, so anything this
    disagrees with the kernel about by more than that is either a real
    selection difference or a gap in this model of the kernel -- and the two
    are told apart by whether the *incumbent* disagrees too.
    """
    import numpy as np
    inner = np.einsum("thd,sd->ths", b["q_exact"], b["kv_exact"])
    scored = np.maximum(inner, 0.0) * b["w_exact"][:, :, None]
    s = scored.sum(axis=1)  # (T, S_valid)
    # `kernel` masks position s_idx for token t when
    # s_idx * comp_ratio > q_abs_pos, with q_abs_pos = (S_total - T) + t.
    cr, T = cfg["comp_ratio"], cfg["T_seq"]
    q_abs = (cfg["S_seq"] - T) + np.arange(T)
    s_idx = np.arange(s.shape[1])
    vis = s_idx[None, :] * cr <= q_abs[:, None]
    return np.where(vis, s, -np.inf), int((~vis).sum())


def gate2(arms, cfg, seed=0):
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sx
    import streamindex_topk_c as sxc

    b = _build_g2(cfg, seed)
    k, T = cfg["k"], cfg["T_seq"]
    scores, n_masked = _oracle(b, cfg)
    order = np.argsort(-scores, axis=-1)
    oracle_top = order[:, :k]
    # the f64 score at the k-th and (k+1)-th place: the decision boundary
    boundary = np.take_along_axis(scores, order[:, k - 1:k + 1], axis=-1)
    _p(f"oracle built: T={T} S_valid={b['S_valid']} k={k} H={cfg['H_I']} "
       f"masked={n_masked}; "
       f"boundary gap min={float(np.min(boundary[:,0]-boundary[:,1])):.3e} "
       f"max={float(np.max(boundary[:,0]-boundary[:,1])):.3e}")

    got_sets, rows = {}, []
    for arm in arms:
        fn = (sx.streamindex_topk if arm == "incumbent"
              else bind_layout(sxc, arm))
        out = np.asarray(fn(
            q=b["q"], indexer_weights=b["w"], cache_kv=b["cache_kv"],
            seq_lens=b["seq_lens"], page_indices=b["page_indices"],
            cu_q_lens=b["cu_q_lens"], distribution=b["distribution"],
            k=k, compression_ratio=cfg["comp_ratio"],
            num_kv_pages_per_block=cfg["bkv_p"],
            num_queries_per_block=cfg["bq_sz"]))
        got_sets[arm] = [set(int(i) for i in out[t] if i >= 0)
                         for t in range(T)]
        recalls, worst_gap, n_missed = [], 0.0, 0
        for t in range(T):
            want = set(int(i) for i in oracle_top[t])
            hit = len(got_sets[arm][t] & want)
            recalls.append(hit / max(1, len(want)))
            missed = want - got_sets[arm][t]
            n_missed += len(missed)
            for i in missed:
                # how far above the (k+1)-th score was the item it dropped?
                # near zero => a boundary tie; large => a real selection miss.
                worst_gap = max(worst_gap,
                                float(scores[t, i] - boundary[t, 1]))
        rows.append(dict(arm=arm, min_recall=float(np.min(recalls)),
                         mean_recall=float(np.mean(recalls)),
                         missed_total=n_missed,
                         worst_miss_gap_above_boundary=worst_gap))
        _p(f"{arm:16} vs f64 oracle: min_recall={rows[-1]['min_recall']:.6f} "
           f"mean={rows[-1]['mean_recall']:.6f} missed={n_missed} "
           f"worst_gap={worst_gap:.3e}")

    # ---- the statistic that actually decides Option C ----------------------
    # Recall against the oracle is a joint test of the kernel's selection AND
    # of this oracle's model of it, and the incumbent is subject to both.  The
    # question Option C has to answer is narrower and is answered here: does
    # the reassociation change the index set the *incumbent* produces?  Any
    # oracle modelling gap cancels, because both arms meet it identically.
    ref = "incumbent" if "incumbent" in got_sets else arms[0]
    _p(f"-- index-set agreement vs {ref} (the discriminating statistic) --")
    for arm in arms:
        diff_rows, sym, worst_swap = 0, 0, 0.0
        for t in range(T):
            d = got_sets[arm][t] ^ got_sets[ref][t]
            if d:
                diff_rows += 1
                sym += len(d)
                for i in d:
                    worst_swap = max(
                        worst_swap,
                        abs(float(scores[t, i] - boundary[t, 1])))
        r = next(x for x in rows if x["arm"] == arm)
        r.update(vs_ref=ref, rows_differing=diff_rows,
                 indices_differing=sym,
                 worst_swap_gap_above_boundary=worst_swap,
                 identical_to_ref=(sym == 0))
        _p(f"{arm:16} rows_differing={diff_rows}/{T} indices_differing={sym} "
           f"worst_swap_gap={worst_swap:.3e} "
           f"identical={'YES' if sym == 0 else 'NO'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pytest", "recall"), required=True)
    ap.add_argument("--arm", default="head_major")
    ap.add_argument("--arms", default="incumbent,head_minor,head_major")
    ap.add_argument("--test-file", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("rest", nargs="*")
    args = ap.parse_args()

    if args.mode == "pytest":
        tf = args.test_file
        if not tf:
            root = _HERE.parents[3]
            tf = str(root / "tests" / "kernels" /
                     "test_dsv4_streamindex_topk.py")
        if not os.path.exists(tf):
            _p("MISSING test file", tf)
            return 3
        return gate1(args.arm, tf, args.rest)

    rows = gate2([a for a in args.arms.split(",") if a], G2)
    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump(dict(config=G2, arms=rows), fh, indent=2)
        _p("wrote", args.json)
    # Verdict.  §3.2's predicate is "min-row index-set recall < 0.999 -> revert",
    # and it is reported below -- but it is applied RELATIVE to the incumbent,
    # not absolutely.  An absolute threshold that the incumbent itself fails is
    # not a gate on Option C; it is a gate on the oracle, and a check that
    # fails on the baseline establishes nothing either way.
    ref = next((r for r in rows if r["arm"] == "incumbent"), None)
    verdict, bad = [], False
    for r in rows:
        if r["arm"] == "incumbent":
            verdict.append(f"{r['arm']}: baseline, min_recall="
                           f"{r['min_recall']:.6f}"
                           f"{'  (< 0.999 -- the ORACLE, not Option C)' if r['min_recall'] < 0.999 else ''}")
            continue
        ok_ident = r.get("identical_to_ref", False)
        ok_recall = ref is None or r["min_recall"] >= ref["min_recall"] - 1e-12
        bad = bad or not (ok_ident and ok_recall)
        ref_rc = f"{ref['min_recall']:.6f}" if ref else "n/a"
        verdict.append(
            f"{r['arm']}: index-set identical to incumbent="
            f"{'YES' if ok_ident else 'NO'}, "
            f"min_recall {r['min_recall']:.6f} vs incumbent {ref_rc}")
    _p("-- GATE 2 VERDICT --")
    for v in verdict:
        _p(v)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
