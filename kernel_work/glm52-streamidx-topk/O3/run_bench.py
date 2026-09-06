"""Measure O3 against the incumbent, adjacent arms in one process, shared instruments.

  python3 run_bench.py --rows B-D4-256k,B-D1-1M --sc
  python3 run_bench.py --rows B-P1024-256k

`instruments/bench.py` is imported, never copied and never edited: `interleaved`
is the statistic (paired difference inside each repeat, medianed over repeats),
because run-order drift here is 3-9% and larger than the block-to-block spread.
`shapes.build` is the input builder, so both arms see byte-identical inputs.

The incumbent runs as an arm in the SAME job on the SAME shape. The two full-op
arms need two distinct module objects, not a monkey-patch in place: the
shipping entry point is itself a `jax.jit` object, so patched in place the two
arms would share one pjit cache entry and the second would time the first
arm's executable. `_clone_module` re-executes the shipping source into a fresh
namespace, which gives a fresh jit object and leaves the shipping file
untouched.
"""
import argparse
import functools
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import tiling  # noqa: E402
from pallas_mosaic_tpu import _get_heuristics_config  # noqa: E402
from pallas_mosaic_tpu_kernel import _select_at_p  # noqa: E402
from pallas_mosaic_tpu_kernel import sparsecore_topk as o3_topk  # noqa: E402

from bench import interleaved, synth_scores  # noqa: E402
from shapes import K, build  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    sparsecore_topk as incumbent_topk  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk as incumbent_full  # noqa: E402

_SIT = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk"


def _clone_module(src_name, alias):
    spec = importlib.util.find_spec(src_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_o3_sit = _clone_module(_SIT, "sit_o3")
_o3_sit.sparsecore_topk = o3_topk
o3_full = _o3_sit.streamindex_topk

_ns_sit = _clone_module(_SIT, "sit_ns")
ns_full = _ns_sit.streamindex_topk

_nh_sit = _clone_module(_SIT, "sit_nh")
nh_full = _nh_sit.streamindex_topk


def noswitch_topk_for(p, tn):
    """PROBE ONLY, not a shipping candidate: O3's chosen rung with no switch.

    `p` is resolved on the host here, which the real call site cannot do --
    `seq_lens` is a device array and reading it would cost a sync. The arm
    exists to price the `lax.switch` itself: it runs exactly the branch O3's
    switch selects, so O3 minus this arm is the conditional and nothing else.
    """

    @functools.partial(jax.jit, static_argnames=("k", "write_empty_rows"))
    def f(scores, k, row_lengths=None, *, write_empty_rows=True):
        B, N = scores.shape
        if row_lengths is None:
            row_lengths = jnp.full((B, ), N, jnp.int32)
        words = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)
        return _select_at_p(words, B, N, k, row_lengths.astype(jnp.int32),
                            p=p, tn=tn, write_empty_rows=write_empty_rows)

    return f


def nohoist_topk(scores, k, row_lengths=None, *, write_empty_rows=True):
    """Iteration 1's shape: the same switch, bitcast+reshape inside the branch."""
    return o3_topk(scores, k, row_lengths, write_empty_rows=write_empty_rows,
                   hoist=False)


def lens_live_for(meta):
    """Per-token effective row length, matching `_effective_row_lengths`."""
    qlen = meta.get("qlen", 1)
    out = np.zeros(meta["T"], np.int32)
    t = 0
    for i in range(meta["R"]):
        for u in range(qlen):
            out[t] = min(int(meta["lens"][i]) - qlen + u + 1, meta["n"])
            t += 1
    return out


def rung_for(T, n, live_max):
    cfg = _get_heuristics_config(T, n, K)
    lad = cfg.ladder
    return cfg.p_max, lad[tiling.ladder_index(live_max, cfg.tn, lad)], cfg.tn


def full_arms(row, noswitch=False, nohoist=False):
    args, kwargs, meta = build(row)
    jax.block_until_ready(args[2])
    live = lens_live_for(meta)
    p_max, p, tn = rung_for(meta["T"], meta["n"], int(live.max()))
    waves0 = tiling.num_waves(meta["T"] * p_max, 32)
    waves1 = tiling.num_waves(meta["T"] * p, 32)
    print(f"  {row}: n={meta['n']} live_max={int(live.max())} "
          f"p {p_max}->{p} (tn={tn})  waves {waves0}->{waves1}  "
          f"X_crit {waves0 * tn}->{waves1 * tn}", flush=True)

    f0 = jax.jit(lambda *x: incumbent_full(*x, **kwargs))
    f1 = jax.jit(lambda *x: o3_full(*x, **kwargs))
    i0, i1 = np.asarray(f0(*args)), np.asarray(f1(*args))
    if bool((i0[0] == np.arange(K)).all()):
        raise SystemExit(f"{row}: took _fast_path -- measures nothing")
    diff = sum(len(set(i0[t][i0[t] >= 0].tolist())
                   ^ set(i1[t][i1[t] >= 0].tolist()))
               for t in range(i0.shape[0]))
    print(f"    filled row0 {int((i0[0] >= 0).sum())}/{K}; O3 vs incumbent "
          f"symmetric difference {diff} of {i0.shape[0] * K} slots "
          f"(identical sets: {diff == 0})", flush=True)
    arms = [(f"{row} incumbent", lambda: f0(*args)),
            (f"{row} O3 p={p}", lambda: f1(*args))]
    if nohoist:
        _nh_sit.sparsecore_topk = nohoist_topk
        f3 = jax.jit(lambda *x: nh_full(*x, **kwargs))
        i3 = np.asarray(f3(*args))
        d3 = sum(len(set(i0[t][i0[t] >= 0].tolist())
                     ^ set(i3[t][i3[t] >= 0].tolist()))
                 for t in range(i0.shape[0]))
        print(f"    it1 no-hoist arm: symmetric difference vs incumbent {d3}",
              flush=True)
        arms.append((f"{row} O3-it1-nohoist p={p}", lambda: f3(*args)))
    if noswitch:
        _ns_sit.sparsecore_topk = noswitch_topk_for(p, tn)
        f2 = jax.jit(lambda *x: ns_full(*x, **kwargs))
        i2 = np.asarray(f2(*args))
        d2 = sum(len(set(i0[t][i0[t] >= 0].tolist())
                     ^ set(i2[t][i2[t] >= 0].tolist()))
                 for t in range(i0.shape[0]))
        print(f"    no-switch probe (p={p} resolved on host): symmetric "
              f"difference vs incumbent {d2}", flush=True)
        arms.append((f"{row} O3-noswitch p={p}", lambda: f2(*args)))
    return arms, meta, live, diff


def sc_arms(row, meta, live, noswitch=False):
    """Selector-only arms on the same geometry: where O3's mechanism lives."""
    s = synth_scores(meta["T"], meta["n"], live)
    jax.block_until_ready(s)
    rl = jnp.asarray(live, jnp.int32)
    g0 = jax.jit(lambda x, r: incumbent_topk(x, K, r))
    g1 = jax.jit(lambda x, r: o3_topk(x, K, r))
    a, b = np.asarray(g0(s, rl)), np.asarray(g1(s, rl))
    diff = sum(len(set(a[t][a[t] >= 0].tolist()) ^ set(b[t][b[t] >= 0].tolist()))
               for t in range(a.shape[0]))
    print(f"    sc-level symmetric difference {diff} of {a.shape[0] * K} slots",
          flush=True)
    arms = [(f"{row} sc incumbent", lambda: g0(s, rl)),
            (f"{row} sc O3", lambda: g1(s, rl))]
    if noswitch:
        _, p, tn = rung_for(meta["T"], meta["n"], int(live.max()))
        ns = noswitch_topk_for(p, tn)
        g2 = jax.jit(lambda x, r: ns(x, K, r))
        jax.block_until_ready(g2(s, rl))
        arms.append((f"{row} sc O3-noswitch", lambda: g2(s, rl)))
    return arms, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--repeats", type=int,
                    default=int(os.environ.get("REPEATS", 5)))
    ap.add_argument("--iters", type=int,
                    default=int(os.environ.get("ITERS", 20)))
    ap.add_argument("--sc", action="store_true",
                    help="also time the selector alone on synthetic scores")
    ap.add_argument("--noswitch", action="store_true",
                    help="add the host-resolved-p probe arm that prices "
                         "the lax.switch (probe, not a shipping candidate)")
    ap.add_argument("--nohoist", action="store_true",
                    help="add iteration 1's arm: bitcast+reshape inside the "
                         "switch branch instead of hoisted above it")
    a = ap.parse_args()

    for row in [r.strip() for r in a.rows.split(",")]:
        print(f"\n{'=' * 74}\n{row}\n{'=' * 74}", flush=True)
        arms, meta, live, _ = full_arms(row, noswitch=a.noswitch,
                                        nohoist=a.nohoist)
        print("\n-- full op, arms adjacent --", flush=True)
        interleaved(arms, repeats=a.repeats, iters=a.iters, ref=arms[0][0])
        if a.sc:
            sarms, _ = sc_arms(row, meta, live, noswitch=a.noswitch)
            print("\n-- sparsecore_topk alone, arms adjacent --", flush=True)
            interleaved(sarms, repeats=a.repeats, iters=a.iters,
                        ref=sarms[0][0])


if __name__ == "__main__":
    main()
