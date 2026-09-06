"""Time O8 against `inc`, `inc_copy` and `flat`, adjacently, in one process.

  python3 run_bench.py --rows B-D4-256k --level sc,full

`instruments/bench.py` supplies the statistic and is imported, never copied and
never edited. `interleaved` times every arm back to back inside each repeat and
differences within the repeat, because run-order drift on this block is 3-9% --
larger than the effect being chased. Nothing here forms a delta between two
separately measured numbers.

Arm order is `inc, inc_copy, flat, o8` so the headline pair (`flat`, `o8`) is
adjacent. `interleaved` reports deltas against one reference; `paired()` below
re-differences the SAME per-repeat medians against a second reference, which is
still a within-repeat difference and never a ratio of two medians.

Two levels, because they answer different questions:
  sc    `sparsecore_topk` alone on a synthetic score matrix -- where the change
        lives, and the only level at which it is not diluted.
  full  the whole `streamindex_topk`, with the selector rebound in a cloned
        module. Says how much of the sc-level effect survives the scoring pass.
"""
import argparse
import functools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.environ.get(
    "INST_DIR", os.path.join(_HERE, "..", "instruments")))

import tiling  # noqa: E402
from api import flat_topk, inc_copy_topk, inc_topk, o8_topk  # noqa: E402
from run_correctness import clone_streamindex  # noqa: E402

from bench import interleaved, synth_scores  # noqa: E402
from shapes import K, ROWS, build  # noqa: E402

ARMS = (("inc", inc_topk), ("inc_copy", inc_copy_topk),
        ("flat", flat_topk), ("o8", o8_topk))


def paired(out, ref):
    """Re-difference the per-repeat medians `interleaved` returned, vs `ref`.

    Same arithmetic `interleaved` does internally: delta formed inside each
    repeat, then medianed, with the spread of the per-repeat deltas beside it.
    """
    base = np.array(out[ref]["us"])
    res = {}
    for name, rec in out.items():
        if name == ref:
            continue
        v = np.array(rec["us"])
        d = 100.0 * (v - base) / base
        res[name] = (float(np.median(d)), float(d.max() - d.min()))
    return res


def report(tag, out, refs=("inc", "flat")):
    print(f"\n  -- {tag}: paired d time % (negative is faster) --", flush=True)
    print(f"  {'arm':10} " + "  ".join(
        f"{'vs ' + r:>14} {'spread':>9}" for r in refs), flush=True)
    rows = {}
    tables = {r: paired(out, r) for r in refs if r in out}
    for name, _ in ARMS:
        if name not in out:
            continue
        cells, rec = [], {}
        for r in refs:
            if r not in tables or name not in tables[r]:
                cells.append(f"{'--':>14} {'--':>9}")
                continue
            d, s = tables[r][name]
            verdict = "" if abs(d) > s else "!"
            cells.append(f"{d:>+13.1f}%{verdict:1} {s:8.1f}pt")
            rec[r] = dict(delta_pct=d, spread_pt=s,
                          resolved=bool(abs(d) > s))
        print(f"  {name:10} " + "  ".join(cells), flush=True)
        rows[name] = rec
    print("  (! marks |delta| <= its own spread -- UNRESOLVED, not a result)",
          flush=True)
    return rows


def live_lens(meta):
    """Per-token-row effective lengths, as `bench.py --decompose` derives them."""
    qlen, lens = meta.get("qlen", 1), meta["lens"]
    out = np.zeros(meta["T"], dtype=np.int32)
    t = 0
    for i in range(meta["R"]):
        for u in range(qlen):
            out[t] = min(int(lens[i]) - qlen + u + 1, meta["n"])
            t += 1
    return out


def bench_sc(meta, repeats, iters):
    lens = live_lens(meta)
    sc = synth_scores(meta["T"], meta["n"], lens)
    jax.block_until_ready(sc)
    rl = jnp.asarray(lens, jnp.int32)
    arms = [(nm, functools.partial(fn, sc, K, rl)) for nm, fn in ARMS]
    got = {nm: np.asarray(fn()) for nm, fn in arms}
    ref = got["inc"]
    for nm in got:
        if nm == "inc":
            continue
        sd = sum(len(set(ref[t][ref[t] >= 0].tolist())
                     ^ set(got[nm][t][got[nm][t] >= 0].tolist()))
                 for t in range(meta["T"]))
        print(f"  setdiff {nm:9} vs inc: {sd}", flush=True)
    out = interleaved(arms, repeats=repeats, iters=iters, ref="inc")
    del sc, rl, arms, got, ref
    return out


def bench_gather(meta, repeats, iters):
    """PROBE, not a shipping arm: price the stage-2 gather ON ITS OWN.

    Three arms with identical output volume (T x C), so the paired differences
    isolate the gathering and not the writing:
      g_write       `idx + 1` -- writes T x C i32 and gathers nothing. The
                    floor: harness constant plus the output write.
      g_take_along  `take_along_axis(scores, idx, axis=1)` -- what `inc` does.
      g_flat        `take(words_flat, row_base + idx)` -- what `flat` does.

    The indices are stage 1's REAL output, not random draws: the incumbent's
    own `global_cand_indices`, so the access pattern is the one that ships.

    `08-sparsecore.md:1175-1245` measures a cliff in XLA's gather at exactly
    2^17 indices for a 65 536-row 1-wide f32 table -- 0.45 ns/index below it,
    12 ns above. This block's index counts straddle that number, so the probe
    is also a test of whether that claim reaches this operand shape.
    """
    T, n = meta["T"], meta["n"]
    p, n_p, k_p, C = tiling.stage_extents(T, n, K)
    if p == 1:
        print("  [gather] p==1: no stage-2 gather exists here", flush=True)
        return None
    lens = live_lens(meta)
    sc = synth_scores(T, n, lens)
    rl = jnp.asarray(lens, jnp.int32)

    # Stage 1's real candidate indices, built with the SHIPPING kernel.
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
        _sc_topk_direct as ship_direct
    offs = jnp.arange(p, dtype=jnp.int32) * n_p
    lp = jnp.clip(rl[:, None] - offs[None, :], 0, n_p).reshape(T * p)
    li = ship_direct(sc.reshape(T * p, n_p), k_p, lp,
                     write_empty=True).reshape(T, p, k_p)
    gci = jnp.where(li < 0, -1, offs[None, :, None] + li).reshape(T, C)
    idx = jnp.maximum(gci, 0)
    words = jax.lax.bitcast_convert_type(sc, jnp.int32).reshape(-1)
    base = (jnp.arange(T, dtype=jnp.int32) * jnp.int32(n))[:, None]
    jax.block_until_ready((sc, idx, words))

    f_write = jax.jit(lambda s, i: i + 1)
    f_along = jax.jit(lambda s, i: jnp.take_along_axis(s, i, axis=1))
    f_flat = jax.jit(lambda w, i: jnp.take(
        w, (base + i).reshape(-1), mode="clip").reshape(T, C))
    arms = [("g_write", functools.partial(f_write, sc, idx)),
            ("g_take_along", functools.partial(f_along, sc, idx)),
            ("g_flat", functools.partial(f_flat, words, idx))]
    print(f"  [gather] T={T} N={n} p={p} C={C}  indices={T * C} "
          f"({T * C / 2**17:.2f} x the 2^17 cliff in 08-sparsecore.md)  "
          f"table={T * n} elements", flush=True)
    out = interleaved(arms, repeats=repeats, iters=iters, ref="g_write")
    del sc, idx, words, arms
    return out


def bench_full(name, repeats, iters):
    args, kwargs, meta = build(name)
    jax.block_until_ready(args[2])
    arms = []
    for nm, sel in ARMS:
        mod = clone_streamindex(f"sit_bench_{nm}",
                                None if nm == "inc" else sel)
        f = jax.jit(functools.partial(mod.streamindex_topk, **kwargs))
        arms.append((nm, functools.partial(f, *args)))
    idx = np.asarray(arms[0][1]())
    if bool((idx[0] == np.arange(K)).all()):
        raise SystemExit(f"{name}: took _fast_path -- measures nothing")
    out = interleaved(arms, repeats=repeats, iters=iters, ref="inc")
    del args, arms
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--level", default="sc,full")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    levels = [x.strip() for x in a.level.split(",")]
    results = {}

    for name in [r.strip() for r in a.rows.split(",")]:
        r = ROWS[name]
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}", flush=True)
        if "full" in levels:
            t0 = time.perf_counter()
            out, meta = bench_full(name, a.repeats, a.iters)
            p, _, _, c = tiling.stage_extents(meta["T"], meta["n"], K)
            print(f"  [full] built+ran in {time.perf_counter() - t0:.0f}s  "
                  f"B={meta['T']} N={meta['n']} p={p} C={c}", flush=True)
            results[f"{name}/full"] = dict(
                p=p, C=c, arms=out, table=report(f"{name} full", out))
        if "sc" in levels or "gather" in levels:
            from shapes import padded_n
            lens = np.full(r["R"], r["seqlen"], dtype=np.int64)
            if r.get("seqlens_override") == "ragged":
                lens = np.full(r["R"], 4096, dtype=np.int64)
                lens[::4] = r["seqlen"]
            meta = dict(T=r["T"], R=r["R"], lens=lens,
                        n=padded_n(r["psz"], r["pages_per_seq"]),
                        qlen=r.get("qlen", 1))
            p, _, _, c = tiling.stage_extents(meta["T"], meta["n"], K)
            print(f"\n  [sc/gather] B={meta['T']} N={meta['n']} p={p} C={c}"
                  + ("   p==1: two-stage path NOT taken, O8 is structurally a "
                     "no-op here" if p == 1 else ""), flush=True)
            if "gather" in levels:
                og = bench_gather(meta, a.repeats, a.iters)
                if og is not None:
                    results[f"{name}/gather"] = dict(
                        p=p, C=c, arms=og,
                        table=report(f"{name} gather", og,
                                     refs=("g_write", "g_take_along")))
            if "sc" in levels:
                out = bench_sc(meta, a.repeats, a.iters)
                results[f"{name}/sc"] = dict(
                    p=p, C=c, arms=out, table=report(f"{name} sc", out))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nwrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
