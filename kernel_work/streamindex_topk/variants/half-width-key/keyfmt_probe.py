# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Option C, gate first: what does a 16-bit ranking key do to the ranking?

This probe answers the correctness half of Option C WITHOUT building it. The
kernel returns a SET of indices, so the only thing a narrowed key can change is
WHICH indices come back. A pipeline that stores a bf16 key and top-k's it
returns exactly the same set as a pipeline that stores f32 scores rounded to
bf16 and then widened back to f32, because widening bf16 -> f32 is an
order-preserving injection on the bf16 value set: it maps equal to equal and
less-than to less-than, so every comparison the selector makes is unchanged.

So: patch the module attribute `streamindex_topk.sparsecore_topk` with a
wrapper that rounds its input, and the returned index set is the one Option C
would return. No kernel edit, no SparseCore edit, and the number is not a
simulation of C's ranking -- it IS C's ranking.

(The tie-break AMONG equal keys is unspecified in both pipelines --
`sparsecore_topk`'s docstring says so -- so that part is not claimed identical,
only the set-modulo-ties. Every check below is a set/rank check, which is the
same criterion `instruments/test_correctness.py` applies.)

Nothing in `instruments/` is edited. `oracle_scores` and `check` are imported
and executed exactly as written.

Modes
-----
margins  run the real kernel through the real correctness criterion, once per
         key format, and print `worst_rank=` / `min_recall=` for each. Directly
         comparable to report.md's incumbent figures.
churn    compare the index SETS between key formats. No f64 oracle, so it runs
         at `mixed_64k` where the oracle does not.
dist     the score distribution off the f64 oracle: dynamic range (does fp16
         even reach?), tie multiplicity at the k-th cut under each rounding,
         and a host-side exact top-k on rounded scores. The host arm isolates
         the quantization from the kernel's own f32 arithmetic error; the
         `margins` arm carries both.
"""

import argparse
import json
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

sys.path.insert(0, "/workspace")

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import (  # noqa: E402
    test_correctness as TC)

import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as SIT  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (  # noqa: E402
    sparsecore_topk as REAL_SC)

# Key formats, as EXPLICIT MANTISSA BITS KEPT. f32 has 23, fp16 has 10,
# bf16 has 7.
#
# Narrowing is done by masking f32 mantissa bits, not by `astype`. That is
# deliberate and it cost a second job to learn
# (j-cb875cb9-4177-4756-84af: the -1 sentinel proved the rebind reached the
# traced graph, and every key format STILL returned byte-identical
# worst_rank/min_recall with churn exactly 0.0 on 1024 rows). The reason is
# `--xla_allow_excess_precision`, which defaults to TRUE: XLA is explicitly
# permitted to delete a `convert(convert(x, bf16), f32)` round trip, and it
# does. A mask-and-round on the integer bit pattern is opaque to that
# simplification, and it is also closer to what the shipped kernel would emit
# -- the TensorCore would write the truncated bits, not call a converter.
#
# The mask arms model PRECISION, not RANGE: `f16p` is fp16's 10 mantissa bits
# with f32's exponent range, i.e. the "shifted-exponent i16 / custom
# fixed-point" alternative that measurement_and_design.md 5.1 names. The
# separate `f16ieee` arm carries real IEEE fp16 semantics (overflow to inf,
# subnormals) so the range question is answered by its own arm rather than
# assumed away. `m4` is a POSITIVE CONTROL: 4 mantissa bits must produce large
# churn, and if it does not the narrowing is not reaching the device.
KEEP_BITS = {
    "f32": 23,
    "f16p": 10,
    "bf16": 7,
    "m4": 4,
}
IEEE_KEYS = {"f16ieee": jnp.float16}
NP_KEYS = {
    "f32": np.float32,
    "bf16": ml_dtypes.bfloat16,
    "f16p": None,  # host arm masks bits; see _np_narrow
    "f16ieee": np.float16,
    "m4": None,
}


def _narrow_bits(x, keep: int):
    """Round `x` (f32) to `keep` explicit mantissa bits, round-half-to-even,
    entirely in int32. No `convert` op, so nothing for the algebraic
    simplifier to undo."""
    drop = 23 - keep
    if drop <= 0:
        return x
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    half = jnp.int32(1 << (drop - 1))
    lsb = jnp.bitwise_and(jax.lax.shift_right_logical(bits, drop),
                          jnp.int32(1))
    # +half-1+lsb is round-half-to-even. int32 addition wraps, which would
    # matter only for f32 bit patterns within `half` of 0x7fffffff (|x| within
    # a few ulp of the f32 max); the measured score range is ~2e2.
    r = bits + (half - jnp.int32(1)) + lsb
    keep_mask = jnp.int32(-(1 << drop))  # ~((1<<drop)-1) as a signed int32
    return jax.lax.bitcast_convert_type(jnp.bitwise_and(r, keep_mask),
                                        jnp.float32)


def _np_narrow(x, keep: int):
    """Host mirror of `_narrow_bits`, for the `dist` mode."""
    drop = 23 - keep
    if drop <= 0:
        return np.asarray(x, np.float32)
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    half = np.uint64(1 << (drop - 1))
    lsb = (bits >> np.uint64(drop)) & np.uint64(1)
    r = bits + half - np.uint64(1) + lsb
    r &= np.uint64(0xFFFFFFFF) ^ np.uint64((1 << drop) - 1)
    return r.astype(np.uint32).view(np.float32)


def _flush_traces():
    """Force a genuine retrace after the module global has been rebound.

    `SIT.streamindex_topk.clear_cache()` alone is NOT enough and this cost a
    whole job to learn (j-9d6276c2-8eaa-4dc6-9bef, where all three key formats
    reported byte-identical worst_rank/min_recall and churn was exactly 0.0 on
    1024 rows). It clears pjit's C++ fastpath and its params cache, but the
    jaxpr itself comes from `jax._src.linear_util`'s memo, which is keyed on
    the wrapped function object and its transforms -- it has no idea a module
    global changed, so it hands back the first arm's jaxpr and every arm runs
    f32. `jax.clear_caches()` clears that memo too.
    """
    SIT.streamindex_topk.clear_cache()
    jax.clear_caches()


def _install(key: str):
    """Rebind the module attribute. Never edits a file."""
    if key == "f32":
        SIT.sparsecore_topk = REAL_SC
    elif key in IEEE_KEYS:
        dt = IEEE_KEYS[key]

        def patched(scores, k, row_lengths=None, *, write_empty_rows=True):
            # optimization_barrier between the two converts: without it XLA
            # deletes the pair under --xla_allow_excess_precision.
            narrow = jax.lax.optimization_barrier(scores.astype(dt))
            wide = jax.lax.optimization_barrier(narrow.astype(jnp.float32))
            return REAL_SC(wide, k, row_lengths,
                           write_empty_rows=write_empty_rows)

        SIT.sparsecore_topk = patched
    else:
        keep = KEEP_BITS[key]

        def patched(scores, k, row_lengths=None, *, write_empty_rows=True):
            return REAL_SC(_narrow_bits(scores, keep), k, row_lengths,
                           write_empty_rows=write_empty_rows)

        SIT.sparsecore_topk = patched
    _flush_traces()


def _all_keys():
    return list(KEEP_BITS) + list(IEEE_KEYS)


def _install_sentinel():
    """A patch whose effect cannot be mistaken for anything else."""

    def sentinel(scores, k, row_lengths=None, *, write_empty_rows=True):
        return jnp.full((scores.shape[0], k), -1, jnp.int32)

    SIT.sparsecore_topk = sentinel
    _flush_traces()


def verify_patch_mechanism(d, label=""):
    """Prove the rebind reaches the traced graph, before trusting any arm.

    Installs a `sparsecore_topk` that returns all -1 and checks the kernel
    returns all -1. If it does not, the rebind is not taking and every
    comparison below this line is vacuous -- so this raises rather than
    printing a warning nobody reads.
    """
    _install_sentinel()
    out = _run_kernel(d)
    ok = bool((out == -1).all())
    print(f"[{label}] patch mechanism: "
          f"{'OK -- sentinel reached the traced graph' if ok else 'VACUOUS'}"
          f" (out==-1 fraction {float((out == -1).mean()):.6f})", flush=True)
    _install("f32")
    if not ok:
        raise SystemExit(
            "patch mechanism is VACUOUS: rebinding "
            "streamindex_topk.sparsecore_topk did not change the traced "
            "graph, so every key-format arm would report the f32 number. "
            "Refusing to produce numbers.")


def _run_kernel(d, negative_control=False):
    k = S.INDEX_TOPK
    q = d["q"]
    if negative_control:
        qm = np.array(q.astype(jnp.float32), copy=True)
        qm[:, S.INDEX_N_HEADS // 2:, :] = 0.0
        q = jnp.asarray(qm).astype(jnp.float8_e4m3fn)
    idxs = SIT.streamindex_topk(
        q,
        d["indexer_weights"],
        d["cache_kv"],
        d["seq_lens"],
        d["page_indices"],
        d["cu_q_lens"],
        d["distribution"],
        k=k,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    )
    return np.asarray(jax.block_until_ready(idxs))


def mode_margins(regime, keys, do_control):
    d = I.build(regime, seed=0, with_dequant=True)
    verify_patch_mechanism(d, regime.name)
    print(f"[{regime.name}] building float64 oracle "
          f"({regime.num_tokens} x {regime.kv_len}) ...", flush=True)
    sc = TC.oracle_scores(d, regime)
    rows = []
    for key in keys:
        _install(key)
        idxs = _run_kernel(d)
        n, failures, st = TC.check(idxs, sc, S.INDEX_TOPK)
        allowed = S.INDEX_TOPK + TC.RANK_SLACK
        verdict = "FAIL" if failures else "PASS"
        print(f"[{regime.name}] key={key:5s} {verdict} -- rows={n} "
              f"worst_rank={st['worst_rank_seen']} of allowed {allowed}, "
              f"min_recall={st['min_recall']:.5f} of required {TC.MIN_RECALL}"
              f"{'' if not failures else f'  ({len(failures)} bad rows)'}",
              flush=True)
        for f in failures[:3]:
            print("    ", f, flush=True)
        rows.append(
            dict(regime=regime.name, key=key, verdict=verdict, rows=n,
                 worst_rank=st["worst_rank_seen"], allowed=allowed,
                 min_recall=st["min_recall"], n_failures=len(failures)))
        if do_control:
            idxs_c = _run_kernel(d, negative_control=True)
            nc, fc, stc = TC.check(idxs_c, sc, S.INDEX_TOPK)
            fired = bool(fc)
            print(f"[{regime.name}] key={key:5s} NEGATIVE CONTROL "
                  f"{'FIRED' if fired else 'DID NOT FIRE -- VACUOUS'} "
                  f"({len(fc)} of {nc} rows bad) worst_rank="
                  f"{stc['worst_rank_seen']}, min_recall="
                  f"{stc['min_recall']:.5f}", flush=True)
            rows[-1].update(control_fired=fired,
                            control_worst_rank=stc["worst_rank_seen"],
                            control_min_recall=stc["min_recall"],
                            control_bad_rows=len(fc))
    return rows


def mode_churn(regime, keys):
    """Index-set churn between key formats. No oracle -- runs anywhere."""
    d = I.build(regime, seed=0, with_dequant=False)
    verify_patch_mechanism(d, regime.name)
    got = {}
    for key in keys:
        _install(key)
        got[key] = _run_kernel(d)
    base = got["f32"]
    rows = []
    for key in keys:
        if key == "f32":
            continue
        a, b = base, got[key]
        diffs = []
        for t in range(a.shape[0]):
            sa = set(int(x) for x in a[t] if x >= 0)
            sb = set(int(x) for x in b[t] if x >= 0)
            if not sa and not sb:
                continue
            diffs.append(len(sa - sb))
        diffs = np.asarray(diffs) if diffs else np.zeros(1)
        n = max(1, int(np.median([len([x for x in a[t] if x >= 0])
                                  for t in range(a.shape[0])])))
        r = dict(regime=regime.name, key=key, rows=int(len(diffs)),
                 median_k=n,
                 swapped_mean=float(diffs.mean()),
                 swapped_max=int(diffs.max()),
                 swapped_frac_mean=float(diffs.mean() / n),
                 rows_identical=int((diffs == 0).sum()))
        print(f"[{regime.name}] churn f32 -> {key}: mean {r['swapped_mean']:.1f} "
              f"of {n} indices swapped per row "
              f"({100*r['swapped_frac_mean']:.3f}%), max {r['swapped_max']}, "
              f"{r['rows_identical']}/{r['rows']} rows identical", flush=True)
        rows.append(r)
    # POSITIVE CONTROL. 4 explicit mantissa bits is a 5-bit significand; on a
    # 2048-of-65536 cut it must move a large number of indices. If it moves
    # none, the narrowing never reached the device and every other churn
    # number on this regime is zero for the same reason, not because 16 bits
    # are enough.
    ctrl = [r for r in rows if r["key"] == "m4"]
    if ctrl:
        moved = ctrl[0]["swapped_mean"]
        ok = moved > 0
        print(f"[{regime.name}] narrowing positive control (m4, 4 mantissa "
              f"bits): {'OK' if ok else 'VACUOUS'} -- {moved:.1f} indices "
              f"moved per row", flush=True)
        if not ok:
            raise SystemExit(
                "narrowing positive control is VACUOUS: masking the f32 "
                "mantissa to 4 bits changed nothing, so the narrowing is not "
                "reaching the device and no churn number here means "
                "anything.")
    return rows


def mode_dist(regime, keys):
    """Score-distribution facts and a host-only quantization rank movement."""
    d = I.build(regime, seed=0, with_dequant=True)
    print(f"[{regime.name}] building float64 oracle ...", flush=True)
    sc = TC.oracle_scores(d, regime)          # f64 [T, kv_len], -inf masked
    k = S.INDEX_TOPK
    finite = np.isfinite(sc)
    vals = sc[finite]
    nz = np.abs(vals[vals != 0])
    stats = dict(
        regime=regime.name,
        n_finite=int(finite.sum()),
        max_abs=float(np.abs(vals).max()),
        min_abs_nonzero=float(nz.min()) if nz.size else 0.0,
        frac_negative=float((vals < 0).mean()),
        # fp16: finite range is +-65504, smallest normal 6.104e-5,
        # smallest subnormal 5.96e-8.
        f16_overflow=int((np.abs(vals) > 65504.0).sum()),
        f16_subnormal=int(((nz > 0) & (nz < 6.103515625e-05)).sum()),
        f16_flush_to_zero=int((nz < 5.9604644775390625e-08).sum()),
    )
    print(f"[{regime.name}] score range: max|s|={stats['max_abs']:.6g} "
          f"min|s|(nonzero)={stats['min_abs_nonzero']:.6g} "
          f"neg frac={stats['frac_negative']:.4f}", flush=True)
    print(f"[{regime.name}] fp16 reach: overflow={stats['f16_overflow']} "
          f"subnormal={stats['f16_subnormal']} "
          f"flush_to_zero={stats['f16_flush_to_zero']} "
          f"of {stats['n_finite']} finite scores", flush=True)

    out = [stats]
    for key in keys:
        if key == "f32":
            continue
        npdt = NP_KEYS[key]
        if npdt is None:  # bit-masked precision arm, mirrors _narrow_bits
            rq = _np_narrow(sc.astype(np.float32),
                            KEEP_BITS[key]).astype(np.float64)
        else:
            rq = sc.astype(np.float32).astype(npdt).astype(np.float64)
        worst = 0
        min_recall = 1.0
        tie_at_cut = []
        for t in range(sc.shape[0]):
            row = sc[t]
            fin = np.isfinite(row)
            nv = int(fin.sum())
            if nv == 0:
                continue
            exp_n = min(k, nv)
            rr = rq[t]
            # exact top-exp_n on the ROUNDED scores; ties broken arbitrarily,
            # exactly as an exact selector on a 16-bit key would.
            sel = np.argpartition(-rr, exp_n - 1)[:exp_n]
            order = np.argsort(-row, kind="stable")
            rank_of = np.empty(len(row), dtype=np.int64)
            rank_of[order] = np.arange(len(row))
            ranks = rank_of[sel]
            worst = max(worst, int(ranks.max()) + 1)
            min_recall = min(min_recall, float((ranks < exp_n).mean()))
            cut = np.partition(rr, len(rr) - exp_n)[len(rr) - exp_n]
            if np.isfinite(cut):
                tie_at_cut.append(int((rr == cut).sum()))
        tie_at_cut = np.asarray(tie_at_cut) if tie_at_cut else np.zeros(1)
        r = dict(regime=regime.name, key=key, host_worst_rank=worst,
                 host_min_recall=min_recall,
                 tie_at_cut_mean=float(tie_at_cut.mean()),
                 tie_at_cut_max=int(tie_at_cut.max()))
        print(f"[{regime.name}] host-only quantization, key={key}: "
              f"worst_rank={worst} of allowed {k + TC.RANK_SLACK}, "
              f"min_recall={min_recall:.5f}; tie multiplicity at the "
              f"k-th cut mean {r['tie_at_cut_mean']:.1f} max "
              f"{r['tie_at_cut_max']}", flush=True)
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["margins", "churn", "dist"])
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--keys", default="f32,m4,bf16,f16p,f16ieee")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    keys = [x for x in a.keys.split(",") if x]
    for key in keys:
        if key not in _all_keys():
            raise SystemExit(f"unknown key format {key!r}; "
                             f"known: {_all_keys()}")

    rows = []
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        if a.mode == "margins":
            rows += mode_margins(regime, keys, a.control)
        elif a.mode == "churn":
            rows += mode_churn(regime, keys)
        else:
            rows += mode_dist(regime, keys)

    for r in rows:
        print("ROW " + json.dumps(r), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
