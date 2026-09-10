#!/usr/bin/env python3
"""Probe 15 -- the two mandatory step-2 probes for `v10` and `v3`.

Probe A (feeds `v10`, `design.md:1119`): the **count-pass rate**. `v10`'s
sample-and-count root replaces a histogram pass with a bare compare + population
count. Its whole band (-25% .. +15%) is set by how that pass prices against the
L0 dense scatter it replaces. Measured as a **slope** over pass count
(0,1,2,4) so the launch intercept, the entry relayout and the per-wave score DMA
all cancel -- the same ladder `probes/03` used, rebuilt (not edited: the
`instruments/` tree is frozen).

Probe B (feeds `v3`, `design.md:929`): the **marginal cost of one conditional
`plsc.store_compressed` inside the L0 walk**. `v3` deletes the separate emit
pass by storing eligible elements during the histogram walk; that is the only
thing it adds. `probes/03` priced `emit1` at 27.456 us but that arm bundles a
`cumsum`, a popcount, a carry and a wrap with the store, so it is an upper
bound on the store, not the store. Here `B_ns_d*` and `B_st_d*` are the *same*
program except for one line, so the paired difference is the store alone -- and
it is swept across six store densities, because a masked store's cost is a
function of density, not of element count.

Both probes are one interleaved pool in one job. Every arm sees the **production
score distribution** (`probes/13/level_sim.py`, its validated generator), never
`rng.standard_normal`.

Entry shape: production hands SparseCore `f32[1024,80,128]{2,1,0:T(8,128)}` and
flattens it with a `bitcast_convert_type` + `bitcast` pair. `probes/11` showed a
`f32[b, n]` entry instead forces XLA to untile 40 MiB and invents 34-60 us. This
probe's jit entry is therefore `f32[b, n/128, 128]` and the flatten happens
*inside* the jit. One arm's optimized HLO is dumped and grepped to prove it.

NOTE: nothing here modifies the subject kernel. It is read for its constants and
its idioms only.
"""

import argparse
import functools
import hashlib
import json
import os
import sys
import time

import numpy as np

# --- constants, mirrored from the subject (sparsecore_topk.py:37-50) ---------
LANES = 16
NUM_BUCKETS = 256
L0_SHIFT = 24
MONO_MASK = 0x7FFFFFFF

# Production prefill geometry for GLM-5.2 DSA: b=1024, n=10240, k=2048, P=1,
# slice_len=10240, num_waves=cdiv(1024,32)=32  (sparsecore_topk.py:441-442,
# _pick_partition:493 -> P=1 because b>=32).
B = 1024
N = 10240
K = 2048
NUM_WAVES = 32
SLICE_LEN = N            # already a multiple of LANES
LANE_TILE = 128          # the minor dim that makes the entry a free bitcast
STRAD_WORDS = 8192       # power of two, so `woff & (STRAD_WORDS-1)` is in range

# A bit pattern no fp32 score can hold (a quiet NaN); used as the always-true,
# data-dependent, un-foldable `valid` mask for the dense L0 scatter. Verified
# absent from the generated scores host-side before any arm runs.
NEVER_BITS = 0x7FC00000

# Density thresholds t: sel = (bits & 255) < t  ->  density ~ t/256, exactly
# controlled and independent of the score distribution (the low mantissa bits
# are uniform). t=0 is NOT used: `x & 255 < 0` is provably false and a control
# the compiler can fold away is not a control (PROBING.md).
DENS_T = {"d004": 1, "d05": 13, "d15": 38, "d30": 77, "d45": 115, "d65": 166}

# v10's count-pass predicate. Threshold 0 on the raw fp32 bits selects the
# positive scores (~18% of production scores). The compare and the population
# count are vector ops whose cost does not depend on how many lanes are set --
# `probes/03` scatter_sparse vs scatter showed density matters for a *scatter*,
# not for a compare -- so one threshold suffices for this pass kind.
CNT_THRESH = 0

SEED = 20260910


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Scores: the production distribution, from probes/13's validated generator.
# ---------------------------------------------------------------------------
def make_production_scores(b=B, n=N):
    """rho = 1: all `n` columns are real production-distribution scores.

    This is `level_sim.py`'s STEP-4b construction (`make_kv(N, ...)` rather than
    `make_kv(KV_ROWS, ...)`), chosen deliberately: every arm must walk the *same*
    number of elements or the density slope in probe B is confounded by walk
    length, and a ragged `-inf` tail would give the arms different live-lane
    counts. Production rho (0.40005 prefill, 0.850 decode) is reapplied
    analytically in the README, and that reweighting is labelled `modelled`.
    """
    import level_sim as ls
    rng = np.random.default_rng(SEED)
    kq, ks = ls.make_kv(n, ls.HEAD_DIM, 0.0, rng)
    sc = ls.make_scores(kq, ks, b, ls.N_HEAD, ls.HEAD_DIM, rng)
    return np.ascontiguousarray(sc.astype(np.float32))


def score_facts(scores):
    bits = scores.view(np.int32)
    perrow_neg = np.mean(scores < 0, axis=1)
    facts = {
        "shape": list(scores.shape),
        "finite_frac": float(np.isfinite(scores).mean()),
        # probes/13/README.md:81's "81.7% negative" is a ROW-0 statistic
        # (level_sim.py:459-465 takes `sc[0, :8192]`), not a population one:
        # each row's signed head weights shift its own mean, so the per-row
        # negative fraction spans 7%-95% with a ~46.5% median. Both are
        # recorded so the comparison is against the right one.
        "neg_frac_pop": float((scores < 0).mean()),
        "neg_frac_row0": float(perrow_neg[0]),
        "neg_frac_perrow_min_med_max": [float(perrow_neg.min()),
                                        float(np.median(perrow_neg)),
                                        float(perrow_neg.max())],
        "exact_zeros": int((scores == 0).sum()),
        "distinct_frac_row0": float(len(np.unique(scores[0])) / scores.shape[1]),
        "never_bits_hits": int((bits == np.int32(NEVER_BITS)).sum()),
    }
    for name, t in DENS_T.items():
        facts[f"realized_density_{name}"] = float(
            ((bits & 255) < t).mean())
    facts["realized_density_cnt"] = float((bits > np.int32(CNT_THRESH)).mean())
    return facts


# ---------------------------------------------------------------------------
# The kernel. One body, one `kind`, one `npasses`, one density -- so every arm
# differs from its partner in exactly the term being priced.
# ---------------------------------------------------------------------------
def _probe_kernel(src_hbm, out_hbm, resident_vmem, priv_vmem, strad_vmem,
                  strad2_vmem, acc_vmem, sems, *, b, slice_len, num_waves,
                  npasses, kind, dens_t, nstores):
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    num_subcores = 16
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub
    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    walk_chunks = slice_len // LANES

    acc_vmem[pl.ds(0, LANES)] = jnp.zeros((LANES, ), jnp.int32)

    def wave_body(w, _):
        row = w * (2 * num_subcores) + flat
        row_c = jnp.minimum(row, b - 1)
        cp = pltpu.make_async_copy(
            src_hbm.at[pl.ds(row_c * slice_len, slice_len)], resident_vmem,
            sems.at[0])
        cp.start()
        cp.wait()

        for _p in range(npasses):
            if kind == "cvt":
                # sparsecore_topk.py:169-172, verbatim.
                def body(i):
                    bits = resident_vmem[pl.ds(i * LANES, LANES)]
                    m = jnp.bitwise_and(jnp.right_shift(bits, 31),
                                        jnp.int32(MONO_MASK))
                    resident_vmem[pl.ds(i * LANES, LANES)] = jnp.bitwise_xor(
                        bits, m)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(body)

            elif kind == "scat":
                # The L0 histogram pass, sparsecore_topk.py:186-193: dense
                # `addupdate_scatter` with the 16-lane-replicated addressing.
                # This is v10's denominator -- the pass a count pass replaces.
                def body(i):
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                        NUM_BUCKETS // 2)
                    addr = lane_iota * NUM_BUCKETS + bucket
                    valid = key != jnp.int32(NEVER_BITS)
                    plsc.addupdate_scatter(priv_vmem, (addr, ),
                                           jnp.ones((LANES, ), jnp.int32),
                                           mask=valid)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(body)

            elif kind == "cnt":
                # v10's count pass as `design.md:1119` specifies it: a bare
                # compare + `all_reduce_population_count`, scalar carry.
                def body(i, c):
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    m = key > jnp.int32(CNT_THRESH)
                    return c + plsc.all_reduce_population_count(m)[0]

                c = plsc.parallel_loop(0, walk_chunks, unroll=4,
                                       carry=jnp.int32(0))(body)
                acc_vmem[pl.ds(0, LANES)] = acc_vmem[pl.ds(0, LANES)] + c

            elif kind == "cntv":
                # The same count with a *vector* accumulator: no
                # `all_reduce_population_count` and no scalar loop carry. If
                # `cnt` reads expensive, this says whether the cost is the
                # count or the serialization -- which changes v10's verdict.
                def body(i, v):
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    m = key > jnp.int32(CNT_THRESH)
                    return v + m.astype(jnp.int32)

                v = plsc.parallel_loop(
                    0, walk_chunks, unroll=4,
                    carry=jnp.zeros((LANES, ), jnp.int32))(body)
                acc_vmem[pl.ds(0, LANES)] = acc_vmem[pl.ds(0, LANES)] + v

            elif kind == "l0":
                # Probe B. `nstores == 0` is the L0 histogram walk plus the
                # density mask, the popcount and the wrapped write offset;
                # `nstores >= 1` adds ONE LINE, the conditional
                # `store_compressed`. Everything else is byte-identical, so the
                # paired difference is the store and nothing else.
                def body(i, woff, nstores=nstores, dens_t=dens_t):
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    j = i * LANES + lane_iota
                    bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                        NUM_BUCKETS // 2)
                    addr = lane_iota * NUM_BUCKETS + bucket
                    valid = key != jnp.int32(NEVER_BITS)
                    plsc.addupdate_scatter(priv_vmem, (addr, ),
                                           jnp.ones((LANES, ), jnp.int32),
                                           mask=valid)
                    sel = jnp.bitwise_and(key, jnp.int32(255)) < jnp.int32(
                        dens_t)
                    if nstores >= 1:
                        plsc.store_compressed(strad_vmem.at[pl.ds(woff, LANES)],
                                              j, mask=sel)
                    if nstores >= 2:
                        plsc.store_compressed(
                            strad2_vmem.at[pl.ds(woff, LANES)], key, mask=sel)
                    woff = woff + plsc.all_reduce_population_count(sel)[0]
                    return jnp.bitwise_and(woff, jnp.int32(STRAD_WORDS - 1))

                wo = plsc.parallel_loop(0, walk_chunks, unroll=4,
                                        carry=jnp.int32(0))(body)
                acc_vmem[pl.ds(0, LANES)] = acc_vmem[pl.ds(0, LANES)] + wo
            else:
                raise ValueError(kind)

        # Liveness epilogue (instruments/sctopk_2stage_bench.py:896-901): with
        # nothing reading `resident`/`priv`/`strad`, the last pass -- and with
        # it the slope this probe exists to measure -- is dead code.
        acc_vmem[pl.ds(0, LANES)] = (acc_vmem[pl.ds(0, LANES)] +
                                     resident_vmem[pl.ds(0, LANES)] +
                                     priv_vmem[pl.ds(0, LANES)] +
                                     strad_vmem[pl.ds(0, LANES)] +
                                     strad2_vmem[pl.ds(0, LANES)])
        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)

    @pl.when(flat == 0)
    def _():
        cp = pltpu.make_async_copy(acc_vmem.at[pl.ds(0, LANES)],
                                   out_hbm.at[pl.ds(0, LANES)], sems.at[1])
        cp.start()
        cp.wait()


def build_kernel(kind, npasses, dens_t=0, nstores=0):
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.experimental.pallas import tpu_sc as plsc

    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    # Identical scratch in EVERY arm, so no arm gets a different VMEM live set.
    scratch_types = (
        pltpu.VMEM((SLICE_LEN, ), jnp.int32),
        pltpu.VMEM((LANES * NUM_BUCKETS + LANES, ), jnp.int32),
        pltpu.VMEM((STRAD_WORDS + LANES, ), jnp.int32),
        pltpu.VMEM((STRAD_WORDS + LANES, ), jnp.int32),
        pltpu.VMEM((LANES, ), jnp.int32),
        pltpu.SemaphoreType.DMA((2, )),
    )
    return pl.kernel(
        functools.partial(_probe_kernel, b=B, slice_len=SLICE_LEN,
                          num_waves=NUM_WAVES, npasses=npasses, kind=kind,
                          dens_t=dens_t, nstores=nstores),
        out_type=jax.ShapeDtypeStruct((LANES, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=scratch_types,
        mesh=mesh,
        name=f"p15_{kind}_p{npasses}_t{dens_t}_s{nstores}",
    )


def make_thunk(kern, tiled, flat_s32=None):
    """Production-shaped entry: f32[b, n/128, 128] in, flatten inside the jit.

    A tiled TPU layout is row-major iff the minor dimension equals the tile
    width, so `f32[1024,80,128]{2,1,0:T(8,128)}` flattens to 1-D as a free
    `bitcast` -- exactly production's chain (probes/11). An `f32[1024,10240]`
    entry has minor dim 10240 >> 128 and makes XLA untile 40 MiB.
    """
    import jax
    import jax.numpy as jnp

    if flat_s32 is not None:
        f, x = jax.jit(kern), flat_s32
    else:
        f = jax.jit(lambda s: kern(
            jax.lax.bitcast_convert_type(s, jnp.int32).reshape(-1)))
        x = tiled
    return (lambda: f(x), lambda: f.lower(x).compile().as_text())


# ---------------------------------------------------------------------------
def hlo_entry_report(text):
    """Grep the optimized HLO for the three entry-cleanliness facts."""
    lines = text.splitlines()
    rep = {}
    rep["sparse_core_data_format_call"] = sum(
        1 for l in lines if "sparse-core-data-format-call" in l)
    big = f"[{B * N}]"
    rep["copy_on_score_operand"] = [
        l.strip()[:200] for l in lines
        if " copy(" in l and ("10240" in l or big in l)
    ]
    rep["bitcast_lines"] = [
        l.strip()[:200] for l in lines
        if ("bitcast" in l and ("10240" in l or big in l))
    ]
    rep["fusion_on_scores"] = [
        l.strip()[:200] for l in lines
        if "fusion(" in l and big in l
    ]
    rep["clean"] = (rep["sparse_core_data_format_call"] == 0
                    and not rep["copy_on_score_operand"]
                    and not rep["fusion_on_scores"])
    return rep


def fit_slope(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx)**2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    s = sxy / sxx
    a = my - s * mx
    ss_tot = sum((y - my)**2 for y in ys)
    ss_res = sum((y - (a + s * x))**2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return s, a, r2


# ---------------------------------------------------------------------------
def dry_run(args):
    """Host-only gates. A VectorSubcoreMesh kernel cannot run under CPU
    interpret mode, so this is probes/11's shape: facts that must hold before
    any pod time is spent, not an execution."""
    print("=== DRY RUN (host only) ===", flush=True)
    import level_sim as ls
    print(f"level_sim: N={ls.N} K={ls.K} HEAD_DIM={ls.HEAD_DIM} "
          f"N_HEAD={ls.N_HEAD} KV_ROWS={ls.KV_ROWS} SEED={ls.SEED}")
    b = 32
    sc = make_production_scores(b=b, n=N)
    facts = score_facts(sc)
    for k, v in facts.items():
        print(f"  {k}: {v}")
    ok = True
    if facts["never_bits_hits"] != 0:
        print("!! NEVER_BITS occurs in the scores -- the dense mask is not "
              "all-true")
        ok = False
    # The invariants probes/13/README.md:79-81 actually established, which are
    # population facts: no point mass at zero, and 100% distinct within a row.
    if facts["exact_zeros"] != 0:
        print(f"!! exact zeros {facts['exact_zeros']} -- expected 0")
        ok = False
    if facts["distinct_frac_row0"] < 0.999:
        print(f"!! row 0 only {facts['distinct_frac_row0']:.4f} distinct")
        ok = False
    if facts["finite_frac"] != 1.0:
        print("!! non-finite scores at rho=1")
        ok = False
    # Cross-check the generator against probes/13's own printed row-0 number,
    # under probes/13's exact construction (level_sim.py:457-465).
    import level_sim as ls2
    rng2 = np.random.default_rng(ls2.SEED)
    kq2, ks2 = ls2.make_kv(ls2.KV_ROWS, ls2.HEAD_DIM, 0.0, rng2)
    r0 = ls2.make_scores(kq2, ks2, 64, ls2.N_HEAD, ls2.HEAD_DIM, rng2)[0, :8192]
    print(f"  probes/13 row-0 reproduction: neg%={100*(r0<0).mean():.3f} "
          f"(README.md:81 says 81.7)")
    if abs(100 * (r0 < 0).mean() - 81.7) > 0.2:
        print("!! generator does not reproduce probes/13")
        ok = False
    for name, t in DENS_T.items():
        want = t / 256.0
        got = facts[f"realized_density_{name}"]
        if abs(got - want) > 0.01:
            print(f"!! density {name}: want ~{want:.4f} got {got:.4f}")
            ok = False
    print(f"arms planned: {len(plan_arms())}")
    for a in plan_arms():
        print(f"  {a[0]}")
    print("DRY_GATE_OK" if ok else "DRY_GATE_FAIL")
    return 0 if ok else 1


def plan_arms():
    """(name, kind, npasses, dens_t, nstores, entry) -- one pool, both probes."""
    arms = []
    arms.append(("A_p0", "cvt", 0, 0, 0, "tiled"))
    for kind in ("cvt", "scat", "cnt", "cntv"):
        for p in (1, 2, 4):
            arms.append((f"A_{kind}_p{p}", kind, p, 0, 0, "tiled"))
    # Same-program replicates: the floor on the programs actually differenced
    # (PROBING.md -- "measure the floor on the program you are differencing").
    arms.append(("A_cnt_p2#rep", "cnt", 2, 0, 0, "tiled"))
    arms.append(("A_scat_p2#rep", "scat", 2, 0, 0, "tiled"))
    # Entry-cleanliness cross-check: identical work, s32[b*n] flat entry
    # (probes/11's `A0_flatin`, its cleanest arm). Matching `A_cvt_p1` within
    # the floor is a *measured* statement that the tiled entry costs nothing.
    arms.append(("A_cvt_p1#flatin", "cvt", 1, 0, 0, "flat"))
    for name in DENS_T:
        arms.append((f"B_ns_{name}", "l0", 1, DENS_T[name], 0, "tiled"))
        arms.append((f"B_st_{name}", "l0", 1, DENS_T[name], 1, "tiled"))
    arms.append(("B_ns_d30_p2", "l0", 2, DENS_T["d30"], 0, "tiled"))
    arms.append(("B_st_d30_p2", "l0", 2, DENS_T["d30"], 1, "tiled"))
    # v3 stores TWO things at L0 (the strictly-above set and the straddlers),
    # so the second store's marginal is priced too.
    arms.append(("B_st2_d30", "l0", 1, DENS_T["d30"], 2, "tiled"))
    arms.append(("B_st2_d65", "l0", 1, DENS_T["d65"], 2, "tiled"))
    arms.append(("B_st_d30#rep", "l0", 1, DENS_T["d30"], 1, "tiled"))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/probe15")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if args.dry_run:
        return dry_run(args)

    os.makedirs(args.outdir, exist_ok=True)
    import jax
    import jax.numpy as jnp
    import glm_dsa_block_bench as gb
    from glm_dsa_block_bench import measure, dispatch_floor
    print(f"FROZEN INSTRUMENT glm_dsa_block_bench.py md5="
          f"{_md5(gb.__file__)}  REPS_LO={gb.REPS_LO} REPS_HI={gb.REPS_HI}")
    print(f"jax {jax.__version__}  devices={jax.devices()}", flush=True)

    t0 = time.time()
    scores = make_production_scores()
    facts = score_facts(scores)
    print(f"scores built in {time.time()-t0:.1f}s", flush=True)
    for k, v in facts.items():
        print(f"  {k}: {v}")
    assert facts["never_bits_hits"] == 0, "dense mask is not all-true"

    # Production entry shape: f32[b, n/128, 128].
    tiled = jnp.asarray(scores.reshape(B, N // LANE_TILE, LANE_TILE))
    flat_s32 = jax.lax.bitcast_convert_type(
        jnp.asarray(scores), jnp.int32).reshape(-1)
    jax.block_until_ready((tiled, flat_s32))
    print(f"entry array: {tiled.shape} {tiled.dtype}", flush=True)

    # Build AND compile every arm up front, each in its own try. A compile that
    # fails inside `measure`'s warmup would take the whole pool down with it; a
    # dropped arm only costs that arm. The compiles are not extra work -- the
    # results are cached and `measure` reuses them.
    variants, hlos, hlo_len, skipped = {}, {}, {}, []
    print("\n=== BUILD + COMPILE ===", flush=True)
    for name, kind, p, t, ns, entry in plan_arms():
        t0 = time.time()
        try:
            kern = build_kernel(kind, p, dens_t=t, nstores=ns)
            thunk, hlo = make_thunk(kern, tiled,
                                    flat_s32=flat_s32 if entry == "flat" else None)
            hlo_len[name] = len(hlo())
            variants[name] = thunk
            hlos[name] = hlo
            print(f"  {name:<20} ok   {time.time()-t0:6.1f}s  "
                  f"len(HLO)={hlo_len[name]}", flush=True)
        except Exception as e:  # noqa: BLE001
            skipped.append((name, repr(e)[:400]))
            print(f"!! ARM UNSUPPORTED {name}: {e!r}", flush=True)

    # Nulls: the instrument's own floor, three of them (probes/07, probes/11).
    nullf = jax.jit(lambda a: a + 1.0)
    nullx = jnp.ones((8, 128), jnp.float32)
    for j in range(3):
        variants[f"#null{j}"] = (lambda f=nullf, x=nullx: f(x))

    # ---- structural gate: prove the one-line edit is real and identical ----
    print("\n=== HLO STRUCTURAL GATE ===", flush=True)
    deltas = {}
    for d in DENS_T:
        if f"B_st_{d}" in hlo_len and f"B_ns_{d}" in hlo_len:
            deltas[d] = hlo_len[f"B_st_{d}"] - hlo_len[f"B_ns_{d}"]
            print(f"  len(HLO) st-ns @ {d}: {deltas[d]:+d}  "
                  f"(ns={hlo_len[f'B_ns_{d}']} st={hlo_len[f'B_st_{d}']})")
    if deltas:
        vals = set(deltas.values())
        print(f"  st-ns HLO delta identical across densities: "
              f"{len(vals) == 1}  values={sorted(vals)}")
        print(f"  st != ns everywhere: {all(v != 0 for v in deltas.values())}")

    # ---- entry gate: dump one arm's optimized HLO and grep it ----
    print("\n=== ENTRY HLO GATE ===", flush=True)
    entry_rep = None
    try:
        txt = hlos["A_cvt_p1"]()
        with open(os.path.join(args.outdir, "A_cvt_p1.optimized.hlo"), "w") as f:
            f.write(txt)
        entry_rep = hlo_entry_report(txt)
        for k, v in entry_rep.items():
            print(f"  {k}: {v}")
    except Exception as e:  # noqa: BLE001
        print(f"!! entry HLO dump failed: {e!r}")
    try:
        txt2 = hlos["A_cvt_p1#flatin"]()
        with open(os.path.join(args.outdir, "A_cvt_p1_flatin.optimized.hlo"),
                  "w") as f:
            f.write(txt2)
    except Exception as e:  # noqa: BLE001
        print(f"!! flatin HLO dump failed: {e!r}")

    # ---- measure: ONE pool, interleaved, reshuffled, `rounds` rounds ----
    print(f"\n=== MEASURE  {len(variants)} arms x {args.rounds} rounds ===",
          flush=True)
    floor_us = dispatch_floor()
    print(f"  dispatch floor: {floor_us:.2f} us/call", flush=True)
    t0 = time.time()
    res = measure(variants, rounds=args.rounds)
    print(f"  measured in {time.time()-t0:.1f}s", flush=True)

    print(f"\n  {'arm':<20} {'us_median':>11} {'us':>11} {'us_mean':>11} "
          f"{'hostbnd%':>9}  suspect")
    for name in variants:
        r = res[name]
        print(f"  {name:<20} {r['us_median']:11.3f} {r['us']:11.3f} "
              f"{r['us_mean']:11.3f} {r.get('hostbound_pct', float('nan')):9.2f}"
              f"  {r['suspect']}")

    def um(n):
        return res[n]["us_median"]

    # ---- floors ----
    nulls = [um(f"#null{j}") for j in range(3)]
    null_spread = max(nulls) - min(nulls)
    reps = []
    for a, b_ in (("A_cnt_p2", "A_cnt_p2#rep"), ("A_scat_p2", "A_scat_p2#rep"),
                  ("B_st_d30", "B_st_d30#rep")):
        if a in res and b_ in res:
            reps.append((a, abs(um(a) - um(b_))))
    prog_floor = max(v for _, v in reps) if reps else float("nan")
    print(f"\n=== FLOORS ===")
    print(f"  three nulls: {[round(x,3) for x in nulls]}  "
          f"spread {null_spread:.3f} us   <- instrument floor")
    for a, v in reps:
        print(f"  same-program replicate |{a} - {a}#rep| = {v:.3f} us")
    print(f"  RESOLUTION FLOOR (max same-program replicate) = "
          f"{prog_floor:.3f} us")

    # ---- Probe A: slopes ----
    print(f"\n=== PROBE A: pass-price ladder (slope over p in 0,1,2,4) ===")
    elems_chip = B * SLICE_LEN                 # 10,485,760 per pass
    elems_sub = SLICE_LEN * NUM_WAVES          #    327,680 per subcore per pass
    zero = um("A_p0")
    slopes = {}
    print(f"  A_p0 (0 passes, DMA + launch only) = {zero:.3f} us")
    print(f"  {'kind':<8} {'p1':>10} {'p2':>10} {'p4':>10} {'us/pass':>10} "
          f"{'R2':>8} {'ps/elem':>9} {'x cvt':>7} {'x scat':>7}")
    for kind in ("cvt", "scat", "cnt", "cntv"):
        ns = [f"A_{kind}_p{p}" for p in (1, 2, 4)]
        if not all(n in res for n in ns):
            print(f"  {kind:<8} UNSUPPORTED")
            continue
        ys = [zero] + [um(n) for n in ns]
        s, a, r2 = fit_slope([0, 1, 2, 4], ys)
        slopes[kind] = s
        print(f"  {kind:<8} {ys[1]:10.3f} {ys[2]:10.3f} {ys[3]:10.3f} "
              f"{s:10.3f} {r2:8.5f} {s*1e6/elems_chip:9.3f} "
              f"{s/slopes['cvt']:7.3f} "
              f"{(s/slopes['scat'] if 'scat' in slopes else float('nan')):7.3f}")

    # ---- Probe B: the marginal store, as a slope in density ----
    print(f"\n=== PROBE B: marginal conditional store_compressed at L0 ===")
    print(f"  {'density':<8} {'t':>4} {'realized':>9} {'B_ns':>10} {'B_st':>10} "
          f"{'marginal':>10} {'d time %':>9} {'x floor':>8}")
    pts = []
    for name, t in DENS_T.items():
        a, b_ = f"B_ns_{name}", f"B_st_{name}"
        if a not in res or b_ not in res:
            continue
        d = facts[f"realized_density_{name}"]
        marg = um(b_) - um(a)
        pts.append((d, marg))
        print(f"  {name:<8} {t:4d} {d:9.4f} {um(a):10.3f} {um(b_):10.3f} "
              f"{marg:+10.3f} {100.0*marg/um(a):+9.3f} "
              f"{abs(marg)/prog_floor:8.2f}")
    if len(pts) >= 3:
        s, a, r2 = fit_slope([p[0] for p in pts], [p[1] for p in pts])
        print(f"\n  marginal(density) = {a:+.3f} + {s:+.3f} * density  us"
              f"   (R2 {r2:.5f})")
        print(f"  intercept {a:+.3f} us = the per-vector ISSUE cost of a store "
              f"that stores nothing")
        print(f"  slope     {s:+.3f} us per unit density "
              f"= {s*1e6/elems_chip:+.4f} ps per stored element")
    for lbl, a, b_ in (("p2 linearity @d30", "B_ns_d30_p2", "B_st_d30_p2"),
                       ("2nd store  @d30", "B_st_d30", "B_st2_d30"),
                       ("2nd store  @d65", "B_st_d65", "B_st2_d65")):
        if a in res and b_ in res:
            print(f"  {lbl}: {um(b_) - um(a):+.3f} us")

    out = {
        "facts": facts,
        "geometry": {"b": B, "n": N, "k": K, "slice_len": SLICE_LEN,
                     "num_waves": NUM_WAVES, "elems_chip_per_pass": elems_chip,
                     "elems_subcore_per_pass": elems_sub},
        "dispatch_floor_us": floor_us,
        "null_spread_us": null_spread,
        "replicate_floor_us": prog_floor,
        "slopes_us_per_pass": slopes,
        "hlo_len": hlo_len,
        "hlo_st_minus_ns": deltas,
        "entry_hlo": entry_rep,
        "skipped": skipped,
        "visit_order": gb.LAST_ORDER,
        "density_points": pts,
        "results": {k: dict(v) for k, v in res.items()},
    }
    with open(os.path.join(args.outdir, "probe15.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {args.outdir}/probe15.json")
    print("PROBE15_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
