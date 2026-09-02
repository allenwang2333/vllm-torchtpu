"""Step-2 probe P6: is a narrow strided HBM read charged for the bytes it asks
for, or for the granule/tile it lands in?

The dominant term in nightly decode HBM traffic is `_fetch_bkv`'s DMA of the
indexer K cache (`streamindex_topk.py:160-198`), which copies whole
`[page_size_per_kv_packing, kv_packing, lkv_dim]` pages out of a
`u8[..., 4, 256]` cache. Only 129 of every 256 minor bytes are consumed
(128 fp8 key bytes + 1 e8m0 scale byte); the remaining 127 are padding
introduced by `align_to(129, 128) == 256`.

If a strided read that takes only the useful prefix is charged for the prefix,
narrowing the DMA is a one-line, movement-only change worth ~37% of the
dominant decode term. If it is charged for the full 256-byte stride -- because
the HBM/DMA granule is >= 128 bytes and 129 rounds up to 256 -- the change is
worth exactly nothing and the padding can only be reclaimed by repacking the
cache itself.

Nothing in the source settles this. It is a property of the DMA engine's
granule, so it is measured.

Design
------
Two arms, both holding the DMA COUNT and the ROW COUNT fixed and varying only
how many bytes of each 256-byte row are requested. Same number of descriptors,
same number of rows, same grid: any time difference is bytes/granules moved.

  arm "minor"  -- source u8[R, 256]; copy x[rows, 0:w] for w in a sweep that
                  straddles every plausible granule (32 / 64 / 96 / 128 / 129 /
                  160 / 192 / 224 / 256). This is the direct experiment and the
                  one that can locate the granule. Some widths may not lower;
                  each width is attempted independently and failures reported.

  arm "major"  -- source u8[R, S, 128]; copy x[rows, 0:t, :] for t = 1..S. Every
                  slice is 128-lane aligned so it is guaranteed to lower. It
                  reads t*128 bytes out of every S*128-byte stride. This is the
                  robust fallback: at S=2 it is exactly "128 of every 256
                  bytes", which brackets the 129-byte question from below.

Calibration is mandatory and is reported: the full-width point must approach
the 2.97e12 B/s achieved-stream figure from `report.md` §4, or the measurement
is latency-bound rather than bandwidth-bound and the ratios mean nothing.

A physical-allocation check runs first (same idiom as `probe_lkv_dim.py`): if
a reshaped source array is charged more HBM than its logical size, its minor
dim was padded by the layout assignment and that arm is invalid.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# 2.97e12 B/s measured read-write stream, 3.70e12 B/s reported spec.
# `report.md` §4, from `references/hardware-facts.md`.
HBM_ACHIEVED_BPS = 2.97e12
HBM_SPEC_BPS = 3.70e12

STRIDE_BYTES = 256  # lkv_dim, the indexer K cache's minor dim
USEFUL_BYTES = 129  # D_IDX + D_IDX // QUANT_BLOCK

MINOR_WIDTHS = (32, 64, 96, 128, 129, 160, 192, 224, 256)


def _p(tag, *rest):
    print(f"[P6] {tag}:", *rest, flush=True)


def hbm_in_use(dev):
    st = dev.memory_stats() or {}
    for key in ("bytes_in_use", "bytes_in_use_hbm", "hbm_bytes_in_use"):
        if key in st:
            return st[key]
    return None


def check_physical_size(shape, dtype=jnp.uint8):
    """Actual HBM charged vs logical bytes. Ratio > 1 means the layout padded."""
    dev = jax.devices()[0]
    before = hbm_in_use(dev)
    if before is None:
        return None
    x = jax.block_until_ready(jnp.zeros(shape, dtype))
    after = hbm_in_use(dev)
    logical = int(np.prod(shape)) * np.dtype(dtype).itemsize
    del x
    return (after - before), logical


# ---------------------------------------------------------------------------
# The read kernel. A 4-deep software pipeline of explicit HBM->VMEM DMAs, which
# is the same mechanism `_fetch_bkv` uses (`pltpu.make_async_copy` on an
# `ANY`-memory-space ref), so the granule this measures is the granule that
# term pays.
# ---------------------------------------------------------------------------

NBUF = 4


def _build_reader(rows_per_copy, ngrid, slicer):
    """A Pallas kernel that streams `ngrid` slices of `src` HBM->VMEM.

    `slicer(x_ref, row_start)` returns the sliced source ref for one copy. The
    caller sizes the VMEM scratch to match.

    `ngrid` is a Python int, so every pipeline-boundary test below is resolved
    at trace time and no `pl.num_programs` comparison is needed.

    There is deliberately NO consumer of the fetched bytes. Pallas DMAs are
    effectful -- they carry semaphore side effects and the `wait` forces
    completion -- so they are not dead-code eliminated, and adding a load of
    the VMEM buffer would put a second, width-dependent cost on the very
    comparison this probe is making. The check that the DMAs really happened is
    the calibration in `interpret()`: the full-width arm has to land near the
    2.97e12 B/s achieved-stream reference, and no arm may report a useful
    bandwidth above the 3.70e12 B/s spec.
    """

    def _slot(j):
        return j % NBUF if isinstance(j, int) else jax.lax.rem(j, NBUF)

    def kernel(x_hbm, o_ref, buf, sems):
        i = pl.program_id(0)

        def start(j):
            s = _slot(j)
            pltpu.make_async_copy(slicer(x_hbm, j * rows_per_copy),
                                  buf.at[s], sems.at[s]).start()

        def wait(j):
            s = _slot(j)
            pltpu.make_async_copy(slicer(x_hbm, j * rows_per_copy),
                                  buf.at[s], sems.at[s]).wait()

        # Prologue: fill the pipeline on the first step.
        @pl.when(i == 0)
        def _():
            for j in range(min(NBUF - 1, ngrid)):
                start(j)

        if ngrid > NBUF - 1:
            @pl.when(i < ngrid - (NBUF - 1))
            def _():
                start(i + NBUF - 1)

        wait(i)
        o_ref[0] = i

    return kernel


def _time_pallas(fn, x, warmup=2, trials=5, target_s=0.05):
    out = fn(x)
    jax.block_until_ready(out)
    for _ in range(warmup):
        jax.block_until_ready(fn(x))
    t0 = time.perf_counter()
    jax.block_until_ready(fn(x))
    single = time.perf_counter() - t0
    iters = max(1, min(200, int(target_s / max(single, 1e-9))))
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(x)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / iters)
    return best, iters


def run_minor_arm(total_rows, rows_per_copy, widths):
    """u8[R, 256]; copy x[rows, 0:w]. Same rows, same DMA count, varying w."""
    ngrid = total_rows // rows_per_copy
    src_shape = (total_rows, STRIDE_BYTES)
    key = jax.random.key(0)
    x = jax.block_until_ready(jax.random.bits(key, src_shape, dtype=jnp.uint8))

    results = {}
    for w in widths:
        buf_shape = (NBUF, rows_per_copy, w)

        def slicer(x_ref, row_start, w=w):
            return x_ref.at[pl.ds(row_start, rows_per_copy), pl.ds(0, w)]

        kernel = _build_reader(rows_per_copy, ngrid, slicer)
        try:
            f = pl.pallas_call(
                kernel,
                grid=(ngrid, ),
                in_specs=[pl.BlockSpec(memory_space=pltpu.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                out_shape=jax.ShapeDtypeStruct((1, ), jnp.int32),
                scratch_shapes=[
                    pltpu.VMEM(buf_shape, jnp.uint8),
                    pltpu.SemaphoreType.DMA((NBUF, )),
                ],
            )
            f = jax.jit(f)
            t, iters = _time_pallas(f, x)
        except Exception as e:  # noqa: BLE001 - a width that will not lower is a result
            _p(f"minor w={w}", "FAIL", repr(e)[:300])
            results[w] = None
            continue

        useful = total_rows * w
        strided = total_rows * STRIDE_BYTES
        results[w] = {
            "t_s": t,
            "iters": iters,
            "useful_bytes": useful,
            "bps_useful": useful / t,
            "bps_if_charged_full_stride": strided / t,
        }
        _p(
            f"minor w={w:>3}", f"t={t*1e6:9.2f}us",
            f"useful={useful/1e6:8.2f}MB", f"B/s(useful)={useful/t:.3e}",
            f"B/s(if full stride)={strided/t:.3e}")
    return results


def run_major_arm(total_rows, rows_per_copy, S):
    """u8[R, S, 128]; copy x[rows, 0:t, :]. Every slice is 128-lane aligned."""
    ngrid = total_rows // rows_per_copy
    src_shape = (total_rows, S, 128)

    phys = check_physical_size(src_shape)
    if phys is not None:
        actual, logical = phys
        _p(f"major S={S} layout", f"logical={logical/1e6:.2f}MB",
           f"actual={actual/1e6:.2f}MB", f"ratio={actual/max(logical,1):.3f}")
        if logical and actual / logical > 1.05:
            _p(f"major S={S}", "SKIP - source array is physically padded, "
               "the stride is not what this arm assumes")
            return {}

    key = jax.random.key(1)
    x = jax.block_until_ready(jax.random.bits(key, src_shape, dtype=jnp.uint8))

    results = {}
    for t_sub in range(1, S + 1):
        buf_shape = (NBUF, rows_per_copy, t_sub, 128)

        def slicer(x_ref, row_start, t_sub=t_sub):
            return x_ref.at[pl.ds(row_start, rows_per_copy),
                            pl.ds(0, t_sub), :]

        kernel = _build_reader(rows_per_copy, ngrid, slicer)
        try:
            f = pl.pallas_call(
                kernel,
                grid=(ngrid, ),
                in_specs=[pl.BlockSpec(memory_space=pltpu.ANY)],
                out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
                out_shape=jax.ShapeDtypeStruct((1, ), jnp.int32),
                scratch_shapes=[
                    pltpu.VMEM(buf_shape, jnp.uint8),
                    pltpu.SemaphoreType.DMA((NBUF, )),
                ],
            )
            f = jax.jit(f)
            tt, iters = _time_pallas(f, x)
        except Exception as e:  # noqa: BLE001
            _p(f"major S={S} t={t_sub}", "FAIL", repr(e)[:300])
            results[t_sub] = None
            continue

        useful = total_rows * t_sub * 128
        strided = total_rows * S * 128
        results[t_sub] = {
            "t_s": tt,
            "iters": iters,
            "useful_bytes": useful,
            "bps_useful": useful / tt,
            "bps_if_charged_full_stride": strided / tt,
        }
        _p(
            f"major S={S} t={t_sub} ({t_sub*128}/{S*128}B)",
            f"t={tt*1e6:9.2f}us", f"useful={useful/1e6:8.2f}MB",
            f"B/s(useful)={useful/tt:.3e}",
            f"B/s(if full stride)={strided/tt:.3e}")
    return results


def interpret(minor, major, total_rows):
    full = minor.get(STRIDE_BYTES)
    if not full:
        _p("VERDICT", "INCONCLUSIVE - the full-width control did not run")
        return
    t_full = full["t_s"]
    achieved = full["bps_useful"]

    # Elision check. If a DMA were removed, the arm would report a useful
    # bandwidth above what the hardware can deliver.
    for w, r in sorted(minor.items()):
        if r and r["bps_useful"] > HBM_SPEC_BPS:
            _p("ELISION WARNING",
               f"w={w} reports {r['bps_useful']:.3e} B/s useful, above the "
               f"{HBM_SPEC_BPS:.2e} B/s spec. That arm's DMA did not happen; "
               "discard it.")
    _p("calibration",
       f"full-width achieved {achieved:.3e} B/s = "
       f"{100*achieved/HBM_ACHIEVED_BPS:.1f}% of the 2.97e12 achieved-stream "
       f"reference, {100*achieved/HBM_SPEC_BPS:.1f}% of the 3.70e12 spec")
    if achieved < 0.5 * HBM_ACHIEVED_BPS:
        _p("calibration",
           "WARNING - the full-width arm is well below the achieved-stream "
           "reference, so this measurement is not bandwidth-limited and the "
           "cost ratios below are about latency, not about bytes charged")

    _p("cost ratios", "time(w) / time(256); a value of 1.0 means the narrow "
       "read costs the same as the full-width one")
    prev_t = None
    prev_w = None
    for w in sorted(k for k, v in minor.items() if v):
        r = minor[w]["t_s"] / t_full
        step = ""
        if prev_t is not None:
            d = (minor[w]["t_s"] - prev_t) / t_full
            step = f"  step_from_{prev_w}={d:+.3f}"
        _p(f"  w={w:>3}", f"ratio={r:.3f}  (bytes ratio={w/STRIDE_BYTES:.3f})"
           + step)
        prev_t, prev_w = minor[w]["t_s"], w

    r129 = minor.get(USEFUL_BYTES)
    if r129:
        ratio = r129["t_s"] / t_full
        _p("P6_CHARGE_RATIO_129",
           f"{ratio:.3f}  (bytes asked = {USEFUL_BYTES/STRIDE_BYTES:.3f} of a "
           "full-width read)")
        if ratio > 0.92:
            _p("P6_VERDICT",
               "CHARGED-FULL-STRIDE - a 129-of-256 read costs the same as a "
               "256-byte read. Narrowing the DMA recovers nothing; the padding "
               "can only be reclaimed by repacking the cache.")
        elif ratio < 0.62:
            _p("P6_VERDICT",
               "CHARGED-FOR-BYTES - a 129-of-256 read costs close to what it "
               "asks for. The one-line narrowing of _fetch_bkv is live.")
        else:
            _p("P6_VERDICT",
               f"PARTIAL - ratio {ratio:.3f} sits between the two hypotheses; "
               "read the granule from the step structure above.")
    else:
        _p("P6_VERDICT",
           "INCONCLUSIVE at w=129 - that width did not lower; infer from the "
           "surrounding widths and the major arm.")

    _p("minor-arm confound",
       "for w < 128 the VMEM destination is a partial 128-lane row, so Mosaic "
       "pads it and the DMA writes strided into VMEM. Any excess cost at "
       "w < 128 may be that write pattern rather than the HBM read. The major "
       "arm below has no such confound -- every destination is lane-full.")

    # The major arm at S=2, t=1 is the clean, destination-confound-free
    # reading of "128 bytes out of every 256".
    m2 = major.get(2) or {}
    if m2.get(1) and m2.get(2):
        r = m2[1]["t_s"] / m2[2]["t_s"]
        _p("P6_MAJOR_128_OF_256",
           f"ratio={r:.3f} (bytes asked = 0.500). Lane-full destination, so "
           "this is the HBM read alone.")
        if r > 0.92:
            _p("P6_MAJOR_VERDICT", "CHARGED-FULL-STRIDE at the 128/256 point")
        elif r < 0.62:
            _p("P6_MAJOR_VERDICT", "CHARGED-FOR-BYTES at the 128/256 point")
        else:
            _p("P6_MAJOR_VERDICT", f"PARTIAL at the 128/256 point ({r:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-rows", type=int, default=2 * 1024 * 1024,
                    help="rows of 256 B; default 2Mi rows = 512 MiB source")
    ap.add_argument("--rows-per-copy", type=int, default=16384,
                    help="rows per DMA; 16384 x 256 B = 4 MiB at full width")
    args = ap.parse_args()

    dev = jax.devices()[0]
    _p("device", dev.device_kind, "jax", jax.__version__)
    _p("geometry", f"total_rows={args.total_rows}",
       f"rows_per_copy={args.rows_per_copy}",
       f"ngrid={args.total_rows // args.rows_per_copy}",
       f"stride={STRIDE_BYTES}B", f"useful={USEFUL_BYTES}B",
       f"nbuf={NBUF}")

    phys = check_physical_size((args.total_rows, STRIDE_BYTES))
    if phys:
        a, l = phys
        _p("minor arm layout", f"logical={l/1e6:.2f}MB actual={a/1e6:.2f}MB "
           f"ratio={a/max(l,1):.3f}")

    _p("---", "arm 1: minor-dim prefix sweep (the direct experiment)")
    minor = run_minor_arm(args.total_rows, args.rows_per_copy, MINOR_WIDTHS)

    _p("---", "arm 2: 128-aligned major-dim sweep (the robust fallback)")
    major = {}
    for S in (2, 4):
        major[S] = run_major_arm(args.total_rows * 2 // S,
                                 args.rows_per_copy * 2 // S, S)

    _p("---", "interpretation")
    interpret(minor, major, args.total_rows)


if __name__ == "__main__":
    main()
