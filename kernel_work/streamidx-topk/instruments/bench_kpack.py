"""R1 / Flip 1 -- does halving the payload of a *page-table-indirect,
dynamically-clamped* `make_async_copy` halve its time?

Why this file exists
--------------------
Option A ("128-byte K record", `measurement_and_design.md` §3.1) is rank 1 on
the strength of one number that was measured on the wrong shape.  P6
(`probes/P6-strided-read-charge.md`) established that HBM charges for the bytes
a DMA asks for -- charge ratio 1.02, 98.0 % of contiguous bandwidth -- but it
measured that on a **flat array read at a statically computed, affine offset**.

`_fetch_bkv` (`streamindex_topk.py:160-198`) is not that.  For each `batch_idx`
in `range(seq_batch_size)` and each `i` in `range(bkv_p)` it computes

    page_idx         = min(page_indices_offset + i, num_page_indices - 1)
    safe_page_offset = min(page_indices_ref[page_idx] * page_size_per_kv_packing,
                           max(0, max_hbm_pages - page_size_per_kv_packing))

and issues one `make_async_copy` at that offset -- **read out of a scalar-
prefetch page table and clamped at runtime**.  If the cost of that DMA is set
by descriptor issue and address indirection rather than by payload bytes, then
halving the record buys single-digit percent, Option A falls below the ±8.5 %
instrument spread, and the ranking becomes Candidate 0 > C > B > A.

Gate (`measurement_and_design.md` §5, R1): **if halving the bytes saves ≤33 %
of the time, A drops to rank 4.**

Definition of the reported charge ratio
---------------------------------------
    charge_ratio = t(262144 B) / t(132096 B)   at the SAME DMA count

    2.00  purely byte-proportional  -- halving bytes halves time, A wins
    1.00  purely descriptor-bound   -- halving bytes buys nothing, A dies
    1.50  the gate: exactly 33.3 % of the time saved

Note this is the *reciprocal-style* definition given in the R1 brief, not P6's
`time_ratio / byte_ratio` (P6's 1.02 = "2 % worse than perfectly
byte-proportional").  Both are printed, spelled out, so neither can be misread.

The arms
--------
Every arm holds the **DMA count, the grid, the pipeline depth and the number of
in-flight DMAs fixed** and changes only the shape of the record each DMA moves.

  record 256 (the incumbent):  cache_kv u8[P, 256, 4, 256] -> 262,144 B per DMA
  record 132 (Option A)     :  cache_kv u8[P, 258, 4, 128] -> 132,096 B per DMA

and three *forms* of offset arithmetic, so the indirection cost is isolated
rather than assumed:

  form "indirect"        -- byte-for-byte `_fetch_bkv`: page table in scalar
                            prefetch, both `min()`/`max()` clamps.  THIS IS THE
                            ARM R1 IS ABOUT.
  form "affine"          -- same DMAs at an offset computed affinely from
                            `program_id`, no page table, no clamps.  This is
                            P6's form at `_fetch_bkv`'s geometry; the
                            indirect/affine gap is the price of indirection.
  form "indirect_dynlen" -- "indirect" plus a runtime-clamped *length*
                            (`jnp.minimum`) on both the source and destination
                            slice, in the idiom `_fetch_bq` uses.  `_fetch_bkv`
                            itself clamps only the offset -- its extent is the
                            Python int `page_size_per_kv_packing` -- so this arm
                            is a bound on what a dynamic extent would add, not
                            a replication of the incumbent.

Geometry is nightly decode (`measurement_and_design.md` §Area 2): grid `(16,)`,
`seq_batch_size = 4`, `bkv_p = 2`, `num_bkv = 5` -> 640 DMAs per call, matching
the 640 x 262,144 B = 167.8 MB the design attributes to the decode call.
`bkv_p = 3` (P7a's measured win, and the configuration Option A would most
likely ship on top of) is swept too, at 768 DMAs per call.

There is deliberately **no consumer** of the fetched bytes, exactly as in P6:
Pallas DMAs are effectful and are not DCE'd, and a consumer would put a second,
width-dependent cost on the comparison.  The guard against an elided DMA is the
bandwidth check -- no arm may report useful bytes above the 3.70e12 B/s spec --
plus the physical-layout gate on the source arrays.

Timing methodology is P6's, unchanged: `probes/_probe_util.time_arm`, warm up,
size an iteration count so the host clock is not measuring itself, best-of-7.
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# `time_arm` / `emit_json` live next door in probes/.  Reused rather than
# copied so the timing methodology is literally P6's, not a paraphrase of it.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "probes"))
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

HBM_ACHIEVED_BPS = 2.97e12  # measured read-write stream, references/hardware-facts.md
HBM_SPEC_BPS = 3.70e12  # reported spec

# `pltpu.ANY` does not exist on the image's jax 0.10.2; `pltpu.HBM` is what the
# incumbent kernel itself uses for `cache_kv` (`streamindex_topk.py:648`).
_HBM_SPACE = (getattr(pltpu, "HBM", None)
              or getattr(getattr(pltpu, "MemorySpace", None), "ANY", None)
              or getattr(pltpu, "ANY", None))

VMEM_LIMIT_BYTES = 100 * 1024 * 1024

# The record layouts. (page_size_per_kv_packing, kv_packing, lkv_dim).
#   256: today's u8[..., 256, 4, 256]  -> 256*4*256 = 262144 B per page DMA
#   132: Option A's u8[..., 258, 4, 128] -> 258*4*128 = 132096 B per page DMA
#   135: Option A with a 4 KiB-ALIGNED page. 258 rows gives a page stride of
#        132096 B = 1024 mod 4096, so three of every four DMAs start
#        4 KiB-misaligned, while every rec256 DMA is 256 KiB-aligned. That is a
#        confound between the arms that is NOT the byte reduction: it is a
#        property of the packing Option A happens to specify. 264 rows gives
#        135168 B = 0 mod 4096 for 2.3 % more bytes. If the charge ratio moves
#        between 132 and 135, Option A should be re-specified with a padded
#        page rather than dropped. (Raised by the R1 red-team review as the one
#        untested alternative explanation; P6 never exercised a non-4 KiB-
#        aligned DMA start, since its source had power-of-two row strides.)
RECORDS = {
    256: (256, 4, 256),
    132: (258, 4, 128),
    135: (264, 4, 128),
}


def record_bytes(rec: int) -> int:
    psp, kvp, lkv = RECORDS[rec]
    return psp * kvp * lkv


def hbm_in_use(dev):
    st = dev.memory_stats() or {}
    for key in ("bytes_in_use", "bytes_in_use_hbm", "hbm_bytes_in_use"):
        if key in st:
            return st[key]
    return None


def check_physical_size(shape, dtype=jnp.uint8):
    """Actual HBM charged vs logical bytes. Ratio > 1 means the layout padded.

    If the source array is physically padded, the DMA's stride is not the one
    the arm assumes and the two arms are not comparable.  P6's gate, kept.
    """
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
# The kernel. A transcription of `_fetch_bkv`'s DMA schedule with the compute
# removed: same double-buffered start/wait, same per-(batch, page) descriptor,
# same offset arithmetic, same self-copy wait idiom.
# ---------------------------------------------------------------------------


def _make_kernel(*, form, psp, kvp, lkv, bkv_p, seq_batch_size, num_bkv,
                 pages_per_seq, num_grid):

    def kernel(page_indices_ref, eff_len_ref, cache_kv_hbm_ref, o_ref,
               bkv_x2_ref, sems):
        total_num_pages = cache_kv_hbm_ref.shape[0]
        num_src_pages = total_num_pages
        num_page_indices = page_indices_ref.shape[0]

        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * psp, kvp, lkv)
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

        i = pl.program_id(0)
        seq_idx = i * seq_batch_size

        def start(seq_idx, bkv_idx, bkv_sem_idx):
            # `_fetch_bkv(..., wait=False)`, verbatim in structure.
            for batch_idx in range(seq_batch_size):
                sem = sems.at[bkv_sem_idx, batch_idx]
                bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]

                kv_p_start = bkv_idx * bkv_p
                page_indices_offset = ((seq_idx + batch_idx) * pages_per_seq +
                                       kv_p_start)

                for j in range(bkv_p):
                    if form == "affine":
                        # P6's form at this geometry: no page table, no
                        # data-dependent load. The `remainder` keeps the offset
                        # in bounds and is NOT a fidelity concession -- an
                        # earlier version used a bare `(page_indices_offset+j)
                        # * psp`, which at --num-grid 128 reaches index 5119
                        # against a 2048-page array and, with
                        # `disable_bounds_checks=True`, reads 2.5x past the end
                        # of the source. In bounds at grid 16, so the two grids
                        # agreed, but the grid-128 affine numbers from that
                        # version are not quotable. The wrap costs one scalar
                        # op and leaves the arm affine: no page table, no
                        # clamp against loaded data.
                        safe_page_offset = jnp.remainder(
                            page_indices_offset + j, num_src_pages) * psp
                        sz = psp
                    else:
                        page_idx = jnp.minimum(page_indices_offset + j,
                                               num_page_indices - 1)
                        safe_page_offset = jnp.minimum(
                            page_indices_ref[page_idx] * psp,
                            jnp.maximum(0, max_hbm_pages - psp),
                        )
                        if form == "indirect_dynlen":
                            # A runtime-clamped extent, `_fetch_bq`'s idiom.
                            # `eff_len_ref` is a scalar-prefetch operand set so
                            # the clamp always resolves to `psp` -- the bytes
                            # moved are identical, only the extent's
                            # staticness changes.
                            sz = jnp.minimum(psp, eff_len_ref[0])
                        else:
                            sz = psp

                    pltpu.make_async_copy(
                        reshaped_cache_hbm_ref.at[pl.ds(safe_page_offset, sz)],
                        bkv_vmem_ref.at[pl.ds(j * psp, sz)],
                        sem,
                    ).start()

        def wait(bkv_sem_idx):
            # `_fetch_bkv(..., wait=True)`: a self-copy on the destination
            # extent, which is how the incumbent drains the semaphore.
            for batch_idx in range(seq_batch_size):
                sem = sems.at[bkv_sem_idx, batch_idx]
                dst = bkv_x2_ref.at[bkv_sem_idx,
                                    batch_idx].at[pl.ds(0, bkv_p * psp)]
                pltpu.make_async_copy(dst, dst, sem).wait()

        # Prologue on the first grid step; every later step's first block was
        # started by the previous step, exactly as `process()` does.
        @pl.when(i == 0)
        def _():
            start(seq_idx, 0, 0)

        # The buffer parity is a RUNTIME value, keyed on the global block index
        # `i * num_bkv + b`, not on `b % 2`.
        #
        # This is not a stylistic choice. With `num_bkv` odd -- and nightly
        # decode's num_bkv IS 5 -- a compile-time `b % 2` flips parity at every
        # grid boundary: grid step k+1 waits on semaphore 0 for a block that
        # grid step k started into semaphore 1, and the kernel deadlocks. That
        # is measured, not hypothesized: job j-248bfcd5 hung for 57 minutes
        # after printing the dispatch floor and produced no arm. The incumbent
        # has the same hazard and avoids it the same way, by carrying
        # `bkv_sem_idx` as a traced value (`streamindex_topk.py:292-302`,
        # `get_next_bkv_ids`), which makes the dynamic `sems.at[idx]` and
        # `bkv_x2_ref.at[idx]` indexing below a fidelity gain as well as a fix.
        base = i * num_bkv
        for b in range(num_bkv):
            cur = jnp.remainder(base + b, 2)
            nxt = 1 - cur
            if b + 1 < num_bkv:
                start(seq_idx, b + 1, nxt)
            else:
                # Wrap into the next grid step, so the pipeline is not drained
                # at every grid boundary. Guarded on the last step so no DMA is
                # left in flight at kernel exit.
                @pl.when(i < num_grid - 1)
                def _(seq_idx=seq_idx, nxt=nxt):
                    start(seq_idx + seq_batch_size, 0, nxt)

            wait(cur)

        o_ref[0] = i

    return kernel


def build_arm(*, form, rec, bkv_p, seq_batch_size, num_grid, num_bkv,
              pages_per_seq, num_pages):
    psp, kvp, lkv = RECORDS[rec]
    max_num_seqs = num_grid * seq_batch_size
    num_page_indices = max_num_seqs * pages_per_seq

    src_shape = (num_pages, psp, kvp, lkv)
    buf_shape = (2, seq_batch_size, bkv_p * psp, kvp, lkv)

    kernel = _make_kernel(form=form, psp=psp, kvp=kvp, lkv=lkv, bkv_p=bkv_p,
                          seq_batch_size=seq_batch_size, num_bkv=num_bkv,
                          pages_per_seq=pages_per_seq, num_grid=num_grid)

    f = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=_HBM_SPACE)],
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            grid=(num_grid, ),
            scratch_shapes=[
                pltpu.VMEM(buf_shape, jnp.uint8),
                pltpu.SemaphoreType.DMA((2, seq_batch_size)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((1, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )
    return jax.jit(f), src_shape, num_page_indices


def make_page_table(num_page_indices, num_pages, seed=0):
    """A page table that touches each page at most once per call.

    A random permutation, not `arange`: an affine page table would let the
    compiler or the memory system see a sequential stream and would make the
    indirect arm secretly affine.
    """
    rng = np.random.default_rng(seed)
    if num_page_indices <= num_pages:
        idx = rng.permutation(num_pages)[:num_page_indices]
    else:
        idx = rng.integers(0, num_pages, size=num_page_indices)
    return jnp.asarray(np.ascontiguousarray(idx, dtype=np.int32))


def run_arm(*, form, rec, bkv_p, seq_batch_size, num_grid, num_bkv,
            pages_per_seq, num_pages, trials):
    psp, kvp, lkv = RECORDS[rec]
    rb = record_bytes(rec)
    n_dma = num_grid * num_bkv * seq_batch_size * bkv_p
    total_bytes = n_dma * rb
    label = f"{form}/rec{rec}/bkvp{bkv_p}"

    # Progress is printed before every blocking phase. A Pallas kernel can hang
    # rather than fail (see the parity note in `_make_kernel`), and a job that
    # dies silently with no marker costs a whole pod reservation to diagnose.
    p(label, "building")
    try:
        f, src_shape, num_page_indices = build_arm(
            form=form, rec=rec, bkv_p=bkv_p, seq_batch_size=seq_batch_size,
            num_grid=num_grid, num_bkv=num_bkv, pages_per_seq=pages_per_seq,
            num_pages=num_pages)
    except Exception as e:  # noqa: BLE001 -- a form that will not lower is a result
        p(label, "BUILD FAIL", repr(e)[:400])
        return {"label": label, "form": form, "rec": rec, "bkv_p": bkv_p,
                "ok": False, "error": repr(e)[:1200]}

    page_indices = make_page_table(num_page_indices, num_pages)
    eff_len = jnp.asarray([psp * 4096], jnp.int32)  # clamp always resolves to psp
    key = jax.random.key(rec + bkv_p)
    x = jax.block_until_ready(jax.random.bits(key, src_shape, dtype=jnp.uint8))

    p(label, f"compiling + timing (n_dma={n_dma}, {total_bytes/1e6:.1f}MB "
      f"per call)")
    try:
        t = time_arm(f, (page_indices, eff_len, x), warmup=3, trials=trials)
    except Exception as e:  # noqa: BLE001
        p(label, "RUN FAIL", repr(e)[:400])
        del x
        return {"label": label, "form": form, "rec": rec, "bkv_p": bkv_p,
                "ok": False, "error": repr(e)[:1200]}
    del x

    best = t["best_s"]
    bps = total_bytes / best
    res = {
        "label": label, "form": form, "rec": rec, "bkv_p": bkv_p, "ok": True,
        "record_bytes": rb, "n_dma": n_dma, "total_bytes": total_bytes,
        "best_s": best, "median_s": t["median_s"],
        "max_s": max(t["trials_s"]), "trials_s": t["trials_s"],
        "iters": t["iters"], "bps": bps,
        "src_shape": list(src_shape),
    }
    p(label,
      f"best={best*1e6:8.2f}us median={t['median_s']*1e6:8.2f}us "
      f"max={max(t['trials_s'])*1e6:8.2f}us "
      f"spread={(max(t['trials_s'])/best-1)*100:5.1f}% "
      f"n_dma={n_dma} rec={rb}B total={total_bytes/1e6:7.2f}MB "
      f"B/s={bps:.3e} iters={t['iters']}")
    if bps > HBM_SPEC_BPS:
        p(label, "ELISION WARNING -- useful bandwidth above the 3.70e12 B/s "
          "spec. That arm's DMA did not happen; discard it.")
    return res


def check_schedule(num_grid, num_bkv):
    """Simulate the start/wait schedule in Python before touching the TPU.

    A Pallas semaphore mismatch does not raise -- it hangs, and a hang costs a
    whole pod reservation to notice (j-248bfcd5, 57 minutes, zero arms). This
    replays the exact sequence `_make_kernel` emits and asserts the two
    properties that a double buffer needs: every wait finds the block it is
    waiting for, and no buffer is refilled while its previous block is still in
    flight. Cheap, and it is the check that would have caught the odd-`num_bkv`
    parity bug on the host.
    """
    inflight = {0: None, 1: None}
    inflight[0] = 0  # the `pl.when(i == 0)` prologue
    for i in range(num_grid):
        for b in range(num_bkv):
            g = i * num_bkv + b
            cur = g % 2
            nxt_g = None
            if b + 1 < num_bkv:
                nxt_g = g + 1
            elif i < num_grid - 1:
                nxt_g = g + 1
            if nxt_g is not None:
                nxt = 1 - cur
                assert nxt == nxt_g % 2, (
                    f"parity mismatch: block {nxt_g} started into buffer "
                    f"{nxt} but will be waited on buffer {nxt_g % 2}")
                assert inflight[nxt] is None, (
                    f"buffer {nxt} refilled with block {nxt_g} while block "
                    f"{inflight[nxt]} is still in flight")
                inflight[nxt] = nxt_g
            assert inflight[cur] == g, (
                f"wait on buffer {cur} at block {g} finds "
                f"{inflight[cur]!r} -- this schedule deadlocks")
            inflight[cur] = None
    assert all(v is None for v in inflight.values()), (
        f"DMAs left in flight at kernel exit: {inflight}")


def run_dispatch_floor(trials):
    """How long a *zero-DMA* pallas_call of the same launch shape takes.

    This is the contamination guard.  Both arms are timed by a host loop, so if
    per-call dispatch is comparable to the device time the ratio is pulled
    toward 1.0 -- which would look exactly like "descriptor-bound" and would
    fail Option A for the wrong reason.  Both arm times must be large against
    this floor for the charge ratio to mean anything.
    """

    def kernel(o_ref):
        o_ref[0] = 0

    f = jax.jit(
        pl.pallas_call(
            kernel,
            grid=(1, ),
            out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
            out_shape=jax.ShapeDtypeStruct((1, ), jnp.int32),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", )),
        ))
    try:
        t = time_arm(f, (), warmup=3, trials=trials)
    except Exception as e:  # noqa: BLE001
        p("dispatch floor", "FAIL", repr(e)[:300])
        return None
    p("dispatch floor",
      f"best={t['best_s']*1e6:8.2f}us median={t['median_s']*1e6:8.2f}us "
      f"iters={t['iters']} -- an empty pallas_call, host-loop timed the same "
      "way as every arm below")
    return {"best_s": t["best_s"], "median_s": t["median_s"],
            "iters": t["iters"], "trials_s": t["trials_s"]}


def pair_report(results, form, bkv_p, small=132):
    a = next((r for r in results
              if r["ok"] and r["form"] == form and r["rec"] == 256
              and r["bkv_p"] == bkv_p), None)
    b = next((r for r in results
              if r["ok"] and r["form"] == form and r["rec"] == small
              and r["bkv_p"] == bkv_p), None)
    tag = f"{form}/bkvp{bkv_p}" + ("" if small == 132 else f"/small{small}")
    if not (a and b):
        p(f"PAIR {tag}", "INCOMPLETE -- one arm did not run")
        return None
    for stat in ("best_s", "median_s"):
        t256, t132 = a[stat], b[stat]
        charge = t256 / t132
        saved = 1.0 - t132 / t256
        byte_ratio = b["record_bytes"] / a["record_bytes"]
        p6_style = (t132 / t256) / byte_ratio
        p(f"PAIR {tag} [{stat}]",
          f"t({a['record_bytes']}B)={t256*1e6:8.2f}us "
          f"t({b['record_bytes']}B)={t132*1e6:8.2f}us "
          f"CHARGE_RATIO={charge:.3f} time_saved={saved*100:5.1f}% "
          f"(byte ratio {byte_ratio:.4f}; P6-style charge "
          f"time_ratio/byte_ratio={p6_style:.3f})")
    return {
        "form": form, "bkv_p": bkv_p, "small_rec": small,
        "t256_best_s": a["best_s"], "t132_best_s": b["best_s"],
        "t256_median_s": a["median_s"], "t132_median_s": b["median_s"],
        "charge_ratio_best": a["best_s"] / b["best_s"],
        "charge_ratio_median": a["median_s"] / b["median_s"],
        "time_saved_best": 1.0 - b["best_s"] / a["best_s"],
        "time_saved_median": 1.0 - b["median_s"] / a["median_s"],
        "byte_ratio": b["record_bytes"] / a["record_bytes"],
    }


def interpret(pairs, results, floor, gate_bkvp=2):
    if floor:
        worst = min((r["best_s"] for r in results if r["ok"]), default=None)
        if worst is not None:
            frac = floor["best_s"] / worst
            p("dispatch contamination",
              f"the fastest arm is {worst*1e6:.2f}us against a "
              f"{floor['best_s']*1e6:.2f}us empty-launch floor "
              f"({frac*100:.1f}%).")
            if frac > 0.10:
                p("dispatch contamination", "WARNING -- the floor is more "
                  "than 10% of the fastest arm, so a fixed per-call cost is "
                  "compressing the ratio toward 1.0 and the gate is biased "
                  "AGAINST Option A. Re-run with a larger --num-grid.")
    key = next((q for q in pairs
                if q and q["form"] == "indirect" and q["bkv_p"] == gate_bkvp
                and q.get("small_rec", 132) == 132), None)
    if key is None:
        p("R1_VERDICT", f"INCONCLUSIVE -- the indirect/bkv_p={gate_bkvp} pair, "
          "which is the configuration the gate is being read at, did not "
          "produce two arms.")
        return
    cr = key["charge_ratio_best"]
    saved = key["time_saved_best"] * 100
    p("R1_CHARGE_RATIO", f"{cr:.3f}  (t(262144B)/t(132096B), best-of-N, "
      f"indirect form, bkv_p={gate_bkvp}, same DMA count)")
    p("R1_TIME_SAVED_BY_HALVING", f"{saved:.1f}%")
    if saved <= 33.0:
        p("R1_VERDICT", "GATE FAILED -- halving the bytes saves <=33% of the "
          "time. Per measurement_and_design.md §5 R1, Option A drops to "
          "rank 4.")
    else:
        p("R1_VERDICT", "GATE PASSED -- halving the bytes saves >33% of the "
          "time on the page-table-indirect, clamped DMA. P6's "
          "byte-proportionality transfers; Option A survives at rank 1 on "
          "this criterion.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-grid", type=int, default=128,
                    help="grid steps. Nightly decode stage A is (16,); the "
                         "default repeats that schedule 8x so both arms sit "
                         "far above the host dispatch floor. Pass 16 for the "
                         "nightly-exact call, and read the dispatch-floor "
                         "line before trusting its ratio.")
    ap.add_argument("--seq-batch-size", type=int, default=4)
    ap.add_argument("--pages-per-seq", type=int, default=None,
                    help="page-table entries per sequence. Defaults to 10 "
                         "(nightly decode) for the 'main' and 'followup' "
                         "suites, and to 18 for 'deep', where bkv_p*num_bkv "
                         "reaches 18 and a 10-entry stride would make the "
                         "tail DMAs clamp onto the last page-table entry.")
    ap.add_argument("--num-pages", type=int, default=2048,
                    help="pages in the synthetic cache; must exceed "
                         "num_grid*seq_batch_size*pages_per_seq for the page "
                         "table to be a permutation")
    ap.add_argument("--trials", type=int, default=9,
                    help="best-of-N; >=7 is required, the pod's run-to-run "
                         "spread is +-8.5%%")
    ap.add_argument("--suite", choices=("main", "followup", "deep"),
                    default="main",
                    help="'main' is the R1 gate itself: three DMA forms at "
                         "bkv_p 2 and 3. 'followup' answers the two questions "
                         "the R1 red-team review raised -- is the residual an "
                         "ALIGNMENT artifact of Option A's 258-row page "
                         "(rec135 control), and does the saving keep climbing "
                         "with DMA concurrency (bkv_p 4 and 6)? 'deep' reads "
                         "the gate at the bkv_p the extended Candidate 0 "
                         "sweep actually selects -- cdiv(ctx, page_size) = 8 "
                         "at ctx 8192 and 9 at ctx 8704 -- which is past the "
                         "top of the followup sweep, so the saving there was "
                         "an extrapolation and is measured here instead. "
                         "bkv_p 6 is repeated as an anchor against 'followup'.")
    args = ap.parse_args()
    if args.pages_per_seq is None:
        args.pages_per_seq = 18 if args.suite == "deep" else 10

    tpu_banner()
    p("hbm memory space", repr(_HBM_SPACE))
    if _HBM_SPACE is None:
        p("ABORT", "no usable unsliced memory-space name on this jax")
        return 2

    # bkv_p -> num_bkv, so `num_bkv * bkv_p` covers pages_per_seq (=10).
    # bkv_p 4 and 6 each fetch 12 pages/seq, the same as bkv_p 3, so the three
    # concurrency points move a comparable number of bytes; only the number of
    # DMAs in flight per wait differs (16, 24 vs 12 at seq_batch_size 4).
    bkvp_to_numbkv = {2: 5, 3: 4, 4: 3, 6: 2}
    if args.suite == "main":
        bkvp_to_numbkv = {2: 5, 3: 4}
    elif args.suite == "deep":
        # bkv_p 6 (anchor, also in `followup`), then the two the extended
        # Candidate 0 sweep selects. num_bkv is 2 throughout so the pipeline
        # depth is identical across the three and only the DMAs in flight per
        # wait change: 24, 32, 36 at seq_batch_size 4.
        bkvp_to_numbkv = {6: 2, 8: 2, 9: 2}

    for bkv_p, num_bkv in bkvp_to_numbkv.items():
        check_schedule(args.num_grid, num_bkv)
    p("schedule self-check",
      f"OK for num_bkv in {sorted(bkvp_to_numbkv.values())} at "
      f"num_grid={args.num_grid}")

    p("geometry",
      f"num_grid={args.num_grid} seq_batch_size={args.seq_batch_size} "
      f"pages_per_seq={args.pages_per_seq} num_pages={args.num_pages} "
      f"trials={args.trials}")
    recs = (256, 132, 135) if args.suite == "followup" else (256, 132)
    for rec in recs:
        psp, kvp, lkv = RECORDS[rec]
        p(f"record {rec}",
          f"cache_kv u8[{args.num_pages}, {psp}, {kvp}, {lkv}] -> "
          f"{record_bytes(rec)} B per page DMA, page stride "
          f"{record_bytes(rec) % 4096} mod 4096 "
          f"({'4KiB-ALIGNED' if record_bytes(rec) % 4096 == 0 else 'misaligned'})")

    # Physical-layout gate: if either source array is charged more HBM than
    # its logical size, its stride is not what the arm assumes.
    layout = {}
    for rec in recs:
        psp, kvp, lkv = RECORDS[rec]
        phys = check_physical_size((args.num_pages, psp, kvp, lkv))
        if phys:
            actual, logical = phys
            ratio = actual / max(logical, 1)
            layout[rec] = {"actual": actual, "logical": logical,
                           "ratio": ratio}
            p(f"layout rec{rec}",
              f"logical={logical/1e6:.2f}MB actual={actual/1e6:.2f}MB "
              f"ratio={ratio:.3f}")
            if ratio > 1.05:
                p(f"layout rec{rec}", "WARNING -- the source array is "
                  "physically padded; the two arms are not comparable and "
                  "the ratio below is not the charge ratio.")

    floor = run_dispatch_floor(args.trials)

    arms = []
    if args.suite == "main":
        for bkv_p in (2, 3):
            for rec in (256, 132):
                arms.append(("indirect", rec, bkv_p))
        for rec in (256, 132):
            arms.append(("affine", rec, 2))
        for rec in (256, 132):
            arms.append(("indirect_dynlen", rec, 2))
    elif args.suite == "deep":
        for bkv_p in (6, 8, 9):
            for rec in (256, 132):
                arms.append(("indirect", rec, bkv_p))
    else:
        # Alignment control: 132 (misaligned) and 135 (4 KiB-aligned) against
        # the same 256 arm, in the same process, at both bkv_p the gate cares
        # about. Then the concurrency sweep, which turns the main suite's
        # two-point trend into four points.
        for bkv_p in (2, 3):
            for rec in (256, 132, 135):
                arms.append(("indirect", rec, bkv_p))
        for bkv_p in (4, 6):
            for rec in (256, 132):
                arms.append(("indirect", rec, bkv_p))
        # The main suite's grid-128 affine numbers were taken with an
        # out-of-bounds offset; re-measure them with the wrap in place.
        for rec in (256, 132):
            arms.append(("affine", rec, 2))

    results = []
    for form, rec, bkv_p in arms:
        results.append(
            run_arm(form=form, rec=rec, bkv_p=bkv_p,
                    seq_batch_size=args.seq_batch_size,
                    num_grid=args.num_grid,
                    num_bkv=bkvp_to_numbkv[bkv_p],
                    pages_per_seq=args.pages_per_seq,
                    num_pages=args.num_pages, trials=args.trials))

    p("---", "pairwise charge ratios")
    if args.suite == "main":
        pairs = [
            pair_report(results, "indirect", 2),
            pair_report(results, "indirect", 3),
            pair_report(results, "affine", 2),
            pair_report(results, "indirect_dynlen", 2),
        ]
    elif args.suite == "deep":
        pairs = [pair_report(results, "indirect", b) for b in (6, 8, 9)]
        p("---", "concurrency sweep: per-DMA time vs DMAs in flight")
        for rec in (256, 132):
            row = []
            for bkv_p in (6, 8, 9):
                r = next((x for x in results if x["ok"] and x["rec"] == rec
                          and x["form"] == "indirect"
                          and x["bkv_p"] == bkv_p), None)
                if r:
                    row.append(f"bkvp{bkv_p}(inflight "
                               f"{bkv_p*args.seq_batch_size}): "
                               f"{r['best_s']/r['n_dma']*1e9:6.1f}ns/DMA "
                               f"{r['bps']:.3e}B/s")
            p(f"CONCURRENCY rec{rec}", "  ".join(row))
        # The absolute stake, normalised to the nightly decode call's own DMA
        # count (640 = 16 grid steps x 4 sequences x 10 pages). The probe moves
        # a different number of DMAs per call, so the per-DMA saving is what
        # transfers, not the per-call one.
        for bkv_p in (6, 8, 9):
            a = next((x for x in results if x["ok"] and x["rec"] == 256
                      and x["form"] == "indirect" and x["bkv_p"] == bkv_p),
                     None)
            b = next((x for x in results if x["ok"] and x["rec"] == 132
                      and x["form"] == "indirect" and x["bkv_p"] == bkv_p),
                     None)
            if a and b:
                per_dma = (a["best_s"] - b["best_s"]) / a["n_dma"]
                p(f"STAKE bkvp{bkv_p}",
                  f"delta={per_dma*1e9:6.1f}ns/DMA -> "
                  f"{per_dma*640*1e6:6.2f}us over the nightly call's 640 page "
                  f"DMAs (DMA-only, no compute to overlap with, so an upper "
                  f"bound on the call-level saving)")
    else:
        pairs = [
            pair_report(results, "indirect", 2),
            pair_report(results, "indirect", 3),
            pair_report(results, "indirect", 4),
            pair_report(results, "indirect", 6),
            pair_report(results, "indirect", 2, small=135),
            pair_report(results, "indirect", 3, small=135),
            pair_report(results, "affine", 2),
        ]
        p("---", "alignment control: does the 258-row page's 4 KiB "
          "misalignment explain the residual?")
        for bkv_p in (2, 3):
            r132 = next((r for r in results if r["ok"] and r["rec"] == 132
                         and r["form"] == "indirect" and r["bkv_p"] == bkv_p),
                        None)
            r135 = next((r for r in results if r["ok"] and r["rec"] == 135
                         and r["form"] == "indirect" and r["bkv_p"] == bkv_p),
                        None)
            if r132 and r135:
                # 135 moves 2.3 % MORE bytes than 132. If alignment is free,
                # 135 should be very slightly SLOWER. If it is faster, the
                # misalignment is real and Option A should be re-specified
                # with a 264-row page rather than dropped.
                p(f"ALIGNMENT bkvp{bkv_p}",
                  f"t(132096B, misaligned)={r132['best_s']*1e6:8.2f}us  "
                  f"t(135168B, 4KiB-aligned)={r135['best_s']*1e6:8.2f}us  "
                  f"aligned/misaligned={r135['best_s']/r132['best_s']:.4f} "
                  f"(bytes ratio {r135['record_bytes']/r132['record_bytes']:.4f} "
                  f"-- aligned arm moves MORE bytes, so <1.0 means "
                  "misalignment is costing real time)")
        p("---", "concurrency sweep: per-DMA time vs DMAs in flight")
        for rec in (256, 132):
            row = []
            for bkv_p in (2, 3, 4, 6):
                r = next((x for x in results if x["ok"] and x["rec"] == rec
                          and x["form"] == "indirect"
                          and x["bkv_p"] == bkv_p), None)
                if r:
                    row.append(f"bkvp{bkv_p}(inflight "
                               f"{bkv_p*args.seq_batch_size}): "
                               f"{r['best_s']/r['n_dma']*1e9:6.1f}ns/DMA "
                               f"{r['bps']:.3e}B/s")
            p(f"CONCURRENCY rec{rec}", "  ".join(row))

    # Cost of indirection itself, at fixed record size.
    for rec in (256, 132):
        aff = next((r for r in results if r["ok"] and r["form"] == "affine"
                    and r["rec"] == rec and r["bkv_p"] == 2), None)
        ind = next((r for r in results if r["ok"] and r["form"] == "indirect"
                    and r["rec"] == rec and r["bkv_p"] == 2), None)
        if aff and ind:
            p(f"INDIRECTION COST rec{rec}",
              f"indirect/affine = {ind['best_s']/aff['best_s']:.3f} "
              f"({ind['best_s']*1e6:.2f}us vs {aff['best_s']*1e6:.2f}us)")

    p("---", "interpretation")
    if args.suite == "deep":
        # Read the gate at 9 (ctx 8704) and at 8 (ctx 8192) separately: they
        # are two different production points, not a headline and a check.
        for gate_bkvp in (9, 8):
            interpret([q for q in pairs if q], results, floor,
                      gate_bkvp=gate_bkvp)
    else:
        interpret([q for q in pairs if q], results, floor)

    emit_json(f"R1_kpack_{args.suite}_grid{args.num_grid}", {
        "geometry": vars(args),
        "dispatch_floor": floor,
        "records": {str(k): {"shape": RECORDS[k], "bytes": record_bytes(k)}
                    for k in RECORDS},
        "layout": layout,
        "arms": results,
        "pairs": [q for q in pairs if q],
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
