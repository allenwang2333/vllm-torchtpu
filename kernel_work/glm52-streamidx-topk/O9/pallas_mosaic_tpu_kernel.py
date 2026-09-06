"""O9's SparseCore top-k kernel: the score-slice fetch, and what hides it.

The only thing O9 changes is WHEN the score slice arrives. The incumbent issues
one blocking `slice_len`-word copy at the top of a wave and starts no compute
until it lands; everything downstream (convert, histogram, three refinement
levels, emit) is byte-identical here. Four knobs, all off by default:

  `dma_repeat`  attribution, semantics-preserving: copy the same slice R times
                before converting. The slope of wave time in R is the serial
                cost of one slice fetch, measured rather than modelled, and the
                output is unchanged so a correctness gate still applies.
  `dma_denom`   attribution ABLATION, NOT semantics-preserving: fetch only
                `slice_len // denom` words and walk the whole slice anyway. Its
                output is garbage past the fetched prefix; it exists only to
                bracket `dma_repeat` from below and must never be a candidate.
  `chunks`/`depth`  the change: split the one copy into `chunks` static
                sub-copies over disjoint sub-ranges of the SAME buffer, keep
                `depth` in flight, and convert chunk c while chunk c+depth is
                landing. Costs zero extra VMEM -- that is the whole reason this
                spelling was chosen over a second resident buffer.
  `second_buffer`  allocation probe: claim a second `slice_len` scratch. The
                wave-level double buffer the design sheet names needs one, and
                at (B) `slice_len` is 256 KiB against a 512 KiB vector subcore,
                so the question is whether it can exist at all before it is
                worth scheduling around. Nothing reads it.

Compromise this file makes and the reason: the chunk loop is a PYTHON loop over
static chunk indices, not a `fori_loop`. Semaphore slots have to be picked
statically (`sems.at[c % depth]`) and the copy extents are static, so a traced
loop would buy nothing and would forbid the start/wait skew. `chunks` is
therefore code size, and the config caps it.

Dimension glossary
  B  score rows                     N  padded score columns
  K  selected indices per row       S  resident slice words (`slice_len`)
  L  lanes (16 on tpu7x)            W  waves over the row axis
  c  chunk index      i  vector index      w  wave index
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

NUM_BUCKETS = 256
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
KEY_NEG_INF = int(
    np.int32(np.array(-np.inf, np.float32).view(np.int32))
    ^ np.int32(0x7FFFFFFF))

# Semaphore slots. 0-2 are the incumbent's; the chunk and repeat copies take
# their own so no two live copies ever share one, and so the compiler cannot
# read two `dma_repeat` copies as the same operation.
SEM_LEN, SEM_SLICE, SEM_OUT = 0, 1, 2
SEM_EXTRA = 4


def num_semaphores(chunks, depth, dma_repeat):
    """Semaphore array extent for one (chunks, depth, dma_repeat) point."""
    return SEM_EXTRA + max(1, depth if chunks > 1 else 0, dma_repeat - 1)


def topk_kernel(
    # Inputs.
    scores_hbm,  # [B * N] i32   (f32 score bits)
    lengths_hbm,  # [B_pad] i32
    # Outputs.
    out_hbm,  # [B * K] i32
    # Scratch.
    resident_vmem,  # [S] i32          the score slice, monotone-keyed in place
    priv_vmem,  # [L * 256 + L] i32    lane-private histograms
    glob_vmem,  # [256 + L] i32        folded histogram, then its prefix sum
    rowbuf_vmem,  # [K + L] i32        winner indices for one row
    len_vmem,  # [B_pad] i32
    spare_vmem,  # [S or L] i32        second resident buffer, probe only
    sems,  # DMA[num_semaphores(...)]
    *,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    lanes: int,
    chunks: int,
    depth: int,
    dma_repeat: int,
    dma_denom: int,
):
    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(lanes, dtype=jnp.int32)
    b_pad = len_vmem.shape[0]

    cp = pltpu.make_async_copy(lengths_hbm.at[pl.ds(0, b_pad)], len_vmem,
                               sems.at[SEM_LEN])
    cp.start()
    cp.wait()

    # Touch the probe buffer so the allocation cannot be folded away before the
    # compiler has had to find room for it. One vector, once, outside the loop.
    spare_vmem[pl.ds(0, lanes)] = jnp.zeros((lanes, ), jnp.int32)

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, lanes)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def convert(lo_vec, count):
        """Monotone-key `count` vectors of `resident_vmem`, starting at `lo_vec`."""

        def cvt(i):
            off = (lo_vec + i) * lanes
            resident_vmem[pl.ds(off, lanes)] = monotone_key(
                resident_vmem[pl.ds(off, lanes)])

        plsc.parallel_loop(0, count, unroll=4)(cvt)

    def fetch_and_convert(s, walk_chunks):
        """The one thing O9 varies. Same bytes, same conversions, new schedule."""
        if chunks > 1:
            # Static sub-copies over disjoint sub-ranges of the one buffer.
            cw = slice_len // chunks  # words per chunk
            vpc = cw // lanes  # vectors per chunk
            copies = [
                pltpu.make_async_copy(
                    scores_hbm.at[pl.ds(s + c * cw, cw)],
                    resident_vmem.at[pl.ds(c * cw, cw)],
                    sems.at[SEM_EXTRA + (c % depth)],
                ) for c in range(chunks)
            ]
            for c in range(min(depth, chunks)):
                copies[c].start()
            for c in range(chunks):
                copies[c].wait()
                if c + depth < chunks:
                    # Issued before this chunk's compute, waited before the
                    # next one's: the whole point of the variant.
                    copies[c + depth].start()
                convert(c * vpc, jnp.clip(walk_chunks - c * vpc, 0, vpc))
            return

        fetch_words = slice_len // dma_denom
        dma(scores_hbm.at[pl.ds(s, fetch_words)],
            resident_vmem.at[pl.ds(0, fetch_words)], SEM_SLICE)
        for r in range(1, dma_repeat):
            dma(scores_hbm.at[pl.ds(s, fetch_words)],
                resident_vmem.at[pl.ds(0, fetch_words)], SEM_EXTRA + r - 1)
        convert(0, walk_chunks)

    def zero_priv(bound):

        def body(i):
            priv_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes, ),
                                                           jnp.int32)

        plsc.parallel_loop(0, bound, unroll=4)(body)

    def fold_priv(bound):

        def body(i):
            off = i * lanes
            acc = jnp.zeros((lanes, ), jnp.int32)
            for m in range(lanes):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, lanes)]
            glob_vmem[pl.ds(off, lanes)] = acc

        plsc.parallel_loop(0, bound, unroll=2)(body)

    def scan_glob(bound):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * lanes, lanes)]) + carry
            glob_vmem[pl.ds(i * lanes, lanes)] = c
            return c[lanes - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * lanes, lanes)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def wave_body(w, _):
        row = w * (num_sc_cores * num_subcores) + flat
        active = row < b
        row_c = jnp.minimum(row, b - 1)
        eff_len = jnp.minimum(vec_at(len_vmem, row_c), n)
        eff_len = jnp.where(active, eff_len, 0)

        s = row_c * n
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = jnp.right_shift(i_hi + (lanes - 1), 4)

        @pl.when(i_hi > 0)
        def _():
            fetch_and_convert(s, walk_chunks)

        def load_keys(i):
            key = resident_vmem[pl.ds(i * lanes, lanes)]
            j = i * lanes + lane_iota
            valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))
            return key, valid, j

        @pl.when(i_hi > 0)
        def _():
            zero_priv(NUM_BUCKETS)

            def hist0_body(i):
                key, valid, _ = load_keys(i)
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                    NUM_BUCKETS // 2)
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((lanes, ), jnp.int32),
                                       mask=valid)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            fold_priv(NUM_BUCKETS // lanes)

        @pl.when(i_hi == 0)
        def _():

            def zg(i):
                glob_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes, ),
                                                               jnp.int32)

            plsc.parallel_loop(0, NUM_BUCKETS // lanes, unroll=4)(zg)

        total = scan_glob(NUM_BUCKETS // lanes)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // lanes, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0,
                           vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(NUM_BUCKETS // 2)
        shift = jnp.int32(L0_SHIFT)
        done = keep_all | (quota == cnt)

        for lvl_shift, lvl_mask in REFINEMENT_LEVELS:
            zero_priv(jnp.where(done, 0, NUM_BUCKETS))
            active_walk = jnp.where(done, 0, walk_chunks)

            def ref_body(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                prefix=prefix,
                shift=shift,
            ):
                key, valid, _ = load_keys(i)
                match = valid & (jnp.right_shift(key, shift) == prefix)
                bucket = jnp.bitwise_and(jnp.right_shift(key, lvl_shift),
                                         jnp.int32(lvl_mask))
                addr = lane_iota * NUM_BUCKETS + bucket
                plsc.addupdate_scatter(priv_vmem, (addr, ),
                                       jnp.ones((lanes, ), jnp.int32),
                                       mask=match)

            plsc.parallel_loop(0, active_walk, unroll=4)(ref_body)
            fold_bound = jnp.where(done, 0, NUM_BUCKETS // lanes)
            fold_priv(fold_bound)

            sub_total = scan_glob(fold_bound)
            b2 = find_bucket(fold_bound, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0,
                                vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)
            c_above2 = sub_total - cum_at2
            cnt2 = cum_at2 - cum_lo2
            quota2 = quota - c_above2

            width = shift - lvl_shift
            new_prefix = jnp.left_shift(prefix, width) | b2
            new_done = done | (quota2 == cnt2) | (quota2 == 0)

            prefix = jnp.where(done, prefix, new_prefix)
            shift = jnp.where(done, shift, jnp.int32(lvl_shift))
            c_hi = jnp.where(done, c_hi, c_hi + c_above2)
            quota = jnp.where(done, quota, quota2)
            cnt = jnp.where(done, cnt, cnt2)
            done = new_done

        my_take = jnp.where(keep_all, 0, quota)
        out_gate = active if write_empty else (active & (i_hi > 0))

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * lanes, lanes)] = jnp.full((lanes, ), -1,
                                                            jnp.int32)

        fill_bound = jnp.where(out_gate & (total < k), k // lanes, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        def emit_body(i, carry):
            woff, rank = carry
            key, valid, j = load_keys(i)
            hi = jnp.right_shift(key, shift)
            strict = valid & (hi > prefix)
            tie = valid & (hi == prefix)
            tie_rank = plsc.cumsum(tie.astype(jnp.int32)) + rank
            take = tie & (tie_rank <= my_take)
            sel = jnp.where(keep_all, valid, strict | take)
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)], j,
                                  mask=sel)
            woff = woff + plsc.all_reduce_population_count(sel)[0]
            rank = tie_rank[lanes - 1]
            return woff, rank

        plsc.parallel_loop(
            0,
            jnp.where(active, walk_chunks, 0),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_body)

        @pl.when(out_gate)
        def _():
            dma(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(row_c * k, k)],
                SEM_OUT)

        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)


def build_call(b, n, k, slice_len, num_waves, write_empty, lanes, cfg):
    """The `pl.kernel` for one shape and one config. Picks no config itself."""
    sc = pltpu.get_tpu_info().sparse_core
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    b_pad = ((b + lanes - 1) // lanes) * lanes
    spare_words = slice_len if cfg.second_buffer else lanes
    return pl.kernel(
        functools.partial(
            topk_kernel,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
            lanes=lanes,
            chunks=cfg.chunks,
            depth=cfg.depth,
            dma_repeat=cfg.dma_repeat,
            dma_denom=cfg.dma_denom,
        ),
        out_type=jax.ShapeDtypeStruct((b * k, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((lanes * NUM_BUCKETS + lanes, ), jnp.int32),
            pltpu.VMEM((NUM_BUCKETS + lanes, ), jnp.int32),
            pltpu.VMEM((k + lanes, ), jnp.int32),
            pltpu.VMEM((b_pad, ), jnp.int32),
            pltpu.VMEM((spare_words, ), jnp.int32),
            pltpu.SemaphoreType.DMA(
                (num_semaphores(cfg.chunks, cfg.depth, cfg.dma_repeat), )),
        ),
        mesh=mesh,
        name=f"o9_topk_b{b}_n{n}_k{k}_c{cfg.chunks}d{cfg.depth}"
        f"r{cfg.dma_repeat}q{cfg.dma_denom}",
    )
