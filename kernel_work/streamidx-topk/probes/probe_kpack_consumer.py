"""R1b -- what Option A's changed *consumer* costs, measured rather than assumed.

R1 (probes/R1.md) priced the fetch side of Option A: halving the K record from
262,144 B to 132,096 B per page saves 21.0 % of DMA time at bkv_p=2 and 42.7 %
at bkv_p=6, all of it recovered per-descriptor-issue cost.  R1 also named the
largest untested item in A's case, and this file is that item:

    measurement_and_design.md:714-716 asserts "The scale byte is unpacked on the
    VPU exactly as it is today -- the fetch changes, the unpack does not."

That is false as stated, and the design's own §3.1 layout table is why.  Today
the scale byte sits at lane `head_dim` of a 256-lane record, so `load_bkv`
recovers it with

    scale = bitcast(flat_bkv[:, 128:129].T, e8m0).astype(bf16)   # (1, bkv_sz)

-- one lane column, transposed out of `bkv_sz` sublanes.  Under Option A the
record is 128 lanes wide with no room for a scale lane, so the scales live in
8 extra sublane rows per page (`rows 256..257` x `kv_packing` 4) and must be
collapsed *sublanes-into-lanes* instead:

    scale = bitcast(b4[:, 256:258, :, :].reshape(1, bkv_sz), e8m0).astype(bf16)

Both are relayouts to the same (1, bkv_sz) bf16 operand; neither is free; which
is cheaper is an empirical question about Mosaic's relayout lowering and cannot
be settled by reading the design.  It matters more than it looks: a sibling
probe established that prefill stage A is 75.4 % issue/VPU with no roofline
headroom, so A is a decode-only play, and any VPU cost the new unpack adds
lands on the axis that is already the binding constraint at prefill and is a
real term at decode.

WHAT THIS MEASURES

Three modes, so the fetch saving and the unpack cost can be separated instead
of being read off one composite number:

  fetch    the DMA schedule alone, no compute.  (Same quantity R1 measured;
           repeated here so the decomposition is within one process.)
  compute  the bkv buffer is DMA'd ONCE at grid step 0 and every step runs
           `load_bkv` + `compute_scores` on the resident buffer.  No per-step
           DMA at all, so this is the unpack-and-arithmetic cost in isolation.
  full     both -- the real stage-A inner loop.

Three consumers:

  inc          the incumbent `load_bkv`, transcribed from
               streamindex_topk.py:257-272, on the 256-lane record.
  a_reshape    Option A's layout, scales recovered by the direct
               (bkv_p, 2, 4, 128) -> (1, bkv_sz) reshape.
  a_concat     Option A's layout, scales recovered by concatenating the
               bkv_p*8 individual (1, 128) scale rows along lanes.  A fallback
               formulation: if `a_reshape` lowers badly, this bounds how much
               of that is Mosaic's reshape rather than Option A's layout.

`compute_scores` after the unpack is a verbatim transcription of
streamindex_topk.py:304-354 and is IDENTICAL across all three consumers -- the
einsum, the relu, the head reduction, the scale multiply and both masks.  The
only difference between arms is how `bkvs` and `bkv_scales` are produced.  Both
layouts feed the same shapes to the same arithmetic, so any time difference is
the unpack.

WHAT IT DOES NOT MEASURE

Option A's *writer*.  §3.1 area 5 prices it as "one extra scalar scatter per
step in `_streamidx_topk_jax`, on TensorCore, moving 1 KB".  That is on the
caller side, outside this kernel, and is not exercised here.

Nothing in `src/` is modified.  This is a standalone replication, in the same
spirit as `_stage_a.py`, and `source_md5` below records the md5 of the module
the transcription was taken from so drift shows up in the job log.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry. Every one of these is the nightly decode value, cited to the file
# it comes from via instruments/bench_streamidx_topk.py:80-121.
# ---------------------------------------------------------------------------
H_IDX = 32            # indexer query heads
D_IDX = 128           # indexer head dim
PAGE_SIZE = 1024      # PallasMLAttentionBackend.get_page_size
KV_PACKING = 4        # get_dtype_packing(uint8)
PSP = PAGE_SIZE // KV_PACKING          # 256 groups per page
LKV_DIM = 256         # 129 useful bytes rounded up to a 128 multiple
SEQ_BATCH_SIZE = 4    # mla_attention_op decode path
BQ_SZ = 1             # decode: min(num_queries_per_block[0]=1, static_q_len=1)
CTX = 8704            # the ctx bucket the extended Candidate 0 sweep argmins at

VMEM_LIMIT_BYTES = 100 * 1024 * 1024   # streamindex_topk.py:28
HBM_SPEC_BPS = 3.70e12

# (rows per page on the group axis, lanes per row). Bytes per page DMA is
# rows * KV_PACKING * lanes.
#   inc: 256 * 4 * 256 = 262144   -- today
#   a  : 258 * 4 * 128 = 132096   -- measurement_and_design.md:678-684
LAYOUTS = {
    "inc": (256, 256),
    "a": (258, 128),
}
CONSUMER_LAYOUT = {"inc": "inc", "a_reshape": "a", "a_concat": "a"}


def page_bytes(layout: str) -> int:
    rows, lanes = LAYOUTS[layout]
    return rows * KV_PACKING * lanes


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def source_md5() -> str:
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sx
    return hashlib.md5(pathlib.Path(sx.__file__).read_bytes()).hexdigest()


_HBM_SPACE = None
for _name in ("HBM", "ANY"):
    if hasattr(pltpu, _name):
        _HBM_SPACE = getattr(pltpu, _name)
        break


# ---------------------------------------------------------------------------
# The two unpacks.
# ---------------------------------------------------------------------------


def _unpack_inc(bkv, bkv_p):
    """streamindex_topk.py:261-271, verbatim, for one batch element.

    `bkv` is (bkv_p*256, 4, 256) uint8.
    """
    bkv_sz = bkv_p * PAGE_SIZE
    flat = bkv.reshape(-1, bkv.shape[-1])              # (bkv_sz, 256)
    fp8 = pltpu.bitcast(flat[:, :D_IDX], jnp.float8_e4m3fn)
    scale = pltpu.bitcast(flat[:, D_IDX:D_IDX + 1].T,
                          jnp.float8_e8m0fnu).astype(jnp.bfloat16)
    return fp8.reshape(bkv_sz, D_IDX), scale


def _unpack_a(bkv, bkv_p, *, how):
    """Option A's layout: (bkv_p*258, 4, 128) uint8.

    Rows 0..255 of each page's group axis hold 1024 positions x 128 payload
    bytes; rows 256..257 hold the 1024 scale bytes, 128 per lane row.  The
    design's index arithmetic is `256 + (slot//128)//4, (slot//128)%4,
    slot%128` (measurement_and_design.md:705), so a row-major flatten of the
    (2, 4, 128) scale block yields the scales in slot order and the payload's
    row-major flatten yields the payload in the same slot order.  The two
    orders therefore agree with no permutation -- that is the property that
    makes the layout usable at all, and it is asserted here so a future edit
    to the layout cannot silently break it.
    """
    bkv_sz = bkv_p * PAGE_SIZE
    b4 = bkv.reshape(bkv_p, PSP + 2, KV_PACKING, D_IDX)

    payload = b4[:, :PSP].reshape(bkv_sz, D_IDX)
    fp8 = pltpu.bitcast(payload, jnp.float8_e4m3fn)

    srows = b4[:, PSP:PSP + 2]                          # (bkv_p, 2, 4, 128)
    if how == "reshape":
        su8 = srows.reshape(1, bkv_sz)
    elif how == "concat":
        parts = [srows[q, r, k].reshape(1, D_IDX)
                 for q in range(bkv_p)
                 for r in range(2)
                 for k in range(KV_PACKING)]
        su8 = jnp.concatenate(parts, axis=1)            # (1, bkv_sz)
    else:
        raise ValueError(how)
    scale = pltpu.bitcast(su8, jnp.float8_e8m0fnu).astype(jnp.bfloat16)
    return fp8, scale


def check_slot_order(bkv_p=2):
    """Host-side proof that A's flatten order matches the payload's.

    Cheap, exact, and it is the one correctness property of Option A's layout
    that this probe can establish without running the real kernel: for every
    position `s` in a page, the scale byte the flatten puts at lane `s` is the
    one the design's index arithmetic stores for slot `s`.
    """
    slots = np.arange(PAGE_SIZE)
    block = np.zeros((2, KV_PACKING, D_IDX), dtype=np.int64)
    block[(slots // 128) // KV_PACKING,
          (slots // 128) % KV_PACKING,
          slots % 128] = slots
    assert np.array_equal(block.reshape(-1), slots), (
        "row-major flatten of the (2,4,128) scale block is NOT slot order")
    # And the payload's own flatten: (256, 4, 128) rows -> slot = g*4 + k.
    pay = np.zeros((PSP, KV_PACKING), dtype=np.int64)
    pay[slots // KV_PACKING, slots % KV_PACKING] = slots
    assert np.array_equal(pay.reshape(-1), slots)
    return True


# ---------------------------------------------------------------------------
# compute_scores: streamindex_topk.py:304-354, transcribed. Identical for all
# three consumers -- only its `bkv_vec` / `scale_val_vec` inputs differ.
# ---------------------------------------------------------------------------


def _compute_scores(bq_vec, bkv_vec, scale_val_vec, bq_weights_vec,
                    bq_pos_compressed_vec, bkv_idx, kv_lens, bkv_sz):
    bq_stacked = jnp.stack(bq_vec)
    bkv_stacked = jnp.stack(bkv_vec)
    scale_val_stacked = jnp.stack(scale_val_vec)
    bq_weights_stacked = jnp.stack(bq_weights_vec)
    bq_pos_compressed_stacked = jnp.stack(bq_pos_compressed_vec)

    s = jnp.einsum("bnd,bmd->bnm", bq_stacked, bkv_stacked,
                   preferred_element_type=jnp.float32)
    s = s.reshape(SEQ_BATCH_SIZE, BQ_SZ, H_IDX, bkv_sz)
    s = jnp.maximum(s, 0.0)
    s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
    s_summed = s.sum(axis=2)
    s_summed = s_summed * scale_val_stacked

    k_span = bkv_idx * bkv_sz + jnp.arange(bkv_sz, dtype=jnp.int32)
    kv_lens_stacked = jnp.stack(kv_lens)
    valid_mask = k_span[None, None, :] < kv_lens_stacked[:, None, None]
    causal_mask = (k_span[None, None, :]
                   <= bq_pos_compressed_stacked[:, :, None])
    mask = jnp.logical_and(valid_mask, causal_mask)
    return jnp.where(mask, s_summed, -jnp.inf)


# ---------------------------------------------------------------------------
# The kernel.
# ---------------------------------------------------------------------------


def _make_kernel(*, consumer, mode, bkv_p, num_bkv, num_grid, pages_per_seq):
    layout = CONSUMER_LAYOUT[consumer]
    rows, lanes = LAYOUTS[layout]
    bkv_sz = bkv_p * PAGE_SIZE
    do_dma = mode in ("fetch", "full")
    do_compute = mode in ("compute", "full")

    def kernel(page_indices_ref, meta_ref, cache_kv_hbm_ref, q_hbm_ref,
               w_hbm_ref, o_ref, bkv_x2_ref, bq_ref, bw_ref, acc_ref, sems,
               qsems):
        total_num_pages = cache_kv_hbm_ref.shape[0]
        num_page_indices = page_indices_ref.shape[0]
        src = cache_kv_hbm_ref.reshape(total_num_pages * rows, KV_PACKING,
                                       lanes)
        max_hbm_rows = src.shape[0]

        i = pl.program_id(0)
        seq_idx = i * SEQ_BATCH_SIZE

        def start(seq_idx, bkv_idx, sem_idx):
            # `_fetch_bkv(..., wait=False)`, streamindex_topk.py:160-198: one
            # descriptor per (sequence, page), offset read out of the page
            # table and clamped with `min()`, static extent.
            for b in range(SEQ_BATCH_SIZE):
                sem = sems.at[sem_idx, b]
                dst = bkv_x2_ref.at[sem_idx, b]
                base = (seq_idx + b) * pages_per_seq + bkv_idx * bkv_p
                for j in range(bkv_p):
                    page_idx = jnp.minimum(base + j, num_page_indices - 1)
                    off = jnp.minimum(page_indices_ref[page_idx] * rows,
                                      jnp.maximum(0, max_hbm_rows - rows))
                    pltpu.make_async_copy(src.at[pl.ds(off, rows)],
                                          dst.at[pl.ds(j * rows, rows)],
                                          sem).start()

        def wait(sem_idx):
            for b in range(SEQ_BATCH_SIZE):
                d = bkv_x2_ref.at[sem_idx, b].at[pl.ds(0, bkv_p * rows)]
                pltpu.make_async_copy(d, d, sems.at[sem_idx, b]).wait()

        # q and the per-head weights are fetched once, at step 0. They are
        # 16 KB and 512 B; the real kernel double-buffers them across bq
        # blocks, but decode has exactly one bq block per sequence batch so
        # there is nothing to pipeline and their cost is not what is under
        # study here.
        @pl.when(i == 0)
        def _():
            pltpu.make_async_copy(q_hbm_ref, bq_ref, qsems.at[0]).start()
            pltpu.make_async_copy(w_hbm_ref, bw_ref, qsems.at[1]).start()
            pltpu.make_async_copy(q_hbm_ref, bq_ref, qsems.at[0]).wait()
            pltpu.make_async_copy(w_hbm_ref, bw_ref, qsems.at[1]).wait()
            for x in range(acc_ref.shape[0]):
                acc_ref[x] = jnp.zeros(acc_ref.shape[1:], jnp.float32)
            if do_dma:
                start(seq_idx, 0, 0)
            else:
                # `compute` mode: fill buffer 0 once and never touch the DMA
                # engine again, so every later grid step measures unpack plus
                # arithmetic with nothing to overlap it.
                start(0, 0, 0)
                wait(0)

        kv_lens = [meta_ref[b] for b in range(SEQ_BATCH_SIZE)]
        bq_pos = [jnp.full((BQ_SZ, ), meta_ref[b] - 1, jnp.int32)
                  for b in range(SEQ_BATCH_SIZE)]

        def compute(sem_idx, bkv_idx):
            if not do_dma:
                # LICM guard, and it is load-bearing. In `compute` mode nothing
                # writes the bkv buffer inside the grid loop, every step reads
                # the same bytes, and a compiler is entitled to hoist the whole
                # unpack out of the loop -- which would report Option A's
                # consumer as free for the same reason an elided DMA reports as
                # infinitely fast. One store per step, of one (4, lanes) row
                # keyed on the step index, makes the buffer loop-variant. It is
                # ~0.5-1 KB against a multi-MB unpack. The `--linearity-check`
                # arms below verify the guard held.
                bkv_x2_ref[0, 0, 0] = jnp.full(
                    (KV_PACKING, lanes),
                    jnp.remainder(i, 251).astype(jnp.uint8), jnp.uint8)
            bq_vec, bkvs, scales, w_vec = [], [], [], []
            for b in range(SEQ_BATCH_SIZE):
                blk = bkv_x2_ref.at[sem_idx, b, :bkv_p * rows][...]
                if consumer == "inc":
                    f, sc = _unpack_inc(blk, bkv_p)
                else:
                    f, sc = _unpack_a(
                        blk, bkv_p,
                        how="reshape" if consumer == "a_reshape" else "concat")
                bkvs.append(f)
                scales.append(sc)
                bq_vec.append(
                    pltpu.bitcast(bq_ref[b], jnp.float8_e4m3fn))
                w_vec.append(bw_ref[b])
            s = _compute_scores(bq_vec, bkvs, scales, w_vec, bq_pos, bkv_idx,
                                kv_lens, bkv_sz)
            # Consume the result so nothing above can be dead-coded, and make
            # the consumption step-dependent (`bkv_idx` already enters through
            # `k_span`) so it cannot be hoisted out of the grid loop either.
            for b in range(SEQ_BATCH_SIZE):
                acc_ref[b] += s[b]

        def finish():
            # `acc_ref` is scratch. If nothing reads it, the whole unpack and
            # einsum above are dead and a compiler may delete them -- the
            # compute-side equivalent of an elided DMA. Reducing it into the
            # kernel's only real output closes that, and the reduction doubles
            # as a correctness check: the two layouts are fed byte-equivalent
            # source data (see `make_source`), so a matching checksum means
            # Option A's unpack recovered the same payload and the same scale
            # for every slot. §3.1 claims A is "bit-identical"; nothing until
            # now has run the code that would make that true or false.
            #
            # The reduction stops at a (128,) lane vector rather than a
            # scalar: a full vector-to-SMEM-scalar reduction is not reliably
            # lowerable in Mosaic, and the host can finish the sum. The
            # partial is still order-deterministic and still identical between
            # the arms, which is all the check needs.
            a = acc_ref[...]
            fin = jnp.isfinite(a)
            o_ref[0] = jnp.sum(jnp.where(fin, a, 0.0).reshape(-1, 128), axis=0)
            o_ref[1] = jnp.sum(fin.astype(jnp.float32).reshape(-1, 128),
                               axis=0)

        base_blk = i * num_bkv
        for blk_i in range(num_bkv):
            # Runtime parity, keyed on the global block index -- a compile-time
            # `blk_i % 2` deadlocks whenever num_bkv is odd, which nightly
            # decode's num_bkv (5 at bkv_p=2) is. Measured: j-248bfcd5 hung 57
            # minutes. See bench_kpack.py's note.
            cur = jnp.remainder(base_blk + blk_i, 2)
            nxt = 1 - cur
            if do_dma:
                if blk_i + 1 < num_bkv:
                    start(seq_idx, blk_i + 1, nxt)
                else:
                    @pl.when(i < num_grid - 1)
                    def _(seq_idx=seq_idx, nxt=nxt):
                        start(seq_idx + SEQ_BATCH_SIZE, 0, nxt)
                wait(cur)
                if do_compute:
                    compute(cur, blk_i)
            else:
                if do_compute:
                    compute(0, blk_i)

        if do_compute:
            @pl.when(i == num_grid - 1)
            def _():
                finish()
        else:
            o_ref[0] = jnp.full((128, ), i.astype(jnp.float32))
            o_ref[1] = jnp.zeros((128, ), jnp.float32)

    return kernel


def build_arm(*, consumer, mode, bkv_p, num_bkv, num_grid, pages_per_seq,
              num_pages):
    layout = CONSUMER_LAYOUT[consumer]
    rows, lanes = LAYOUTS[layout]
    bkv_sz = bkv_p * PAGE_SIZE
    num_page_indices = num_grid * SEQ_BATCH_SIZE * pages_per_seq

    src_shape = (num_pages, rows, KV_PACKING, lanes)
    q_shape = (SEQ_BATCH_SIZE, BQ_SZ * H_IDX, D_IDX)
    w_shape = (SEQ_BATCH_SIZE, BQ_SZ, H_IDX)

    kernel = _make_kernel(consumer=consumer, mode=mode, bkv_p=bkv_p,
                          num_bkv=num_bkv, num_grid=num_grid,
                          pages_per_seq=pages_per_seq)

    f = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=_HBM_SPACE)] * 3,
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            grid=(num_grid, ),
            scratch_shapes=[
                pltpu.VMEM((2, SEQ_BATCH_SIZE, bkv_p * rows, KV_PACKING,
                            lanes), jnp.uint8),
                pltpu.VMEM(q_shape, jnp.uint8),
                pltpu.VMEM(w_shape, jnp.float32),
                pltpu.VMEM((SEQ_BATCH_SIZE, BQ_SZ, bkv_sz), jnp.float32),
                pltpu.SemaphoreType.DMA((2, SEQ_BATCH_SIZE)),
                pltpu.SemaphoreType.DMA((2, )),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((2, 128), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )
    return jax.jit(f), src_shape, q_shape, w_shape, num_page_indices


def make_source(layout: str, num_pages: int, seed: int = 0):
    """The K cache, built so the two layouts hold **byte-equivalent** content.

    One payload array and one scale array are drawn from a fixed seed and then
    placed into whichever layout is asked for:

      inc  u8[P, 256, 4, 256] -- lanes 0..127 payload, lane 128 the e8m0 scale,
                                 lanes 129..255 the dead pad (this is today's
                                 cache; `bench_streamidx_topk._make_cache`
                                 builds the same thing).
      a    u8[P, 258, 4, 128] -- rows 0..255 payload; rows 256..257 the scales,
                                 reshaped (2, 4, 128) so index (r, k, l) holds
                                 the scale for slot (r*4+k)*128 + l, which is
                                 measurement_and_design.md:705's arithmetic.

    Because `scale` is indexed by slot and the payload's own (256, 4) flatten
    is also slot-ordered, a plain `reshape(P, 2, 4, 128)` is the whole of the
    placement -- that equivalence is the layout's load-bearing property and
    building both arrays from the same two source tensors is what lets the
    kernel's output checksum test it.

    The fp8 bytes have their exponent clamped into [5, 8]: finite, NaN-free
    (e4m3fn NaN needs exp 15) and spread over +-[0.25, 3.75].
    """
    k1, k2 = jax.random.split(jax.random.key(seed))
    raw = jax.random.bits(k1, (num_pages, PAGE_SIZE, D_IDX), jnp.uint8)
    exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
    payload = (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))
    del raw, exp
    # e8m0: 127 is 2**0, so this spans 2**-3 .. 2**3.
    scale = jax.random.randint(k2, (num_pages, PAGE_SIZE), 124, 131,
                               jnp.int32).astype(jnp.uint8)

    pay4 = payload.reshape(num_pages, PSP, KV_PACKING, D_IDX)
    del payload
    if layout == "a":
        srows = scale.reshape(num_pages, 2, KV_PACKING, D_IDX)
        return jnp.concatenate([pay4, srows], axis=1)
    sc4 = scale.reshape(num_pages, PSP, KV_PACKING, 1)
    pad = jnp.zeros((num_pages, PSP, KV_PACKING, LKV_DIM - D_IDX - 1),
                    jnp.uint8)
    return jnp.concatenate([pay4, sc4, pad], axis=-1)


def make_page_table(n, num_pages, seed=0):
    rng = np.random.default_rng(seed)
    if n <= num_pages:
        idx = rng.permutation(num_pages)[:n]
    else:
        idx = rng.integers(0, num_pages, size=n)
    return jnp.asarray(np.ascontiguousarray(idx, dtype=np.int32))


def vmem_estimate(consumer, bkv_p):
    rows, lanes = LAYOUTS[CONSUMER_LAYOUT[consumer]]
    buf = 2 * SEQ_BATCH_SIZE * bkv_p * rows * KV_PACKING * lanes
    acc = SEQ_BATCH_SIZE * BQ_SZ * bkv_p * PAGE_SIZE * 4
    return buf + acc


def run_arm(*, consumer, mode, bkv_p, num_bkv, num_grid, pages_per_seq,
            num_pages, trials):
    rows, lanes = LAYOUTS[CONSUMER_LAYOUT[consumer]]
    pb = page_bytes(CONSUMER_LAYOUT[consumer])
    n_dma = (num_grid * num_bkv * SEQ_BATCH_SIZE * bkv_p) if mode != "compute" \
        else (SEQ_BATCH_SIZE * bkv_p)
    n_blocks = num_grid * num_bkv
    label = f"{consumer}/{mode}/bkvp{bkv_p}"

    p(label, f"building (buf+acc VMEM ~"
      f"{vmem_estimate(consumer, bkv_p)/1e6:.2f}MB)")
    try:
        f, src_shape, q_shape, w_shape, npi = build_arm(
            consumer=consumer, mode=mode, bkv_p=bkv_p, num_bkv=num_bkv,
            num_grid=num_grid, pages_per_seq=pages_per_seq,
            num_pages=num_pages)
    except Exception as e:  # noqa: BLE001 -- a form that will not lower is a result
        p(label, "BUILD FAIL", repr(e)[:500])
        return {"label": label, "consumer": consumer, "mode": mode,
                "bkv_p": bkv_p, "ok": False, "error": repr(e)[:2000]}

    keys = jax.random.split(jax.random.key(11), 2)
    page_indices = make_page_table(npi, num_pages)
    meta = jnp.full((SEQ_BATCH_SIZE, ), CTX, jnp.int32)

    def fp8_bytes(key, shape):
        raw = jax.random.bits(key, shape, dtype=jnp.uint8)
        exp = jnp.uint8(5) + ((raw >> jnp.uint8(3)) & jnp.uint8(3))
        return (raw & jnp.uint8(0x87)) | (exp << jnp.uint8(3))

    # Every arm gets the same q, the same weights, the same page table, and a
    # cache holding the same payload and scale per slot -- only the layout
    # differs. That is what makes the output checksums comparable.
    x = jax.block_until_ready(
        make_source(CONSUMER_LAYOUT[consumer], num_pages))
    assert tuple(x.shape) == tuple(src_shape), (x.shape, src_shape)
    q = jax.block_until_ready(fp8_bytes(keys[0], q_shape))
    w = jax.block_until_ready(
        jax.random.uniform(keys[1], w_shape, jnp.float32, 0.25, 1.75))

    p(label, f"compiling + timing (n_blocks={n_blocks} n_dma={n_dma} "
      f"page={pb}B)")
    try:
        t = time_arm(f, (page_indices, meta, x, q, w), warmup=3, trials=trials)
        out = np.asarray(jax.block_until_ready(f(page_indices, meta, x, q, w)))
    except Exception as e:  # noqa: BLE001
        p(label, "RUN FAIL", repr(e)[:500])
        del x, q, w
        return {"label": label, "consumer": consumer, "mode": mode,
                "bkv_p": bkv_p, "ok": False, "error": repr(e)[:2000]}
    del x, q, w

    best = t["best_s"]
    dma_bytes = n_dma * pb if mode != "compute" else 0
    res = {
        "label": label, "consumer": consumer, "mode": mode, "bkv_p": bkv_p,
        "num_bkv": num_bkv, "ok": True, "page_bytes": pb, "n_dma": n_dma,
        "n_blocks": n_blocks, "dma_bytes": dma_bytes,
        "best_s": best, "median_s": t["median_s"],
        "max_s": max(t["trials_s"]), "trials_s": t["trials_s"],
        "iters": t["iters"], "per_block_ns": best / n_blocks * 1e9,
        "bps": (dma_bytes / best) if dma_bytes else None,
        "vmem_est": vmem_estimate(consumer, bkv_p),
        "checksum": float(np.float64(out[0]).sum()),
        "n_finite": float(np.float64(out[1]).sum()),
    }
    p(label,
      f"best={best*1e6:9.2f}us median={t['median_s']*1e6:9.2f}us "
      f"max={max(t['trials_s'])*1e6:9.2f}us "
      f"spread={(max(t['trials_s'])/best-1)*100:5.1f}% "
      f"per_block={res['per_block_ns']:8.1f}ns "
      + (f"B/s={res['bps']:.3e}" if res["bps"] else "no DMA")
      + f" iters={t['iters']} checksum={res['checksum']:.9g} "
      f"n_finite={res['n_finite']:.0f}")
    if res["bps"] and res["bps"] > HBM_SPEC_BPS:
        p(label, "ELISION WARNING -- useful bandwidth above the 3.70e12 B/s "
          "spec; that arm's DMA did not happen. Discard it.")
    return res


def run_dispatch_floor(trials):
    def k(o_ref):
        o_ref[0] = 0

    f = jax.jit(pl.pallas_call(
        k, grid=(1, ), out_specs=pl.BlockSpec(memory_space=pltpu.SMEM),
        out_shape=jax.ShapeDtypeStruct((1, ), jnp.int32)))
    t = time_arm(f, (), warmup=3, trials=trials)
    p("dispatch floor", f"best={t['best_s']*1e6:.2f}us "
      f"median={t['median_s']*1e6:.2f}us -- empty pallas_call, host-loop "
      "timed the same way as every arm")
    return {"best_s": t["best_s"], "median_s": t["median_s"],
            "iters": t["iters"], "trials_s": t["trials_s"]}


def get(results, consumer, mode, bkv_p):
    return next((r for r in results if r["ok"] and r["consumer"] == consumer
                 and r["mode"] == mode and r["bkv_p"] == bkv_p), None)


def report(results, bkv_ps, a_consumer, floor):
    """The decomposition. All three numbers per bkv_p, then the net."""
    out = []
    for bkv_p in bkv_ps:
        row = {"bkv_p": bkv_p, "a_consumer": a_consumer}
        inc_c = get(results, "inc", "compute", bkv_p)
        a_c = get(results, a_consumer, "compute", bkv_p)
        inc_f = get(results, "inc", "fetch", bkv_p)
        a_f = get(results, a_consumer, "fetch", bkv_p)
        inc_x = get(results, "inc", "full", bkv_p)
        a_x = get(results, a_consumer, "full", bkv_p)

        if inc_c and a_c:
            d = (a_c["per_block_ns"] - inc_c["per_block_ns"])
            row["unpack_inc_ns_per_block"] = inc_c["per_block_ns"]
            row["unpack_a_ns_per_block"] = a_c["per_block_ns"]
            row["unpack_delta_ns_per_block"] = d
            p(f"UNPACK bkvp{bkv_p}",
              f"inc={inc_c['per_block_ns']:8.1f}ns/block  "
              f"{a_consumer}={a_c['per_block_ns']:8.1f}ns/block  "
              f"delta={d:+8.1f}ns/block ({d/inc_c['per_block_ns']*100:+6.1f}%) "
              "-- no DMA in either arm, so this is the unpack plus the "
              "identical arithmetic")
        if inc_f and a_f:
            d = (a_f["per_block_ns"] - inc_f["per_block_ns"])
            row["fetch_inc_ns_per_block"] = inc_f["per_block_ns"]
            row["fetch_a_ns_per_block"] = a_f["per_block_ns"]
            row["fetch_delta_ns_per_block"] = d
            p(f"FETCH  bkvp{bkv_p}",
              f"inc={inc_f['per_block_ns']:8.1f}ns/block  "
              f"a={a_f['per_block_ns']:8.1f}ns/block  "
              f"delta={d:+8.1f}ns/block "
              f"(saved {(1-a_f['best_s']/inc_f['best_s'])*100:5.1f}%, "
              f"charge ratio "
              f"{inc_f['best_s']/a_f['best_s']:.3f})")
        if inc_x and a_x:
            # Correctness, before any timing is quoted. Same payload, same
            # scales, same q, same weights, same page table -- only the layout
            # and the unpack differ, so the reduced score must agree exactly.
            same = (inc_x["checksum"] == a_x["checksum"]
                    and inc_x["n_finite"] == a_x["n_finite"])
            row["checksum_inc"] = inc_x["checksum"]
            row["checksum_a"] = a_x["checksum"]
            row["n_finite_inc"] = inc_x["n_finite"]
            row["n_finite_a"] = a_x["n_finite"]
            row["bit_identical"] = bool(same)
            rel = (abs(a_x["checksum"] - inc_x["checksum"])
                   / max(abs(inc_x["checksum"]), 1e-30))
            p(f"CORRECT bkvp{bkv_p}",
              ("BIT-IDENTICAL" if same else "MISMATCH") +
              f" -- inc checksum={inc_x['checksum']:.9g} "
              f"({inc_x['n_finite']:.0f} finite)  "
              f"{a_consumer} checksum={a_x['checksum']:.9g} "
              f"({a_x['n_finite']:.0f} finite)  rel diff={rel:.3e}"
              + ("" if same else "  <-- Option A's unpack does NOT recover "
                 "the same operands; the timing below is meaningless until "
                 "this is fixed."))
            d = (a_x["per_block_ns"] - inc_x["per_block_ns"])
            saved = 1 - a_x["best_s"] / inc_x["best_s"]
            row["full_inc_ns_per_block"] = inc_x["per_block_ns"]
            row["full_a_ns_per_block"] = a_x["per_block_ns"]
            row["full_delta_ns_per_block"] = d
            row["full_saved_frac"] = saved
            p(f"FULL   bkvp{bkv_p}",
              f"inc={inc_x['per_block_ns']:8.1f}ns/block  "
              f"a={a_x['per_block_ns']:8.1f}ns/block  "
              f"delta={d:+8.1f}ns/block  "
              f"NET={saved*100:+6.1f}% on the stage-A inner loop")
            # The nightly decode call runs `cdiv(ctx, bkv_p*PAGE_SIZE)` blocks
            # per sequence batch and 16 sequence batches (64 seqs / 4).
            nb = max(1, cdiv(CTX, bkv_p * PAGE_SIZE))
            blocks = 16 * nb
            row["nightly_blocks"] = blocks
            row["nightly_delta_us"] = d * blocks / 1e3
            p(f"STAKE  bkvp{bkv_p}",
              f"{d:+.1f}ns/block x {blocks} blocks "
              f"(16 sequence batches x num_bkv {nb} at ctx {CTX}) = "
              f"{d*blocks/1e3:+.2f}us on the decode call")
        out.append(row)
    if floor:
        fastest = min((r["best_s"] for r in results if r["ok"]), default=None)
        if fastest:
            frac = floor["best_s"] / fastest
            p("dispatch contamination",
              f"fastest arm {fastest*1e6:.2f}us vs {floor['best_s']*1e6:.2f}us "
              f"empty-launch floor ({frac*100:.1f}%)"
              + (" -- WARNING, above 10%, deltas are compressed"
                 if frac > 0.10 else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-grid", type=int, default=128,
                    help="grid steps. Nightly decode stage A is (16,) at "
                         "64 sequences / seq_batch_size 4; the default repeats "
                         "it 8x so every arm sits far above the dispatch floor.")
    ap.add_argument("--bkv-p", type=int, nargs="+", default=[9, 2],
                    help="9 = cdiv(ctx 8704, page 1024), the point the "
                         "extended Candidate 0 sweep argmins at; 2 = nightly.")
    ap.add_argument("--a-consumer", choices=("a_reshape", "a_concat", "both"),
                    default="both")
    ap.add_argument("--modes", nargs="+",
                    default=["compute", "fetch", "full"])
    ap.add_argument("--pages-per-seq", type=int, default=None)
    ap.add_argument("--num-pages", type=int, default=2048)
    ap.add_argument("--trials", type=int, default=9)
    ap.add_argument("--no-linearity-check", dest="linearity_check",
                    action="store_false", default=True,
                    help="skip the quarter-grid repeat that verifies the "
                         "`compute` arms were not hoisted out of the grid "
                         "loop. Only for a smoke pass.")
    args = ap.parse_args()

    bkv_ps = sorted(set(args.bkv_p), reverse=True)
    # num_bkv is the ctx-faithful block count, so each bkv_p covers the same
    # 8704 positions rather than an arbitrary number of pages.
    num_bkv = {b: max(1, cdiv(CTX, b * PAGE_SIZE)) for b in bkv_ps}
    if args.pages_per_seq is None:
        args.pages_per_seq = max(b * num_bkv[b] for b in bkv_ps)

    tpu_banner()
    p("hbm memory space", repr(_HBM_SPACE))
    if _HBM_SPACE is None:
        p("ABORT", "no usable unsliced memory-space name on this jax")
        return 2
    try:
        p("source md5", f"{source_md5()} (streamindex_topk.py, the module "
          "load_bkv and compute_scores were transcribed from)")
    except Exception as e:  # noqa: BLE001
        p("source md5", f"unavailable: {e!r}")

    check_slot_order()
    p("slot-order self-check",
      "OK -- Option A's (2,4,128) scale block flattens to slot order, and the "
      "(256,4) payload flattens to the same order, so no permutation is "
      "needed between them")

    p("geometry",
      f"H={H_IDX} D={D_IDX} page={PAGE_SIZE} kv_packing={KV_PACKING} "
      f"seq_batch={SEQ_BATCH_SIZE} bq_sz={BQ_SZ} ctx={CTX} "
      f"num_grid={args.num_grid} pages_per_seq={args.pages_per_seq} "
      f"num_pages={args.num_pages} trials={args.trials}")
    for b in bkv_ps:
        p(f"bkv_p {b}", f"num_bkv={num_bkv[b]} (ctx {CTX} / {b*PAGE_SIZE}) "
          f"bkv_sz={b*PAGE_SIZE} pages/seq={b*num_bkv[b]}")
    for name, (rows, lanes) in LAYOUTS.items():
        p(f"layout {name}",
          f"u8[pages, {rows}, {KV_PACKING}, {lanes}] -> "
          f"{rows*KV_PACKING*lanes} B per page DMA")

    consumers = ["inc"]
    if args.a_consumer == "both":
        consumers += ["a_reshape", "a_concat"]
    else:
        consumers += [args.a_consumer]

    floor = run_dispatch_floor(args.trials)

    results = []
    for bkv_p in bkv_ps:
        for mode in args.modes:
            for c in consumers:
                # `fetch` has no consumer, only a layout: a_reshape and
                # a_concat would run the identical kernel. Skip the duplicate.
                if mode == "fetch" and c == "a_concat":
                    continue
                results.append(run_arm(
                    consumer=c, mode=mode, bkv_p=bkv_p,
                    num_bkv=num_bkv[bkv_p], num_grid=args.num_grid,
                    pages_per_seq=args.pages_per_seq,
                    num_pages=args.num_pages, trials=args.trials))

    # Linearity check: the `compute` arms are the ones a compiler could hoist
    # out of the grid loop, and a hoisted arm reports a per-block cost that
    # falls as the grid grows. Re-run each at a quarter of the grid; if the
    # LICM guard held, per-block time is the same at both.
    lin = []
    if args.linearity_check and args.num_grid >= 8 and "compute" in args.modes:
        g4 = args.num_grid // 4
        b0 = bkv_ps[0]
        for c in consumers:
            r = run_arm(consumer=c, mode="compute", bkv_p=b0,
                        num_bkv=num_bkv[b0], num_grid=g4,
                        pages_per_seq=args.pages_per_seq,
                        num_pages=args.num_pages, trials=args.trials)
            full = get(results, c, "compute", b0)
            if r["ok"] and full:
                ratio = r["per_block_ns"] / full["per_block_ns"]
                r["linearity_grid"] = g4
                lin.append({"consumer": c, "bkv_p": b0, "grid_small": g4,
                            "grid_full": args.num_grid,
                            "per_block_ns_small": r["per_block_ns"],
                            "per_block_ns_full": full["per_block_ns"],
                            "ratio": ratio})
                p(f"LINEARITY {c}",
                  f"grid {g4}: {r['per_block_ns']:8.1f}ns/block   "
                  f"grid {args.num_grid}: {full['per_block_ns']:8.1f}ns/block   "
                  f"ratio={ratio:.3f}"
                  + ("" if 0.85 <= ratio <= 1.18 else
                     "  <-- WARNING: per-block cost is NOT grid-invariant. "
                     "Part of the unpack was hoisted out of the grid loop, or "
                     "the dispatch floor dominates the small grid. This arm's "
                     "unpack cost is not trustworthy."))
            # Re-tag so `get()` cannot confuse the quarter-grid arm with the
            # full-grid one it is checking.
            r["mode"] = "compute_lin"
            results.append(r)

    p("---", "decomposition: unpack cost, fetch saving, and the net")
    rows_out = []
    for ac in [c for c in consumers if c != "inc"]:
        p("---", f"Option A consumer = {ac}")
        rows_out += report(results, bkv_ps, ac, floor)

    emit_json(f"R1b_consumer_grid{args.num_grid}", {
        "geometry": vars(args),
        "num_bkv": num_bkv,
        "layouts": {k: {"rows": v[0], "lanes": v[1], "page_bytes": page_bytes(k)}
                    for k, v in LAYOUTS.items()},
        "dispatch_floor": floor,
        "arms": results,
        "linearity": lin,
        "decomposition": rows_out,
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
