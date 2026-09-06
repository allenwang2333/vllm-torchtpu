"""SparseCore top-k whose stage 1 emits (index, key) pairs, so stage 2 needs no gather.

Dimension glossary
------------------
  B   token rows presented to the selector (`scores.shape[0]`)
  N   score columns per row, the PADDED extent (`scores.shape[1]`)
  P   stage-1 partition factor; each row is cut into P slices
  NP  columns per slice, `N // P`   (a tile extent, hence the n-prefix spirit)
  KP  candidates kept per slice, `min(k, NP)`
  C   stage-2 candidate count per row, `P * KP`
  k   final selection width
  L   LANES, the SparseCore vector width (16)

Everything below `_topk_kernel` is a copy of the shipping
`vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk` with exactly two additions,
both static-flag gated so the copy can also be run as a byte-equivalent
incumbent (`emit_keys=False, premapped=False`):

  emit_keys   also compress-store the selected monotone KEY beside its index,
              and DMA it out. The keys are already in `resident_vmem`; the
              incumbent throws them away and stage 2 buys them back with a
              random gather out of the full `[B, N]` score array.
  premapped   the input words are ALREADY monotone keys, so skip the `cvt`
              pass. Stage 2 consumes stage 1's keys directly in key space and
              never round-trips through f32.

The -inf sentinel moves with them. In score space an unfilled candidate slot is
`-inf`; in key space it is `KEY_NEG_INF`, which is exactly
`monotone_key(bitcast(-inf))`, and `load_keys` already rejects it with
`key > KEY_NEG_INF`. The rowbuf prefill writes that constant, so the masking the
incumbent does on the host (`where(global_cand_indices >= 0, ..., -inf)`)
happens in the same place the -1 index padding does and needs no second pass.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

LANES = 16
NUM_BUCKETS = 256
# Largest per-subcore resident slice, in 32-bit words.
MAX_SLICE_WORDS = 64 * 1024

L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
KEY_NEG_INF = int(
    np.int32(np.array(-np.inf, np.float32).view(np.int32))
    ^ np.int32(0x7FFFFFFF))


def _cdiv(a, b):
    return (a + b - 1) // b


def _align_to(x, a):
    return _cdiv(x, a) * a


def _cdiv_dyn(x):
    return jnp.right_shift(x + (LANES - 1), 4)


def _topk_kernel(
    *refs,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    emit_keys: bool,
    premapped: bool,
):
    """Radix-select top-k on 32 subcores, optionally emitting keys with indices.

    The ref list is variadic only because `emit_keys` adds one output and one
    scratch; the trailing shape comments below are the glossary entry the
    checklist asks for.
    """
    if emit_keys:
        (
            scores_hbm,  # [B * N] i32   whole array (score bits, or keys)
            lengths_hbm,  # [B_PAD] i32  whole array
            out_hbm,  # [B * k] i32      whole array
            keys_hbm,  # [B * k] i32     whole array
            resident_vmem,  # [slice_len] i32   one row's slice
            priv_vmem,  # [L * 256 + L] i32     lane-private histograms
            glob_vmem,  # [256 + L] i32         folded hist, then prefix sum
            rowbuf_vmem,  # [k + L] i32         winner indices, one row
            rowkey_vmem,  # [k + L] i32         winner keys, one row
            len_vmem,  # [B_PAD] i32
            sems,  # DMA[4]
        ) = refs
    else:
        (
            scores_hbm,  # [B * N] i32
            lengths_hbm,  # [B_PAD] i32
            out_hbm,  # [B * k] i32
            resident_vmem,  # [slice_len] i32
            priv_vmem,  # [L * 256 + L] i32
            glob_vmem,  # [256 + L] i32
            rowbuf_vmem,  # [k + L] i32
            len_vmem,  # [B_PAD] i32
            sems,  # DMA[4]
        ) = refs
        keys_hbm = None
        rowkey_vmem = None

    num_subcores = 16
    num_sc_cores = 2
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    b_pad = len_vmem.shape[0]

    cp = pltpu.make_async_copy(lengths_hbm.at[pl.ds(0, b_pad)], len_vmem,
                               sems.at[0])
    cp.start()
    cp.wait()

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def zero_priv(bound):

        def body(i):
            priv_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES, ),
                                                           jnp.int32)

        plsc.parallel_loop(0, bound, unroll=4)(body)

    def fold_priv(bound):

        def body(i):
            off = i * LANES
            acc = jnp.zeros((LANES, ), jnp.int32)
            for m in range(LANES):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, LANES)]
            glob_vmem[pl.ds(off, LANES)] = acc

        plsc.parallel_loop(0, bound, unroll=2)(body)

    def scan_glob(bound):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(0, bound, unroll=2, carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def find_bucket(bound, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
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
        walk_chunks = _cdiv_dyn(i_hi)

        @pl.when(i_hi > 0)
        def _():
            dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)

            # `premapped` inputs are already monotone keys -- stage 2 of the
            # paired path receives stage 1's own keys, so this pass would be
            # the second of two involutions and is skipped, not reordered.
            if not premapped:

                def cvt(i):
                    bits = resident_vmem[pl.ds(i * LANES, LANES)]
                    resident_vmem[pl.ds(i * LANES, LANES)] = monotone_key(bits)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)

        def load_keys(i):
            key = resident_vmem[pl.ds(i * LANES, LANES)]
            j = i * LANES + lane_iota
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
                                       jnp.ones((LANES, ), jnp.int32),
                                       mask=valid)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            fold_priv(NUM_BUCKETS // LANES)

        @pl.when(i_hi == 0)
        def _():

            def zg(i):
                glob_vmem[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES, ),
                                                               jnp.int32)

            plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=4)(zg)

        total = scan_glob(NUM_BUCKETS // LANES)
        keep_all = total <= k

        b_star = find_bucket(NUM_BUCKETS // LANES, total - k)
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
                                       jnp.ones((LANES, ), jnp.int32),
                                       mask=match)

            plsc.parallel_loop(0, active_walk, unroll=4)(ref_body)
            fold_bound = jnp.where(done, 0, NUM_BUCKETS // LANES)
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
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES, ), -1,
                                                            jnp.int32)
            if emit_keys:
                # The score-space -inf sentinel, in key space. Written in the
                # same pass as the -1 index padding so the two arrays agree
                # slot for slot and stage 2 needs no host-side mask.
                rowkey_vmem[pl.ds(i * LANES, LANES)] = jnp.full(
                    (LANES, ), KEY_NEG_INF, jnp.int32)

        fill_bound = jnp.where(out_gate & (total < k), k // LANES, 0)
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
            pos = j
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)],
                                  pos,
                                  mask=sel)
            if emit_keys:
                # Same `woff`, same `sel`: the two compressed stores stay
                # slot-aligned by construction, which is the whole point --
                # a separate pass would have to re-derive the permutation.
                plsc.store_compressed(rowkey_vmem.at[pl.ds(woff, LANES)],
                                      key,
                                      mask=sel)
            woff = woff + plsc.all_reduce_population_count(sel)[0]
            rank = tie_rank[LANES - 1]
            return woff, rank

        n_win, _ = plsc.parallel_loop(
            0,
            jnp.where(active, walk_chunks, 0),
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_body)

        @pl.when(out_gate)
        def _():
            dma(
                rowbuf_vmem.at[pl.ds(0, k)],
                out_hbm.at[pl.ds(row_c * k, k)],
                2,
            )
            if emit_keys:
                dma(
                    rowkey_vmem.at[pl.ds(0, k)],
                    keys_hbm.at[pl.ds(row_c * k, k)],
                    3,
                )

        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)


def _launch(
    words: jax.Array,  # [B * N] i32, score bits or (if premapped) keys
    b: int,
    n: int,
    k: int,
    row_lengths: jax.Array,  # [B] i32
    *,
    write_empty: bool,
    emit_keys: bool,
    premapped: bool,
):
    """The single `pl.kernel` in this variant. Returns indices, or (idx, keys)."""
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")

    b_pad = _align_to(b, LANES)
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, b_pad - b))

    slice_len = _align_to(n, LANES)
    num_waves = _cdiv(b, 32)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    idx_type = jax.ShapeDtypeStruct((b * k, ), jnp.int32)
    out_type = (idx_type, idx_type) if emit_keys else idx_type
    scratch = [
        pltpu.VMEM((slice_len, ), jnp.int32),
        pltpu.VMEM((LANES * NUM_BUCKETS + LANES, ), jnp.int32),
        pltpu.VMEM((NUM_BUCKETS + LANES, ), jnp.int32),
        pltpu.VMEM((k + LANES, ), jnp.int32),
    ]
    if emit_keys:
        scratch.append(pltpu.VMEM((k + LANES, ), jnp.int32))
    scratch += [
        pltpu.VMEM((b_pad, ), jnp.int32),
        pltpu.SemaphoreType.DMA((4, )),
    ]

    tag = ("pair" if emit_keys else ("key" if premapped else "raw"))
    out = pl.kernel(
        functools.partial(
            _topk_kernel,
            b=b,
            n=n,
            k=k,
            slice_len=slice_len,
            num_waves=num_waves,
            write_empty=write_empty,
            emit_keys=emit_keys,
            premapped=premapped,
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=tuple(scratch),
        mesh=mesh,
        name=f"o8_topk_{tag}_b{b}_n{n}_k{k}",
    )(words, lengths)
    if emit_keys:
        return out[0].reshape(b, k), out[1].reshape(b, k)
    return out.reshape(b, k)


def sc_topk_scores(scores, k, row_lengths, *, write_empty=True):
    """Incumbent-equivalent single stage: f32 in, indices out."""
    b, n = scores.shape
    words = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)
    return _launch(words, b, n, k, row_lengths, write_empty=write_empty,
                   emit_keys=False, premapped=False)


def sc_topk_scores_paired(scores, k, row_lengths, *, write_empty=True):
    """Single stage, f32 in, (indices, monotone keys) out."""
    b, n = scores.shape
    words = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)
    return _launch(words, b, n, k, row_lengths, write_empty=write_empty,
                   emit_keys=True, premapped=False)


def sc_topk_keys(keys, k, row_lengths, *, write_empty=True):
    """Single stage over ALREADY-monotone keys: i32 in, indices out."""
    b, n = keys.shape
    return _launch(keys.reshape(-1), b, n, k, row_lengths,
                   write_empty=write_empty, emit_keys=False, premapped=True)


# ---------------------------------------------------------------------------
# Assembly: the two-stage composition at a GIVEN p. `p` is chosen in
# `pallas_mosaic_tpu.py`; nothing here picks it.
# ---------------------------------------------------------------------------


def select_inc_at_p(scores, k, row_lengths, *, p, write_empty_rows):
    """Incumbent shape, rebuilt on this file's kernel copy.

    Exists so `inc_copy` can be timed beside the shipping incumbent: if the two
    differ, the difference is this copy and not the change under test.
    """
    b, n = scores.shape
    if p == 1:
        return sc_topk_scores(scores, k, row_lengths,
                              write_empty=write_empty_rows)
    n_p = n // p
    k_p = min(k, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)
    local_indices = sc_topk_scores(scores.reshape(b * p, n_p), k_p, lengths_p,
                                   write_empty=True).reshape(b, p, k_p)
    global_cand_indices = jnp.where(
        local_indices < 0, -1,
        offsets[None, :, None] + local_indices).reshape(b, p * k_p)
    safe_cand_indices = jnp.maximum(global_cand_indices, 0)
    cand_scores = jnp.take_along_axis(scores, safe_cand_indices, axis=1)
    cand_scores = jnp.where(global_cand_indices >= 0, cand_scores, -jnp.inf)
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))
    final_cand_slots = sc_topk_scores(cand_scores, k, cand_lengths,
                                      write_empty=write_empty_rows)
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


def select_o8_at_p(scores, k, row_lengths, *, p, write_empty_rows):
    """O8: stage 1 hands stage 2 its keys, so the [B, N] gather does not exist.

    The only lines that differ from `select_inc_at_p` are the paired stage-1
    call and the absence of the three that follow it there (max, take_along,
    where). Stage 2 runs in key space, so no bitcast happens at the seam and
    the kernel's own `cvt` pass is skipped on the candidate array as well.
    """
    b, n = scores.shape
    if p == 1:
        return sc_topk_scores(scores, k, row_lengths,
                              write_empty=write_empty_rows)
    n_p = n // p
    k_p = min(k, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)
    local_indices, local_keys = sc_topk_scores_paired(
        scores.reshape(b * p, n_p), k_p, lengths_p, write_empty=True)
    global_cand_indices = jnp.where(
        local_indices.reshape(b, p, k_p) < 0, -1,
        offsets[None, :, None] + local_indices.reshape(b, p, k_p)).reshape(
            b, p * k_p)
    # No mask here: the kernel prefilled unfilled slots with KEY_NEG_INF, which
    # is the key-space -inf, and it did so from the same rowbuf pass that wrote
    # the -1 indices -- so the two arrays already agree slot for slot.
    cand_keys = local_keys.reshape(b, p * k_p)
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))
    final_cand_slots = sc_topk_keys(cand_keys, k, cand_lengths,
                                    write_empty=write_empty_rows)
    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)
