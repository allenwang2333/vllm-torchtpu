"""Every `pallas_call` in the composed candidate, and both kernel bodies.

This is the one shipping candidate step 4 composes: O8 + O3's wave packing +
O7's `scan_count` histogram in the SparseCore selector, and O1's two launch
eliminations in the TensorCore scoring half. O2 is a tile choice and appears in
`pallas_mosaic_tpu.tile_config`, not here -- its kernel body is the
incumbent's, unchanged.

Each change is behind its own field of `pallas_mosaic_tpu.Config`, and each
carries the gating it was measured with. `Config()` is the candidate.

Dimension glossary
------------------
  B      token rows presented to the selector (`scores.shape[0]`)
  N      score columns per row, the PADDED extent (`scores.shape[1]`)
  P      column partition actually run; each row is cut into P slices
  P_MAX  the static ceiling on P (capacity, then the growth guard)
  tn     columns per slice, `N // P_MAX`, held fixed across the ladder
  n_sl   slices in flight, `B * P`
  kp     candidates kept per slice, `min(k, tn)`
  C      stage-2 candidate count per row, `P * kp`
  k      final selection width
  L      SparseCore lanes (16 on tpu7x), read from the platform
  T,H,DH padded token count, index heads, head dim -- the scoring half
  psz    page size; bkv_p KV pages per grid step; bq_sz queries per grid step

The two rewrites of stage 1, and why they compose
-------------------------------------------------
O8 changes WHAT stage 1 emits: alongside the winning index it compress-stores
the winning monotone KEY at the same offset under the same mask, and DMAs both
out. Stage 2 then runs in key space (`premapped`) and the incumbent's
`take_along_axis` over the whole `[B, N]` score array at the seam does not
exist. The `-inf` sentinel moves with it: an unfilled candidate slot is
prefilled with `KEY_NEG_INF`, which is exactly `monotone_key(bitcast(-inf))`,
in the same pass that writes the `-1` index padding -- so the host-side mask
the incumbent applies is not needed either.

O3 changes WHERE stage 1 reads: the incumbent addresses slice `slot` at
`slot * tn`, which is only correct when the slices tile every row completely,
i.e. at `p == P_MAX`. The wave packing lets `p` fall below the ceiling at
runtime, so a slice is tile `slot % p` of row `slot // p` and the row stride N
must be applied explicitly. At `p == P_MAX` the two are arithmetically
identical.

They touch different lines of the same body -- emission versus addressing --
which is what makes the merge a merge rather than a redesign. It is still the
riskiest seam in this file, and `run_correctness.py` checks the composed
selector's index set against the incumbent's as a symmetric difference, not
only against a tolerance.

Deliberate deviations from `references/code-style.md`, declared per its S10;
the full list with reasons is in `DEVIATIONS.md`.

  1. Two operations share these four files rather than taking four each. The
     scoring half is only present because O1 changes it, and splitting it out
     would put a 700-line verbatim copy in a file of its own for no gain.
  2. The copied TensorCore body keeps the incumbent's literal 128 lane width
     instead of `pltpu.get_tpu_info().num_lanes`, against S13. Rewriting ~370
     lines of index arithmetic is unrelated to any adopted change and is pure
     regression risk in a half whose contract is that its output does not move.
     The SparseCore body, which IS changed, has all four of its hardware
     constants -- lane count, subcore count, core count, tile-memory capacity
     -- read from `pltpu.get_tpu_info()` via `pallas_mosaic_tpu.hardware()`.
  3. No jaxtyping annotations: `jaxtyping` is not installed in the benchmark
     image. Shapes are carried as trailing comments in the glossary's letters.
"""

import enum
import functools
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

import pallas_mosaic_tpu as policy
import tiling
from tiling import align_to, cdiv

Enum = enum.Enum

# The exact source text of the selector gate that makes the -inf pre-fill
# unnecessary. `_assert_valid_gate_intact` looks for it before eliding the fill.
_VALID_GATE_SRC = "valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))"


def _assert_valid_gate_intact():
    """Fail loudly if the selector could read past a row's live extent.

    `elide_score_prefill` replaces the `-inf` pre-fill of the score buffer with
    an uninitialised allocation. Nothing downstream notices, because
    `_topk_kernel.load_keys` masks every lane on `j < i_hi` and `i_hi` is
    `min(tn, eff_len)`. If that mask is ever relaxed the fill stops being dead
    weight and starts being the only thing standing between the selector and
    garbage -- and the failure is silent, because uninitialised HBM usually
    looks small.
    """
    src = inspect.getsource(_topk_kernel)
    if _VALID_GATE_SRC not in src:
        raise ValueError(
            "the selector's live-extent gate is gone: expected "
            f"{_VALID_GATE_SRC!r} in _topk_kernel. This candidate elides the "
            "-inf pre-fill of the score buffer and is only correct while that "
            "gate holds. Re-derive the fill before removing the gate.")


# --------------------------------------------------------------------------
# SparseCore selection.
# --------------------------------------------------------------------------

NUM_BUCKETS = 256
L0_SHIFT = 24
REFINEMENT_LEVELS = ((16, 0xFF), (8, 0xFF), (0, 0xFF))
MONO_MASK = 0x7FFFFFFF
KEY_NEG_INF = int(
    np.int32(np.array(-np.inf, np.float32).view(np.int32))
    ^ np.int32(0x7FFFFFFF))


def _cdiv_dyn(x, lanes):
    """ceil(x / lanes) for a traced x; `lanes` is a power of two, so a shift."""
    return jnp.right_shift(x + (lanes - 1), lanes.bit_length() - 1)


def _topk_kernel(
    *refs,
    n_sl: int,
    tn: int,
    k: int,
    p: int,
    row_stride: int,
    nw: int,
    lanes: int,
    subcores_per_wave: int,
    num_subcores: int,
    num_sc_cores: int,
    write_empty: bool,
    emit_keys: bool,
    premapped: bool,
    scan_count: bool,
):
    """Radix-select top-k on the subcore mesh, one slice per subcore.

    The ref list is variadic because two of the flags add refs: `emit_keys`
    adds an output and a scratch buffer, and `scan_count` removes one. Popping
    in a fixed order keeps the unpacking and the `scratch_types` tuple in
    `_sc_topk_slices` in step by construction -- a positional mismatch between
    the two is silent and would read as a kernel bug.
    """
    refs = list(refs)
    scores_hbm = refs.pop(0)        # [n_sl_rows * row_stride] i32  score bits,
    lengths_hbm = refs.pop(0)       # [n_sl_pad] i32               or keys
    out_hbm = refs.pop(0)           # [n_sl * k] i32
    keys_hbm = refs.pop(0) if emit_keys else None       # [n_sl * k] i32
    resident_vmem = refs.pop(0)     # [tn] i32          one slice, keys in place
    priv_vmem = refs.pop(0) if not scan_count else None  # [L*256 + L] i32
    glob_vmem = refs.pop(0)         # [256 + L] i32     histogram, then prefix
    rowbuf_vmem = refs.pop(0)       # [k + L] i32       winner indices, one slice
    rowkey_vmem = refs.pop(0) if emit_keys else None     # [k + L] i32
    len_vmem = refs.pop(0)          # [n_sl_pad] i32
    sems = refs.pop(0)              # DMA[4]
    if refs:
        raise AssertionError(f"{len(refs)} refs left unbound")

    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(lanes, dtype=jnp.int32)
    zeros = jnp.zeros((lanes, ), jnp.int32)
    n_sl_pad = len_vmem.shape[0]

    # Slot -> (row, tile). `log2p` and `part_mask` are Python ints: `p` is
    # static per branch, so these are shifts and masks, not divides.
    log2p = int(p).bit_length() - 1
    part_mask = jnp.int32(p - 1)

    cp = pltpu.make_async_copy(lengths_hbm.at[pl.ds(0, n_sl_pad)], len_vmem,
                               sems.at[0])
    cp.start()
    cp.wait()

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, lanes)][0]

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def zero_words(ref, chunks):
        """Zero `chunks` vectors at the front of `ref`. `chunks` may be traced."""

        def body(i):
            ref[pl.ds(i * lanes, lanes)] = zeros

        plsc.parallel_loop(0, chunks, unroll=4)(body)

    # ---- histogram back end ------------------------------------------------
    # Two spellings of "count keys per bucket". The lane-private one is the
    # incumbent's: 16 private histograms sidestep the fact that a scatter cannot
    # combine duplicate addresses inside one vector. `scan_count` removes that
    # need -- it computes the duplicate multiplicity in-register and scatters
    # one combined count per distinct bucket -- and with it the 16x replication,
    # its zeroing and its fold.

    def hist_zero(gate):
        if scan_count:
            zero_words(glob_vmem, gate * (NUM_BUCKETS // lanes))
        else:
            zero_words(priv_vmem, gate * NUM_BUCKETS)

    def hist_add(bucket, mask):
        if scan_count:
            run, last = plsc.scan_count(bucket, mask=mask)
            # `run` is the 1-BASED running occurrence count, so at the final
            # occurrence of a bucket it is exactly that bucket's multiplicity in
            # this vector. Scattering `run + 1` would over-count every duplicate
            # run by one, silently.
            plsc.addupdate_scatter(glob_vmem, (bucket, ), run, mask=last)
        else:
            addr = lane_iota * NUM_BUCKETS + bucket
            plsc.addupdate_scatter(priv_vmem, (addr, ),
                                   jnp.ones((lanes, ), jnp.int32), mask=mask)

    def hist_fold(gate):
        if scan_count:
            return  # nothing was replicated, so there is nothing to fold.

        def body(i):
            off = i * lanes
            acc = zeros
            for m in range(lanes):
                acc = acc + priv_vmem[pl.ds(m * NUM_BUCKETS + off, lanes)]
            glob_vmem[pl.ds(off, lanes)] = acc

        plsc.parallel_loop(0, gate * (NUM_BUCKETS // lanes), unroll=2)(body)

    def scan_glob(chunks):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * lanes, lanes)]) + carry
            glob_vmem[pl.ds(i * lanes, lanes)] = c
            return c[lanes - 1]

        return plsc.parallel_loop(0, chunks, unroll=2,
                                  carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def find_bucket(chunks, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * lanes, lanes)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, chunks, unroll=2,
                                  carry=jnp.int32(0))(body)  # pytype: disable=bad-argument-type

    def wave_body(w, _):
        slot = w * subcores_per_wave + flat
        active = slot < n_sl
        slot_c = jnp.minimum(slot, n_sl - 1)
        eff_len = jnp.minimum(vec_at(len_vmem, slot_c), tn)
        eff_len = jnp.where(active, eff_len, 0)

        # THE WAVE-PACKING LINE. The incumbent reads slice `slot_c` at
        # `slot_c * tn`, which assumes the slices tile every row completely --
        # true only at p == P_MAX. Here p <= P_MAX, so a slice is tile
        # (slot_c % p) of row (slot_c // p) and the row stride has to be
        # applied explicitly. At p == P_MAX this is arithmetically identical to
        # `slot_c * tn`, so the ladder's top rung reproduces the incumbent's
        # addressing exactly rather than approximating it.
        row_id = jnp.right_shift(slot_c, log2p)
        part_id = jnp.bitwise_and(slot_c, part_mask)
        s = row_id * row_stride + part_id * tn

        i_hi = jnp.where(eff_len > 0, jnp.minimum(tn, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi, lanes)

        @pl.when(i_hi > 0)
        def _():
            dma(scores_hbm.at[pl.ds(s, tn)], resident_vmem, 1)

            # `premapped` inputs are already monotone keys -- stage 2 of the
            # paired path receives stage 1's own keys, so this pass would be
            # the second of two involutions and is skipped, not reordered.
            if not premapped:

                def cvt(i):
                    bits = resident_vmem[pl.ds(i * lanes, lanes)]
                    resident_vmem[pl.ds(i * lanes, lanes)] = monotone_key(bits)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)

        def load_keys(i):
            key = resident_vmem[pl.ds(i * lanes, lanes)]
            j = i * lanes + lane_iota
            valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))
            return key, valid, j

        @pl.when(i_hi > 0)
        def _():
            hist_zero(1)

            def hist0_body(i):
                key, valid, _ = load_keys(i)
                bucket = jnp.right_shift(key, L0_SHIFT) + jnp.int32(
                    NUM_BUCKETS // 2)
                hist_add(bucket, valid)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            hist_fold(1)

        @pl.when(i_hi == 0)
        def _():
            zero_words(glob_vmem, NUM_BUCKETS // lanes)

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
            gate = jnp.where(done, 0, 1)
            hist_zero(gate)
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
                hist_add(bucket, match)

            plsc.parallel_loop(0, active_walk, unroll=4)(ref_body)
            hist_fold(gate)
            fold_bound = gate * (NUM_BUCKETS // lanes)

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
            if emit_keys:
                # The score-space -inf sentinel, in key space. Written in the
                # same pass as the -1 index padding so the two arrays agree slot
                # for slot and stage 2 needs no host-side mask.
                rowkey_vmem[pl.ds(i * lanes, lanes)] = jnp.full(
                    (lanes, ), KEY_NEG_INF, jnp.int32)

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
            if emit_keys:
                # Same `woff`, same `sel`: the two compressed stores stay
                # slot-aligned by construction, which is the whole point -- a
                # separate pass would have to re-derive the permutation.
                plsc.store_compressed(rowkey_vmem.at[pl.ds(woff, lanes)], key,
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
            dma(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(slot_c * k, k)],
                2)
            if emit_keys:
                dma(rowkey_vmem.at[pl.ds(0, k)],
                    keys_hbm.at[pl.ds(slot_c * k, k)], 3)

        return 0

    lax.fori_loop(0, nw, wave_body, 0)


def _sc_topk_slices(
    words_flat: jax.Array,  # [B * row_stride] i32   already flat, already i32
    b: int,
    row_stride: int,
    k: int,
    lengths: jax.Array,  # [B * p] i32   per-SLICE, already clipped to tn
    *,
    p: int,
    tn: int,
    write_empty: bool,
    emit_keys: bool,
    premapped: bool,
    scan_count: bool,
):
    """One ladder rung: B*p slices of `tn` words each, over the subcore mesh.

    `p == 1, tn == row_stride` is the incumbent's `_sc_topk_direct`; every other
    (p, tn) covers only the first `p * tn` columns of each row, which is legal
    exactly when the caller bucketed upward.

    Takes the ALREADY flat int32 view rather than a `[B, N]` f32 array. The
    bitcast and the reshape are free on their own, but performed inside a
    `lax.switch` branch they are not: `measured (j-fad1a0f8)` caught the
    conditional materialising a whole copy of the operand at 1.30 TB/s
    effective -- 51 us at 64 MiB and 3310 us at 4 GiB, the same rate at both
    sizes. Hoisting them above the switch is the fix, and it only works if
    nothing inside a branch needs the `[B, N]` f32 view.

    Returns `[B*p, k]` indices, or `(indices, keys)` when `emit_keys`.
    """
    hw = policy.hardware()
    lanes = hw["lanes"]
    subcores_per_wave = hw["subcores_per_wave"]

    if tn % lanes:
        raise ValueError(f"{tn=} must be a multiple of {lanes}")
    if p * tn > row_stride:
        raise ValueError(f"p*tn={p * tn} overruns the row extent {row_stride}")
    if words_flat.dtype != jnp.int32:
        raise ValueError(f"words_flat must be i32, got {words_flat.dtype}")

    n_sl = b * p
    n_sl_pad = align_to(n_sl, lanes)
    lengths = jnp.pad(lengths.astype(jnp.int32), (0, n_sl_pad - n_sl))
    nw = tiling.num_waves(b, p, subcores_per_wave)

    mesh = plsc.VectorSubcoreMesh(
        num_cores=hw["num_cores"],
        num_subcores=hw["num_subcores"],
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    idx_type = jax.ShapeDtypeStruct((n_sl * k, ), jnp.int32)
    out_type = (idx_type, idx_type) if emit_keys else idx_type

    # Same order the kernel pops them in. Keep the two in step.
    scratch = [pltpu.VMEM((tn, ), jnp.int32)]
    if not scan_count:
        scratch.append(pltpu.VMEM((lanes * NUM_BUCKETS + lanes, ), jnp.int32))
    scratch.append(pltpu.VMEM((NUM_BUCKETS + lanes, ), jnp.int32))
    scratch.append(pltpu.VMEM((k + lanes, ), jnp.int32))
    if emit_keys:
        scratch.append(pltpu.VMEM((k + lanes, ), jnp.int32))
    scratch += [
        pltpu.VMEM((n_sl_pad, ), jnp.int32),
        pltpu.SemaphoreType.DMA((4, )),
    ]

    tag = "pair" if emit_keys else ("key" if premapped else "raw")
    out = pl.kernel(
        functools.partial(
            _topk_kernel,
            n_sl=n_sl,
            tn=tn,
            k=k,
            p=p,
            row_stride=row_stride,
            nw=nw,
            lanes=lanes,
            subcores_per_wave=subcores_per_wave,
            num_subcores=hw["num_subcores"],
            num_sc_cores=hw["num_cores"],
            write_empty=write_empty,
            emit_keys=emit_keys,
            premapped=premapped,
            scan_count=scan_count,
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=tuple(scratch),
        mesh=mesh,
        name=f"sc_topk_{tag}_b{b}_p{p}_tn{tn}_k{k}",
    )(words_flat, lengths)
    if emit_keys:
        return out[0].reshape(n_sl, k), out[1].reshape(n_sl, k)
    return out.reshape(n_sl, k)


def _select_at_p(
    words_flat: jax.Array,  # [B * N] i32
    scores: jax.Array | None,  # [B, N] f32, only when the seam gather is used
    b: int,
    n: int,
    k: int,
    row_lengths: jax.Array,  # [B] i32
    *,
    p: int,
    tn: int,
    write_empty_rows: bool,
    emit_keys: bool,
    scan_count: bool,
):  # -> [B, k] i32
    """The whole selection at a fixed `p`; one precompiled `lax.switch` branch.

    Nothing here may reshape or bitcast `words_flat` or `scores` -- doing so
    inside a branch is what `j-fad1a0f8` priced at a full copy of the array.
    Both views are passed in already in the shape they are used in.
    """
    if p == 1:
        # No stage 2 at all, and this is the largest single step on the ladder
        # rather than a special case: dropping to p == 1 deletes a pl.kernel
        # launch and the seam entirely, on top of the wave saving.
        return _sc_topk_slices(words_flat, b, n, k,
                               jnp.minimum(row_lengths, tn), p=1, tn=tn,
                               write_empty=write_empty_rows, emit_keys=False,
                               premapped=False, scan_count=scan_count)

    kp = min(k, tn)
    offsets = jnp.arange(p, dtype=jnp.int32) * tn
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         tn).reshape(b * p)
    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * kp), jnp.int32(0))

    if emit_keys:
        local_indices, local_keys = _sc_topk_slices(
            words_flat, b, n, kp, lengths_p, p=p, tn=tn, write_empty=True,
            emit_keys=True, premapped=False, scan_count=scan_count)
        global_cand_indices = jnp.where(
            local_indices.reshape(b, p, kp) < 0, -1,
            offsets[None, :, None] + local_indices.reshape(b, p, kp)).reshape(
                b, p * kp)
        # No mask here: the kernel prefilled unfilled slots with KEY_NEG_INF,
        # the key-space -inf, from the same pass that wrote the -1 indices, so
        # the two arrays already agree slot for slot.
        cand_keys = local_keys.reshape(b, p * kp)
        final_cand_slots = _sc_topk_slices(
            cand_keys.reshape(-1), b, p * kp, k, cand_lengths, p=1, tn=p * kp,
            write_empty=write_empty_rows, emit_keys=False, premapped=True,
            scan_count=scan_count)
    else:
        # The incumbent's seam, kept so `emit_keys=False` is a faithful
        # leave-one-out arm: buy the candidate scores back out of the full
        # [B, N] array with a gather. `scores` is a switch operand, so no
        # reshape or bitcast happens inside the branch.
        local_indices = _sc_topk_slices(
            words_flat, b, n, kp, lengths_p, p=p, tn=tn, write_empty=True,
            emit_keys=False, premapped=False,
            scan_count=scan_count).reshape(b, p, kp)
        global_cand_indices = jnp.where(
            local_indices < 0, -1,
            offsets[None, :, None] + local_indices).reshape(b, p * kp)
        safe_cand_indices = jnp.maximum(global_cand_indices, 0)
        cand_scores = jnp.take_along_axis(scores, safe_cand_indices, axis=1)
        cand_scores = jnp.where(global_cand_indices >= 0, cand_scores,
                                -jnp.inf)
        # [B, p*kp] is 128x smaller than the scores it came from, so this
        # bitcast inside the branch is not the one that was priced.
        final_cand_slots = _sc_topk_slices(
            jax.lax.bitcast_convert_type(cand_scores, jnp.int32).reshape(-1),
            b, p * kp, k, cand_lengths, p=1, tn=p * kp,
            write_empty=write_empty_rows, emit_keys=False, premapped=False,
            scan_count=scan_count)

    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices, safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


def sparsecore_topk_impl(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array | None = None,  # [B] i32, defaults to N
    *,
    write_empty_rows: bool = True,
    cfg: policy.Config,
) -> jax.Array:  # [B, k] i32
    """Exact top-k indices per row, unsorted, -1 suffix-padded.

    Same contract as the incumbent, and deliberately the same selected set: the
    partition decides HOW the exact top-k is found, not WHAT it is. Ties at the
    cut may break differently, which the shared instrument's churn metric
    covers and the incumbent's own docstring already permits.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    b, n = scores.shape
    policy.select_supported_on(b, n, k, fix_growth=cfg.fix_growth)
    geo = policy.select_geometry(b, n, k, fix_growth=cfg.fix_growth)

    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)

    if cfg.elide_score_prefill:
        # The caller may have handed us a score buffer whose columns past
        # `row_lengths` were never written. That is legal only while this
        # kernel refuses to read there. Check it, here, at the one call that
        # depends on it.
        _assert_valid_gate_intact()

    if b * n >= 2**31:
        raise NotImplementedError(
            f"{b}*{n} flat elements overrun the int32 addressing used for the "
            "flat score view")
    words_flat = jax.lax.bitcast_convert_type(scores, jnp.int32).reshape(-1)

    ladder = geo.ladder if cfg.pack_waves else (geo.p_max, )

    def at(p, words_, scores_, row_lengths_):
        return _select_at_p(words_, scores_, b, n, k, row_lengths_, p=p,
                            tn=geo.tn, write_empty_rows=write_empty_rows,
                            emit_keys=cfg.emit_keys and p > 1,
                            scan_count=cfg.scan_count_hist)

    if len(ladder) == 1:
        # Either the wave packing is off, or `p_max == 1` and there is no
        # choice to make and no switch to pay for. The second is regime (A)
        # once the growth guard has collapsed P, and it is why the packing is
        # worth exactly zero there.
        return at(ladder[0], words_flat, scores, row_lengths)

    # Bucket UPWARD over the ladder. `coverage` is what each rung below the top
    # can reach; the index is the count of rungs that fall short, so the rung
    # selected is the smallest one that covers the batch's longest live row.
    # Downward would truncate a live candidate set silently -- see
    # `tiling.ladder_index`, whose Python spelling this must agree with.
    live_max = jnp.max(row_lengths)
    coverage = jnp.asarray([q * geo.tn for q in ladder[:-1]], jnp.int32)
    rung = jnp.sum((coverage < live_max).astype(jnp.int32))

    # `k` and `p` stay closed over: passing `p` as a switch operand would trace
    # it and `min(k, tn)` would stop being a Python int. Both views of the
    # score buffer cross the boundary as operands, already in the shape their
    # branch uses, so no branch reshapes or bitcasts them.
    if cfg.emit_keys:
        # `scores` is not read by any branch under `emit_keys`; leaving it out
        # of the operand list keeps the conditional from carrying a second live
        # buffer.
        def branch(p):

            def go(words_, row_lengths_):
                return at(p, words_, None, row_lengths_)

            return go

        return lax.switch(rung, tuple(branch(q) for q in ladder), words_flat,
                          row_lengths)

    def branch_gather(p):

        def go(words_, scores_, row_lengths_):
            return at(p, words_, scores_, row_lengths_)

        return go

    return lax.switch(rung, tuple(branch_gather(q) for q in ladder),
                      words_flat, scores, row_lengths)


# --------------------------------------------------------------------------
# TensorCore scoring. Copied from streamindex_topk.py; the only changes are
# `skip_empty_launches` and `elide_score_prefill`, both flag-gated.
# --------------------------------------------------------------------------
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


class MlaCase(Enum):
    """Represents the different cases for MLA."""

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]
def kernel(
    # Prefetch
    seq_lens_ref,  # Shape: [max_num_seqs], Memory: SMEM
    page_indices_ref,  # Shape: [max_num_seqs * pages_per_seq], Memory: SMEM
    cu_q_lens_ref,  # Shape: [max_num_seqs + 1], Memory: SMEM
    start_end_seq_idx_ref,  # Shape: [2], Memory: SMEM
    sem_ids_ref,  # Shape: [3], Memory: SMEM
    bo_sz_ref,  # Shape: [2], Memory: SMEM
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim], Memory: HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM
    # Scratch
    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim], Memory: VMEM
    bq_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads, head_dim], Memory: VMEM
    bq_weights_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads], Memory: VMEM
    scores_block_x2_ref,  # Shape: [2, seq_batch_size * bq_sz, num_sublanes_bkv, 128], Memory: VMEM
    sems,  # Shape: [4, 2, seq_batch_size], Memory: Semaphore
    *,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
):
    """Core kernel logic that operates on memory references."""

    _, num_q_heads, head_dim = q_hbm_ref.shape
    lkv_dim = cache_kv_hbm_ref.shape[-1]

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    num_sublanes_bkv = bkv_sz // 128

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * seq_batch_size
    batch_end_seq_idx = batch_start_seq_idx + seq_batch_size - 1

    q_lens = []
    kv_lens = []
    seq_lens = []
    for batch_idx in range(seq_batch_size):
        q_start = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
        q_end = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
        q_len = q_end - q_start
        q_lens.append(q_len)
        seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
        seq_lens.append(seq_len)
        kv_len = seq_len // compression_ratio
        kv_lens.append(kv_len)

    def wait_send_scores(bo_sem_idx):
        old_sz = bo_sz_ref[bo_sem_idx]

        @pl.when(old_sz >= 0)
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

    def start_send_scores(bo_sem_idx, sz, token_start, bkv_idx):
        # All sequences in the batch have the same sz. Issue a single DMA for the
        # entire batch.
        bo_sz_ref[bo_sem_idx] = sz
        sublane_start = bkv_idx * num_sublanes_bkv
        _async_copy(
            scores_block_x2_ref.at[bo_sem_idx,
                                   pl.ds(0, seq_batch_size * sz)],
            scores_hbm_ref.at[
                pl.ds(token_start, seq_batch_size * sz),
                pl.ds(sublane_start, num_sublanes_bkv),
            ],
            sems.at[2, bo_sem_idx, 0],
            wait=False,
        )

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            kv_packing,
            lkv_dim,
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

        for batch_idx in range(seq_batch_size):
            sem = sems.at[0, bkv_sem_idx, batch_idx]
            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]

            kv_p_start = bkv_idx * bkv_p
            page_indices_offset = (seq_idx +
                                   batch_idx) * pages_per_seq + kv_p_start

            if not wait:
                for i in range(bkv_p):
                    sz_per_kv_packing = page_size_per_kv_packing
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    safe_page_offset = jnp.minimum(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        jnp.maximum(0,
                                    max_hbm_pages - page_size_per_kv_packing),
                    )

                    _async_copy(
                        reshaped_cache_hbm_ref.at[pl.ds(
                            safe_page_offset, sz_per_kv_packing)],
                        bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                              sz_per_kv_packing)],
                        sem,
                        wait=False,
                    )
            else:
                dma_bkv_sz = bkv_p * page_size_per_kv_packing
                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        for batch_idx in range(seq_batch_size):
            sem = sems.at[1, bq_sem_idx, batch_idx]
            weights_sem = sems.at[3, bq_sem_idx, batch_idx]
            bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, batch_idx]
            bq_weights_vmem_ref = bq_weights_x2_ref.at[bq_sem_idx, batch_idx]

            q_len_start = cu_q_lens_ref[seq_idx + batch_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[seq_idx + batch_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))

            if not wait:
                _async_copy(
                    q_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    indexer_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_weights_vmem_ref.at[pl.ds(0, sz)],
                    weights_sem,
                    wait=False,
                )
            else:
                dst = bq_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst, dst, sem, wait=True)
                dst_w = bq_weights_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst_w, dst_w, weights_sem, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def load_bq(bq_sem_idx):
        data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].reshape(
            seq_batch_size, bq_sz * num_q_heads, head_dim)
        bqs = []
        for batch_idx in range(seq_batch_size):
            bqs.append(data[batch_idx])
        return bqs

    def load_bq_weights(bq_sem_idx):
        data = bq_weights_x2_ref.at[bq_sem_idx, :, :bq_sz][...]
        bq_weights = []
        for batch_idx in range(seq_batch_size):
            bq_weights.append(data[batch_idx])
        return bq_weights

    def load_bkv(bkv_sem_idx):
        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            bkv = bkv_x2_ref.at[bkv_sem_idx,
                                batch_idx, :bkv_sz_per_kv_packing][...]

            flat_bkv = bkv.reshape(-1, bkv.shape[-1])
            fp8_val = flat_bkv[:, :head_dim]
            fp8_val = pltpu.bitcast(fp8_val, jnp.float8_e4m3fn)
            scale_val = pltpu.bitcast(flat_bkv[:, head_dim:head_dim + 1].T,
                                      jnp.float8_e8m0fnu).astype(jnp.bfloat16)

            bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
            bkv_scales.append(scale_val)
        return bkvs, bkv_scales

    def process():
        kv_len_max = jnp.max(jnp.stack(kv_lens))
        num_bkv = jnp.maximum(1, cdiv(kv_len_max, bkv_sz))
        if static_q_len is None:
            assert seq_batch_size == 1
            num_bq = jnp.maximum(1, cdiv(q_lens[0], bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_scores(
            bq_vec,
            bkv_vec,
            scale_val_vec,
            bq_weights_vec,
            bq_pos_compressed_vec,
            bkv_idx,
        ):
            # Vectorized batched matmul
            bq_stacked = jnp.stack(
                bq_vec
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, head_dim)
            bkv_stacked = jnp.stack(
                bkv_vec)  # Shape: (seq_batch_size, bkv_sz, head_dim)
            scale_val_stacked = jnp.stack(
                scale_val_vec)  # Shape: (seq_batch_size, 1, bkv_sz)
            bq_weights_stacked = jnp.stack(
                bq_weights_vec)  # Shape: (seq_batch_size, bq_sz, num_q_heads)
            bq_pos_compressed_stacked = jnp.stack(
                bq_pos_compressed_vec)  # Shape: (seq_batch_size, bq_sz)

            # Compute in Registers
            s = jnp.einsum(
                "bnd,bmd->bnm",
                bq_stacked,
                bkv_stacked,
                preferred_element_type=jnp.float32,
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, bkv_sz)
            s = s.reshape(seq_batch_size, bq_sz, num_q_heads, bkv_sz)
            s = jnp.maximum(s, 0.0)
            s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
            s_summed = s.sum(axis=2)  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            s_summed = (s_summed * scale_val_stacked
                        )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            k_span = bkv_idx * bkv_sz + jnp.arange(
                bkv_sz, dtype=jnp.int32)  # Shape: (bkv_sz,)

            kv_lens_stacked = jnp.stack(kv_lens)  # Shape: (seq_batch_size,)
            valid_mask = (k_span[None, None, :] < kv_lens_stacked[:, None,
                                                                  None]
                          )  # Shape: (seq_batch_size, 1, bkv_sz)
            causal_mask = (k_span[None, None, :]
                           <= bq_pos_compressed_stacked[:, :, None]
                           )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            mask = jnp.logical_and(valid_mask, causal_mask)
            s_summed = jnp.where(mask, s_summed, -jnp.inf)

            return s_summed  # Shape: (seq_batch_size, bq_sz, bkv_sz)

        def compute_with_bq(bq_idx, _):

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            bq_pos_compressed_vec = []
            for batch_idx in range(seq_batch_size):
                q_pos = (seq_lens[batch_idx] - q_lens[batch_idx] +
                         bq_idx * bq_sz + jnp.arange(bq_sz, dtype=jnp.int32))
                bq_pos_compressed_vec.append(q_pos // compression_ratio)

            # Wait for cur bq if not ready yet
            wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)
            bq_vec = load_bq(bq_sem_idx)
            bq_weights_vec = load_bq_weights(bq_sem_idx)

            # If seq_batch_size > 1, static_q_len is always 1, therefore sz is always
            # 1 for all sequences within the batch.
            token_start = cu_q_lens_ref[batch_start_seq_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[batch_start_seq_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - token_start))

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)
                bkv_vec, scale_val_vec = load_bkv(bkv_sem_idx)

                s_summed = compute_scores(
                    bq_vec,
                    bkv_vec,
                    scale_val_vec,
                    bq_weights_vec,
                    bq_pos_compressed_vec,
                    bkv_idx,
                )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

                bo_sem_idx = sem_ids_ref[2]
                wait_send_scores(bo_sem_idx)
                scores_block_x2_ref[bo_sem_idx, ...] = s_summed.astype(
                    scores_block_x2_ref.dtype).reshape(seq_batch_size * bq_sz,
                                                       num_sublanes_bkv, 128)
                start_send_scores(bo_sem_idx, sz, token_start, bkv_idx)
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(start_seq_idx, 0, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_scores(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
        q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


def streamindex_topk_impl(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, head_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    cfg: policy.Config = policy.Config(),
) -> jax.Array:
    """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    num_kv_pages_per_block: number of kv pages to be processed in one block in
      the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    num_queries_per_block: number of queries to be processed in one block in the
      pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
      kv_len for all sequences in the batch. Defaults to False.
    decode_req_batch_size: maximum decode batch size per iteration.

  Returns:
    Top-K indices (in compressed space).
  """
    # Scale factors for the FP8 index cache format are packed directly inside
    # `cache_kv` along the width dimension, keeping HBM transactions fused.

    # The two eliminations this half owns, read out of `cfg` once so the body
    # below stays the incumbent's text with two `if`s added to it.
    skip_empty_launches = cfg.skip_empty_launches
    elide_score_prefill = cfg.elide_score_prefill

    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )

    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype

    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    q = prepare_q_inputs(q)
    lkv_dim = cache_kv.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        out_dtype,
        case=MlaCase.MIXED,
    ):
        _, num_q_heads, head_dim = q.shape
        # Only support batching for decode sequences.
        # TODO: support batching for decode sequences with speculative decoding
        # enabled, e.g. static_q_len = gamma + 1.
        if seq_batch_size > 1:
            assert static_q_len == 1

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        # If seq_batch_size > 1, caller already guaranteed that
        # end_seq_idx - start_seq_idx % seq_batch_size == 0.
        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(
                memory_space=pltpu.HBM),  # scores_init (aliased to out)
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # scores

        bkv_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bkv_buf_sz_per_kv_packing,
                kv_packing,
                lkv_dim,
            ),
            cache_kv.dtype,
        )
        bq_double_bufq = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
                head_dim,
            ),
            q.dtype,
        )
        bq_weights_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
            ),
            prepared_indexer_weights.dtype,
        )
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), out_dtype)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),  # (bq, bkv, bo) sem indices
            jnp.full((2, ), -1, jnp.int32),  # in-flight out DMA row counts
        )

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
        )

    def _common_path(_):
        out_dtype = jnp.float32
        score_shape = (q.shape[0], num_sublanes_total, 128)

        if elide_score_prefill:
            # O1, item 3. No pre-fill: `lax.empty` lowers to an
            # `AllocateBuffer` custom call, which reserves the buffer and
            # writes nothing. Every column at or past a row's
            # `eff_row_lengths` is now uninitialised, and that is safe only
            # because the selector masks each lane on `j < i_hi`.
            # `sparsecore_topk_impl` asserts that gate before it reads here.
            #
            # The fill this replaces is 28-59% of the block's modelled HBM
            # traffic and 1.7-2.3% of its measured time -- the clearest case
            # in the block that the byte roofline is the wrong instrument.
            scores_init = lax.empty(score_shape, out_dtype)
        else:
            # Pre-fill the output with -inf and alias it, so masked /
            # unwritten columns are already -inf.
            scores_init = jnp.full(score_shape, -jnp.inf, dtype=out_dtype)

        def _maybe_run(scores_in, *, start_seq_idx, end_seq_idx, **kw):
            """One scoring launch, skipped when its sequence range is empty.

            O1, item 2. 42 of the 63 launches a 21-layer decode step makes run
            a zero-extent grid, and a zero-extent grid still pays a full
            Pallas dispatch. The range is traced, so the skip is a `lax.cond`
            and not a Python `if`; the false branch threads the score buffer
            through untouched, so the alias chain still holds.
            """

            def _run(scores_prev):
                return run_scores_kernel(
                    q,
                    prepared_indexer_weights,
                    cache_kv,
                    scores_prev,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    start_seq_idx=start_seq_idx,
                    end_seq_idx=end_seq_idx,
                    **kw,
                )

            if not skip_empty_launches:
                return _run(scores_in)
            return lax.cond(end_seq_idx > start_seq_idx, _run, lambda s: s,
                            scores_in)

        # TODO: we shall sort the sequences by length, so that multiple decode
        # sequences in one batch have similar lengths to reduce waste of compute.
        # With the same batch size, the longest sequence will determine number of
        # blocks to run computation for.
        decode_batch_end = (distribution[0] // decode_req_batch_size *
                            decode_req_batch_size)
        scores = _maybe_run(
            scores_init,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=jnp.array(0),
            end_seq_idx=decode_batch_end,
            static_q_len=1,
            seq_batch_size=decode_req_batch_size,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )
        # Handle num_decode_seqs % decode_req_batch_size != 0 case.
        scores = _maybe_run(
            scores,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[1],
            static_q_len=1,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )

        scores = _maybe_run(
            scores,
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            static_q_len=None,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.MIXED,
        )

        scores = scores.reshape(q.shape[0], -1)
        if scores.shape[1] < k:
            scores = jnp.pad(
                scores,
                ((0, 0), (0, k - scores.shape[1])),
                constant_values=-jnp.inf,
            )

        scores_to_reduce = scores
        if scores_to_reduce.dtype != jnp.float32:
            scores_to_reduce = scores_to_reduce.astype(jnp.float32)

        eff_row_lengths = _effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=q.shape[0],
            num_positions=scores_to_reduce.shape[1],
            compression_ratio=compression_ratio,
        )
        topk_idxs = sparsecore_topk_impl(
            scores_to_reduce,
            k,
            row_lengths=eff_row_lengths,
            write_empty_rows=True,
            cfg=cfg,
        )

        return topk_idxs[:q.shape[0], :k]

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        seq_idx = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            q_abs_pos // compression_ratio,
        )
        max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        _fast_path,
        _common_path,
        None,
    )
