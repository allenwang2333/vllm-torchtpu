"""O7 kernel body: the SparseCore radix-select inner loop, rewritten.

Four changes over the incumbent (`sparsecore_topk.py`), each behind its own
`Config` flag so it can be landed and measured on its own:

  fold_and_zero    `fold_priv` writes zeros back into the slots it reads, so the
                   lane-private histogram is left clean for the next level and
                   every per-level `zero_priv(256)` disappears. One zeroing pass
                   at kernel entry re-establishes the invariant once.
  scan_count_hist  `plsc.scan_count` pre-aggregates duplicate buckets inside the
                   vector, so one scatter of counts into a single 256-word
                   histogram replaces 16 lane-private histograms of ones. This
                   subsumes fold_and_zero: there is nothing left to fold.
  carry_forward    After level 0, one pass compacts the slice: the strictly
                   winning positions go straight to the output buffer and the
                   tie-bucket positions to a candidate buffer. Every later level
                   -- and the emit -- walks the candidates instead of re-walking
                   the slice under a mask.
  fused_convert    Drop the separate f32->monotone-key conversion pass and apply
                   `monotone_key` at every read instead. One fewer full-slice
                   pass, at the cost of two ALU ops per loaded vector.
  l0_shift         How many bits level 0 splits on: 24 is the incumbent's 256
                   buckets, 22 is 1024, 20 is 4096. This is not a change in its
                   own right -- it is what decides how big the level-0 tie
                   bucket is, and therefore how much work `carry_forward` has
                   left to carry. The top byte of a monotone f32 key is sign
                   plus seven exponent bits, so one incumbent bucket spans a
                   factor of four in magnitude and on a concentrated score
                   distribution holds most of the slice.

Dimension glossary
  B   score rows (already multiplied by the partition factor)
  N   score columns per row
  W   words in one subcore's resident slice, `slice_len` = align(N, L)
  L   SparseCore lanes, 16 on tpu7x (read from the platform, not hardcoded)
  K   selected indices per row
  NB  buckets in one radix level's histogram
  C   compacted candidates carried forward past level 0, C <= `cand_cap`

Ordering hazard, declared. `plsc.parallel_loop` documents its iterations as
independent, yet `compact_body` and both emit bodies write a ref at an offset
carried between iterations. This is not new: the incumbent's `emit_body` does
exactly the same thing at `sparsecore_topk.py:269-291` and ships. The candidate
compaction inherits that assumption and adds no new one -- but it is an
assumption, and if `parallel_loop` ever reorders across a carry, both the
incumbent and this variant are wrong together.
"""
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from compute import (KEY_NEG_INF, descend, digit, monotone_key,
                     refinement_levels, top_digit)

# The incumbent's level-0 geometry, and the default. `Config.l0_shift` may widen
# it; `l0_buckets` is the only place the relationship between the two is
# written, so scratch sizing and the kernel cannot disagree about it.
L0_SHIFT = 24


def l0_buckets(l0_shift):
    """Distinct values of a signed 32-bit key shifted right by `l0_shift`."""
    return 1 << (32 - l0_shift)


NUM_L0_BUCKETS = l0_buckets(L0_SHIFT)


def _cdiv_dyn(x, lanes):
    return jnp.right_shift(x + (lanes - 1), 4)


def topk_kernel(
    # Inputs.
    scores_hbm,  # [B*N] i32   (f32 score bits)
    lengths_hbm,  # [B_pad] i32
    # Outputs.
    out_hbm,  # [B*K] i32
    # Scratch.
    resident_vmem,  # [W] i32          one row slice, keys in place
    priv_vmem,  # [L*NB0 + L] i32      lane-private histograms (unused under scan_count)
    glob_vmem,  # [NB0 + L] i32        folded histogram, then its prefix sum
    rowbuf_vmem,  # [K + L] i32        winner indices for one row
    cand_vmem,  # [cand_cap + L] i32   carried-forward tie positions
    len_vmem,  # [B_pad] i32
    sems,  # DMA[4]
    *,
    cfg,
    b: int,
    n: int,
    k: int,
    slice_len: int,
    num_waves: int,
    write_empty: bool,
    lanes: int,
    num_subcores: int,
    num_sc_cores: int,
    cand_cap: int,
):
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    lane_iota = jnp.arange(lanes, dtype=jnp.int32)
    b_pad = len_vmem.shape[0]
    l0_shift = cfg.l0_shift
    nb0 = l0_buckets(l0_shift)
    ref_levels = refinement_levels(l0_shift, cfg.digit_width)
    # The last level may be narrower than the rest (see `refinement_levels`), so
    # each level's histogram is sized from its own mask.
    max_ref_buckets = max(m + 1 for _, m in ref_levels)
    zeros = jnp.zeros((lanes, ), jnp.int32)

    cp = pltpu.make_async_copy(lengths_hbm.at[pl.ds(0, b_pad)], len_vmem,
                               sems.at[0])
    cp.start()
    cp.wait()

    def dma(src, dst, sem_idx):
        cp = pltpu.make_async_copy(src, dst, sems.at[sem_idx])
        cp.start()
        cp.wait()

    def vec_at(ref, idx):
        return ref[pl.ds(idx, lanes)][0]

    def zero_words(ref, chunks):
        """Zero `chunks` vectors at the front of `ref`. `chunks` may be dynamic."""

        def body(i):
            ref[pl.ds(i * lanes, lanes)] = zeros

        plsc.parallel_loop(0, chunks, unroll=4)(body)

    # ---- histogram back end -------------------------------------------------
    # Two spellings of "count keys per bucket". The lane-private one is the
    # incumbent's: 16 private histograms sidestep the fact that a scatter cannot
    # combine duplicate addresses inside one vector. The scan_count one removes
    # that need -- it computes the duplicate multiplicity in-register and
    # scatters one combined count per distinct bucket -- and with it the whole
    # 16x replication, its zeroing and its fold.

    def hist_zero(nb, gate):
        if cfg.scan_count_hist:
            zero_words(glob_vmem, gate * (nb // lanes))
        elif cfg.fold_and_zero:
            pass  # `hist_fold` left it zero; re-established at kernel entry.
        else:
            zero_words(priv_vmem, gate * nb)

    def hist_add(bucket, mask, nb):
        if cfg.scan_count_hist:
            run, last = plsc.scan_count(bucket, mask=mask)
            # `run` is the 1-BASED running occurrence count, so at the final
            # occurrence of a bucket it is exactly that bucket's multiplicity in
            # this vector. Scattering `run + 1` would over-count every duplicate
            # run by one, silently.
            plsc.addupdate_scatter(glob_vmem, (bucket, ), run, mask=last)
        else:
            addr = lane_iota * nb + bucket
            plsc.addupdate_scatter(priv_vmem, (addr, ),
                                   jnp.ones((lanes, ), jnp.int32),
                                   mask=mask)

    def hist_fold(nb, gate):
        if cfg.scan_count_hist:
            return

        def body(i):
            off = i * lanes
            acc = zeros
            for m in range(lanes):
                acc = acc + priv_vmem[pl.ds(m * nb + off, lanes)]
                if cfg.fold_and_zero:
                    priv_vmem[pl.ds(m * nb + off, lanes)] = zeros
            glob_vmem[pl.ds(off, lanes)] = acc

        plsc.parallel_loop(0, gate * (nb // lanes), unroll=2)(body)

    def scan_glob(chunks):

        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * lanes, lanes)]) + carry
            glob_vmem[pl.ds(i * lanes, lanes)] = c
            return c[lanes - 1]

        return plsc.parallel_loop(0, chunks, unroll=2,
                                  carry=jnp.int32(0))(body)

    def find_bucket(chunks, thresh):

        def body(i, cnt):
            below = glob_vmem[pl.ds(i * lanes, lanes)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(0, chunks, unroll=2,
                                  carry=jnp.int32(0))(body)

    # `fold_and_zero` keeps the lane-private histogram clean by construction,
    # but only once it has been clean once. Pallas scratch is not zeroed.
    if cfg.fold_and_zero and not cfg.scan_count_hist:
        zero_words(priv_vmem, max(nb0, max_ref_buckets))

    def wave_body(w, _):
        row = w * (num_sc_cores * num_subcores) + flat
        active = row < b
        row_c = jnp.minimum(row, b - 1)
        eff_len = jnp.minimum(vec_at(len_vmem, row_c), n)
        eff_len = jnp.where(active, eff_len, 0)

        s = row_c * n
        i_hi = jnp.where(eff_len > 0, jnp.minimum(slice_len, eff_len), 0)
        walk_chunks = _cdiv_dyn(i_hi, lanes)

        @pl.when(i_hi > 0)
        def _():
            dma(scores_hbm.at[pl.ds(s, slice_len)], resident_vmem, 1)

            if not cfg.fused_convert:

                def cvt(i):
                    bits = resident_vmem[pl.ds(i * lanes, lanes)]
                    resident_vmem[pl.ds(i * lanes, lanes)] = monotone_key(bits)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(cvt)

        def load_keys(i):
            raw = resident_vmem[pl.ds(i * lanes, lanes)]
            key = monotone_key(raw) if cfg.fused_convert else raw
            j = i * lanes + lane_iota
            valid = (j < i_hi) & (key > jnp.int32(KEY_NEG_INF))
            return key, valid, j

        def gather_keys(i, cnt):
            """Read the i-th vector of carried-forward candidates."""
            pos = cand_vmem[pl.ds(i * lanes, lanes)]
            live = (i * lanes + lane_iota) < cnt
            # Out-of-range lanes hold whatever the previous row left behind, and
            # bounds checks are off -- force them to a legal index rather than
            # relying on the gather's mask to suppress the address.
            raw = plsc.load_gather(resident_vmem,
                                   (jnp.where(live, pos, 0), ))
            key = monotone_key(raw) if cfg.fused_convert else raw
            return key, live, pos

        # ---- level 0: histogram the whole slice -----------------------------
        @pl.when(i_hi > 0)
        def _():
            hist_zero(nb0, 1)

            def hist0_body(i):
                key, valid, _ = load_keys(i)
                bucket = top_digit(key, l0_shift, nb0 // 2)
                hist_add(bucket, valid, nb0)

            plsc.parallel_loop(0, walk_chunks, unroll=4)(hist0_body)
            hist_fold(nb0, 1)

        @pl.when(i_hi == 0)
        def _():
            zero_words(glob_vmem, nb0 // lanes)

        l0_chunks = nb0 // lanes
        total = scan_glob(l0_chunks)
        keep_all = total <= k

        b_star = find_bucket(l0_chunks, total - k)
        cum_at = vec_at(glob_vmem, b_star)
        cum_lo = jnp.where(b_star > 0,
                           vec_at(glob_vmem, jnp.maximum(b_star - 1, 0)), 0)
        c_hi = total - cum_at
        cnt = cum_at - cum_lo
        quota = k - c_hi

        prefix = b_star - jnp.int32(nb0 // 2)
        shift = jnp.int32(l0_shift)
        done = keep_all | (quota == cnt)

        # ---- carry-forward: compact once, then never touch the slice again --
        # The strictly-winning positions are final at this point, so they are
        # written straight into the output buffer; only the tie bucket needs
        # refining, and it is the only thing later levels walk.
        #
        # `cand_cap` bounds the candidate buffer. A tie bucket wider than the cap
        # falls back to the incumbent's masked re-scan for that row: every loop
        # below is bounded by a dynamic count that is zero on the path not
        # taken, so both paths are present and exactly one of them runs.
        if cfg.carry_forward:
            cf = (i_hi > 0) & jnp.logical_not(done) & (cnt <= cand_cap)

            def compact_body(i, carry):
                woff, coff = carry
                key, valid, j = load_keys(i)
                hi = jnp.right_shift(key, l0_shift)
                strict = valid & (hi > prefix)
                tie = valid & (hi == prefix)
                plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)],
                                      j,
                                      mask=strict)
                woff = woff + plsc.all_reduce_population_count(strict)[0]
                plsc.store_compressed(cand_vmem.at[pl.ds(coff, lanes)],
                                      j,
                                      mask=tie)
                coff = coff + plsc.all_reduce_population_count(tie)[0]
                return woff, coff

            cf_woff, cf_cnt = plsc.parallel_loop(
                0,
                jnp.where(cf, walk_chunks, 0),
                unroll=4,
                carry=(jnp.int32(0), jnp.int32(0)),
            )(compact_body)
            cand_chunks = _cdiv_dyn(cf_cnt, lanes)
        else:
            cf = False
            cf_woff = jnp.int32(0)
            cf_cnt = jnp.int32(0)
            cand_chunks = jnp.int32(0)

        # ---- refinement ladder ----------------------------------------------
        for lvl_shift, lvl_mask in ref_levels:
            num_ref_buckets = lvl_mask + 1
            ref_chunks = num_ref_buckets // lanes
            live = jnp.logical_not(done)
            hist_zero(num_ref_buckets, live.astype(jnp.int32))

            def ref_body_full(
                i,
                lvl_shift=lvl_shift,
                lvl_mask=lvl_mask,
                nrb=num_ref_buckets,
                prefix=prefix,
                shift=shift,
            ):
                key, valid, _ = load_keys(i)
                match = valid & (jnp.right_shift(key, shift) == prefix)
                hist_add(digit(key, lvl_shift, lvl_mask), match, nrb)

            if cfg.carry_forward:
                full_bound = jnp.where(done | cf, 0, walk_chunks)
            else:
                full_bound = jnp.where(done, 0, walk_chunks)
            plsc.parallel_loop(0, full_bound, unroll=4)(ref_body_full)

            if cfg.carry_forward:

                def ref_body_cand(
                    i,
                    lvl_shift=lvl_shift,
                    lvl_mask=lvl_mask,
                    prefix=prefix,
                    shift=shift,
                    nrb=num_ref_buckets,
                    cf_cnt=cf_cnt,
                ):
                    key, alive, _ = gather_keys(i, cf_cnt)
                    match = alive & (jnp.right_shift(key, shift) == prefix)
                    hist_add(digit(key, lvl_shift, lvl_mask), match, nrb)

                cand_bound = jnp.where(done, 0,
                                       jnp.where(cf, cand_chunks, 0))
                plsc.parallel_loop(0, cand_bound, unroll=4)(ref_body_cand)

            gate = live.astype(jnp.int32)
            hist_fold(num_ref_buckets, gate)
            fold_chunks = gate * ref_chunks

            sub_total = scan_glob(fold_chunks)
            b2 = find_bucket(fold_chunks, sub_total - quota)
            cum_at2 = vec_at(glob_vmem, b2)
            cum_lo2 = jnp.where(b2 > 0,
                                vec_at(glob_vmem, jnp.maximum(b2 - 1, 0)), 0)

            prefix, shift, c_hi, quota, cnt, done = descend(
                prefix, shift, c_hi, quota, cnt, done,
                b2=b2, sub_total=sub_total, cum_at2=cum_at2,
                cum_lo2=cum_lo2, lvl_shift=lvl_shift)

        my_take = jnp.where(keep_all, 0, quota)
        out_gate = active if write_empty else (active & (i_hi > 0))

        def fill_body(i):
            rowbuf_vmem[pl.ds(i * lanes, lanes)] = jnp.full((lanes, ), -1,
                                                            jnp.int32)

        # A short row is -1-padded before the winners are written. Under
        # carry-forward the compaction has already written into `rowbuf_vmem`,
        # so this must not run then -- and it cannot: `cf` requires `not done`,
        # `done` includes `total <= k`, so `cf` implies `total > k`.
        fill_bound = jnp.where(out_gate & (total < k), k // lanes, 0)
        plsc.parallel_loop(0, fill_bound, unroll=4)(fill_body)

        def emit_full_body(i, carry):
            woff, rank = carry
            key, valid, j = load_keys(i)
            hi = jnp.right_shift(key, shift)
            strict = valid & (hi > prefix)
            tie = valid & (hi == prefix)
            tie_rank = plsc.cumsum(tie.astype(jnp.int32)) + rank
            take = tie & (tie_rank <= my_take)
            sel = jnp.where(keep_all, valid, strict | take)
            plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)],
                                  j,
                                  mask=sel)
            woff = woff + plsc.all_reduce_population_count(sel)[0]
            return woff, tie_rank[lanes - 1]

        if cfg.carry_forward:
            emit_full_bound = jnp.where(active & jnp.logical_not(cf),
                                        walk_chunks, 0)
        else:
            emit_full_bound = jnp.where(active, walk_chunks, 0)
        plsc.parallel_loop(
            0,
            emit_full_bound,
            unroll=4,
            carry=(jnp.int32(0), jnp.int32(0)),
        )(emit_full_body)

        if cfg.carry_forward:
            # Every candidate matched the level-0 prefix, so `strict` here counts
            # only what the refinement promoted; the level-0 strict winners were
            # already written by `compact_body` and `woff` resumes after them.
            # c_hi(final) = c_hi(level 0) + promoted, and the take is the final
            # quota, so the total is k exactly as in the full-slice emit.
            def emit_cand_body(i, carry, cf_cnt=cf_cnt):
                woff, rank = carry
                key, alive, pos = gather_keys(i, cf_cnt)
                hi = jnp.right_shift(key, shift)
                strict = alive & (hi > prefix)
                tie = alive & (hi == prefix)
                tie_rank = plsc.cumsum(tie.astype(jnp.int32)) + rank
                take = tie & (tie_rank <= my_take)
                sel = strict | take
                plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, lanes)],
                                      pos,
                                      mask=sel)
                woff = woff + plsc.all_reduce_population_count(sel)[0]
                return woff, tie_rank[lanes - 1]

            plsc.parallel_loop(
                0,
                jnp.where(cf, cand_chunks, 0),
                unroll=4,
                carry=(cf_woff, jnp.int32(0)),
            )(emit_cand_body)

        @pl.when(out_gate)
        def _():
            dma(rowbuf_vmem.at[pl.ds(0, k)], out_hbm.at[pl.ds(row_c * k, k)],
                2)

        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)
