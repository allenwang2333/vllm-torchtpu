"""Integer geometry for the composed streamindex top-k. Ints in, ints out.

No `jax`, no refs, no device, so every rule here is testable without a TPU.
Hardware quantities appear only as *parameters*; they are read from
`pltpu.get_tpu_info()` in `pallas_mosaic_tpu.py` and passed down
(`code-style.md` S13).

This file is the union of the three variants' arithmetic that survived step 3,
with each rule attributed to the option that owns it, because step 4's whole
job is to keep them separable:

  `pick_partition(..., fix_growth=)`      O1  -- the growth guard
  `partition_ladder` / `ladder_index`     O3  -- the runtime wave packing
  `padded_n` / `scratch_bytes` / ...      O2  -- the KV tile ledger
  `launch_ranges` / `num_nonempty_...`    O1  -- the empty-launch rule

# Dimensions.
#   B          rows handed to the selector; B == T, the padded token count
#   N          padded score extent per row (`padded_n`, NOT pages*page_size)
#   K          selected count per row (2048 on GLM-5.2)
#   P          SparseCore column partition; there are B*P slices in flight
#   P_MAX      the capacity/policy ceiling on P, from static N
#   tn         slice width in words = N // P_MAX, one subcore's resident tile
#   n_sl       slice count = B * P
#   nw         wave count = cdiv(n_sl, subcores_per_wave)
#   psz        page_size in compressed KV positions (1024 at (A), 512 at (B))
#   bkv_p      KV pages per grid step; bkv_sz = bkv_p * psz positions
#   bq_sz      queries per grid step
#   sbs        seq_batch_size, decode sequences fused into one grid step
"""

# Native TensorCore lane width. Literal because it is a shape fact that appears
# in shape comments, which code-style.md S13 exempts; it is not a capacity
# constant. The SparseCore lane count is NOT literal anywhere -- it is read off
# the platform and threaded through as `lanes`.
LANES_TC = 128


def cdiv(a: int, b: int) -> int:
    assert b != 0
    return -(-a // b)


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


# ---------------------------------------------------------------------------
# SparseCore selection geometry.
# ---------------------------------------------------------------------------


def max_slice_words(sc_vmem_capacity_bytes: int) -> int:
    """Largest per-subcore resident slice, in 32-bit words.

    Half the subcore's tile memory: the histograms, the row buffer and the
    length vector need the other half. On tpu7x's 512 KiB subcore this is
    65536, the same number `sparsecore_topk.py:45` hardcodes as
    `MAX_SLICE_WORDS`, so the derived form reproduces the incumbent today and
    tracks the chip if it changes.
    """
    return sc_vmem_capacity_bytes // 2 // 4


def num_waves(b: int, p: int, subcores_per_wave: int) -> int:
    """`nw`: how many rounds of `subcores_per_wave` slices the selector runs.

    This is the only quantity `p` can move that costs time. A slice is walked
    to `min(tn, eff_len)` words and a wave's cost is its *longest* slice, so
    narrowing slices without lowering `nw` buys nothing.
    """
    return cdiv(b * p, subcores_per_wave)


def pick_partition(b: int, n: int, k: int, *, lanes: int,
                   subcores_per_wave: int, sc_vmem_capacity_bytes: int,
                   fix_growth: bool = True) -> int:
    """The power-of-two column partition ceiling P_MAX, 1..subcores_per_wave.

    Two loops with different jobs. The first is a *capacity* constraint: a
    slice must fit one subcore's tile memory, so `P >= cdiv(N, max_slice)`.
    There is no choice in it.

    The second is the incumbent's tuning loop, and it is where O1 changes the
    answer. The incumbent raises P whenever the resulting slice is still legal
    and never asks whether raising it *buys* anything. It does not: the
    selector's cost is `nw` rounds of the longest slice in each round, so P
    pays off only when it lowers `nw`.

    `fix_growth=False` reproduces the incumbent's `_pick_partition` exactly and
    exists so the two can be timed adjacently in one process. Under
    `fix_growth=True` the guard fires at (A) with B=16 -- P falls 2 -> 1, which
    deletes a whole `pl.kernel` dispatch and both stage-2 gathers.
    """
    p = 1
    p_min = cdiv(n, max_slice_words(sc_vmem_capacity_bytes))
    while p < p_min:
        p *= 2

    max_p = subcores_per_wave // b if b < subcores_per_wave else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np < k or next_np < lanes or (n % (p * 2)) != 0:
            break
        if fix_growth and (num_waves(b, p * 2, subcores_per_wave)
                           >= num_waves(b, p, subcores_per_wave)):
            break
        p *= 2
    return p


def partition_ladder(p_max: int) -> tuple[int, ...]:
    """The precompiled ladder (1, 2, 4, ... p_max). One `lax.switch` branch each."""
    if p_max < 1 or (p_max & (p_max - 1)) != 0:
        raise ValueError(f"p_max must be a power of two, got {p_max}")
    out, p = [], 1
    while p <= p_max:
        out.append(p)
        p *= 2
    return tuple(out)


def ladder_index(live_max: int, tn: int, ladder: tuple[int, ...]) -> int:
    """Index of the smallest ladder entry whose coverage `p*tn` reaches `live_max`.

    BUCKET UPWARD. The direction is the whole correctness argument and it is
    asymmetric: rounding up covers columns that are dead, which costs a walk
    the wave was paying anyway; rounding down drops columns that are LIVE, and
    the selector has no way to notice -- it returns k indices, all unique, all
    in range, drawn from a truncated candidate set.

    Implemented as a count of entries that fall short, so the traced form is a
    sum of comparisons rather than a search. The device spelling in
    `pallas_mosaic_tpu_kernel.select` must agree with this one; it is written
    here so it can be tested without a chip.
    """
    return sum(1 for p in ladder[:-1] if p * tn < live_max)


def x_crit(b: int, p: int, tn: int, subcores_per_wave: int) -> int:
    """Sum over waves of the longest walked slice in that wave.

    The quantity the wave packing moves: p falls, n_sl falls, nw falls, and
    this falls with them. Shorter ROWS do not appear here, which is why
    narrowing slices at a fixed wave count buys nothing.
    """
    return num_waves(b, p, subcores_per_wave) * tn


# ---------------------------------------------------------------------------
# TensorCore scoring geometry (the O2 half).
# ---------------------------------------------------------------------------


def padded_n(page_size: int, pages_per_seq: int, bkv_ps) -> int:
    """The score-row width the launcher allocates, over ALL THREE tile slots.

    `streamindex_topk.py:603-605` takes the max over the whole
    `num_kv_pages_per_block` tuple, including the PREFILL slot the launcher
    never uses. A slot raised for a case that never runs still widens the score
    matrix -- which is the (A) trap the regime split exists to avoid: at
    pages_per_seq=9, align_to(9, 16) = 16 pages against 9 live, and N_pad goes
    10240 -> 16384, a 60% wider extent into a SparseCore whose cost is driven
    by that extent.
    """
    return max(align_to(pages_per_seq, bkv_p) * page_size for bkv_p in bkv_ps)


def kv_iterations(kv_len: int, page_size: int, bkv_p: int, q_len: int,
                  bq_sz: int) -> int:
    """In-body iterations for one sequence.

    `measured (j-69ee326d)`: 0.500 us per iteration at B-D1-1M, linear to
    +/-0.8% over an 8x range, with copies, bytes, bytes/copy and the score
    extent all held constant.
    """
    return max(1, cdiv(q_len, bq_sz)) * max(1, cdiv(kv_len, bkv_p * page_size))


def scratch_bytes(*, sbs, bq_sz, bkv_p, page_size, kv_packing, lkv_dim,
                  num_q_heads, head_dim, q_itemsize, w_itemsize,
                  score_itemsize):
    """The four DECLARED double-buffered scratch arrays, in bytes."""
    page_size_per_kv_packing = page_size // kv_packing
    return dict(
        # bkv_x2_ref [2, sbs, bkv_p*PSZ/kv_packing, kv_packing, lkv_dim] u8
        bkv=2 * sbs * bkv_p * page_size_per_kv_packing * kv_packing * lkv_dim,
        # bq_x2_ref [2, sbs, tq, H, D] fp8
        bq=2 * sbs * bq_sz * num_q_heads * head_dim * q_itemsize,
        # bq_weights_x2_ref [2, sbs, tq, H] f32
        bqw=2 * sbs * bq_sz * num_q_heads * w_itemsize,
        # scores_block_x2_ref [2, sbs*tq, bkv_sz/128, 128] f32
        bo=2 * sbs * bq_sz * bkv_p * page_size * score_itemsize,
    )


def einsum_temp_bytes(*, sbs, bq_sz, bkv_p, page_size, num_q_heads):
    """The two UNDECLARED f32 intermediates inside `compute_scores`.

        s       [sbs, tq*H, bkv_sz] f32   the einsum output, before the head sum
        s_summed[sbs, tq,   bkv_sz] f32   after the sum over H

    AN UPPER BOUND, NOT AN OCCUPANCY, and reported rather than charged.
    `measured (j-5136b90a-9f9e-4144-b066)`: on a chip whose
    `vmem_capacity_bytes` is 64.0 MiB, tiles priced here at 289.25 MiB and
    74.62 MiB both compiled and produced correct output, so Mosaic tiles the
    einsum internally and this quantity never all exists at once. It is kept
    because the ratio says which tiles would fail first if it ever stopped.
    """
    bkv_sz = bkv_p * page_size
    return dict(
        s=sbs * bq_sz * num_q_heads * bkv_sz * 4,
        s_summed=sbs * bq_sz * bkv_sz * 4,
    )


def vmem_bytes(**kw):
    """Total VMEM a scoring launch would need: declared scratch + einsum temps."""
    parts = dict(scratch_bytes(**kw))
    parts.update(einsum_temp_bytes(
        sbs=kw["sbs"], bq_sz=kw["bq_sz"], bkv_p=kw["bkv_p"],
        page_size=kw["page_size"], num_q_heads=kw["num_q_heads"]))
    parts["total"] = sum(parts.values())
    return parts


def is_legal_tile(page_size: int, bkv_p: int) -> bool:
    """streamindex_topk.py:596-601: bkv_sz must be a multiple of 128."""
    return (page_size * bkv_p) % LANES_TC == 0


# ---------------------------------------------------------------------------
# Launch emptiness (the O1 `skip_empty_launches` rule).
# ---------------------------------------------------------------------------


def launch_ranges(num_decode_seqs: int, num_decode_and_prefill_seqs: int,
                  num_seqs: int, decode_req_batch_size: int):
    """The `[start, end)` sequence range each of the three scoring launches gets.

    Ints in, ints out, so the emptiness accounting can be asserted in a unit
    test. At trace time `distribution` is a device array and the same rule is
    evaluated on device -- `end > start` inside a `lax.cond` -- which is why
    the kernel file cannot simply call this.
    """
    decode_batch_end = (num_decode_seqs // decode_req_batch_size *
                        decode_req_batch_size)
    return ((0, decode_batch_end),
            (decode_batch_end, num_decode_and_prefill_seqs),
            (num_decode_and_prefill_seqs, num_seqs))


def num_nonempty_launches(*args) -> int:
    """How many of `launch_ranges`' three launches do any work.

    Never 3 on a GLM-5.2 step. The incumbent dispatches all three regardless,
    and a zero-extent Pallas grid still pays a full dispatch.
    """
    return sum(1 for start, end in launch_ranges(*args) if end > start)
