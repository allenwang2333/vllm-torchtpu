"""Integer geometry for the streamindex top-k block: extents, waves, partition.

Ints in, ints out. No `jax`, no refs, no device -- so every rule below can be
unit-tested on a laptop, which is the point of the file boundary.

Hardware quantities appear here only as *parameters*. They are read from
`pltpu.get_tpu_info()` in `pallas_mosaic_tpu.py` and passed down; nothing in
this file knows what chip it is on. That split is what lets the wave arithmetic
be tested without a TPU while still obeying code-style.md §13.
"""

# Dimensions.
#   B          rows handed to the selector; B == T, the padded token count
#   N          padded score extent per row, `padded_n()` -- NOT pages*page_size
#   K          selected count per row
#   P          SparseCore column partition; there are B*P slices in flight
#   nw         grid extent over waves: nw = cdiv(B*P, subcores_per_wave)
#   ns         grid extent over score sublanes: ns = N // 128


def cdiv(a: int, b: int) -> int:
    assert b != 0
    return -(-a // b)


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def padded_n(pages_per_seq: int, page_size: int, bkv_ps, lanes_tc: int) -> int:
    """The score extent the scoring kernel allocates, in columns.

    `align_to(pages_per_seq, bkv_p) * page_size`, maximised over the three
    per-case `bkv_p` values, because one buffer serves all three launches. At
    (A) -- 9 pages, bkv_p 2 -- this is 10240, 11.1% wider than max_model_len,
    and the SparseCore is driven by this number rather than by the live length.
    """
    return num_sublanes_total(pages_per_seq, page_size, bkv_ps) * lanes_tc


def num_sublanes_total(pages_per_seq: int, page_size: int, bkv_ps) -> int:
    """`ns`: the score buffer's middle extent, in 128-lane sublanes."""
    return max(align_to(pages_per_seq, bkv_p) * page_size // 128
               for bkv_p in bkv_ps)


def launch_ranges(num_decode_seqs: int, num_decode_and_prefill_seqs: int,
                  num_seqs: int, decode_req_batch_size: int):
    """The `[start, end)` sequence range each of the three scoring launches gets.

    The three are: the batched decode launch, the ragged decode remainder, and
    the mixed prefill launch. `distribution` is `(D, D, R)` in the caller's
    spelling; the batched launch is truncated to a whole multiple of
    `decode_req_batch_size` and the remainder picks up what is left.

    Ints in, ints out, so the emptiness accounting can be asserted in a unit
    test. At trace time `distribution` is a device array and the same rule is
    evaluated on device -- `end > start` inside a `lax.cond` -- which is why the
    kernel file cannot simply call this.
    """
    decode_batch_end = (num_decode_seqs // decode_req_batch_size *
                        decode_req_batch_size)
    return ((0, decode_batch_end),
            (decode_batch_end, num_decode_and_prefill_seqs),
            (num_decode_and_prefill_seqs, num_seqs))


def num_nonempty_launches(*args) -> int:
    """How many of `launch_ranges`' three launches do any work.

    Never 3 on a GLM-5.2 step: a pure-decode step leaves the prefill launch
    empty and (unless R is not a multiple of `decode_req_batch_size`) the
    remainder launch too; a pure-prefill step leaves both decode launches
    empty. The incumbent dispatches all three regardless, and a zero-extent
    Pallas grid still pays a full dispatch.
    """
    return sum(1 for start, end in launch_ranges(*args) if end > start)


def num_waves(b: int, p: int, subcores_per_wave: int) -> int:
    """`nw`: how many rounds of `subcores_per_wave` slices the selector runs.

    This is the only quantity `p` can move that costs time. A slice is walked
    to `min(slice_len, eff_len)` words and the wave's cost is its *longest*
    slice, so halving the slice width without halving the wave count buys
    nothing and pays for a second kernel dispatch.
    """
    return cdiv(b * p, subcores_per_wave)


def max_slice_words(sc_vmem_capacity_bytes: int) -> int:
    """Largest per-subcore resident slice, in 32-bit words.

    Half the subcore's tile memory: the histograms, the row buffer and the
    length vector need the other half (~100 KiB of it in practice). Derived
    rather than written down -- on tpu7x's 512 KiB subcore this is 65536, the
    same number `sparsecore_topk.py:45` hardcodes, and on a chip with a
    different tile memory it moves with the hardware instead of silently
    over-committing it.
    """
    return sc_vmem_capacity_bytes // 2 // 4


def pick_partition(b: int, n: int, k: int, *, lanes: int,
                   subcores_per_wave: int, sc_vmem_capacity_bytes: int,
                   fix_growth: bool = True) -> int:
    """The power-of-two column partition `P`, 1..subcores_per_wave.

    Two loops with different jobs. The first is a *capacity* constraint: a
    slice must fit one subcore's tile memory, so `P >= cdiv(N, max_slice)`.
    There is no choice in it.

    The second is the tuning loop, and it is where the incumbent is wrong.
    It raises `P` whenever the resulting slice is still wide enough to be
    legal, and never asks whether raising it *buys* anything. It does not:
    the selector's cost is `nw` rounds of the longest slice in each round, so
    `P` pays off only when it lowers `nw`. At B=16 on tpu7x it raises P to 2
    with `nw` pinned at 1 either way, and the price is a second `pl.kernel`
    dispatch plus two `take_along_axis` gathers over a [16, 10240] row --
    measured at 13.7% of the whole op (j-cefba169), with the selected sets
    byte-identical, so it was pure loss.

    `fix_growth=False` reproduces the incumbent exactly and exists so the two
    can be timed adjacently in one process.
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
        # The guard the incumbent is missing: refuse to raise P unless doing
        # so lowers the wave count.
        #
        # It never does. Doubling P doubles the slice count, so `nw` can only
        # hold level or rise -- which means this guard always fires and P
        # collapses to the capacity minimum `p_min`. The loop is kept rather
        # than deleted so the incumbent's form stays visible next to the one
        # line that disarms it, and so `fix_growth=False` is a one-flag
        # reproduction of the incumbent.
        #
        # What the guard also blocks, stated rather than hidden: at B*P
        # already >= subcores_per_wave, raising P narrows every stage-1 slice
        # without adding a stage-2 dispatch, so it is not obviously a loss --
        # but it lengthens stage 2's own row by the same P*k_p words it
        # removes from stage 1, so it is a wash by M2 rather than a win. It is
        # unreachable at GLM-5.2's token ladder in any case: B >= 16 there, so
        # `max_p` is at most 2 and P has already been raised past it by
        # `p_min` wherever N needs more than one slice.
        if fix_growth and (num_waves(b, p * 2, subcores_per_wave)
                           >= num_waves(b, p, subcores_per_wave)):
            break
        p *= 2
    return p
