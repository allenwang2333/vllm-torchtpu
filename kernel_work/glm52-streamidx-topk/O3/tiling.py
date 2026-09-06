"""O3 partition arithmetic: ints only, no jax, no device -- so it unit-tests on a laptop."""

# Dimensions.
#   B        rows of the score matrix = padded token count `T`
#   N        padded score extent per row (shapes.padded_n(), NOT pages*page_size)
#   K        top-k
#   tn       slice width in words: the tile extent of N one subcore holds resident
#   p        number of N-tiles covered this call. code-style spells a grid extent
#            `nn`, but every document in this exercise calls it `p`, so `p` it is.
#   P_MAX    the capacity ceiling on p: what the incumbent picks from static N
#   n_sl     number of (row, tile) slices = B * p
#   nw       wave count = cdiv(n_sl, slices_per_wave)


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def slice_words_budget(sc_vmem_capacity_bytes: int) -> int:
    """Largest resident slice in 32-bit words, from the platform's subcore VMEM.

    Half the subcore tile memory, in words. The other half carries the 16
    lane-private histograms, the folded histogram, the row buffer and the
    lengths vector -- ~26 KiB at K=2048, so half is generous and matches what
    the incumbent hardcodes as MAX_SLICE_WORDS = 65536 on a 512 KiB subcore.
    Derived rather than written down because a chip with a different subcore
    VMEM changes `p`, and a stale constant would change it silently
    (code-style.md 13).
    """
    return sc_vmem_capacity_bytes // 2 // 4


def static_partition(B: int, N: int, K: int, *, max_slice_words: int,
                     lanes: int, slices_per_wave: int) -> int:
    """The incumbent's `_pick_partition`, reproduced exactly.

    O3 uses it as the CEILING on p, not as the choice. Reproducing it -- growth
    branch and all -- is deliberate: O1 is the variant that fixes the growth
    branch, and if O3 also removed it the two would not be separable in step 4.
    """
    p_min = cdiv(N, max_slice_words)
    p = 1
    while p < p_min:
        p *= 2

    max_p = slices_per_wave // B if B < slices_per_wave else 1
    while p * 2 <= max_p:
        next_np = N // (p * 2)
        if next_np >= K and next_np >= lanes and (N % (p * 2)) == 0:
            p *= 2
        else:
            break
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
    in range, drawn from a truncated candidate set. That is area 7's
    silent-failure hazard (axis 14, model step): a table keyed on a length
    bucket fails loudly in the growing direction and silently in the shrinking
    one. `dec-ragged` in the correctness instrument, whose L=2049 row sits one
    word above k=2048, is where it would show up first.

    Implemented as a count of entries that fall short, so the traced form is a
    sum of comparisons rather than a search.
    """
    return sum(1 for p in ladder[:-1] if p * tn < live_max)


def num_waves(n_sl: int, slices_per_wave: int) -> int:
    return cdiv(n_sl, slices_per_wave)


def x_crit(B: int, p: int, tn: int, slices_per_wave: int) -> int:
    """M2's `X_crit` = sum over waves of the longest walked slice in that wave.

    Every slice in flight is `tn` words wide and full-length rows walk all of
    it, so with no per-row information this is `waves * tn`. It is the quantity
    O3 moves: p falls, n_sl falls, waves fall, X_crit falls with them. Shorter
    ROWS do not appear here, which is exactly why E3 lost.
    """
    return num_waves(B * p, slices_per_wave) * tn
