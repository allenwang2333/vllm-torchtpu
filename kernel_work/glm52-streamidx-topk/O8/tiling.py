"""Extent arithmetic for the O8 selector. Ints only -- no jax, no device.

`code-style.md` §2 puts "ints only -- block extents, grid extents" on the
ref-free side of the boundary so it is testable with no device at all. The one
decision that lives here is P, and it is copied unchanged from the shipping
`_pick_partition`: O8 changes what stage 1 *emits*, not how the row is cut. The
copy is asserted equal to the shipping function in `run_selfcheck.py`, because a
silent divergence in P would move both arms at once and look like a result.
"""

LANES = 16
MAX_SLICE_WORDS = 64 * 1024


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def pick_partition(b: int, n: int, k: int) -> int:
    """P: the power-of-two slice count per row. Copy of the shipping policy."""
    p_min = cdiv(n, MAX_SLICE_WORDS)
    p = 1
    while p < p_min:
        p *= 2

    max_p = 32 // b if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p


def stage_extents(b: int, n: int, k: int):
    """(P, NP, KP, C) for a row -- C is what stage 2 actually reads.

    C = P*KP is the size of the candidate array the incumbent gathers into out
    of B*N, and the size of the array O8 gets for free from stage 1.
    """
    p = pick_partition(b, n, k)
    if p == 1:
        return 1, n, k, 0
    n_p = n // p
    k_p = min(k, n_p)
    return p, n_p, k_p, p * k_p
