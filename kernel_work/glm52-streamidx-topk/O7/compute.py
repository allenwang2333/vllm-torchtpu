"""Pure array algebra for the SparseCore radix top-k. Arrays in, arrays out.

No refs, no pallas primitives, no config -- everything here is testable on CPU
with plain jnp arrays, which is the point of keeping it separate (code-style
"compute.py: arrays only").

Dimension glossary
  L   lanes in one SparseCore vector register (16 on tpu7x)
  W   words in a resident slice (`slice_len`)
  NB  buckets in one radix level's histogram
"""
import jax.numpy as jnp
import numpy as np

# The order-preserving f32 -> i32 map. For a non-negative float the sign bit is
# 0 and the bit pattern already orders correctly; for a negative float the whole
# magnitude must be reversed, which xor-ing with 0x7fffffff does.
MONO_MASK = 0x7FFFFFFF

# `monotone_key(bitcast(-inf))`. Every key at or below this is "not a score":
# a padded column, a masked position, or a genuine -inf.
KEY_NEG_INF = int(
    np.int32(np.array(-np.inf, np.float32).view(np.int32))
    ^ np.int32(MONO_MASK))


def monotone_key(bits):  # [L] i32 -> [L] i32
    """Map f32 bit patterns to i32 keys that compare in float order."""
    m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
    return jnp.bitwise_xor(bits, m)


def top_digit(key, shift, half):  # [L] i32 -> [L] i32
    """The level-0 bucket: the signed top byte, biased into [0, 2*half)."""
    return jnp.right_shift(key, shift) + jnp.int32(half)


def digit(key, shift, mask):  # [L] i32 -> [L] i32
    """One refinement digit of `key`."""
    return jnp.bitwise_and(jnp.right_shift(key, shift), jnp.int32(mask))


def descend(prefix, shift, c_hi, quota, cnt, done, *, b2, sub_total, cum_at2,
            cum_lo2, lvl_shift):
    """One level of the radix descent, as scalars.

    Given the histogram statistics of the current tie group, refine the prefix
    to the bucket that straddles the cut and re-split the quota. `done` latches:
    once the tie group is exactly the quota (take all) or the quota is zero
    (take none), no further level can change the answer, and every subsequent
    level is skipped by zeroing its loop bounds rather than by branching.

    Returns the same six scalars, advanced.
    """
    c_above2 = sub_total - cum_at2
    cnt2 = cum_at2 - cum_lo2
    quota2 = quota - c_above2

    width = shift - lvl_shift
    new_prefix = jnp.left_shift(prefix, width) | b2
    new_done = done | (quota2 == cnt2) | (quota2 == 0)

    return (
        jnp.where(done, prefix, new_prefix),
        jnp.where(done, shift, jnp.int32(lvl_shift)),
        jnp.where(done, c_hi, c_hi + c_above2),
        jnp.where(done, quota, quota2),
        jnp.where(done, cnt, cnt2),
        new_done,
    )


def refinement_levels(l0_shift, digit_width):
    """The (shift, mask) ladder below the level-0 digit.

    At the incumbent's `l0_shift=24, digit_width=8` this is the incumbent's own
    three levels, ((16, 0xff), (8, 0xff), (0, 0xff)); `digit_width=4` gives six
    narrower ones. Narrow digits shrink the per-level histogram (and so the
    fixed per-level loop count) at the cost of more levels, which is only a
    trade worth making when each level walks a compacted candidate set rather
    than the whole slice.

    A widened level 0 need not leave a multiple of `digit_width` bits behind, so
    the LAST level is allowed to be narrower than the rest -- its mask says how
    wide it is, and the caller must size that level's histogram from the mask
    rather than from `digit_width`. The ladder always covers bits
    `l0_shift-1 .. 0` exactly once, which is what makes the descent exact.
    """
    levels = []
    shift = l0_shift
    while shift > 0:
        w = min(digit_width, shift)
        shift -= w
        levels.append((shift, (1 << w) - 1))
    return tuple(levels)
