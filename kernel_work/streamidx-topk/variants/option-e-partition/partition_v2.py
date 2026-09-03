"""Option E -- a cost-aware replacement for `sparsecore_topk._pick_partition`.

Nothing in `src/` is edited.  This module *imports* the incumbent kernel
(`_sc_topk_direct`) unchanged and re-implements only the two things around it:
the partition rule, and the two-stage wrapper that consumes it.  The incumbent
therefore stays measurable as a candidate in its own right.

--------------------------------------------------------------------------
What `_pick_partition` is actually for
--------------------------------------------------------------------------
Read off `sparsecore_topk.py:360-377`, it does two unrelated jobs in one loop
and only the first is a constraint:

1. **Capacity (`p_min`, mandatory).**  `_sc_topk_direct` allocates a resident
   VMEM scratch of `slice_len = align_to(n, 16)` int32 words per subcore
   (`sparsecore_topk.py:324, 347`) and DMAs the whole slice into it
   (`:160`).  SparseCore tile memory is 512 KiB and the module reserves at most
   `MAX_SLICE_WORDS = 64*1024` words (256 KiB) of it for that slice.  So
   `p >= ceil(n / 65536)`, rounded up to a power of two, or the kernel cannot
   be built.  **This half is not negotiable.**

2. **Occupancy (`max_p = 32 // b`, discretionary).**  The mesh is 2 cores x 16
   subcores = 32 subcores and one *wave* processes 32 rows
   (`:148, num_waves = cdiv(b, 32)`).  When `b < 32` most subcores idle in the
   single wave, so splitting each row into `p` column slices manufactures
   `b*p <= 32` rows and fills them.  The docstring's "maximize subcore
   utilization" is this clause.  It is a **performance heuristic with no
   correctness content**: the loop is entered only when `p_min` is already
   satisfied, and every `p` it can return is a legal kernel.

The exactness argument for `p > 1` is independent of both and is a property of
the wrapper, not of the choice of `p`: stage 1 returns each slice's local
top-`k_p` with `k_p = min(k, n_p)`, and the global top-`k` can draw at most
`min(k, n_p)` entries from any one slice, so the `p * k_p` candidates are a
superset of the answer.  Lowering `p` (the only thing this module ever does --
see the invariant below) cannot break that, and `p = 1` is the degenerate case
where the candidate set is the input.

--------------------------------------------------------------------------
Why the occupancy clause loses at the production decode shape
--------------------------------------------------------------------------
`p > 1` buys parallelism the shape did not need and pays for it three times:
a second `_sc_topk_direct` launch, a stage-2 scan over `p * k_p` candidates,
and two `jnp.take_along_axis` gathers of `b x p*k_p` on the critical path.  At
`b=8, n=10240, k=2048` the incumbent picks `p=4`, and stage 2 alone rescans
`4 * 2048 = 8192` candidates -- as much work as the `eff_len = 8704` row it was
supposed to be splitting up.  Measured device time (probes/stage-b-ksweep.md
Sect 5): 50.09 us at `p=4` against 17.22 us at `p=1`.

--------------------------------------------------------------------------
The rule
--------------------------------------------------------------------------
Keep the incumbent's feasible set exactly, and add one predicate: raise `p`
above `p_min` only when the modelled cost actually falls.  The model is the
measured stage-B cost model from `probes/stage-b-ksweep.md` Sect 3 (r^2 = 0.9982
over 35 single-stage points, waves 1-32, n 2048-65536, eff_len 1024-10240) plus
a two-stage glue term measured by `probe_option_e.py --arm glue`.

**Invariant: `p_min <= pick_partition(...) <= _pick_partition(...)`.**  The
feasible chain is the incumbent's own; this only ever stops walking it earlier.
So no shape gains a partition it did not already have, and no shape loses the
capacity partition it needs.

`eff_len` is a runtime value (`row_lengths`) and the partition is a trace-time
decision, so the model is evaluated at `eff_len = n`, the upper bound.  That is
the assumption *most* favourable to partitioning -- a full row is the most work
there is to split -- so the rule is conservative in the direction that matters:
where it declines to partition, it would decline at every shorter row too.
"""

from __future__ import annotations

LANES = 16
NUM_SUBCORES = 32
MAX_SLICE_WORDS = 64 * 1024

# ---------------------------------------------------------------------------
# Measured constants.  Every one of these is a hardware measurement on tpu7x,
# not a fit to a single point and not a guess.  Sources are named so a later
# reader can re-derive them from the committed JSON rather than from prose.
# ---------------------------------------------------------------------------

# probes/stage-b-ksweep.md Sect 3, jobs j-245d71f0 and j-7a694535:
#   t_device(us) = 8.152 + waves * (0.702 + 0.05424e-3*n + 1.01975e-3*eff_len)
T_CALL_US = 8.152  # per-launch fixed
T_WAVE_US = 0.702  # per-wave fixed
T_DMA_US_PER_WORD = 0.05424e-3  # resident-slice DMA, tracks n
T_SCAN_US_PER_ELEM = 1.01975e-3  # the length-bounded scan, tracks eff_len

# The XLA glue the two-stage path adds on top of its two kernel launches: the
# offset arithmetic, and two `jnp.take_along_axis` gathers of b x (p*k_p).
# Measured by `probe_option_e.py --arm glue` (see RESULTS.md); the default here
# is deliberately the *low* end of what was measured, so the rule under-charges
# the two-stage path rather than rejecting it on an inflated constant.
GLUE_FIXED_US = 16.0
GLUE_US_PER_ELEM = 3.5e-5  # per element of one b x (p*k_p) gather


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _stage_us(rows: int, cols: int, eff: int) -> float:
    """Modelled device time of one `_sc_topk_direct` launch."""
    waves = _cdiv(rows, NUM_SUBCORES)
    return T_CALL_US + waves * (T_WAVE_US + T_DMA_US_PER_WORD * cols +
                                T_SCAN_US_PER_ELEM * eff)


def modelled_us(b: int, n: int, k: int, p: int) -> float:
    """Modelled device time of the whole public call at partition `p`.

    MODELLED, not measured.  Used only to rank candidate `p` against each
    other at trace time; never report it as a timing.
    """
    if p == 1:
        return _stage_us(b, n, n)
    n_p = n // p
    k_p = min(k, n_p)
    cand = p * k_p
    glue = GLUE_FIXED_US + GLUE_US_PER_ELEM * (b * cand)
    return _stage_us(b * p, n_p, n_p) + _stage_us(b, cand, cand) + glue


def feasible_partitions(b: int, n: int, k: int) -> list[int]:
    """The incumbent's own feasible chain, `p_min` first.

    Mirrors `sparsecore_topk._pick_partition` exactly, except that it collects
    the chain instead of stopping at its end.  The guards are the incumbent's:
    `n // (2p) >= k` (so `k_p == k` and the candidate set stays a superset),
    `>= LANES`, `n` divisible by `2p`, and `b * p <= 32`.
    """
    p_min = _cdiv(n, MAX_SLICE_WORDS)
    p = 1
    while p < p_min:
        p *= 2
    chain = [p]
    max_p = NUM_SUBCORES // b if b < NUM_SUBCORES else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= LANES and (n % (p * 2)) == 0:
            p *= 2
            chain.append(p)
        else:
            break
    return chain


def pick_partition(b: int, n: int, k: int) -> int:
    """Option E's partition rule.  Drop-in for `_pick_partition`."""
    chain = feasible_partitions(b, n, k)
    return min(chain, key=lambda p: (modelled_us(b, n, k, p), p))


# ---------------------------------------------------------------------------
# The wrapper.  A copy of `sparsecore_topk.sparsecore_topk`'s body with the
# partition rule swapped and `p` optionally forced, so the probe can measure
# any point of the feasible chain rather than only the one a rule picks.
# ---------------------------------------------------------------------------


def _topk_at_p(scores, k, row_lengths, p, write_empty_rows):
    """The incumbent's two-stage body, at an explicit `p`.

    Verbatim from `sparsecore_topk.py:411-457` apart from `p` being a
    parameter.  `_sc_topk_direct` is the unmodified kernel from `src/`.
    """
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sc

    if p == 1:
        return _sc._sc_topk_direct(scores,
                                   k,
                                   row_lengths,
                                   write_empty=write_empty_rows)

    b, n = scores.shape
    n_p = n // p
    k_p = min(k, n_p)
    scores_p = scores.reshape(b * p, n_p)

    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(row_lengths[:, None] - offsets[None, :], 0,
                         n_p).reshape(b * p)

    local_indices = _sc._sc_topk_direct(scores_p,
                                        k_p,
                                        lengths_p,
                                        write_empty=True).reshape(b, p, k_p)

    global_cand_indices = jnp.where(
        local_indices < 0,
        -1,
        offsets[None, :, None] + local_indices,
    ).reshape(b, p * k_p)

    safe_cand_indices = jnp.maximum(global_cand_indices, 0)
    cand_scores = jnp.take_along_axis(scores, safe_cand_indices, axis=1)
    cand_scores = jnp.where(global_cand_indices >= 0, cand_scores, -jnp.inf)

    cand_lengths = jnp.where(row_lengths > 0, jnp.int32(p * k_p), jnp.int32(0))

    final_cand_slots = _sc._sc_topk_direct(cand_scores,
                                           k,
                                           cand_lengths,
                                           write_empty=write_empty_rows)

    safe_slots = jnp.maximum(final_cand_slots, 0)
    final_indices = jnp.take_along_axis(global_cand_indices,
                                        safe_slots,
                                        axis=1)
    return jnp.where(final_cand_slots >= 0, final_indices, -1)


def _validate(scores, k):
    import jax.numpy as jnp
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    _, n = scores.shape
    if n % LANES != 0:
        raise ValueError(f"{n=} must be a multiple of {LANES}")
    if k % LANES != 0 or not 0 < k <= 4096:
        raise ValueError(
            f"{k=} must be a positive multiple of {LANES}, at most 4096")


def _jitted(p: int, write_empty_rows: bool):
    """One cached `jax.jit` per (p, write_empty_rows).

    Cached deliberately: a fresh `functools.partial` on every call is a fresh
    cache key, so an uncached version would recompile the Pallas kernel on
    every invocation and every timing taken through it would be a compile.
    """
    import functools

    import jax

    key = (p, bool(write_empty_rows))
    hit = _JIT_CACHE.get(key)
    if hit is None:
        fn = functools.partial(_topk_at_p,
                               p=p,
                               write_empty_rows=bool(write_empty_rows))
        hit = jax.jit(fn, static_argnums=(1, ))
        _JIT_CACHE[key] = hit
    return hit


_JIT_CACHE: dict = {}


def sparsecore_topk_v2(scores,
                       k: int,
                       row_lengths=None,
                       *,
                       write_empty_rows: bool = True):
    """Option E's public entry.  Same contract as `sparsecore_topk`."""
    import jax.numpy as jnp

    _validate(scores, k)
    b, n = scores.shape
    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)
    p = pick_partition(b, n, k)
    return _jitted(p, write_empty_rows)(scores, k, row_lengths)


def sparsecore_topk_forced(scores,
                           k: int,
                           row_lengths=None,
                           *,
                           p: int = 1,
                           write_empty_rows: bool = True):
    """Same, at an explicit `p`.  For the probe; not a shipping entry point."""
    import jax.numpy as jnp

    _validate(scores, k)
    b, n = scores.shape
    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    else:
        row_lengths = row_lengths.astype(jnp.int32)
    return _jitted(p, write_empty_rows)(scores, k, row_lengths)
