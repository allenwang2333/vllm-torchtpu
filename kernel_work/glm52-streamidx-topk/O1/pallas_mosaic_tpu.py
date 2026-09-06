"""Policy for the O1 streamindex top-k: what the hardware is, and what to skip.
"""

# Dimensions.
#   B          rows handed to the selector; B == T, the padded token count
#   N          padded score extent per row
#   K          selected count per row (2048 on GLM-5.2)
#   P          SparseCore column partition; B*P slices, nw waves of them
#
# No `pl.pallas_call` in this file (code-style.md §10). It reads the chip once
# (§13), turns that into the integer arguments `tiling.py` needs, and decides
# which of O1's three eliminations are on. `pallas_mosaic_tpu_kernel.py` uses
# the answers and does not re-derive them.

import functools

import pydantic
from jax.experimental.pallas import tpu as pltpu

import tiling


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """Which of O1's three fixed costs to stop paying.

    All three default on. They are separate fields, and separate `static_argnames`
    at the API boundary, for one reason: the effect of the smallest of them
    (`elide_score_prefill`, modelled at 1.7-2.3% of the op) is below this
    block's 3-9% run-order drift, so it can only be resolved by timing arms
    adjacently in one process. Fusing the three into a single `enable_o1` flag
    would make that measurement impossible.

    None of the three changes what the op computes. The selected index set is
    byte-identical to the incumbent's on every shape where the incumbent is
    already correct; that is the variant's contract and `run_correctness.py`
    checks it as a set difference, not only as a tolerance.
    """

    # `_pick_partition` refuses to raise P unless doing so lowers the wave
    # count. It never does, so P collapses to the capacity minimum.
    fix_growth: bool = True
    # Each of the three scoring launches runs under a `lax.cond` on its own
    # sequence range being non-empty.
    skip_empty_launches: bool = True
    # The `-inf` pre-fill of the score buffer becomes `lax.empty`. Legal only
    # while the selector masks every lane on `j < i_hi`; the kernel file
    # asserts that gate before it relies on this.
    elide_score_prefill: bool = True


def _sparse_core_info():
    """The chip's SparseCore geometry, or `NotImplementedError` if it has none."""
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError(
            "streamindex_topk's selector runs on SparseCore; this chip reports "
            "none. Fall back to lax.top_k.")
    return sc


def supported_on() -> bool:
    """Whether this backend can serve the op at all on the current chip."""
    try:
        _sparse_core_info()
    except NotImplementedError:
        return False
    return True


def _get_heuristics_config(*, num_decode_seqs: int | None = None,
                           num_seqs: int | None = None) -> Config:
    """The default O1 config.

    There is nothing to tune. Each of the three eliminations is either a
    correctness-neutral removal of work that was never used, or -- in
    `fix_growth`'s case -- the restoration of a guard whose absence is a bug in
    the incumbent's own stated intent. So the heuristic is "all three on", and
    the fields exist to be turned *off*, by the microbenchmark, one at a time.

    The sequence counts are accepted and ignored on purpose: the launch
    emptiness that `skip_empty_launches` exploits is not known at trace time.
    `distribution` is a device array, so the predicate has to be evaluated on
    device inside a `lax.cond`. What is static is the *rule*, and it lives in
    `tiling.launch_ranges` where it can be unit-tested without a chip. Passing
    them here and dropping them would be a lie; the parameters are kept only so
    a future caller with static counts has somewhere to put them, and if you are
    that caller, note that specializing the launches statically is a different
    variant, not this one.
    """
    del num_decode_seqs, num_seqs  # see docstring
    return Config()


@functools.lru_cache(maxsize=None)
def _hardware_partition_args() -> dict[str, int]:
    """The three chip numbers `tiling.pick_partition` needs, read once.

    Cached because `_pick_partition` is called on every trace and
    `get_tpu_info()` is not free. Every one of these was a literal in the
    incumbent -- `LANES = 16`, an inline `32`, `MAX_SLICE_WORDS = 64 * 1024` --
    and all three are tpu7x facts rather than algorithm facts.
    """
    sc = _sparse_core_info()
    return dict(
        lanes=sc.num_lanes,
        subcores_per_wave=sc.num_cores * sc.num_subcores,
        sc_vmem_capacity_bytes=sc.vmem_capacity_bytes,
    )


def _pick_partition(b: int, n: int, k: int, *, fix_growth: bool = True) -> int:
    """The column partition P for a [B, N] top-K, 1..subcores_per_wave.

    A thin binding of `tiling.pick_partition` to this chip. The arithmetic --
    including the guard `fix_growth` adds -- is in `tiling.py` so it can be
    tested against a table of (B, N, K, chip) without a device.
    """
    return tiling.pick_partition(b, n, k, fix_growth=fix_growth,
                                 **_hardware_partition_args())
