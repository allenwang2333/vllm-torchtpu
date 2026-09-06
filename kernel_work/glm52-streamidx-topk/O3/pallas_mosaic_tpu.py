"""O3 policy: which column partition `p` the SparseCore selector runs at. No pallas_call here."""

# Dimensions.
#   B        rows of the score matrix = padded token count
#   N        padded score extent per row
#   K        top-k
#   tn       slice width in words = N // P_MAX, the tile extent of N
#   p        number of N-tiles covered this call (code-style's `nn`; the
#            exercise calls it `p` everywhere, so it stays `p`)
#   P_MAX    capacity ceiling on p, from static N

import functools

import pydantic
from jax.experimental.pallas import tpu as pltpu

import tiling


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """What O3 decides. `p_max` and `tn` are the ceiling; `p` itself is traced."""

    p_max: pydantic.PositiveInt
    tn: pydantic.PositiveInt
    lanes: pydantic.PositiveInt
    slices_per_wave: pydantic.PositiveInt

    def __post_init__(self):
        # Cross-field invariants an annotation cannot state (code-style.md 11).
        if self.p_max & (self.p_max - 1):
            raise ValueError(f"p_max must be a power of two, got {self.p_max}")
        if self.tn % self.lanes:
            raise ValueError(f"tn={self.tn} must be a multiple of lanes="
                             f"{self.lanes}")

    @property
    def ladder(self) -> tuple[int, ...]:
        return tiling.partition_ladder(self.p_max)


@functools.cache
def hardware() -> tuple[int, int, int]:
    """(lanes, slices_per_wave, max_slice_words), read from the platform.

    The incumbent hardcodes all three -- LANES = 16 at sparsecore_topk.py:41,
    `num_subcores = 16` / `num_sc_cores = 2` at :85-86, and MAX_SLICE_WORDS =
    64*1024 at :45. On tpu7x `get_tpu_info().sparse_core` reports num_lanes=16,
    num_subcores=16, num_cores=2 and 512 KiB per subcore, so these reproduce the
    incumbent's numbers exactly today and track the chip if it changes. That is
    the point of code-style.md 13: the hardcoded form is not wrong now, it is
    wrong later and silently.
    """
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available on this device")
    return (sc.num_lanes,
            sc.num_cores * sc.num_subcores,
            tiling.slice_words_budget(sc.vmem_capacity_bytes))


def supported_on(B: int, N: int, K: int) -> None:
    """Raise NotImplementedError if this backend cannot serve the shape.

    NotImplementedError, not ValueError: a fallback chain catches only the
    former, and a shape this backend simply cannot hold is not a caller
    mistake (code-style.md 9).
    """
    lanes, _, max_slice_words = hardware()
    if N % lanes:
        raise ValueError(f"{N=} must be a multiple of {lanes}")
    if K % lanes or not 0 < K <= 4096:
        raise ValueError(f"{K=} must be a positive multiple of {lanes}, "
                         f"at most 4096")
    cfg = _get_heuristics_config(B, N, K)
    if cfg.tn > max_slice_words:
        raise NotImplementedError(
            f"slice width {cfg.tn} exceeds the {max_slice_words}-word subcore "
            f"budget; N={N} needs windowing this kernel does not do")


@functools.cache
def _get_heuristics_config(B: int, N: int, K: int) -> Config:
    """The ceiling, and the slice width every ladder rung shares.

    O3's whole change is downstream of this: `p_max` is what the incumbent
    would have run at unconditionally, `tn = N // p_max` is the slice width it
    would have used, and O3 keeps `tn` fixed while letting the NUMBER of slices
    per row fall to whatever the batch's live extent needs. Keeping `tn` fixed
    rather than widening it is what makes the saving a wave-count saving: M2
    charges `f_w` per word of the longest slice in each wave, so a wider slice
    at a lower p would buy nothing (P-WALK dead-half, j-69ee326d).
    """
    lanes, slices_per_wave, max_slice_words = hardware()
    p_max = tiling.static_partition(B, N, K, max_slice_words=max_slice_words,
                                    lanes=lanes,
                                    slices_per_wave=slices_per_wave)
    return Config(p_max=p_max, tn=N // p_max, lanes=lanes,
                  slices_per_wave=slices_per_wave)


def _get_autotuning_configs(B: int, N: int, K: int) -> tuple[Config, ...]:
    """Nothing to sweep: p_max is a VMEM capacity constraint, not a knob.

    Recorded as a one-element tuple rather than omitted, so the absence is a
    statement. Below p_max the choice is made at runtime from the live extent
    and no offline sweep can see it; above it the slice does not fit.
    """
    return (_get_heuristics_config(B, N, K), )
