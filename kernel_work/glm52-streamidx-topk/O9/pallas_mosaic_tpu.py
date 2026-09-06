"""O9 config and the SparseCore launch. Every tuning decision lives here.

`Config()` with all defaults is meant to be the incumbent's schedule -- one
blocking full-slice copy per wave -- so it is the control that separates "the
change" from "the rewrite". Each arm moves exactly one knob off it.

Declared deviations from `code-style.md`:

  * §11 asks for a *pydantic* frozen/kw_only/slots dataclass. This uses
    `dataclasses.dataclass(frozen=True, kw_only=True, slots=True)`; pydantic is
    not a dependency of the measurement image and adding one would be a
    difference between the arms that is not the change under test. The
    cross-field checks are in `__post_init__` as the rule requires.
  * §9 asks unsupported features to raise `NotImplementedError`. They do -- and
    so do out-of-range knob values, because every rejected point here is a
    schedule this variant chose not to build rather than a caller error.
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from pallas_mosaic_tpu_kernel import build_call

# The incumbent's partitioning contract, not a platform constant: `_pick_partition`
# raises P until one row-slice fits a vector subcore, and changing this number
# changes which shapes take the two-stage path. That is a different variant's
# business, so O9 inherits it unchanged.
MAX_SLICE_WORDS = 64 * 1024

MAX_CHUNKS = 32


def _cdiv(a, b):
    return (a + b - 1) // b


def _align_to(x, a):
    return _cdiv(x, a) * a


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """One point in O9's fetch-schedule space. All defaults == the incumbent."""

    # The change.
    chunks: int = 1  # sub-copies the slice fetch is split into
    depth: int = 1  # sub-copies in flight at once
    # Attribution, semantics-preserving.
    dma_repeat: int = 1
    # Attribution ablation. NOT semantics-preserving; never a candidate.
    dma_denom: int = 1
    # Allocation probe: claim a second `slice_len` scratch, read by nothing.
    second_buffer: bool = False

    def __post_init__(self):
        if not 1 <= self.chunks <= MAX_CHUNKS:
            raise NotImplementedError(
                f"chunks must be in 1..{MAX_CHUNKS}, got {self.chunks}; the "
                f"chunk loop is unrolled in Python and this caps code size")
        if self.chunks & (self.chunks - 1):
            raise NotImplementedError(
                f"chunks must be a power of two so `slice_len // chunks` is "
                f"exact at every shape, got {self.chunks}")
        if not 1 <= self.depth <= self.chunks:
            raise NotImplementedError(
                f"depth must be in 1..chunks ({self.chunks}), got {self.depth}")
        if self.chunks > 1 and self.depth == 1:
            raise NotImplementedError(
                "chunks>1 at depth 1 is the incumbent's schedule paid for in "
                "extra descriptors -- it measures the split, not the overlap. "
                "Use chunks=1 for the control")
        if not 1 <= self.dma_repeat <= 8:
            raise NotImplementedError(f"dma_repeat out of range: "
                                      f"{self.dma_repeat}")
        if self.dma_denom not in (1, 2, 4, 8, 16):
            raise NotImplementedError(f"dma_denom must be a power of two in "
                                      f"1..16, got {self.dma_denom}")
        if self.chunks > 1 and (self.dma_repeat > 1 or self.dma_denom > 1):
            raise NotImplementedError(
                "the attribution knobs only apply to the single-copy path; "
                "combining them with chunking would measure neither")

    @property
    def is_semantics_preserving(self):
        """False for the ablation arm, whose output is deliberately wrong."""
        return self.dma_denom == 1


def _get_heuristics_config(b, n, k):
    """The config a caller who does not care gets. O9 ships the incumbent."""
    del b, n, k
    return Config()


def _get_autotuning_configs(b, n, k):
    """The schedules worth sweeping at this shape."""
    del b, n, k
    return (Config(), Config(chunks=4, depth=2), Config(chunks=8, depth=2),
            Config(chunks=8, depth=4), Config(chunks=16, depth=4))


def supported_on(info=None):
    """SparseCore with 16 lanes is the only place this kernel lowers."""
    info = info if info is not None else pltpu.get_tpu_info()
    sc = info.sparse_core
    return sc is not None and sc.num_lanes == 16


def pick_partition(b, n, k):
    """Power-of-two slices per row, maximizing subcore use within VMEM."""
    p_min = _cdiv(n, MAX_SLICE_WORDS)
    p = 1
    while p < p_min:
        p *= 2
    max_p = 32 // b if b < 32 else 1
    while p * 2 <= max_p:
        next_np = n // (p * 2)
        if next_np >= k and next_np >= 16 and (n % (p * 2)) == 0:
            p *= 2
        else:
            break
    return p


def slice_geometry(b, n, k):
    """(slice_len, num_waves) for one direct launch -- ints only, no device."""
    del k
    return _align_to(n, 16), _cdiv(b, 32)


def sc_topk_direct(scores, k, row_lengths, *, cfg, write_empty=True):
    """One single-stage lock-free SparseCore launch over 32 subcores."""
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    lanes = sc.num_lanes
    b, n = scores.shape

    words = jax.lax.bitcast_convert_type(scores, jnp.int32)
    b_pad = _align_to(b, lanes)
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, b_pad - b))

    slice_len, num_waves = slice_geometry(b, n, k)
    if cfg.chunks > 1 and slice_len % (cfg.chunks * lanes):
        raise NotImplementedError(
            f"{slice_len=} does not split into {cfg.chunks} whole-vector "
            f"chunks")

    call = build_call(b, n, k, slice_len, num_waves, write_empty, lanes, cfg)
    return call(words.reshape(-1), lengths).reshape(b, k)


def vmem_ledger(b, n, k, cfg):
    """Per-vector-subcore scratch bytes this config claims, and the ceiling.

    Ints only, no device: the wave-level double buffer the design sheet names
    needs a second `slice_len` buffer, and whether that can exist is arithmetic
    on a platform constant, not an experiment. Reported by `probe.py` and
    checked against a real compile there.
    """
    info = pltpu.get_tpu_info()
    lanes = info.sparse_core.num_lanes
    slice_len, _ = slice_geometry(b, n, k)
    items = {
        "resident": slice_len,
        "priv": lanes * 256 + lanes,
        "glob": 256 + lanes,
        "rowbuf": k + lanes,
        "len": _align_to(b, lanes),
        "spare": slice_len if cfg.second_buffer else lanes,
    }
    used = sum(items.values()) * 4
    return dict(items={n_: v * 4 for n_, v in items.items()},
                used_bytes=used,
                capacity_bytes=info.sparse_core.vmem_capacity_bytes,
                fits=used <= info.sparse_core.vmem_capacity_bytes)
