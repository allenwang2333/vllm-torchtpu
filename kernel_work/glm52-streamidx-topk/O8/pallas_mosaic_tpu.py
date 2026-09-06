"""Backend policy for the O8 selector: the Config, the heuristic, `supported_on`.

No `pl.pallas_call` here -- every launch is in `pallas_mosaic_tpu_kernel.py`.
This file decides P and the arm; the kernel file receives P and never picks it.
"""

import dataclasses

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

import tiling
from pallas_mosaic_tpu_kernel import select_inc_at_p, select_o8_at_p


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Config:
    """Tuning knobs for one selector launch.

    `pydantic` is not installed in the benchmark image, so this is a stdlib
    frozen/kw_only/slots dataclass with the cross-field checks in
    `__post_init__`. Declared as a deviation in `DEVIATIONS.md`.
    """
    p: int
    arm: str  # "inc" | "flat" | "o8"

    def __post_init__(self):
        if self.p < 1 or (self.p & (self.p - 1)) != 0:
            raise ValueError(f"p must be a power of two, got {self.p}")
        if self.arm not in ("inc", "flat", "o8"):
            raise ValueError(f"unknown arm {self.arm!r}")


def supported_on(scores, k) -> bool:
    """SparseCore present, and the shape on the legality ladder."""
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        return False
    _, n = scores.shape
    return n % 16 == 0 and k % 16 == 0 and 0 < k <= 4096


def _get_heuristics_config(scores, k, *, arm: str) -> Config:
    """P from the shipping policy; the arm is the caller's A/B choice."""
    b, n = scores.shape
    return Config(p=tiling.pick_partition(b, n, k), arm=arm)


def _get_autotuning_configs(scores, k, *, arm: str):
    """The P ladder that fits. Not swept in this exercise -- O8 is measured at
    the SHIPPING P, because a P change is a different experiment (that is O3's)
    and moving both at once makes neither attributable."""
    b, n = scores.shape
    out, p = [], 1
    while p <= 32:
        if n % p == 0 and n // p <= tiling.MAX_SLICE_WORDS * 1 and (
                p == 1 or n // p >= k):
            out.append(Config(p=p, arm=arm))
        p *= 2
    return out


def select(scores, k, row_lengths, *, cfg: Config, write_empty_rows: bool):
    """Dispatch one selection under `cfg`."""
    if cfg.arm == "o8":
        return select_o8_at_p(scores, k, row_lengths, p=cfg.p,
                              write_empty_rows=write_empty_rows)
    if cfg.arm == "inc":
        return select_inc_at_p(scores, k, row_lengths, p=cfg.p,
                               write_empty_rows=write_empty_rows)
    raise NotImplementedError(
        "the `flat` arm is built on the SHIPPING kernel, not this copy; "
        "see api.py:flat_topk")
