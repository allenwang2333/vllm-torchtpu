"""Utility functions for ragged paged attention."""

import functools

import jax
from jax._src import dtypes

_ITEMSIZE_BITS = dtypes.itemsize_bits


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    assert a != 0
    return ((x + a - 1) // a) * a


@functools.lru_cache(maxsize=64)
def get_dtype_bitwidth(dtype):
    return _ITEMSIZE_BITS(dtype)


@functools.lru_cache(maxsize=64)
def get_dtype_packing(dtype):
    return 32 // _ITEMSIZE_BITS(dtype)


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    return 1 << (x - 1).bit_length()


@functools.lru_cache(maxsize=16)
def _parse_tpu_version(kind: str) -> int:
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[:-5]
    if kind.endswith(("p", "e")):
        kind = kind[:-1]
    if kind == "TPU7x":
        return 7
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    return _parse_tpu_version(jax.devices()[0].device_kind)
