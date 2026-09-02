"""Step-1 probe: is the indexer K cache 129 bytes wide per position, or 256?

This decides the decode roofline. The K-cache stream is 95% of decode HBM
traffic, so if the trailing dim is physically 256 with only 129 bytes used,
roughly half of the dominant term is padding and the decode floor is 47.5us
where it could be 25.0us. If it is physically 129, there is no such waste and
decode is VPU-bound at 39us instead. Opposite verdicts about what to optimize.

The code does not settle it. Two indexer paths disagree in a way that looks
like a contradiction but may not be:

  deepseek_v4_indexer.py:111   head_size=align_to(128 + 1, 128)   -> 256, explicit
  deepseek_v2.py:631 (upstream, inherited by the SparseAttnIndexer path this
                      op is on)  head_size=self.head_dim          -> 129, no padding

and mla_attention_op.py:76 comments that lkv_dim is "D + 1 padded to a 128
multiple" either way. Both can be true if XLA's tiled layout pads the minor
dimension of a (..., 129) uint8 array up to 256 on its own -- which is the
actual question, and it is about the physical layout rather than the logical
shape, so reading more Python cannot answer it.

Measured against device HBM accounting, not against a formula.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _p(tag, *rest):
    print(f"[probe] {tag}:", *rest, flush=True)


def hbm_in_use(dev):
    st = dev.memory_stats() or {}
    # Field name has moved across jaxlib versions; take the first that exists.
    for key in ("bytes_in_use", "bytes_in_use_hbm", "hbm_bytes_in_use"):
        if key in st:
            return st[key]
    raise RuntimeError(f"no bytes_in_use field in memory_stats: {sorted(st)}")


def measure_alloc(shape, dtype=jnp.uint8):
    """Bytes HBM actually charges for an array of this shape."""
    dev = jax.devices()[0]
    before = hbm_in_use(dev)
    x = jax.block_until_ready(jnp.zeros(shape, dtype))
    after = hbm_in_use(dev)
    logical = int(np.prod(shape)) * np.dtype(dtype).itemsize
    del x
    return after - before, logical


def main():
    dev = jax.devices()[0]
    _p("device", dev.device_kind)
    _p("memory_stats_keys", sorted((dev.memory_stats() or {}).keys()))

    # GLM-5.2 indexer cache geometry. page_size 1024, kv_packing 4, so the
    # 4-D view the kernel takes is [num_pages, 256, 4, lkv_dim]. Use enough
    # pages that the allocation is far above any fixed overhead.
    NUM_PAGES = 512
    cases = [
        ("3D logical, 129 wide", (NUM_PAGES, 1024, 129)),
        ("3D logical, 256 wide", (NUM_PAGES, 1024, 256)),
        ("4D kernel view, 129 wide", (NUM_PAGES, 256, 4, 129)),
        ("4D kernel view, 256 wide", (NUM_PAGES, 256, 4, 256)),
        # Control: a shape with no padding question at all, to show the
        # accounting is reading real allocations and not noise.
        ("control, 128 wide", (NUM_PAGES, 1024, 128)),
    ]
    results = {}
    for name, shape in cases:
        try:
            actual, logical = measure_alloc(shape)
            ratio = actual / logical if logical else float("nan")
            results[name] = actual
            _p(name, f"shape={shape}", f"logical={logical/1e6:.2f}MB",
               f"actual={actual/1e6:.2f}MB", f"ratio={ratio:.3f}")
        except Exception as e:  # noqa: BLE001 - report, do not abort the sweep
            _p(name, "FAIL", repr(e))

    a129 = results.get("4D kernel view, 129 wide")
    a256 = results.get("4D kernel view, 256 wide")
    if a129 and a256:
        if abs(a129 - a256) / a256 < 0.02:
            _p("VERDICT_lkv_dim",
               "256 - a 129-wide array is charged the same HBM as a 256-wide "
               "one, so the minor dim is padded and ~49.6% of the K-cache "
               "stream is padding. Decode floor 47.5us, HBM-bound.")
        elif a129 < a256 * 0.6:
            _p("VERDICT_lkv_dim",
               "129 - the array is charged for what it uses, there is no "
               "padding to reclaim. Decode floor 39.0us, VPU-bound.")
        else:
            _p("VERDICT_lkv_dim",
               f"AMBIGUOUS - 129-wide charged {a129/a256:.3f} of 256-wide; "
               "neither hypothesis fits, inspect the ratios above.")

    # Second, independent reading. The allocation accounting above says what
    # HBM is charged; the tiled layout says *why*. `tiling` is empty on CPU and
    # populated on TPU, and its minor tile is what pads a 129-wide dim. If the
    # two readings disagree, believe neither and say so -- that is a probe
    # result, not a number to average.
    for name, shape in [("129", (NUM_PAGES, 256, 4, 129)),
                        ("256", (NUM_PAGES, 256, 4, 256))]:
        try:
            x = jax.block_until_ready(jnp.zeros(shape, jnp.uint8))
            fmt = getattr(x, "format", None)
            _p(f"layout_{name}", fmt if fmt is not None else x.shape)
            del x
        except Exception as e:  # noqa: BLE001
            _p(f"layout_{name}", "UNAVAILABLE", repr(e))


if __name__ == "__main__":
    main()
