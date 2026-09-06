"""The arms O9 is measured as, and the plumbing that swaps a selector in.

`../instruments/` is the shared yardstick: it is imported and run, never edited
and never copied. The way an arm gets in front of it is that
`streamindex_topk` looks up `sparsecore_topk` as a MODULE GLOBAL at trace time,
so rebinding that attribute is enough. One wrinkle: `streamindex_topk` is a
`jax.jit` object with a single cache, so patching, calling, patching again and
calling again hands the SECOND arm the FIRST arm's compiled executable. Each arm
therefore gets its own `jax.jit` wrapper around the undecorated function, forced
to trace while its own patch is live.

Arm names, and what each is for:

  inc     the shipping kernel, untouched. The incumbent every delta is against.
  base    this file's kernel with every knob at its default. Must be
          algorithmically identical to `inc`; it is the control that says
          whether a difference in the other arms is the change or the rewrite.
  r2 r3   dma_repeat 2 and 3. ATTRIBUTION, semantics-preserving: the slope of
          time in the repeat count is the serial cost of one score-slice fetch.
  q2 q4   dma_denom 2 and 4. ATTRIBUTION ABLATION, output deliberately wrong --
          fetch a fraction of the slice, walk all of it. Brackets r2/r3 from
          below. Never a candidate; `run_correctness.py` refuses to gate them.
  c*d*    the change: `chunks` sub-copies with `depth` in flight, so chunk c is
          converted while chunk c+depth is landing. Zero extra VMEM.
  buf2    allocation probe: a second `slice_len` scratch, read by nothing. The
          wave-level double buffer needs one; this asks whether it can exist.
"""
import functools
import os
import sys

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as _sit  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    sparsecore_topk as incumbent_selector  # noqa: E402

import api  # noqa: E402
from pallas_mosaic_tpu import Config  # noqa: E402

_STATIC = ("k", "compression_ratio", "num_kv_pages_per_block",
           "num_queries_per_block", "vmem_limit_bytes",
           "decode_req_batch_size", "enable_early_exit")

CONFIGS = {
    "inc": None,
    "base": Config(),
    # -- attribution -------------------------------------------------------
    "r2": Config(dma_repeat=2),
    "r3": Config(dma_repeat=3),
    "r4": Config(dma_repeat=4),
    "q2": Config(dma_denom=2),
    "q4": Config(dma_denom=4),
    # -- the change --------------------------------------------------------
    "c2d2": Config(chunks=2, depth=2),
    "c4d2": Config(chunks=4, depth=2),
    "c8d2": Config(chunks=8, depth=2),
    "c8d4": Config(chunks=8, depth=4),
    "c16d4": Config(chunks=16, depth=4),
    # -- allocation probe --------------------------------------------------
    "buf2": Config(second_buffer=True),
}

# Arms whose output is deliberately wrong. They are attribution instruments, and
# nothing may report them as a candidate or run the gate on them.
ABLATION_ARMS = tuple(nm for nm, c in CONFIGS.items()
                      if c is not None and not c.is_semantics_preserving)


def selector(name):
    """The `sparsecore_topk`-compatible callable for one arm."""
    cfg = CONFIGS[name]
    if cfg is None:
        return incumbent_selector

    def sel(scores, k, row_lengths=None, *, write_empty_rows=True, cfg=cfg):
        return api.topk_indices(scores, k, row_lengths,
                                write_empty_rows=write_empty_rows, cfg=cfg)

    return sel


def full_op(name, kwargs, args=None):
    """A `streamindex_topk`-compatible callable wired to arm `name`.

    Passing `args` traces and compiles immediately, which is what makes several
    arms in one process independent -- see the module docstring.
    """
    fn = functools.partial(
        jax.jit(_sit.streamindex_topk.__wrapped__, static_argnames=_STATIC),
        **kwargs)
    prev = _sit.sparsecore_topk
    _sit.sparsecore_topk = selector(name)
    try:
        if args is not None:
            jax.block_until_ready(fn(*args))
    finally:
        _sit.sparsecore_topk = prev
    return fn
