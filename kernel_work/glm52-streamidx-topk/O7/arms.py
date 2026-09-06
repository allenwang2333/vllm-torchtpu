"""The arms O7 is measured as, and the plumbing that swaps a selector in.

`instruments/` is the shared yardstick and is never edited or copied. What is
needed instead is a way to make `instruments/correctness.py` and
`instruments/bench.py` exercise THIS selector while running from their own
files. Two facts make that awkward and one trick resolves both:

  * `streamindex_topk` looks up `sparsecore_topk` as a MODULE GLOBAL at trace
    time, so rebinding that attribute at runtime is enough -- no file is
    touched. But
  * `streamindex_topk` is a `jax.jit` object with one cache, so patching,
    calling, patching again and calling again returns the FIRST arm's compiled
    executable for the second arm (probes.md records exactly this, which is why
    P-P16 copied whole modules rather than patching).

The trick is to build a SEPARATE `jax.jit` around the undecorated function for
each arm, and to force it to trace while that arm's patch is live. Each arm then
owns its own executable and its own baked-in selector, and the patch is
irrelevant afterwards. This keeps one copy of `streamindex_topk.py` -- the
shipping one -- so no arm can drift from production by way of a stale copy.
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

# Arm 0 is the shipping kernel. Arm 1 is THIS file's code with every flag off:
# it must be algorithmically identical to arm 0, and it is the control that says
# whether a difference in arms 2+ is the change or the rewrite.
CONFIGS = {
    "inc": None,
    "base": Config(),
    "fz": Config(fold_and_zero=True),
    "scnt": Config(scan_count_hist=True),
    "cf": Config(carry_forward=True),
    "fc": Config(fused_convert=True),
    "all": Config(scan_count_hist=True, carry_forward=True,
                  fused_convert=True),
    "all4": Config(scan_count_hist=True, carry_forward=True,
                   fused_convert=True, digit_width=4),
    # Iteration 2. The first job measured `cf` at +2.1% / -0.3% -- a wash or
    # worse -- and the reason is that with 256 level-0 buckets the tie bucket
    # holds roughly half the slice, so compaction has almost nothing to discard
    # and pays a full extra pass plus a gather to find that out. These arms
    # widen level 0 so there is something to carry. `w22` isolates the cost of
    # the wider level 0 on its own; the rest pair it with carry-forward.
    "w22": Config(scan_count_hist=True, l0_shift=22),
    "w22cf": Config(scan_count_hist=True, carry_forward=True, l0_shift=22),
    "w22all": Config(scan_count_hist=True, carry_forward=True,
                     fused_convert=True, l0_shift=22),
    "w20all": Config(scan_count_hist=True, carry_forward=True,
                     fused_convert=True, l0_shift=20),
}


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

    If `args` is given the arm is traced and compiled immediately, which is what
    makes several arms in one process independent -- see the module docstring.
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
