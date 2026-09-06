# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetically pre-split cache inputs for variant A ("Narrow Record").

`instruments/inputs.py` is READ-ONLY. This module does not fork it: it calls
`inputs.build` and then *derives* the two split arrays from the single record
array it returned. That derivation is a pure reindexing, so the two arms of the
A/B see bit-identical key values, bit-identical scales, the same page
permutation and the same `q` / `indexer_weights`. Anything else would make the
comparison a comparison of two datasets.

THE INPUT IS SYNTHETICALLY PRE-SPLIT. Nothing here writes the split format the
way production would; the producer-side cost of doing so is a separate number
and is measured by `writer_probe.py`. See RESULTS.md.

Record derivation, from `mla_attention_op.py:230-231` and
`streamindex_topk.py:265-268`:

    cache_kv[g, i, j, 0:128]  -> cache_val[g, i, j, 0:128]   the fp8 values
    cache_kv[g, i, j, 128]    -> cache_scale[g, p // 128, p % 128]  where
                                 p = i*kv_packing + j        the e8m0 scale
    cache_kv[g, i, j, 129:]   -> dropped. This is the 49.6% that never had to
                                 cross HBM.

`i*kv_packing + j` is the position index within the page: `inputs.pack_cache`
builds the cache as `rec.reshape(num_pages, page_size // kv_packing,
kv_packing, width)` from a `[num_pages, page_size, width]` array in position
order, so the flattened `(i, j)` pair is exactly the position.

The scale plane's trailing `(page_size // 128, 128)` split is a pure reshape of
the position axis -- same bytes, same order -- and exists so that Mosaic tiles
the last two dimensions into exactly one page of scales, leaving the page axis
untiled and the reader's per-page DMA tile-aligned. The 2-D
`[num_pages, page_size]` form is what job j-ce74a9f2-a4cd-40cb-9e87 rejected;
see the kernel's module docstring.
"""

import contextlib

import jax.numpy as jnp
import numpy as np

from kernel_work.streamindex_topk.instruments import inputs as I
from kernel_work.streamindex_topk.instruments import shapes as S


def split_cache(cache_kv):
    """(u8[np,pspp,kvp,256]) -> (u8[np,pspp,kvp,128], u8[np,page_size//128,128]).

    Returns the value plane and the scale plane. `cache_scale[g, p//128, p%128]`
    is the e8m0 scale byte of compressed position `p` of page `g` -- the same
    bytes in the same order a flat `[num_pages, page_size]` plane would hold,
    reshaped so that Mosaic's `(8, 128)` HBM tile is exactly one page.
    """
    a = np.asarray(cache_kv)
    num_pages, pspp, kvp, width = a.shape
    if width != S.LKV_DIM:
        raise AssertionError(
            f"expected the shipped record width {S.LKV_DIM}, got {width}")
    hd = S.INDEX_HEAD_DIM
    page_size = pspp * kvp
    if page_size % 128:
        raise AssertionError(
            f"page_size {page_size} is not a multiple of the 128-lane tile")
    val = np.ascontiguousarray(a[..., :hd])
    scale = np.ascontiguousarray(
        a[..., hd].reshape(num_pages, page_size // 128, 128))
    pad = a[..., hd + 1:]
    if pad.size and pad.any():
        raise AssertionError(
            "bytes 129..255 of the shipped record are not all zero; the split "
            "would be dropping live data")
    return jnp.asarray(val), jnp.asarray(scale)


# Captured at import time, before anything rebinds the attribute, so that
# `build` below can never recurse into itself through `patched_inputs`.
_BASE_BUILD = I.build


def build(regime: S.Regime, seed: int = 0, *, with_dequant: bool = False):
    """`instruments.inputs.build`, with `cache_kv` replaced by the split pair."""
    d = _BASE_BUILD(regime, seed=seed, with_dequant=with_dequant)
    d["cache_kv"] = split_cache(d["cache_kv"])
    return d


@contextlib.contextmanager
def patched_inputs():
    """Rebind `instruments.inputs.build` to the split builder, then restore.

    The instruments are read-only, so the variant arm is driven by rebinding
    the module attribute the instrument already calls. `microbench` and
    `test_correctness` both do `from . import inputs as I`, so they share the
    one module object this patches.
    """
    orig = I.build
    I.build = build
    try:
        yield
    finally:
        I.build = orig
