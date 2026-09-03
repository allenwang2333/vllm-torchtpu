#!/usr/bin/env python3
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
"""The callers -- `base` (production, unmodified) and the v128 variants.

## Why the baseline is imported rather than re-typed

`base` is `option-b-runner/runner_op.build_streamidx_topk_jax(..., incumbent=
True)`.  That module is a copy of `_build_streamidx_op`'s inner function from
`src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py`, and it carries
an `ast`-based `check_fidelity()` that reparses the real file and asserts the
KV-cache scatter prologue is *character-identical*.  Reusing it means the
baseline in this exercise is the same object the Option B exercise measured
against, and the fidelity assertion covers exactly the lines this exercise
changes.

## The v128 caller: what the delta is

`_streamidx_topk_jax_v128` differs from the production function in the scatter
and in nothing else.  Production:

    slot = pos % block_size
    cache_kv = cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(
        k_packed)                              # k_packed: [n_tok, 256]

v128, on a cache shaped `[..., kv_packing * 2, 128]`:

    row0 = ROWS * (slot % kv_packing)          # first of the token's two rows
    cache_kv = cache_kv.at[rep(page), rep(slot // kv_packing),
                           rep(row0) + tile(arange(ROWS))].set(
        k_packed.reshape(n_tok * ROWS, 128))

`rep` is `jnp.repeat(x, ROWS)` and `tile` is `jnp.tile(arange(ROWS), n_tok)`,
so the 2 * n_tok index triples enumerate the token's rows in order and the
row-major `reshape` of `k_packed` lines up with them.  Both index arrays are
elementwise arithmetic on values production already computes; nothing new
comes in from vLLM.

`check_fidelity()` prints the full unified diff of the v128 function against
the real file's, so the delta is on the record rather than merely described.

## `v128rt` -- the arm that distinguishes "removed" from "moved"

`v128rt` takes the cache in the production `[415,256,4,256]` shape, reshapes it
to 128-minor, runs the v128 scatter, and reshapes back for the production
kernel.  If route (a) were free as a pure *view*, `v128rt` would be as fast as
`v128` and would need no kernel change at all.  If the two reshapes it adds are
themselves relayouts, `v128rt` shows the cost moved rather than removed -- and
then route (a) is only real if the cache is *allocated* 128-minor, which is
what `v128` measures.
"""

from __future__ import annotations

import ast
import difflib
import os
import sys

import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER_DIR = os.path.abspath(
    os.path.join(_HERE, os.pardir, "option-b-runner"))
if _RUNNER_DIR not in sys.path:
    sys.path.insert(0, _RUNNER_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runner_op  # noqa: E402  the production caller, fidelity-checked

import kvs_common as KC  # noqa: E402

ROWS = KC.ROWS_PER_TOKEN
REAL_CALLER = runner_op.REAL_CALLER


# ---------------------------------------------------------------------------
# The v128 caller.
# ---------------------------------------------------------------------------
DEINTERLEAVE = os.environ.get("KVS_DEINTERLEAVE", "index")


def _v128_scatter(cache_kv, k_packed, page, slot, kv_packing):
    """Write one padded 256-byte row per token as two 128-byte rows.

    `cache_kv` is `[..., kv_packing * ROWS, 128]`; `k_packed` is
    `[n_tok, 128 * ROWS]` and is reshaped row-major so sub-row `r` of token `t`
    is `k_packed[t, r*128:(r+1)*128]`.

    Two sub-row placements, selected by `KVS_DEINTERLEAVE`, which the kernel
    reads too and must agree with:

    * `index` / `stride` -- **interleaved**: token `t`'s two sub-rows are
      adjacent, at rows `2*(t%kv_packing)` and `2*(t%kv_packing)+1`.  The bytes
      then land in exactly the places the production `[...,4,256]` cache has
      them, so this placement is byte-comparable against `base`.
    * `split` -- **grouped**: every token's value sub-row goes in the first
      `kv_packing` rows of the group and every scale sub-row in the last
      `kv_packing`.  This is a *permutation* of the production byte order, so
      byte-equality against `base` is expected to FAIL by construction and only
      the kernel's output is comparable.  It exists because the interleaved
      placement forces the kernel into a stride-2 sub-row read, which is the
      whole of the v128 kernel's measured regression.
    """
    n_tok = k_packed.shape[0]
    lane = k_packed.shape[-1] // ROWS
    sub = slot // kv_packing
    k = slot % kv_packing
    rows = k_packed.reshape(n_tok, ROWS, lane)
    if DEINTERLEAVE == "split":
        page_r = jnp.concatenate([page] * ROWS)
        sub_r = jnp.concatenate([sub] * ROWS)
        row_r = jnp.concatenate(
            [k + r * kv_packing for r in range(ROWS)])
        upd = jnp.concatenate([rows[:, r, :] for r in range(ROWS)], axis=0)
        return cache_kv.at[page_r, sub_r, row_r].set(upd)
    row0 = ROWS * k
    page_r = jnp.repeat(page, ROWS)
    sub_r = jnp.repeat(sub, ROWS)
    row_r = jnp.repeat(row0, ROWS) + jnp.tile(
        jnp.arange(ROWS, dtype=jnp.int32), n_tok)
    return cache_kv.at[page_r, sub_r,
                       row_r].set(k_packed.reshape(n_tok * ROWS, lane))


def build_streamidx_topk_jax_v128(
    streamindex_topk,
    *,
    topk: int,
    num_kv_pages_per_block=KC.INCUMBENT_TUPLE,
    num_queries_per_block=KC.NUM_QUERIES_PER_BLOCK,
    round_trip: bool = False,
    scatter_only: bool = False,
):
    """`_build_streamidx_op`'s inner function with the v128 scatter.

    `round_trip=True` is the `v128rt` arm: the cache arrives and leaves in the
    production `[..., 4, 256]` shape and the 128-minor form exists only across
    the scatter.  `scatter_only=True` drops the kernel call entirely, which is
    the `scat*` arms -- the cleanest isolation of the triad.
    """

    def _streamidx_topk_jax(
            q_bytes: jax.Array,
            weights: jax.Array,
            cache_kv: jax.Array,
            k_packed: jax.Array,
            seq_lens: jax.Array,
            page_indices: jax.Array,
            cu_q_lens: jax.Array,
            distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if round_trip:
            outer_shape = cache_kv.shape
            nb, bspp, kvp, lkv = outer_shape
            cache_kv = cache_kv.reshape(nb, bspp, kvp * ROWS, lkv // ROWS)
        _, block_size_per_kv_packing, rows_per_kv_packing, lane = (
            cache_kv.shape)
        kv_packing = rows_per_kv_packing // ROWS
        lkv_dim = lane * ROWS  # the logical 256-byte row width
        block_size = block_size_per_kv_packing * kv_packing
        # Cache rows are padded to the 128-lane HBM width; pad the packed
        # fp8+scale rows to match before scattering.
        k_packed = jnp.pad(k_packed,
                           ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

        # Insert this step's K rows into the KV cache.
        tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                  side="right").astype(jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                        pos // block_size],
            jnp.int32(2**30))
        slot = pos % block_size
        cache_kv = _v128_scatter(cache_kv, k_packed, page, slot, kv_packing)

        if round_trip:
            cache_kv = cache_kv.reshape(outer_shape)
        if scatter_only:
            return (cache_kv, )

        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        topk_indices = streamindex_topk(
            q,
            weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            k=topk,
            compression_ratio=1,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            enable_early_exit=runner_op.STREAMIDX_ENABLE_EARLY_EXIT,
        )
        return cache_kv, topk_indices

    return _streamidx_topk_jax


def build_streamidx_topk_jax_base(streamindex_topk,
                                  *,
                                  topk: int,
                                  scatter_only: bool = False):
    """The production caller.  `scatter_only` drops the kernel call.

    The non-`scatter_only` path is `runner_op`'s, unmodified, so it carries the
    fidelity assertion.  The `scatter_only` path re-states the same prologue
    (it has to, to be able to stop after it) and `check_fidelity()` diffs it.
    """
    if not scatter_only:
        return runner_op.build_streamidx_topk_jax(
            streamindex_topk,
            topk=topk,
            num_kv_pages_per_block=KC.INCUMBENT_TUPLE,
            num_queries_per_block=KC.NUM_QUERIES_PER_BLOCK,
            incumbent=True,
        )

    def _streamidx_topk_jax_scatter_only(
            q_bytes: jax.Array,
            weights: jax.Array,
            cache_kv: jax.Array,
            k_packed: jax.Array,
            seq_lens: jax.Array,
            page_indices: jax.Array,
            cu_q_lens: jax.Array,
            distribution: jax.Array,
    ):
        _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
        block_size = block_size_per_kv_packing * kv_packing
        k_packed = jnp.pad(k_packed,
                           ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
        tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
        seq_id = jnp.searchsorted(cu_q_lens[1:], tok,
                                  side="right").astype(jnp.int32)
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
        page = jnp.where(
            tok < cu_q_lens[-1],
            page_indices.reshape(seq_lens.shape[0], -1)[seq_id,
                                                        pos // block_size],
            jnp.int32(2**30))
        slot = pos % block_size
        cache_kv = cache_kv.at[page, slot // kv_packing,
                               slot % kv_packing].set(k_packed)
        return (cache_kv, )

    return _streamidx_topk_jax_scatter_only


# ---------------------------------------------------------------------------
# Arm construction.
# ---------------------------------------------------------------------------
def make_arm(mods: dict, arm: str, donate: bool = True):
    """Build and jit one arm.  `donate` mirrors `pallas.jax_op(...,
    donate_argnums=(2,))`, which is how the real op is built."""
    if arm == "base":
        fn = build_streamidx_topk_jax_base(mods["incumbent"].streamindex_topk,
                                           topk=KC.K_IDX)
    elif arm == "scat256":
        fn = build_streamidx_topk_jax_base(None,
                                           topk=KC.K_IDX,
                                           scatter_only=True)
    elif arm == "v128":
        fn = build_streamidx_topk_jax_v128(mods["v128"].streamindex_topk,
                                           topk=KC.K_IDX)
    elif arm == "v128rt":
        fn = build_streamidx_topk_jax_v128(mods["incumbent"].streamindex_topk,
                                           topk=KC.K_IDX,
                                           round_trip=True)
    elif arm == "scat128":
        fn = build_streamidx_topk_jax_v128(None,
                                           topk=KC.K_IDX,
                                           scatter_only=True)
    elif arm == "scat128rt":
        fn = build_streamidx_topk_jax_v128(None,
                                           topk=KC.K_IDX,
                                           round_trip=True,
                                           scatter_only=True)
    else:
        raise ValueError(arm)
    return jax.jit(fn, donate_argnums=(2, )) if donate else jax.jit(fn)


# ---------------------------------------------------------------------------
# Fidelity: the v128 copy against the real file.
# ---------------------------------------------------------------------------
def _extract(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise LookupError(f"{name} not found")


def check_fidelity(log=print) -> dict:
    """Diff every caller in this file against `mla_attention_op.py`'s.

    The `base` arm's fidelity is `runner_op.check_fidelity()`'s job and is
    delegated to it -- that is the assertion that the prologue this exercise
    changes is character-identical in the baseline.  This function adds the
    deliberate deltas: the v128 caller and the scatter-only baseline.
    """
    base = runner_op.check_fidelity(log)
    with open(REAL_CALLER) as fh:
        real = _extract(fh.read(), "_streamidx_topk_jax")
    with open(__file__) as fh:
        mine = fh.read()

    for name in ("_streamidx_topk_jax", "_streamidx_topk_jax_scatter_only"):
        copy = _extract(mine, name)
        log(f"FIDELITY-KVS diff of kvs_op.py::{name} against the real caller:")
        for ln in difflib.unified_diff(
            [l.strip() for l in real.splitlines()],
            [l.strip() for l in copy.splitlines()],
                "mla_attention_op.py::_streamidx_topk_jax",
                f"kvs_op.py::{name}",
                lineterm=""):
            log(f"FIDELITY-KVS {ln}")
    return base


if __name__ == "__main__":
    r = check_fidelity()
    sys.exit(0 if r["prologue_identical"] else 1)
