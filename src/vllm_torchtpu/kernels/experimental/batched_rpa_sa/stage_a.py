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
"""Stage-A levers for batched RPA (close-gaps j-8228ff78 row 2, probe 01).

This package is a copy of `batched_rpa` (stack3 @a0fc38482) with four levers
behind module-level flags. Every flag False reproduces the incumbent's
dataflow; the probe builds one package copy per arm with this file rewritten.

  A1  DECODE prologue reorder. Slot 0's V window (and its K stitch staging
      columns) is zeroed in the prologue, under the schedule fetch, as before;
      slots 1..n_buffer-1 are zeroed in the KV copy_in of their first fill,
      just before their descriptors are issued, so that zeroing runs inside
      the first block's DMA latency instead of in front of it.
  A2  V zeroing as a rolled loop (lax.fori_loop over row chunks) instead of
      ~2,750 straight-line store bundles: less code for the next launch to
      fetch.
  A3  MIXED: skip dead q-row chunks in groups of GROUP chunks, with pl.when on
      the step's live row count (max over lanes of the q tile size). The
      chunks inside a group stay software-pipelined.
  A4  DECODE: a step whose every lane has at most NARROW_KV live KV columns
      runs a narrower copy of the body (window [:NARROW_KV]).
"""

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

A1 = True
A2 = True
A3 = True
A4 = True

GROUP = 4  # q-row chunks per A3 group
NARROW_KV = 8192  # A4 window (64 pages of 128)
ZERO_CHUNK_MAX_ROWS = 512  # A2: rows of 128 lanes stored per loop iteration


def _zero_chunk_rows(rows: int) -> int:
    """Largest multiple of 8 that divides `rows` and is <= ZERO_CHUNK_MAX_ROWS."""
    if rows % 8:
        return 0
    best = 0
    for k in range(1, rows // 8 + 1):
        ch = 8 * k
        if ch > ZERO_CHUNK_MAX_ROWS:
            break
        if rows % ch == 0:
            best = ch
    return best


def zero_flat(v_flat: jax.Ref):
    """Zero a [rows, num_lanes] uint32 ref: rolled loop under A2, else one store."""
    rows, lanes = v_flat.shape
    ch = _zero_chunk_rows(rows) if A2 else 0
    if ch and rows // ch > 1:

        def body(i, c):
            v_flat[pl.ds(pl.multiple_of(i * ch, ch), ch), :] = jnp.zeros((ch, lanes), jnp.uint32)  # SA_ZERO_V_ROLLED
            return c

        lax.fori_loop(0, rows // ch, body, 0)
    else:
        v_flat[...] = jnp.zeros_like(v_flat)  # SA_ZERO_V_FULL


def zero_decode_slot(kv_window: jax.Ref, slot, cfgs):
    """Zero one DECODE KV slot's V window and its K stitch staging columns.

    `slot` may be a Python int or a traced scalar (the copy-in slot)."""
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = (cfgs.bkv_sz + 2 * cfgs.serve.page_size) // num_lanes
    stage_cols = range(cfgs.bkv_sz // num_lanes, lanes_per_col)
    for b in range(cfgs.batch_size):
        for h in range(cfgs.model.num_kv_heads):
            v_flat = (
                kv_window.at[slot]
                .at[b]
                .at[2 * h + 1]
                .bitcast(jnp.uint32)
                .reshape(-1, num_lanes)
            )
            zero_flat(v_flat)
            k_flat = (
                kv_window.at[slot]
                .at[b]
                .at[2 * h]
                .bitcast(jnp.uint32)
                .reshape(-1, num_lanes)
            )
            outer_dim = k_flat.shape[0] // lanes_per_col
            for col in stage_cols:
                k_flat[pl.ds(col, outer_dim, lanes_per_col)] = jnp.zeros(
                    (outer_dim, num_lanes), jnp.uint32
                )


def a4_enabled(cfgs) -> bool:
    from vllm_torchtpu.kernels.experimental.batched_rpa_sa import configs

    return (
        A4
        and cfgs.mode == configs.RpaCase.DECODE
        and cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        and not cfgs.serve.per_token_scale
        and cfgs.serve.dtype_kv != jnp.uint8
        and cfgs.model.sliding_window is None
        and NARROW_KV % 128 == 0
        and NARROW_KV < cfgs.bkv_sz
    )


def a3_enabled(cfgs) -> bool:
    from vllm_torchtpu.kernels.experimental.batched_rpa_sa import configs

    n_chunks = pl.cdiv(cfgs.bq_sz, cfgs.bq_c_sz)
    return (
        A3
        and cfgs.mode != configs.RpaCase.DECODE
        and not cfgs.fuse_accum
        and not cfgs.serve.return_lse
        and n_chunks > GROUP
        and cfgs.bq_sz % cfgs.bq_c_sz == 0
        and cfgs.model.num_kv_heads == 1
    )
