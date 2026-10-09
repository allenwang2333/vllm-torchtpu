# Copyright 20XX Google LLC
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
# ==============================================================================

"""Output relayout and top-k reduction helper from compact to large layout."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

if not hasattr(pltpu, "annotate"):
  pltpu.annotate = lambda ref, **kwargs: ref

TPB = 128  # tokens per batch
TILE = 16  # tokens per reduction tile (multiple of 16)
LANES = 128
FK = 4  # max nonzero picks per token on fast path

SLOTS = 3
STRIDE = 9  # coprime to 32 VMEM banks (8 row-pairs + 1 pad sublane)


def local_pick_weights(
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    expert_lo,
    num_local_experts: int,
) -> jax.Array:
  """``topk_weights`` with the picks other shards serve set to 0."""
  local = jnp.logical_and(
      topk_indices >= expert_lo, topk_indices < expert_lo + num_local_experts
  )
  return jnp.where(local, topk_weights, 0)


def ref_topk_reduce(rows2d, weights, out_dtype=jnp.bfloat16):
  """Pure-JAX reference: rows2d[T * K, H] in (token, pick) order."""
  num_tokens, topk = weights.shape
  w = weights.astype(jnp.float32).reshape(-1, 1)
  x = jnp.where(w != 0, rows2d.astype(jnp.float32), 0.0) * w
  return x.reshape(num_tokens, topk, -1).sum(axis=1).astype(out_dtype)


def _kernel(
    k_lo_smem,
    span_smem,
    n_valid_smem,
    g_cnt_smem,
    idx_smem,
    w_smem,  # scalar prefetch -> SMEM
    rows_hbm,
    out_hbm,  # HBM in / out
    slots,
    outs,
    stage,
    sems,  # scratch
    *,
    K,
    chunks,
    nb,
    out_dtype,
):
  """Three slots: batch b is reduced from slot b % 3 while batch b + 2 is fetched."""
  slots = pltpu.annotate(slots, no_hazard_no_deps=True)
  outs = pltpu.annotate(outs, no_hazard_no_deps=True)
  stage = pltpu.annotate(stage, no_hazard_no_deps=True)
  sem_rows, sem_out = sems.at[0], sems.at[1]
  rows = rows_hbm.bitcast(jnp.uint32).reshape(
      rows_hbm.shape[0] // K, K, chunks, LANES
  )
  nt = TPB // TILE
  fk = min(FK, K)

  def fetch_tok(batch, slot, s, k_lo_b, span_b, t):
    k_lo, span = k_lo_b[t], span_b[t]
    pltpu.make_async_copy(
        rows.at[batch * TPB + t, pl.ds(k_lo, span)],
        slot.at[t, pl.ds(0, span)],
        sem_rows.at[s],
    ).start()

  def fetch(batch, s):
    """Copy each token's active pick span [k_lo, k_lo + span)
    into slot s at [0, span).
    """
    slot = slots.at[s]
    k_lo_b = k_lo_smem.at[pl.ds(batch * TPB, TPB)]
    span_b = span_smem.at[pl.ds(batch * TPB, TPB)]
    for t in range(TPB):
      fetch_tok(batch, slot, s, k_lo_b, span_b, t)

  def wait_rows(batch, s):
    some = rows_hbm.at[pl.ds(0, n_valid_smem[batch])]
    pltpu.make_async_copy(some, some, sem_rows.at[s]).wait()

  def out_copy(batch, s):
    return pltpu.make_async_copy(
        outs.at[s], out_hbm.at[pl.ds(batch * TPB, TPB)], sem_out.at[s]
    )

  def reduce_and_fetch(batch, s, batch_fetch, s_fetch):
    slot = slots.at[s]
    out_slot = outs.at[s]
    w_b = w_smem.at[pl.ds(batch * (TPB * K), TPB * K)]
    idx_base = batch * TPB
    slot_fetch = slots.at[s_fetch]
    k_lo_b = k_lo_smem.at[pl.ds(batch_fetch * TPB, TPB)]
    span_b = span_smem.at[pl.ds(batch_fetch * TPB, TPB)]
    g_stride = 2 * chunks * STRIDE

    def reduce_tok(t, fk_cur):
      acc = None
      for k in range(fk_cur):
        idx = (
            k
            if (k == 0 or fk_cur == K)
            else idx_smem[(k - 1) * (nb * TPB) + idx_base + t]
        )
        w = jnp.broadcast_to(w_b[t * K + k], (chunks, LANES))
        pk = pltpu.bitcast(slot[t, idx], jnp.bfloat16) * pltpu.bitcast(
            w, jnp.bfloat16
        )
        pk = pltpu.bitcast(
            jnp.where(w != 0, pltpu.bitcast(pk, jnp.uint32), 0), jnp.bfloat16
        )
        acc = pk if acc is None else acc + pk
      assert acc is not None
      return pltpu.bitcast(acc, jnp.uint32)

    for g in range(nt):

      def do_tile(fk_cur, g=g):
        g_off = g * g_stride
        for p in range(TILE // 2):
          t0 = g * TILE + 2 * p
          acc0_u32 = reduce_tok(t0, fk_cur)
          f_base = 3 * (g * (TILE // 2) + p)
          if f_base < TPB:
            fetch_tok(batch_fetch, slot_fetch, s_fetch, k_lo_b, span_b, f_base)
          acc1_u32 = reduce_tok(t0 + 1, fk_cur)
          for j in range(2):
            u0 = pltpu.unpack_elementwise(
                acc0_u32,
                index=j,
                packed_dtype=jnp.bfloat16,
                unpacked_dtype=jnp.float32,
            )
            u1 = pltpu.unpack_elementwise(
                acc1_u32,
                index=j,
                packed_dtype=jnp.bfloat16,
                unpacked_dtype=jnp.float32,
            )
            packed = pltpu.pack_elementwise([u0, u1], packed_dtype=jnp.bfloat16)
            off = g_off + j * chunks * STRIDE + p
            stage[off : off + chunks * STRIDE : STRIDE] = pltpu.bitcast(
                packed, jnp.uint32
            )
          if p < TILE // 2 - 1:
            for f_tok in range(f_base + 1, min(f_base + 3, TPB)):
              fetch_tok(batch_fetch, slot_fetch, s_fetch, k_lo_b, span_b, f_tok)
        for c in range(chunks):
          for j in range(2):
            tile_u32 = stage[pl.ds(g_off + (j * chunks + c) * STRIDE, 8)]
            out_slot[
                pl.ds(g * TILE, TILE), pl.ds((2 * c + j) * LANES, LANES)
            ] = pltpu.bitcast(tile_u32, jnp.bfloat16).astype(out_dtype)
        f_last = 3 * (g * (TILE // 2) + TILE // 2 - 1)
        for f_tok in range(f_last + 1, min(f_last + 3, TPB)):
          fetch_tok(batch_fetch, slot_fetch, s_fetch, k_lo_b, span_b, f_tok)

      cnt_val = g_cnt_smem[batch * nt + g]
      if fk < K:
        lax.cond(
            cnt_val <= fk,
            functools.partial(do_tile, fk),
            functools.partial(do_tile, K),
        )
      else:
        do_tile(K)

  fetch(0, 0)
  fetch(1, 1)

  @pl.loop(0, nb)
  def _(batch):
    s = lax.rem(batch, SLOTS)
    s_fetch = lax.rem(batch + 2, SLOTS)
    wait_rows(batch, s)
    out_wait = jnp.where(batch >= SLOTS, TPB, 0)
    pltpu.make_async_copy(
        outs.at[s, pl.ds(0, out_wait)],
        outs.at[s, pl.ds(0, out_wait)],
        sem_out.at[s],
    ).wait()
    reduce_and_fetch(batch, s, batch + 2, s_fetch)
    out_copy(batch, s).start()

  for b in range(max(0, nb - SLOTS), nb):
    out_copy(b, b % SLOTS).wait()


@functools.partial(jax.jit, static_argnames=("num_local_experts", "out_dtype"))
def output_reduce_unpack(
    rows: jax.Array,
    weights: jax.Array,
    topk_indices: jax.Array | None = None,
    group_offset: jax.Array | int | None = None,
    num_local_experts: int | None = None,
    *,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
  if (
      rows.dtype != jnp.bfloat16
      or rows.ndim != 4
      or rows.shape[2:] != (2, LANES)
  ):
    raise ValueError(
        "rows must be bf16[T * K, H / 256, 2, 128], got"
        f" {rows.dtype}{rows.shape}"
    )
  if weights.ndim != 2 or rows.shape[0] != weights.shape[0] * weights.shape[1]:
    raise ValueError(
        f"weights must be [T, K] with T * K == {rows.shape[0]}, got"
        f" {weights.shape}"
    )
  if topk_indices is not None:
    if group_offset is None or num_local_experts is None:
      raise ValueError(
          "group_offset and num_local_experts are required when topk_indices is"
          " provided"
      )
    weights = local_pick_weights(
        weights,
        topk_indices,
        jnp.asarray(group_offset).reshape(()),
        num_local_experts,
    )
  T, K = weights.shape
  chunks = rows.shape[1]
  H = chunks * 2 * LANES
  nb = -(-T // TPB)
  w = jnp.pad(weights.astype(jnp.float32), ((0, nb * TPB - T), (0, 0)))
  valid = w != 0

  k_idx = lax.broadcasted_iota(jnp.int32, w.shape, 1)
  k_lo_raw = jnp.min(jnp.where(valid, k_idx, K), axis=1)
  k_hi = jnp.max(jnp.where(valid, k_idx, -1), axis=1)
  count = jnp.sum(valid.astype(jnp.int32), axis=1)

  fk = min(FK, K)
  g_cnt = count.reshape(-1, TILE).max(axis=1)
  # When K <= FK the kernel has no fast arm (it always runs do_tile(K) with
  # slot index == pick), so the wrapper must emit the raw layout too: compacted
  # weights + a zero-aligned span would mis-pair rows and weights whenever a
  # token has a hole between its local picks.
  use_fast = jnp.broadcast_to(
      ((g_cnt <= fk) & (fk < K))[:, None], (g_cnt.shape[0], TILE)
  ).reshape(-1)

  k_lo = jnp.where(use_fast & (k_lo_raw < K), k_lo_raw, 0)
  span = jnp.where(
      k_hi >= 0, jnp.where(use_fast, k_hi - k_lo_raw + 1, k_hi + 1), 0
  )
  n_valid = span.reshape(nb, TPB).sum(axis=1)

  w16 = lax.bitcast_convert_type(w.astype(jnp.bfloat16), jnp.uint16).astype(
      jnp.uint32
  )
  w_raw = jnp.where(valid, (w16 << 16) | w16, 0)

  # Compact up to `fk` nonzero weights and their relative slot offsets
  # (k_c - k_lo) into static columns [0..fk-1].
  k_cur = k_lo_raw
  w_fast = jnp.zeros_like(w_raw)
  offs = []
  for c in range(fk):
    wc = jnp.max(jnp.where(k_idx == k_cur[:, None], w_raw, 0), axis=1)
    w_fast = jnp.where(k_idx == c, wc[:, None], w_fast)
    if c > 0:
      offs.append(jnp.where(k_cur < K, k_cur - k_lo_raw, 0))
    if c < fk - 1:
      k_cur = (
          jnp.where(count >= fk, k_hi, K)
          if c == fk - 2
          else jnp.min(
              jnp.where(valid & (k_idx > k_cur[:, None]), k_idx, K), axis=1
          )
      )

  idx_u32 = jnp.concatenate(offs) if offs else jnp.zeros(nb * TPB, jnp.int32)
  w_u32 = jnp.where(use_fast[:, None], w_fast, w_raw).reshape(-1)
  k_lo = jnp.pad(k_lo, (0, (SLOTS - 1) * TPB))
  span = jnp.pad(span, (0, (SLOTS - 1) * TPB))
  n_valid = jnp.pad(n_valid, (0, SLOTS - 1))

  out = pl.pallas_call(
      functools.partial(
          _kernel, K=K, chunks=chunks, nb=nb, out_dtype=out_dtype
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,
          in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          grid=(1,),
          scratch_shapes=[
              pltpu.VMEM((SLOTS, TPB, K, chunks, LANES), jnp.uint32),
              pltpu.VMEM((SLOTS, TPB, H), out_dtype),
              pltpu.VMEM(
                  ((TPB // TILE) * 2 * chunks * STRIDE, LANES), jnp.uint32
              ),
              pltpu.SemaphoreType.DMA((2, SLOTS)),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct((nb * TPB, H), out_dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          disable_bounds_checks=True,
          vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
      ),
      name=f"ragged_branch_fk4-k_{K}-h_{H}",
  )(k_lo, span, n_valid, g_cnt, idx_u32, w_u32, rows)
  return out[:T]
