import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

TPB = 128  # tokens per batch
TILE = 16  # tokens per reduction tile (multiple of 16)
LANES = 128


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
    w_smem,  # scalar prefetch -> SMEM
    rows_hbm,
    out_hbm,  # HBM in / out
    slot0,
    slot1,
    out0,
    out1,
    sems,  # scratch
    *,
    K,
    chunks,
    nb,
    out_dtype,
):
  """Two slots: batch b is reduced from slot b % 2 while batch b + 1 lands in the other."""
  slots, outs = (slot0, slot1), (out0, out1)
  sem_rows, sem_out = sems.at[0], sems.at[1]
  rows = rows_hbm.bitcast(jnp.uint32).reshape(
      rows_hbm.shape[0] // K, K, chunks, LANES
  )

  def fetch(batch, s):
    """Scatter each token's non-zero picks into row t of each (pick, chunk) tile."""
    for t in range(TPB):
      tok = batch * TPB + t
      k_lo, span = k_lo_smem[tok], span_smem[tok]
      pltpu.make_async_copy(
          rows.at[tok, pl.ds(k_lo, span)],
          slots[s].at[pl.ds(k_lo, span), :, t],
          sem_rows.at[s],
      ).start()

  def wait_rows(batch, s):
    """One wait for the whole batch: every copy adds its bytes to sem_rows[s]; wait until

    n_valid rows' worth have arrived (phrased as a copy of that size, never
    started).
    """
    some = (
        slots[s].reshape(-1, LANES).at[pl.ds(0, n_valid_smem[batch] * chunks)]
    )
    pltpu.make_async_copy(some, some, sem_rows.at[s]).wait()

  def out_copy(batch, s):
    return pltpu.make_async_copy(
        outs[s], out_hbm.at[pl.ds(batch * TPB, TPB)], sem_out.at[s]
    )

  def reduce(batch, s):
    iota = lax.broadcasted_iota(jnp.uint32, (TILE, LANES), 0)
    for g in range(TPB // TILE):
      toks = pl.ds(g * TILE, TILE)
      sums = []
      for k in range(K):
        base = (batch * TPB + g * TILE) * K + k
        wv = jnp.broadcast_to(w_smem[base], (TILE, LANES))
        for i in range(1, TILE):
          wv = jnp.where(iota == i, w_smem[base + i * K], wv)
        valid_k = wv != 0
        wk = pltpu.bitcast(wv, jnp.bfloat16)
        for c in range(chunks):
          x = jnp.where(valid_k, slots[s][k, c, toks], 0)
          x = pltpu.bitcast(x, jnp.bfloat16) * wk
          if k == 0:
            sums.append(x)
          else:
            sums[c] = sums[c] + x
      for c in range(chunks):
        packed = pltpu.bitcast(sums[c], jnp.uint32)
        for j in range(2):
          outs[s][toks, pl.ds((2 * c + j) * LANES, LANES)] = (
              pltpu.unpack_elementwise(
                  packed,
                  index=j,
                  packed_dtype=jnp.bfloat16,
                  unpacked_dtype=jnp.float32,
              ).astype(out_dtype)
          )

  def one_batch(batch, s, fetch_next):
    if fetch_next:
      fetch(batch + 1, 1 - s)

    wait_rows(batch, s)

    @pl.when(batch >= 2)
    def _():
      out_copy(batch - 2, s).wait()

    reduce(batch, s)
    out_copy(batch, s).start()

  fetch(0, 0)
  pairs = (nb - 1) // 2

  def pair(i, carry):
    one_batch(2 * i, 0, True)
    one_batch(2 * i + 1, 1, True)
    return carry

  lax.fori_loop(0, pairs, pair, 0)
  for b in range(2 * pairs, nb):
    one_batch(b, b % 2, b + 1 < nb)
  for b in range(max(0, nb - 2), nb):
    out_copy(b, b % 2).wait()


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def ragged_dma_reduce(
    rows: jax.Array,
    weights: jax.Array,
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
  T, K = weights.shape
  chunks = rows.shape[1]
  H = chunks * 2 * LANES
  nb = -(-T // TPB)
  w = jnp.pad(weights.astype(jnp.float32), ((0, nb * TPB - T), (0, 0)))
  valid = w != 0

  k_lo = jnp.argmax(valid, axis=1).astype(jnp.int32)
  k_hi = K - 1 - jnp.argmax(valid[:, ::-1], axis=1).astype(jnp.int32)
  span = jnp.where(valid.any(axis=1), k_hi - k_lo + 1, 0).astype(jnp.int32)
  n_valid = span.reshape(nb, TPB).sum(axis=1)
  w16 = lax.bitcast_convert_type(w.astype(jnp.bfloat16), jnp.uint16).astype(
      jnp.uint32
  )
  w_u32 = jnp.where(valid, (w16 << 16) | w16, 0).reshape(-1)

  out = pl.pallas_call(
      functools.partial(
          _kernel, K=K, chunks=chunks, nb=nb, out_dtype=out_dtype
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=4,
          in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
          out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
          grid=(1,),
          scratch_shapes=[
              pltpu.VMEM((K, chunks, TPB, LANES), jnp.uint32),
              pltpu.VMEM((K, chunks, TPB, LANES), jnp.uint32),
              pltpu.VMEM((TPB, H), out_dtype),
              pltpu.VMEM((TPB, H), out_dtype),
              pltpu.SemaphoreType.DMA((2, 2)),
          ],
      ),
      out_shape=jax.ShapeDtypeStruct((nb * TPB, H), out_dtype),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          disable_bounds_checks=True,
          vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
      ),
      name=f"min_ragged_dma_reduce-k_{K}-h_{H}",
  )(k_lo, span, n_valid, w_u32, rows)
  return out[:T]
