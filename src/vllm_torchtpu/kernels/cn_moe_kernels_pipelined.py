# Copyright 2026 Google LLC
"""Implementation that uses emit_pipeline instead of hand-rolled version."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

TOP_K = 10
_CN_K_TILE = 4096
_CN_I_TILE = 2048
_CN_NBUF = 4


def _blockwise_matmul_chunked(lhs, w, scales, total_dim, num_scale_blocks, out_cols):
  """Memory-efficient 128-element chunked matmul keeping VPU registers at 64 KB."""
  CHUNK_SIZE = 128
  num_chunks = total_dim // CHUNK_SIZE
  acc = jnp.zeros((1, out_cols), dtype=jnp.float32)
  for b in range(num_chunks):
    s = b * CHUNK_SIZE
    cur_lhs = lhs[:, s : s + CHUNK_SIZE]
    cur_w = w[s : s + CHUNK_SIZE, :].astype(lhs.dtype)
    blk = jnp.matmul(cur_lhs, cur_w, preferred_element_type=jnp.float32)
    # If subchannel scaling, index per-chunk; if per-channel, index 0
    scale_idx = b if num_scale_blocks == num_chunks else 0
    acc = acc + blk * scales[scale_idx].reshape(1, out_cols)
  return acc


def _inner_kernel(
    w1_tile_ref,  # [1, K, 2ID]
    w1_scale_tile_ref,  # [1, K_BLOCKS, 1, 2ID]
    w2_tile_ref,  # [1, ID, H]
    w2_scale_tile_ref,  # [1, I_BLOCKS, 1, H]
    *,
    weights_ref,
    lhs_ref,  # [C_PAD, 1, K]
    acc_ref,  # [N_TOKENS_, 1, H]
    ID,
):
  """Inner kernel for pipelined weight fetch with 128-element chunked matmul."""
  slot = pl.program_id(0)
  tok = slot // TOP_K
  weight = weights_ref[slot]

  K_lhs = lhs_ref.shape[2]
  lhs_row = lhs_ref[pl.ds(tok, 1), 0, :K_lhs]

  _, K, TWO_ID = w1_tile_ref.shape
  K_BLOCKS = w1_scale_tile_ref.shape[1]
  w1 = w1_tile_ref[...].reshape(K, TWO_ID)
  s1 = w1_scale_tile_ref[...].reshape(K_BLOCKS, 1, TWO_ID)
  gate_up = _blockwise_matmul_chunked(lhs_row, w1, s1, K, K_BLOCKS, TWO_ID)

  gate = gate_up[:, :ID]
  up = gate_up[:, ID:]
  act = (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)

  _, ID_, H = w2_tile_ref.shape
  I_BLOCKS = w2_scale_tile_ref.shape[1]
  w2 = w2_tile_ref[...].reshape(ID_, H)
  s2 = w2_scale_tile_ref[...].reshape(I_BLOCKS, 1, H)
  down = _blockwise_matmul_chunked(act, w2, s2, ID_, I_BLOCKS, H)

  H_acc = acc_ref.shape[2]
  current = acc_ref[pl.ds(tok, 1), 0, :H_acc]
  acc_ref[pl.ds(tok, 1), 0, :H_acc] = current + (down * weight).reshape(
      1, H_acc
  )


def _outer_kernel(
    ids_ref,
    weights_ref,
    lhs_ref,
    w1_ref,
    w1_scale_ref,
    w2_ref,
    w2_scale_ref,
    o_ref,
    acc_ref,
    out_vmem_ref,
    sem_ref,
    *,
    K,
    ID,
    H,
    K_BLOCKS,
    I_BLOCKS,
    N_TOKENS_,
    N_SLOTS,
    NBUF,
    DTYPE_OUT,
):

  acc_ref[...] = jnp.zeros((N_TOKENS_, 1, H), dtype=jnp.float32)

  pipeline_fn = pltpu.emit_pipeline(
      functools.partial(
          _inner_kernel,
          weights_ref=weights_ref,
          lhs_ref=lhs_ref,
          acc_ref=acc_ref,
          ID=ID,
      ),
      grid=(N_SLOTS,),
      in_specs=(
          pl.BlockSpec(
              (1, K, 2 * ID),
              lambda s: (ids_ref[s], 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=NBUF),
          ),
          pl.BlockSpec(
              (1, K_BLOCKS, 1, 2 * ID),
              lambda s: (ids_ref[s], 0, 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=NBUF),
          ),
          pl.BlockSpec(
              (1, ID, H),
              lambda s: (ids_ref[s], 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=NBUF),
          ),
          pl.BlockSpec(
              (1, I_BLOCKS, 1, H),
              lambda s: (ids_ref[s], 0, 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=NBUF),
          ),
      ),
      out_specs=(),
  )
  pipeline_fn(w1_ref, w1_scale_ref, w2_ref, w2_scale_ref)

  for t in range(N_TOKENS_):
    r = acc_ref[pl.ds(t, 1), 0, :H]
    out_vmem_ref[pl.ds(t, 1), :H] = r.reshape(1, H).astype(DTYPE_OUT)
  out_copy = pltpu.make_async_copy(
      out_vmem_ref, o_ref.at[pl.ds(0, N_TOKENS_)], sem_ref.at[0]
  )
  out_copy.start()
  out_copy.wait()


def cn_moe_full_pipelined(
    hidden_state,
    w1,
    w1_scale,
    w2,
    w2_scale,
    active_ids,
    topk_weights,
    *,
    interpret=False,
    **kwargs,
):
  """Unified low-concurrency MoE kernel with pipelined weight streaming (TP-only).

  Optimized specifically for Tensor Parallelism (TP) where all experts are
  resident on each TPU core. Processes expert slots in natural token-major
  order without argsort or scatter/gather overhead.
  """
  if kwargs.get("use_ep", False):
    raise ValueError(
        "cn_moe_full_pipelined currently only supports Tensor Parallelism (TP)."
    )

  C = hidden_state.shape[0]  # concurrency / num_tokens
  G, K, N1 = w1.shape
  ID = N1 // 2
  _, ID2, H = w2.shape
  if w1_scale.ndim == 3:
    w1_scale = jnp.expand_dims(w1_scale, axis=2)
  if w2_scale.ndim == 3:
    w2_scale = jnp.expand_dims(w2_scale, axis=2)
  K_BLOCKS = w1_scale.shape[1]
  I_BLOCKS = w2_scale.shape[1]

  TOPK_TOTAL = C * TOP_K
  N_SLOTS = TOPK_TOTAL
  NBUF = min(_CN_NBUF, G)

  # Slots stay in natural token-major order without argsort or extra masking
  sorted_ids = active_ids.reshape(TOPK_TOTAL)
  sorted_weights = topk_weights.reshape(TOPK_TOTAL)

  # Ensure 3D shape (C_PAD, 1, K) so dim 0 is an untiled batch dimension
  C_PAD = hidden_state.shape[0]
  lhs = hidden_state.reshape(C_PAD, 1, K)

  # Hint XLA to keep lhs in VMEM
  lhs = pltpu.with_memory_space_constraint(lhs, pltpu.VMEM)

  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=2,  # ids, weights (toks computed as slot // TOP_K)
      in_specs=[vmem_spec] + [any_spec] * 4,  # lhs (3D) in VMEM
      out_specs=any_spec,
      grid=(1,),
      scratch_shapes=[
          pltpu.VMEM((C, 1, H), jnp.float32),  # acc (3D)
          pltpu.VMEM((C, H), jnp.bfloat16),  # out_2d (DMA pad)
          pltpu.SemaphoreType.DMA((1,)),
      ],
  )

  compiler_params = (
      None
      if interpret
      else pltpu.CompilerParams(
          vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
      )
  )

  fused_out = pl.pallas_call(
      functools.partial(
          _outer_kernel,
          K=K,
          ID=ID,
          H=H,
          K_BLOCKS=K_BLOCKS,
          I_BLOCKS=I_BLOCKS,
          N_TOKENS_=C,
          N_SLOTS=N_SLOTS,
          NBUF=NBUF,
          DTYPE_OUT=jnp.bfloat16,
      ),
      out_shape=jax.ShapeDtypeStruct((C, H), jnp.bfloat16),
      grid_spec=grid_spec,
      compiler_params=compiler_params,
      interpret=interpret,
      name="cn_moe_full_pipelined",
  )(sorted_ids, sorted_weights, lhs, w1, w1_scale, w2, w2_scale)

  return fused_out[:C, :].astype(jnp.bfloat16)


# Backwards compatibility alias
cn_gemv_w1w2_fused_pipelined = cn_moe_full_pipelined
