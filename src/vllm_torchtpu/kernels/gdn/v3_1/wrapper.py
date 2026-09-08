import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3_1 import compute_conv1d
from vllm_torchtpu.kernels.gdn.v3_1 import compute_gdn
from vllm_torchtpu.kernels.gdn.v3_1 import config
from vllm_torchtpu.kernels.gdn.v3_1 import memory_ref
from vllm_torchtpu.kernels.gdn.v3_1 import metadata
from vllm_torchtpu.kernels.gdn.v3_1 import vmem_ldst


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Array,  # (seq, chunk, 1, dim_size)
    b_slot_ref: jax.Array,  # (seq, chunk, 1, num_v_heads)
    a_slot_ref: jax.Array,  # (seq, chunk, 1, num_v_heads)
    conv_state_slot_ref: jax.Array,  # (seq, prev_kernel_size, 1, dim_size)
    recurrent_slot_ref: jax.Array,  # (seq, num_v_heads, kq_head, v_head)
    # Outputs.
    out_slot_ref: jax.Array,  # (seq * chunk, num_v_heads, v_head)
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
):
  p_id = pl.program_id(0)

  # Prepare states.
  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_mask_states(
      metadata_ref=metadata_ref,
      p_id=p_id,
      conv_state_slot_ref=conv_state_slot_ref,
      recurrent_slot_ref=recurrent_slot_ref,
      carry_conv_scratch_ref=carry_conv_scratch_ref,
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
      cfg=cfg,
  )

  # Step 1: Conv1D.
  # NOTE: Conv1D requires performing sliding window where inputs are slided
  # across rows. If typical 2D layout was used, multiple rows are stored in a
  # single register which necessitate costly shuffling for every sliding.
  # Therefore, it is extremely important to leverage compact layout that
  # ensures 1 register only stores data from 1 row.
  qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
  qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

  # Prepare conv1d weights.
  conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
  conv_bias = None
  if weights_ref.conv.bias is not None:
    conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes,
      lhs=qkv_in_compact,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      cfg=cfg,
  )

  conv_state_slot_ref[...] = new_conv_state.reshape(
      conv_state_slot_ref.shape
  ).astype(conv_state_slot_ref.dtype)
  if carry_conv_scratch_ref is not None:
    carry_conv_scratch_ref[...] = new_conv_state[:, -1]

  # Apply activation function.
  qkv_out_compact = jax.nn.silu(qkv_out_compact)

  # Step 2: GDN.

  # Prepare gdn weights.
  padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
  a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
  dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
  # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
  # Passing refs into gdn.py breaks strict separation of concerns.
  if cfg.chunk_size == 1:
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = compute_gdn.recurrent_gdn(
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  else:
    q_large, k_large, v_large, b_large, a_large = (
        vmem_ldst.load_activation_as_large(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = compute_gdn.chunked_gdn(
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  # Store output and recurrent to vmem.
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    states_ref: jax.Array | tuple[jax.Array, jax.Array],
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    states_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
):
  del states_out_ref

  if cfg.is_shared:
    assert not isinstance(states_ref, tuple)
    if states_ref.ndim == 5:
      num_manager_blocks = (
          states_ref.shape[0] * 128
      ) // cfg.manager_block_tokens
      new_states_ref = states_ref.bitcast(jnp.uint32).reshape(
          num_manager_blocks, cfg.manager_block_tokens, 1, 128
      )
    else:
      num_states = states_ref.shape[0]
      new_states_ref = states_ref.bitcast(jnp.uint32).reshape(
          num_states, -1, 1, 128
      )
    conv_state_ref = recurrent_state_ref = new_states_ref
  else:
    assert len(states_ref) == 2
    conv_state_ref = states_ref[0]
    recurrent_state_ref = states_ref[1]

  qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
      memory_ref.create_allocs(
          metadata_ref=metadata_ref,
          qkv_ref=qkv_ref,
          b_ref=b_ref,
          a_ref=a_ref,
          out_ref=out_ref,
          conv_state_ref=conv_state_ref,
          recurrent_state_ref=recurrent_state_ref,
          cfg=cfg,
      )
  )

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(
          inner_kernel,
          cfg=cfg,
      ),
      grid=(num_tiles,),
      in_specs=(
          qkv_alloc.spec,
          b_alloc.spec,
          a_alloc.spec,
          conv_alloc.spec,
          recurrent_alloc.spec,
      ),
      out_specs=(out_alloc.spec,),
  )

  @pl.with_scoped(
      allocations=(
          qkv_alloc,
          b_alloc,
          a_alloc,
          conv_alloc,
          recurrent_alloc,
          out_alloc,
      ),
  )
  def _run(allocations):
    pipeline_func(
        qkv_ref,
        b_ref,
        a_ref,
        conv_state_ref,
        recurrent_state_ref,
        out_ref,
        scratches=(
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
        ),
        allocations=allocations,
    )

  _run()


@jax.jit(
    donate_argnames=("states",),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
        "conv_dtype",
        "recurrent_dtype",
        "conv_offset_override",
        "recurrent_offset_override",
        "manager_block_tokens",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # (batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size)
    b: jax.Array,  # (batch_size, n_v)
    a: jax.Array,  # (batch_size, n_v)
    states: jax.Array | tuple[jax.Array, jax.Array],  # (num_seqs + 1, num_chunks, num_lanes)
    conv_weight: jax.Array,  # (kernel_size - 1, dim_size)
    conv_bias: jax.Array | None,  # (dim_size,)
    a_log: jax.Array,  # (n_v, )
    dt_bias: jax.Array,  # (n_v, )
    query_start_loc: jax.Array,  # (num_seqs + 1, )
    state_indices: jax.Array,  # (num_seqs,)
    distribution: jax.Array,  # (3, )
    seq_lens: jax.Array,  # (num_seqs, )
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
    conv_dtype: jnp.dtype | None = None,
    recurrent_dtype: jnp.dtype | None = None,
    conv_offset_override: int | None = None,
    recurrent_offset_override: int | None = None,
    manager_block_tokens: int = 4224,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  act_out_dtype = qkv.dtype

  is_shared = isinstance(states, jax.Array)
  if is_shared:
    assert conv_dtype is not None
    assert recurrent_dtype is not None
    if states.ndim == 5:
      if conv_offset_override is None:
        conv_offset_override = 4096
      if recurrent_offset_override is None:
        recurrent_offset_override = 0
  else:
    assert len(states) == 2
    conv_dtype = states[0].dtype
    recurrent_dtype = states[1].dtype

  if conv_dtype not in (jnp.float32, jnp.bfloat16):
    raise NotImplementedError(f"Unsupported conv state dtype: {conv_dtype}")
  if recurrent_dtype != jnp.float32:
    raise NotImplementedError("Only supports fp32 recurrent state.")

  qkv = qkv.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.astype(jnp.float32)

  # Step 1: Validate inputs.
  num_seqs = state_indices.size
  batch_size, dim = qkv.shape
  assert conv_weight.shape == (dim, 1, kernel_size)
  if conv_bias is not None:
    assert conv_bias.shape == (dim,)
  assert query_start_loc.shape == (num_seqs + 1,)
  assert state_indices.shape == (num_seqs,)
  assert distribution.shape == (3,)
  act_in_dtype = qkv.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = pl.cdiv(batch_size, packing) * packing
  decode_tile_size = min(decode_tile_size, batch_size)
  mixed_tile_size = min(mixed_tile_size, batch_size)
  aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

  batch_padding_size = padded_batch_size - batch_size
  num_v_padding_size = aligned_num_v_heads - n_v
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
  a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))

  qkv = qkv.reshape(padded_batch_size, 1, -1)
  b = b.reshape(padded_batch_size, 1, -1)
  a = a.reshape(padded_batch_size, 1, -1)

  # Step 3: Weights pre-processing.
  # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
  # loading stage.
  conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
  conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

  # Step 4: Wrap inputs for the kernel.
  conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
  gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
  weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

  # Step 5: Create specs.
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

  def call_kernel(
      in_states: jax.Array,
      in_act: jax.Array | None,
      mode: config.GDNMode,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    if mode == config.GDNMode.BATCHED:
      tile_size = decode_tile_size
    else:
      tile_size = mixed_tile_size

    cfg = config.GDNConfig(
        mode=mode,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=recurrent_dtype,
            conv_state=conv_dtype,
        ),
        is_shared=is_shared,
        conv_offset_override=conv_offset_override,
        recurrent_offset_override=recurrent_offset_override,
        manager_block_tokens=manager_block_tokens,
    )

    # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
    # but will be CSEed by compiler.
    if mode == config.GDNMode.BATCHED:
      metadata_obj = metadata.compute_batched_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          end_seq=distribution[0],
      )
    else:
      metadata_obj = metadata.compute_per_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          start_seq=distribution[0],
          end_seq=distribution[-1],
      )

    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
    state_spec = jax.tree.map(lambda _: hbm_spec, in_states)

    # Step 7: Handle case where write needs to be done in existing out.
    in_out_spec = None

    input_output_aliases = {}
    offset = len(metadata_obj) + 3
    num_states = 1 if cfg.is_shared else 2
    for i in range(num_states):
      input_output_aliases[offset + i] = 1 + i
    offset += num_states

    out_shape = cfg.get_out_shape()

    if in_act is None and zero_initialize_out:
      in_act = jnp.zeros_like(out_shape)
    if in_act is not None:
      out_shape = in_act
      in_out_spec = hbm_spec
      input_output_aliases[offset] = 0

    return pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg),
        out_shape=(out_shape, in_states),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            state_spec,
            in_out_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, state_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
    )(
        metadata_obj,
        qkv,
        b,
        a,
        in_states,
        in_act,
        weights,
    )

  out_act, out_states = call_kernel(states, None, config.GDNMode.BATCHED)
  out_act, out_states = call_kernel(out_states, out_act, config.GDNMode.PER_SEQ)
  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  return out_states, out_act
