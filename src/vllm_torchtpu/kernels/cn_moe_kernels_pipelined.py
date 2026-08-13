# Copyright 2026 Google LLC
"""
Implementation that uses emit_pipeline instead of hand-rolled version.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TOP_K = 10
M_PAD = 1
_CN_NBUF = 4


def _forward_fill_ids(sorted_ids, sorted_weights):
    valid = (sorted_weights.astype(jnp.float32) != 0.0)

    def _fill_op(a, b):
        a_id, a_v = a
        b_id, b_v = b
        out_id = jnp.where(b_v, b_id, a_id)
        out_v = a_v | b_v
        return (out_id, out_v)

    filled_ids, _ = jax.lax.associative_scan(_fill_op, (sorted_ids, valid))
    return filled_ids


def _inner_kernel(
    # Leading expert dim kept at 1 (not squeezed) for Mosaic layout.
    w1_tile_ref,  # [1, K, 2ID]
    w1_scale_tile_ref,  # [1, K_BLOCKS, 1, 2ID]
    w2_tile_ref,  # [1, ID, H]
    w2_scale_tile_ref,  # [1, I_BLOCKS, 1, H]
    *,
    ids_ref,
    toks_ref,
    weights_ref,
    lhs_full_ref,  # [C_PAD, 1, K] -- 3D so `tok` indexes a leading dim
    acc_ref,  # [N_TOKENS_, 1, H] fp32 -- likewise
    ID,
    DTYPE_LHS,
):
    slot = pl.program_id(0)
    tok = toks_ref[slot]
    weight = weights_ref[slot]

    @pl.when(weight.astype(jnp.float32) != 0.0)
    def _():
        K_lhs = lhs_full_ref.shape[2]
        lhs_row = lhs_full_ref[pl.ds(tok, 1),
                               pl.ds(0, 1),
                               pl.ds(0, K_lhs)].reshape(1, K_lhs)

        _, K, TWO_ID = w1_tile_ref.shape
        K_BLOCKS = w1_scale_tile_ref.shape[1]
        QB = K // K_BLOCKS

        # Upfront vectorized dequantization in VMEM (zero-stride broadcast across sublanes)
        w1_cast = w1_tile_ref[0].astype(DTYPE_LHS).reshape(K_BLOCKS, QB, TWO_ID)
        s1 = w1_scale_tile_ref[0].astype(DTYPE_LHS)
        w1_dequant = (w1_cast * s1).reshape(K, TWO_ID)

        # Single full MXU systolic GEMV (eliminates 32 micro-matmul loop)
        gate_up = jnp.matmul(lhs_row, w1_dequant, preferred_element_type=jnp.float32)

        gate = gate_up[:, :ID]
        up = gate_up[:, ID:]
        intermediate = (gate * jax.nn.sigmoid(gate) * up).astype(DTYPE_LHS)

        _, ID_, H = w2_tile_ref.shape
        I_BLOCKS = w2_scale_tile_ref.shape[1]
        IB = ID_ // I_BLOCKS

        # Upfront vectorized dequantization in VMEM for W2
        w2_cast = w2_tile_ref[0].astype(DTYPE_LHS).reshape(I_BLOCKS, IB, H)
        s2 = w2_scale_tile_ref[0].astype(DTYPE_LHS)
        w2_dequant = (w2_cast * s2).reshape(ID_, H)

        # Single full MXU systolic GEMV
        down = jnp.matmul(intermediate, w2_dequant, preferred_element_type=jnp.float32)

        H_acc = acc_ref.shape[2]
        current = acc_ref[pl.ds(tok, 1), pl.ds(0, 1), pl.ds(0, H_acc)]
        acc_ref[pl.ds(tok, 1), pl.ds(0, 1),
                pl.ds(0, H_acc)] = current + (down * weight).reshape(
                    1, 1, H_acc)


def _outer_kernel(
    ids_ref,
    toks_ref,
    weights_ref,
    lhs_ref,
    w1_ref,
    w1_scale_ref,
    w2_ref,
    w2_scale_ref,
    o_ref,
    lhs_full_ref,
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
    DTYPE_LHS,
    DTYPE_OUT,
):
    C_PAD = lhs_ref.shape[0]

    # ---- lhs is already in VMEM via memory space constraint; reshape 2D -> 3D on chip ----
    for t in range(C_PAD):
        row = lhs_ref[pl.ds(t, 1), pl.ds(0, K)]
        lhs_full_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, K)] = \
            row.reshape(1, 1, K)

    acc_ref[...] = jnp.zeros((N_TOKENS_, 1, H), dtype=jnp.float32)

    def w1_index_map(slot):
        return (ids_ref[slot], 0, 0)

    def w1_scale_index_map(slot):
        return (ids_ref[slot], 0, 0, 0)

    def w2_index_map(slot):
        return (ids_ref[slot], 0, 0)

    def w2_scale_index_map(slot):
        return (ids_ref[slot], 0, 0, 0)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(
            _inner_kernel,
            ids_ref=ids_ref,
            toks_ref=toks_ref,
            weights_ref=weights_ref,
            lhs_full_ref=lhs_full_ref,
            acc_ref=acc_ref,
            ID=ID,
            DTYPE_LHS=DTYPE_LHS,
        ),
        grid=(N_SLOTS, ),
        in_specs=(
            pl.BlockSpec((1, K, 2 * ID),
                        w1_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, K_BLOCKS, 1, 2 * ID),
                        w1_scale_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, ID, H),
                        w2_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
            pl.BlockSpec((1, I_BLOCKS, 1, H),
                        w2_scale_index_map,
                        pipeline_mode=pl.Buffered(buffer_count=NBUF)),
        ),
        out_specs=(),
    )
    pipeline_fn(w1_ref, w1_scale_ref, w2_ref, w2_scale_ref)

    for t in range(N_TOKENS_):
        r = acc_ref[pl.ds(t, 1), pl.ds(0, 1), pl.ds(0, H)]
        out_vmem_ref[pl.ds(t, 1),
                     pl.ds(0, H)] = r.reshape(1, H).astype(DTYPE_OUT)
    out_copy = pltpu.make_async_copy(out_vmem_ref, o_ref.at[pl.ds(0, N_TOKENS_)],
                                     sem_ref.at[0])
    out_copy.start()
    out_copy.wait()


def cn_gemv_w1w2_fused_pipelined(
    lhs,
    w1,
    w1_scale,
    w2,
    w2_scale,
    sorted_ids,
    sorted_toks,
    sorted_weights,
    *,
    n_tokens,
    use_ep=False,
    interpret=False,
):
    """Pipelined-weight-fetch variant of cn_gemv_w1w2_fused_mb_fp8. Only
    supports NUM_K = NUM_I = 1 (the real production shape)."""
    G, K, N1 = w1.shape
    ID = N1 // 2
    _, ID2, H = w2.shape
    assert ID == ID2
    K_BLOCKS = w1_scale.shape[1]
    I_BLOCKS = w2_scale.shape[1]

    TOPK_TOTAL = n_tokens * TOP_K
    N_SLOTS = TOPK_TOTAL
    NBUF = min(_CN_NBUF, G)

    effective_ids = sorted_ids  # sort removed; invalid IDs already clamped to 0

    # Hint XLA to keep lhs in VMEM — small enough (<=8 tokens * H) to avoid
    # the HBM write+DMA read round-trip from the preceding layer output.
    lhs = pltpu.with_memory_space_constraint(lhs, pltpu.VMEM)

    C_PAD = lhs.shape[0]
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,  # ids, toks, weights
        in_specs=[vmem_spec] + [any_spec] * 4,  # lhs in VMEM; weights in ANY
        out_specs=any_spec,
        grid=(1, ),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, 1, K), lhs.dtype),  # full_lhs (3D)
            pltpu.VMEM((n_tokens, 1, H), jnp.float32),  # acc (3D)
            pltpu.VMEM((n_tokens, H), jnp.bfloat16),  # out_2d (DMA pad)
            pltpu.SemaphoreType.DMA((1, )),
        ],
    )

    compiler_params = None if interpret else pltpu.CompilerParams(
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9))

    return pl.pallas_call(
        functools.partial(
            _outer_kernel,
            K=K,
            ID=ID,
            H=H,
            K_BLOCKS=K_BLOCKS,
            I_BLOCKS=I_BLOCKS,
            N_TOKENS_=n_tokens,
            N_SLOTS=N_SLOTS,
            NBUF=NBUF,
            DTYPE_LHS=lhs.dtype,
            DTYPE_OUT=jnp.bfloat16,
        ),
        out_shape=jax.ShapeDtypeStruct((n_tokens, H), jnp.bfloat16),
        grid_spec=grid_spec,
        compiler_params=compiler_params,
        interpret=interpret,
        name="cn_gemv_w1w2_fused_pipelined",
    )(effective_ids, sorted_toks, sorted_weights, lhs, w1, w1_scale, w2,
     w2_scale)


def cn_moe_full_pipelined(hidden_state,
                          w1,
                          w1_scale,
                          w2,
                          w2_scale,
                          active_ids,
                          topk_weights,
                          *,
                          use_ep=False,
                          interpret=False):
    C = hidden_state.shape[0]
    flat_ids = active_ids.reshape(C * TOP_K)
    flat_weights = topk_weights.reshape(C * TOP_K)
    # Skip the argsort permutation — slots stay in natural token-major order.
    # Clamp invalid (zero-weight) expert IDs to 0 so weight DMA stays valid;
    # the @pl.when(weight != 0) guard in _inner_kernel skips the computation.
    sorted_ids = jnp.where(flat_weights.astype(jnp.float32) != 0.0, flat_ids, 0)
    sorted_toks = jnp.arange(C * TOP_K, dtype=jnp.int32) // TOP_K
    sorted_weights = flat_weights

    fused_out = cn_gemv_w1w2_fused_pipelined(hidden_state,
                                             w1,
                                             w1_scale,
                                             w2,
                                             w2_scale,
                                             sorted_ids,
                                             sorted_toks,
                                             sorted_weights,
                                             n_tokens=C,
                                             use_ep=use_ep,
                                             interpret=interpret)
    return fused_out[:C, :].astype(jnp.bfloat16)
