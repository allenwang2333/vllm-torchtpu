"""EXPERIMENTAL: emit_pipeline-based rewrite of the CN kernel's weight-fetch
path. Replaces the hand-rolled NBUF-way buffer rotation + manual DMA/
semaphore code (_select_buf, _start_w1_dma/_wait_w1_dma, seed phase,
lookahead tables) with Mosaic's native pl.Buffered pipelining -- the same
mechanism cn_moe_kernels.py's mainline neighbor (megablox/gmm_v2.py) uses.

UNVALIDATED ON REAL HARDWARE. Only interpret-mode correctness has been
checked here; emit_pipeline's actual lowering/performance on real Mosaic is
unverified. Kept as a separate file from the working cn_moe_kernels.py so
nothing about the existing, validated kernel is disturbed.

Only handles NUM_K = NUM_I = 1 (the real production shape) for now --
multi-tile K/I support would need the grid extended to (N_SLOTS, NUM_K) /
a second pipeline for w2, not attempted here.
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
    valid = (sorted_weights != 0.0)

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
    lhs_full_ref,  # [C_PAD, K] -- fully VMEM-resident, staged once upfront
    acc_ref,  # [N_TOKENS_, H] fp32 -- fully VMEM-resident accumulator
    ID,
    DTYPE_LHS,
):
    slot = pl.program_id(0)
    tok = toks_ref[slot]
    weight = weights_ref[slot]

    @pl.when(weight != 0.0)
    def _():
        lhs_row = lhs_full_ref[pl.ds(tok, 1)]  # [1, K]

        _, K, TWO_ID = w1_tile_ref.shape
        K_BLOCKS = w1_scale_tile_ref.shape[1]
        QB = K // K_BLOCKS
        # Per quant-block matmul, scale the [1, TWO_ID] output (as gmm_v2
        # does) -- no full-tile fp32 dequant materialized.
        w1_cast = w1_tile_ref[...].reshape(K, TWO_ID).astype(DTYPE_LHS)
        s1 = w1_scale_tile_ref[...].reshape(K_BLOCKS, 1, TWO_ID)
        gate_up = jnp.zeros((1, TWO_ID), dtype=jnp.float32)
        for b in range(K_BLOCKS):
            ks = b * QB
            blk = jnp.matmul(lhs_row[:, ks:ks + QB],
                             w1_cast[ks:ks + QB, :],
                             preferred_element_type=jnp.float32)
            gate_up = gate_up + blk * s1[b].reshape(1, TWO_ID)

        gate = gate_up[:, :ID]
        up = gate_up[:, ID:]
        intermediate = (gate * jax.nn.sigmoid(gate) * up).astype(DTYPE_LHS)

        _, ID_, H = w2_tile_ref.shape
        I_BLOCKS = w2_scale_tile_ref.shape[1]
        IB = ID_ // I_BLOCKS
        w2_cast = w2_tile_ref[...].reshape(ID_, H).astype(DTYPE_LHS)
        s2 = w2_scale_tile_ref[...].reshape(I_BLOCKS, 1, H)
        down = jnp.zeros((1, H), dtype=jnp.float32)
        for b in range(I_BLOCKS):
            is_ = b * IB
            blk = jnp.matmul(intermediate[:, is_:is_ + IB],
                             w2_cast[is_:is_ + IB, :],
                             preferred_element_type=jnp.float32)
            down = down + blk * s2[b].reshape(1, H)

        current = acc_ref[pl.ds(tok, 1)]
        acc_ref[pl.ds(tok, 1)] = current + down * weight


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
    C_PAD = lhs_full_ref.shape[0]

    # ---- Stage the full LHS into VMEM once (same as the existing kernel) --
    full_lhs_copy = pltpu.make_async_copy(lhs_ref.at[pl.ds(0, C_PAD)],
                                          lhs_full_ref, sem_ref.at[0])
    full_lhs_copy.start()
    full_lhs_copy.wait()

    acc_ref[...] = jnp.zeros((N_TOKENS_, H), dtype=jnp.float32)

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

    out_vmem_ref[...] = acc_ref[...].astype(DTYPE_OUT)
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

    effective_ids = _forward_fill_ids(sorted_ids, sorted_weights)

    C_PAD = lhs.shape[0]
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,  # ids, toks, weights
        in_specs=[any_spec] * 5,
        out_specs=any_spec,
        grid=(1, ),
        scratch_shapes=[
            pltpu.VMEM((C_PAD, K), lhs.dtype),
            pltpu.VMEM((n_tokens, H), jnp.float32),
            pltpu.VMEM((n_tokens, H), jnp.bfloat16),
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
    sort_order = jnp.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sort_order]
    sorted_toks = (sort_order // TOP_K).astype(jnp.int32)
    sorted_weights = flat_weights[sort_order]

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
