"""V4.1 decoder SWA state and native CSA integration."""

import functools

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_attention_op as attention_ops,
)

from .deepseek_v41_cache import gather_cache_values, masked_row_indices

_attention_op = None


def swa_attention_state(q, pool, records, scale):
    """Compute the BF16 SWA state consumed by the native CSA merger."""
    tokens, heads, dim = q.shape
    tile = 64
    width = records.shape[1]
    if not width:
        return (
            jnp.full((tokens, heads), -1e30, jnp.float32),
            jnp.zeros((tokens, heads), jnp.float32),
            jnp.zeros(q.shape, jnp.float32),
        )
    pad_width = -width % tile
    if pad_width:
        records = jnp.pad(records, ((0, 0), (0, pad_width)), constant_values=-1)

    # Gather BF16 rows once outside; keep FP32 scratch tiled.
    gathered = gather_cache_values(
        pool, jnp.maximum(records, 0).reshape(-1), dim
    ).reshape(tokens, records.shape[1], dim)
    padded_width = records.shape[1]
    steps = padded_width // tile
    q_f32 = q if q.dtype == jnp.float32 else q.astype(jnp.float32)

    valid_all = records >= 0
    scores_all = (
        jnp.einsum(
            "thd,tkd->thk",
            q_f32,
            gathered.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    scores_all = jnp.where(valid_all[:, None], scores_all, -jnp.inf)

    scores0 = scores_all[:, :, :tile]
    kv0 = gathered[:, :tile]
    maximum = jnp.maximum(-1e30, scores0.max(-1))
    exponentials0 = jnp.exp(scores0 - maximum[..., None])
    denominator = exponentials0.sum(-1)
    numerator = jnp.einsum(
        "thk,tkd->thd",
        exponentials0.astype(jnp.bfloat16),
        kv0,
        preferred_element_type=jnp.float32,
    )

    if steps <= 12:
        for start in range(tile, padded_width, tile):
            scores = scores_all[:, :, start : start + tile]
            kv = gathered[:, start : start + tile]
            new_max = jnp.maximum(maximum, scores.max(-1))
            correction = jnp.exp(maximum - new_max)
            exponentials = jnp.exp(scores - new_max[..., None])
            denominator = denominator * correction + exponentials.sum(-1)
            numerator = numerator * correction[..., None] + jnp.einsum(
                "thk,tkd->thd",
                exponentials.astype(jnp.bfloat16),
                kv,
                preferred_element_type=jnp.float32,
            )
            maximum = new_max
        return maximum, denominator, numerator

    scores_tiles = scores_all.reshape(tokens, heads, steps, tile)
    kv_tiles = gathered.reshape(tokens, steps, tile, dim)

    def step(index, carry):
        maximum, denominator, numerator = carry
        scores = scores_tiles[:, :, index, :]
        kv = kv_tiles[:, index, :, :]
        new_max = jnp.maximum(maximum, scores.max(-1))
        correction = jnp.exp(maximum - new_max)
        exponentials = jnp.exp(scores - new_max[..., None])
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction[..., None] + jnp.einsum(
            "thk,tkd->thd",
            exponentials.astype(jnp.bfloat16),
            kv,
            preferred_element_type=jnp.float32,
        )
        return new_max, denominator, numerator

    return jax.lax.fori_loop(1, steps, step, (maximum, denominator, numerator))


def ced_attention_jax(
    q: jax.Array,
    swa: jax.Array,
    main_nope: jax.Array,
    main_rope: jax.Array,
    swa_slots: jax.Array,
    selected: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    sink: jax.Array,
    *,
    scale: float,
    logical_main_page_size: int,
    page_size: int,
    page_stride: int,
    row_offset: int,
) -> jax.Array:
    swa_records = masked_row_indices(swa_slots, page_size, page_stride, row_offset)
    maximum, denominator, numerator = swa_attention_state(q, swa, swa_records, scale)
    sink_f32 = sink if sink.dtype == jnp.float32 else sink.astype(jnp.float32)
    sel_width = selected.shape[1]
    if sel_width == 0:
        denominator += jnp.exp(sink_f32[None] - maximum)
        return (numerator / denominator[..., None]).astype(q.dtype)
    return attention_ops._attention_csa_from_swa(
        q,
        main_nope,
        main_rope,
        selected,
        page_indices,
        cu_q_lens,
        distribution,
        sink_f32,
        numerator.astype(q.dtype),
        denominator,
        maximum,
        sm_scale=scale,
        logical_main_page_size=logical_main_page_size,
        attention_kernel_batch_size=min(4, max(1, 4096 // sel_width)),
    )


def initialize_ced_attention():
    global _attention_op
    if _attention_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    specs = (P(None, "model", None),) + (P(),) * 8 + (P("model"),)

    def attend(
        q: jax.Array,
        swa: jax.Array,
        main_nope: jax.Array,
        main_rope: jax.Array,
        swa_slots: jax.Array,
        selected: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        sink: jax.Array,
        scale: float,
        logical_main_page_size: int,
        page_size: int,
        page_stride: int,
        row_offset: int,
    ) -> jax.Array:
        return jax.shard_map(
            functools.partial(
                ced_attention_jax,
                scale=scale,
                logical_main_page_size=logical_main_page_size,
                page_size=page_size,
                page_stride=page_stride,
                row_offset=row_offset,
            ),
            mesh=mesh,
            in_specs=specs,
            out_specs=P(None, "model", None),
            check_vma=False,
        )(
            q,
            swa,
            main_nope,
            main_rope,
            swa_slots,
            selected,
            page_indices,
            cu_q_lens,
            distribution,
            sink,
        )

    _attention_op = pallas.jax_op(
        "pallas::deepseek_v41_ced_attention",
        attend,
        mesh=mesh,
        input_partition_specs=specs + (None,) * 5,
    )
    _attention_op.register_fake(lambda q, *args: torch.empty_like(q))


def ced_attention(
    q, swa, main, slots, selected, metadata, sink, scale, num_states, *, layout
):
    assert _attention_op is not None
    slots_i32 = slots if slots.dtype == torch.int32 else slots.int()
    selected_i32 = selected if selected.dtype == torch.int32 else selected.int()
    return _attention_op(
        q,
        swa,
        main[0],
        main[1],
        slots_i32,
        selected_i32,
        metadata.block_tables,
        metadata.query_start_loc,
        metadata.request_distribution,
        sink,
        scale,
        num_states,
        layout.page_size,
        layout.page_stride,
        layout.row_offset,
    )
