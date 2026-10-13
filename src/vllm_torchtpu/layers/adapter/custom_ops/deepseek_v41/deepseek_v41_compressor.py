"""V4.1 C1/C2 compression and compressed KV publication."""

import types
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import compressor as compressor_upstream

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.layers.adapter.normalization import rms_norm
from vllm_torchtpu.layers.adapter.upstream_bindings import device_hooks

from .deepseek_v41_backend import AttentionBackend
from .deepseek_v41_cache import (
    NativeCacheBinding,
    cache_slots,
    read_cache,
    row_indices,
    scatter_cache_words,
    token_requests,
    write_cache,
)

_write_op = None
_TPU_OP_CACHE: dict[tuple, Callable] = {}


def _get_tpu_op(key: tuple, fn: Callable) -> Callable:
    op = _TPU_OP_CACHE.get(key)
    if op is not None:
        return op
    name = f"compressor_op_{len(_TPU_OP_CACHE)}"
    specialized = types.FunctionType(
        fn.__code__.replace(co_name=name),
        fn.__globals__,
        name,
        fn.__defaults__,
        fn.__closure__,
    )
    op = torch.compile(specialized, backend="tpu", dynamic=False)
    _TPU_OP_CACHE[key] = op
    return op


def _compress_ratio1_mask_impl(
    result: torch.Tensor,
    slot_mapping: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = slot_mapping[:count]
    valid = slots >= 0
    return valid, result.masked_fill(~valid[:, None], 0)


def _compress_ratio2_prep_impl(
    kv_score_f32: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    capacity: int,
    state_layout=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = kv_score_f32.shape[0]
    slots = slot_mapping[:count]
    req_idx = token_to_req_indices[:count]
    valid = slots >= 0
    # Rows, requests and ring slots are < 2**31: native int32 + 1D gathers.
    rows = torch.arange(count, dtype=torch.int32, device=kv_score_f32.device)
    requests = (req_idx if req_idx.dtype == torch.int32 else req_idx.int()).clamp(
        max=query_start_loc.numel() - 2
    )
    qsl_i32 = (
        query_start_loc
        if query_start_loc.dtype == torch.int32
        else query_start_loc.int()
    )
    starts = qsl_i32.index_select(0, requests)
    ends = qsl_i32.index_select(0, requests + 1)
    previous_slots = (slots if slots.dtype == torch.int32 else slots.int()).clamp(
        min=0
    ) // capacity * capacity + (
        (positions if positions.dtype == torch.int32 else positions.int()) - 1
    ) % capacity
    previous = torch.where(
        (rows > starts)[:, None],
        kv_score_f32.index_select(0, (rows - 1).clamp(min=0)),
        read_cache(state_cache, previous_slots, layout=state_layout),
    )
    # All reads above refer to the old ring. Keep unique physical writes.
    publish = valid & (rows >= ends - capacity)
    valid_out = valid & (positions % 2 == 1)
    return previous, publish, valid_out


def save_compress_norm(
    kv_score,
    positions,
    state_cache,
    slot_mapping,
    query_start_loc,
    token_to_req_indices,
    norm_weight,
    eps,
    ratio,
    latent,
    *,
    state_layout=None,
):
    """Tensor implementation of the compressor_upstream fused compression operation.

    Read the previous row before publishing ring writes. For a long prefill,
    previous rows come directly from this chunk and only the final ring-sized
    suffix is stored, so writes never race aliases from earlier ring wraps.
    """
    count = kv_score.shape[0]
    dim = norm_weight.numel()
    kv_score_f32 = kv_score if kv_score.dtype == torch.float32 else kv_score.float()
    values = kv_score_f32[:, :dim]
    use_tpu_op = kv_score.device.type == "tpu" and not torch.compiler.is_compiling()
    if ratio == 2:
        capacity = (
            state_layout.page_size if state_layout is not None else state_cache.shape[1]
        )
        cache = state_cache
        # Rows, requests and ring slots are < 2**31: native int32 + 1D gathers.
        if use_tpu_op:
            prep_op = _get_tpu_op(
                (
                    "compress_r2_prep",
                    kv_score_f32.shape,
                    positions.shape,
                    positions.dtype,
                    cache.shape,
                    cache.dtype,
                    slot_mapping.shape,
                    slot_mapping.dtype,
                    query_start_loc.shape,
                    query_start_loc.dtype,
                    token_to_req_indices.shape,
                    token_to_req_indices.dtype,
                    capacity,
                    state_layout,
                ),
                lambda ks, pos, sc, sm, qsl, req: _compress_ratio2_prep_impl(
                    ks,
                    pos,
                    sc,
                    sm,
                    qsl,
                    req,
                    capacity,
                    state_layout,
                ),
            )
            previous, publish, valid = prep_op(
                kv_score_f32,
                positions,
                cache,
                slot_mapping,
                query_start_loc,
                token_to_req_indices,
            )
        else:
            previous, publish, valid = _compress_ratio2_prep_impl(
                kv_score_f32,
                positions,
                cache,
                slot_mapping,
                query_start_loc,
                token_to_req_indices,
                capacity,
                state_layout,
            )
        scores = torch.stack((previous[:, dim:], kv_score_f32[:, dim:]), 1)
        pair = torch.stack((previous[:, :dim], values), 1)
        values = (pair * scores.softmax(1)).sum(1)
        # All reads above refer to the old ring. Keep unique physical writes.
        write_cache(
            cache,
            slot_mapping[:count],
            kv_score_f32,
            publish,
            layout=state_layout,
        )
    elif ratio != 1:
        raise ValueError(f"V4.1 compression ratio must be 1 or 2, got {ratio}")
    else:
        result = rms_norm(values, norm_weight, eps)
        if use_tpu_op:
            mask_op = _get_tpu_op(
                (
                    "compress_r1_mask",
                    result.shape,
                    result.dtype,
                    slot_mapping.shape,
                    slot_mapping.dtype,
                    count,
                ),
                lambda res, sm: _compress_ratio1_mask_impl(res, sm, count),
            )
            _, masked = mask_op(result, slot_mapping)
            latent.copy_(masked)
            return
        _, masked = _compress_ratio1_mask_impl(result, slot_mapping, count)
        latent.copy_(masked)
        return
    result = rms_norm(values, norm_weight, eps)
    latent.copy_(result.masked_fill(~valid[:, None], 0))


def pack_csa_values(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Encode already-normalized/rotated V4.1 KV in the V4 lane-periodic layout."""
    assert values.ndim == 2 and values.shape[1] == 512
    assert values.dtype == jnp.bfloat16
    groups = values[:, :448].astype(jnp.float32).reshape(-1, 7, 64)
    amax = jnp.maximum(jnp.abs(groups).max(axis=1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    # Construct an exact power of two: exp2's approximate result can sit just
    # below it, making extraction of the FP32 exponent select the wrong E8M0.
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (groups / scales[:, None]).astype(jnp.float8_e4m3fn)
    fp8_bytes = jax.lax.bitcast_convert_type(quantized, jnp.uint8).reshape(-1, 448)
    slabs = jnp.concatenate((fp8_bytes, scale_bytes), axis=1).reshape(-1, 4, 128)
    return csa_cache_layout.slabs_to_words(slabs), csa_cache_layout.rope_words(
        values[:, 448:]
    )


def write_csa_values(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    records: jax.Array,
    values: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """A logical compressed slot names matching rows in two native arrays."""
    nope, rope = pack_csa_values(values)
    updated_nope = scatter_cache_words(nope_cache.reshape(-1, 128), records, nope)
    updated_rope = scatter_cache_words(rope_cache.reshape(-1, 128), records, rope)
    return updated_nope.reshape(nope_cache.shape), updated_rope.reshape(
        rope_cache.shape
    )


def _compressed_records_impl(
    slots: torch.Tensor,
    valid: torch.Tensor,
    num_states: int,
    page_stride: int,
) -> torch.Tensor:
    return row_indices(
        slots if slots.dtype == torch.int32 else slots.int(),
        num_states,
        page_stride,
        0,
    ).masked_fill(~valid, -1)


def write_compressed_cache(caches, slots, values, valid, num_states):
    nope, rope = caches
    page_stride = nope.shape[1]
    if slots.device.type == "tpu" and not torch.compiler.is_compiling():
        rec_op = _get_tpu_op(
            (
                "compressed_records",
                slots.shape,
                slots.dtype,
                valid.shape,
                num_states,
                page_stride,
            ),
            lambda s, v: _compressed_records_impl(s, v, num_states, page_stride),
        )
        records = rec_op(slots, valid)
    else:
        records = _compressed_records_impl(slots, valid, num_states, page_stride)
    assert _write_op is not None
    updated_nope, updated_rope = _write_op(nope, rope, records, values)
    nope.copy_(updated_nope)
    rope.copy_(updated_rope)


def initialize_compressor_ops():
    global _write_op
    if _write_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    mesh = get_vllm_model_wrapper_context().mesh
    write = jax.shard_map(
        write_csa_values,
        mesh=mesh,
        in_specs=(P(),) * 4,
        out_specs=(P(), P()),
        check_vma=False,
    )
    _write_op = pallas.jax_op(
        "pallas::deepseek_v41_write_compressed_kv",
        write,
        mesh=mesh,
        donate_argnums=(0, 1),
        input_partition_specs=(P(),) * 4,
    )
    _write_op.register_fake(
        lambda nope, rope, *args: (torch.empty_like(nope), torch.empty_like(rope))
    )


class CompressorStateCache(
    NativeCacheBinding, compressor_upstream.CompressorStateCache
):
    """Upstream state ownership with a shared native cache binding."""

    def get_attn_backend(self):
        return AttentionBackend


class DeepseekCompressor(compressor_upstream.DeepseekCompressor):
    """Inherit weights and state ownership; consume the shared TPU metadata."""

    def __init__(self, *args, **kwargs):
        with device_hooks(
            compressor_upstream, CompressorStateCache=CompressorStateCache
        ):
            super().__init__(*args, **kwargs)

    def project(self, hidden_states):
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        layer = self.fused_wkv_wgate
        raw_weight = layer.weight
        flipped = bool(getattr(layer, WEIGHT_FLIPPED_ATTR, False))
        cache_key = (id(raw_weight), raw_weight._version, flipped)
        cached = getattr(layer, "_project_weight_f32_cache", None)
        if cached is not None and cached[0] == cache_key:
            weight = cached[1]
        else:
            weight = (
                raw_weight if raw_weight.dtype == torch.float32 else raw_weight.float()
            )
            if not flipped:
                weight = weight.t()
            layer._project_weight_f32_cache = (cache_key, weight)
        # Upstream accumulates the unquantized KV/score projection in FP32.
        # Keep that precision through pooling and normalization.
        hs_f32 = (
            hidden_states
            if hidden_states.dtype == torch.float32
            else hidden_states.float()
        )
        return torch.mm(hs_f32, weight)

    def forward(self, kv_score, positions):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        if self.state_cache is None:
            state_cache = query_start_loc = requests = None
            state_layout = None
            source = self._static_forward_context[self.k_cache_prefix]
            slots = cache_slots(
                metadata[self.k_cache_prefix],
                positions,
                rows_per_page=source.cache_num_states,
            )
        else:
            state_metadata = metadata[self.state_cache.prefix]
            binding = (
                self.state_cache.native() if self.state_cache.has_native_cache else None
            )
            state_cache = self.state_cache.kv_cache
            state_layout = binding.layout if binding is not None else None
            query_start_loc = state_metadata.query_start_loc
            requests = token_requests(state_metadata, positions)
            slots = cache_slots(
                state_metadata,
                positions,
                ring=True,
                rows_per_page=(
                    state_layout.page_size
                    if state_layout is not None
                    else state_cache.shape[1]
                ),
                requests=requests,
            )
        latent = torch.empty(
            (kv_score.shape[0], self.head_dim),
            dtype=torch.bfloat16,
            device=kv_score.device,
        )
        save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slots,
            query_start_loc,
            requests,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
            state_layout=state_layout,
        )
        return latent
