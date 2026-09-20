# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pallas ragged-paged-attention kernel entries for the FLASH_ATTN backend.

The four functions here are the JAX-level entry points that
`PallasAttentionBackendImpl` (and only it) wraps with `pallas.jax_op` to get a
torch custom op. They are the last callers of the in-tree flash kernel,
`ragged_paged_attention` / `attention_bundled`:

* `_pallas_rpa_kernel_impl`   -- shared body of the two non-bundled entries.
* `_pallas_rpa_kernel_default`-- the class default `_kernel_entry`.
* `_pallas_rpa_kernel_local`  -- the tp=1 eagle3 draft variant: no shard_map,
                                 capped KV-fetch block. `tpu_runner` rebinds it
                                 onto the draft layer per instance.
* `_pallas_rpa_kernel_default_bundled`
                              -- block-major KV (`VLLM_TPU_BLOCK_MAJOR_KV=1`).

Split out of `attention.py` so that module holds only the backend and impl
classes. The CUSTOM backend has no entry here at all: it routes onto tokamax's
`PallasMosaicTpuBatchedRpa` singleton, built inline by
`PallasAttentionBackendImpl._tokamax_rpa_closure`.

Each signature is spelled out in full rather than as `*args, **kwargs` --
`pallas.jax_op`'s `_verify_signature` introspects argument types. Nothing here
imports `attention.py`; the dependency runs one way.
"""

import jax

from vllm_torchtpu.layers.core.attention_interface import (
    attention, attention_bundled, ragged_paged_attention)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


def _pallas_rpa_kernel_impl(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    skip_kv_update: bool,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    shard: bool = True,
    kv_block_cap: int | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Shared body of the two Pallas RPA entries below.

    There used to be a `rpa_func` knob here, because the batched entry passed
    `ragged_paged_attention_batched` through it; the batched path is tokamax's
    now, so the only kernel left on this side is `ragged_paged_attention` and
    it is named directly. The `kv_layout` knob went the same way -- only the
    batched entry ever set it, and tokamax is given HEAD_ALONG_SUBLANE pools
    exclusively.
    """
    metadata = AttentionMetadata(
        input_positions=
        None,  # NOTE: vLLM applies RoPE before attention, so input_positions is not consumed here.
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_kv_cache, outputs = attention(
        kv_cache,
        query,
        key,
        value,
        metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=sinks,
        attention_chunk_size=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=ragged_paged_attention,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        shard=shard,
        kv_block_cap=kv_block_cap,
        use_causal_mask=use_causal_mask,
    )
    return new_kv_cache, outputs


def _pallas_rpa_kernel_default(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Default Pallas RPA kernel entry — used by `PallasAttentionBackendImpl`.

    Full signature (not `*args, **kwargs`) is required so `pallas.jax_op` can
    introspect argument types via `_verify_signature`.
    """
    return _pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )


# KV-fetch block cap (tokens) for the tp=1 eagle3 draft's local RPA kernel.
# Capping bkv shrinks the dominant KV scratch tile; the flash kernel loops over more KV chunks.
# 1024 was validated on Llama-3.1-8B (TP=2) and Qwen3-Coder-480b(TP=8). Models with
# many KV heads or longer sequences may need a smaller value to avoid VMEM OOM.
_DRAFT_KV_BLOCK_CAP = 1024


def _pallas_rpa_kernel_local(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Local (non-shard_map) RPA entry for the tp=1 eagle3 draft.

    Identical to `_pallas_rpa_kernel_default` but invokes the kernel WITHOUT
    shard_map (`shard=False`), so the output lands on the worker's own device
    instead of a global chip-0 partition. Also caps the KV-fetch block via
    `kv_block_cap` for speculative decoding draft model.
    """
    return _pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        shard=False,
        kv_block_cap=_DRAFT_KV_BLOCK_CAP,
        use_causal_mask=use_causal_mask,
    )


def _pallas_rpa_kernel_default_bundled(
    kv_cache_bundle: jax.Array,
    layer_idx: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Default bundled Pallas RPA kernel entry for block-major execution.

    Invoked when `VllmModelWrapperContext.kv_cache_bundle` is set. Matches the
    standard kernel entry point interface, taking the full KV cache bundle as the
    first argument and a dynamic int32 scalar layer_idx as the second argument.
    """
    if sinks is not None:
        # Attention sinks are unsupported in the bundled block-major kernel path.
        raise NotImplementedError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: attention sinks are not supported "
            "by the bundled RPA path")
    metadata = AttentionMetadata(
        input_positions=None,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_bundle, outputs = attention_bundled(
        kv_cache_bundle,
        layer_idx,
        query,
        key,
        value,
        metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_chunk_size=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )
    return new_bundle, outputs
