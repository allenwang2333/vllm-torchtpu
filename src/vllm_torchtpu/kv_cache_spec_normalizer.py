from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec)

from vllm_torchtpu.layers.vllm.attention import (PallasAttentionBackend,
                                                 PallasMLAttentionBackend,
                                                 _resolve_kv_cache_dtype)

# Maps a (layer_name, spec) to the attention backend class serving that layer,
# e.g. TPUModelRunner._resolve_tpu_group_backend. Custom backends (such as
# PallasBatchedRPAAttentionBackend) can have a different physical page layout
# than the default Pallas backend, so page sizes must come from the backend
# that will actually materialize the cache.
BackendResolver = Callable[[str, KVCacheSpec], Any]


def normalize_kv_cache_specs_for_tpu(
    kv_cache_specs: dict[str, KVCacheSpec],
    kv_cache_dtype: str | torch.dtype,
    *,
    enable_unified_kv_layout: bool = False,
    backend_resolver: BackendResolver | None = None,
) -> dict[str, KVCacheSpec]:
    normalized: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_specs.items():
        backend_cls = (backend_resolver(layer_name, spec)
                       if backend_resolver is not None
                       and isinstance(spec, AttentionSpec) else None)
        normalized[layer_name] = _normalize_one_spec(spec, kv_cache_dtype,
                                                     backend_cls)
    if (enable_unified_kv_layout
            and _has_hybrid_attention_and_mamba(normalized)):
        page_size = max(spec.page_size_bytes for spec in normalized.values())
        normalized = {
            layer_name: _pad_page_size(spec, page_size)
            for layer_name, spec in normalized.items()
        }
    return normalized


def _normalize_one_spec(
    spec: KVCacheSpec,
    kv_cache_dtype: str | torch.dtype,
    backend_cls: Any | None = None,
) -> KVCacheSpec:
    if not isinstance(spec, AttentionSpec):
        return spec

    if backend_cls is None:
        backend_cls = (PallasMLAttentionBackend if isinstance(
            spec, MLAAttentionSpec) else PallasAttentionBackend)
    page_size = backend_cls.get_kv_cache_page_size_bytes(
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        kv_cache_dtype,
    )
    padded_page_size = spec.page_size_padded or 0
    page_size = max(page_size, spec.real_page_size_bytes, padded_page_size)
    target_dtype = (_resolve_kv_cache_dtype(kv_cache_dtype)
                    if isinstance(kv_cache_dtype, str) else kv_cache_dtype)
    if spec.page_size_padded == page_size and spec.dtype == target_dtype:
        return spec
    return replace(spec, dtype=target_dtype, page_size_padded=page_size)


def _has_hybrid_attention_and_mamba(
    kv_cache_specs: dict[str, KVCacheSpec], ) -> bool:
    has_attention = any(
        isinstance(spec, AttentionSpec) for spec in kv_cache_specs.values())
    has_mamba = any(
        isinstance(spec, MambaSpec) for spec in kv_cache_specs.values())
    return has_attention and has_mamba


def _pad_page_size(spec: KVCacheSpec, page_size: int) -> KVCacheSpec:
    if not isinstance(spec, (AttentionSpec, MambaSpec)):
        return spec
    if spec.page_size_bytes == page_size:
        return spec
    return replace(spec, page_size_padded=page_size)
