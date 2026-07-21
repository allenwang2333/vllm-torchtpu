from dataclasses import replace

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend


def normalize_kv_cache_specs_for_tpu(
    kv_cache_specs: dict[str, KVCacheSpec],
    kv_cache_dtype: str | torch.dtype,
    *,
    enable_unified_block_pool: bool = False,
) -> dict[str, KVCacheSpec]:
    normalized: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_specs.items():
        normalized[layer_name] = _normalize_one_spec(spec, kv_cache_dtype)
    if (enable_unified_block_pool
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
) -> KVCacheSpec:
    if not isinstance(spec, AttentionSpec):
        return spec

    page_size = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        kv_cache_dtype,
    )
    padded_page_size = spec.page_size_padded or 0
    
    real_page_size = spec.real_page_size_bytes
    if spec.kv_quant_mode.is_per_token_head:
        real_page_size += 2 * spec.block_size * spec.num_kv_heads * 4

    page_size = max(page_size, real_page_size, padded_page_size)
    page_size = ((page_size + 127) // 128) * 128

    import sys
    print(f"[DEBUG_NORM_SPEC] {spec.__class__.__name__} page_size_padded={spec.page_size_padded} -> {page_size} (real={real_page_size})", file=sys.stderr, flush=True)

    target_dtype = spec.dtype if spec.dtype == torch.uint8 else kv_cache_dtype

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
