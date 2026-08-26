import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend


def test_attention_spec_uses_tpu_dtype_for_fp8_page_size():
    gpu_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": gpu_spec},
        torch.float8_e4m3fn,
    )
    spec = normalized["layer"]

    assert spec.dtype == torch.float8_e4m3fn
    assert spec.page_size_padded == max(
        PallasAttentionBackend.get_kv_cache_page_size_bytes(
            16,
            2,
            128,
            torch.float8_e4m3fn,
        ),
        gpu_spec.real_page_size_bytes,
    )


def test_mamba_spec_is_left_unchanged():
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=256,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {"layer": mamba_spec},
        torch.float8_e4m3fn,
    )

    assert normalized["layer"] is mamba_spec


def test_hybrid_specs_keep_separate_page_size_by_default():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
    )

    assert normalized["attn"].page_size_bytes != normalized[
        "mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded is None


def test_hybrid_specs_use_uniform_page_size_when_unified_enabled():
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        page_size_padded=1,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=[(2, 8)],
        dtypes=[torch.bfloat16],
        page_size_padded=None,
    )

    normalized = normalize_kv_cache_specs_for_tpu(
        {
            "attn": attention_spec,
            "mamba": mamba_spec,
        },
        torch.float8_e4m3fn,
        enable_unified_kv_layout=True,
    )

    assert normalized["attn"].page_size_bytes == normalized[
        "mamba"].page_size_bytes
    assert normalized["mamba"].page_size_padded == normalized[
        "attn"].page_size_bytes


def test_backend_resolver_overrides_default_backend_page_size():
    # A custom backend (e.g. PallasBatchedRPAAttentionBackend) can have a
    # smaller physical page than the default PallasAttentionBackend; the
    # normalizer must size pages from the backend that will actually
    # materialize the cache, or the unified pool page desyncs from the
    # allocated tensors (assert in allocate_raw_kv_cache_tensors).
    # 1 KV head x head_size 256 (Qwen3.5 at TP=2): the base backend pads the
    # 2 K/V head lanes up to the fp8 packing of 4, doubling its page size
    # relative to the custom batched-RPA storage layout.
    attention_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float8_e4m3fn,
        page_size_padded=None,
    )
    base_page = PallasAttentionBackend.get_kv_cache_page_size_bytes(
        16, 1, 256, torch.float8_e4m3fn)
    custom_page = attention_spec.real_page_size_bytes

    class StubCustomBackend:

        @classmethod
        def get_kv_cache_page_size_bytes(cls, block_size, num_kv_heads,
                                         head_size, cache_dtype):
            return custom_page

    seen = []

    def resolver(layer_name, spec):
        seen.append(layer_name)
        return StubCustomBackend

    normalized = normalize_kv_cache_specs_for_tpu(
        {"attn": attention_spec},
        torch.float8_e4m3fn,
        backend_resolver=resolver,
    )

    assert seen == ["attn"]
    assert normalized["attn"].page_size_bytes == custom_page
    assert normalized["attn"].page_size_bytes < base_page
