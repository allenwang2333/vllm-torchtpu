from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)


def _qwen35_configs(
    total_q_tokens: int,
) -> tuple[configs.ModelConfigs, configs.ServingConfigs]:
    model = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=2,
        head_dim=256,
        mask_value=-1e9,
    )
    serving = configs.ServingConfigs(
        num_seqs=32,
        page_size=2304,
        total_q_tokens=total_q_tokens,
        num_page_indices=32 * 29,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    return model, serving


@pytest.fixture(autouse=True)
def _qwen35_tpu_geometry(monkeypatch):
    monkeypatch.setattr(
        wrapper.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(
            generation=7,
            chip_version="v7x",
            num_lanes=128,
            mxu_column_size=256,
            vmem_capacity_bytes=64 * 1024 * 1024,
            fp8_ops_per_second=2,
            bf16_ops_per_second=1,
        ),
    )


def test_verify_profile_does_not_retune_decode_or_mixed_profiles():
    """Adding long-context verify must leave other traffic profiles unchanged."""
    model, serving = _qwen35_configs(total_q_tokens=64)

    decode_off, verify_off, mixed_off = wrapper.calculate_block_sizes(
        model,
        serving,
        vmem_limit_bytes=64 * 1024 * 1024,
        verify_q_bucket=None,
    )
    decode_on, verify_on, mixed_on = wrapper.calculate_block_sizes(
        model,
        serving,
        vmem_limit_bytes=64 * 1024 * 1024,
        verify_q_bucket=4,
    )

    assert verify_off is None
    assert verify_on is not None
    assert verify_on.bq_sz == 4
    assert decode_on == decode_off
    assert mixed_on == mixed_off


def test_verify_case_uses_middle_request_range_and_cached_kv():
    """VERIFY consumes the middle segment with mixed-style KV semantics."""
    start, end = configs.RpaCase.VERIFY.get_range(
        jnp.asarray([2, 5, 8], dtype=jnp.int32))
    model, serving = _qwen35_configs(total_q_tokens=64)
    block = configs.BlockSizes(4, 2, 2304, 2, 3)
    verify = configs.RpaConfigs(
        block=block,
        model=model,
        serve=serving,
        mode=configs.RpaCase.VERIFY,
        vmem_limit_bytes=64 * 1024 * 1024,
    )

    assert (int(start), int(end)) == (2, 5)
    assert verify.bkv_p_cache == 1
    assert verify.bkv_p_new == 2
    assert not verify.fuse_accum
