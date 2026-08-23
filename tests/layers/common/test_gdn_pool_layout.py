# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_torchtpu.gdn_pool_layout import (
    POOLED_GDN_SSM_STATE_DTYPE, derive_pooled_gdn_state_layout,
    pooled_gdn_ssm_state_bytes, pooled_gdn_ssm_state_dtype,
    pooled_gdn_ssm_state_itemsize)


@pytest.mark.parametrize(
    ("ssm_bytes", "conv_bytes", "ssm_tokens", "conv_tokens"),
    [
        pytest.param(512 * 1024, 12 * 1024, 512, 16, id="p4"),
        pytest.param(1024 * 1024, 24 * 1024, 1024, 32, id="d2"),
    ],
)
def test_qwen35_layout_includes_conv_tile_padding(
    ssm_bytes: int,
    conv_bytes: int,
    ssm_tokens: int,
    conv_tokens: int,
):
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=1024,
    )

    assert layout.ssm_tokens == ssm_tokens
    assert layout.conv_tokens == conv_tokens
    assert layout.required_tokens == ssm_tokens + conv_tokens
    assert layout.required_bytes == (ssm_tokens + conv_tokens) * 1024


def test_bf16_ssm_checkpoint_halves_only_the_live_ssm_region():
    geometry = dict(num_v_heads=8, head_k_dim=128, head_v_dim=128)

    fp32_bytes = pooled_gdn_ssm_state_bytes(**geometry)
    bf16_bytes = pooled_gdn_ssm_state_bytes(**geometry, use_bf16=True)

    assert pooled_gdn_ssm_state_dtype() == POOLED_GDN_SSM_STATE_DTYPE
    assert pooled_gdn_ssm_state_dtype(use_bf16=True) == "torch.bfloat16"
    assert pooled_gdn_ssm_state_itemsize() == 4
    assert pooled_gdn_ssm_state_itemsize(use_bf16=True) == 2
    assert fp32_bytes == 512 * 1024
    assert bf16_bytes == fp32_bytes // 2

    fp32_layout = derive_pooled_gdn_state_layout(
        ssm_bytes=fp32_bytes, conv_bytes=12 * 1024, token_bytes=1024)
    bf16_layout = derive_pooled_gdn_state_layout(
        ssm_bytes=bf16_bytes, conv_bytes=12 * 1024, token_bytes=1024)
    assert fp32_layout.ssm_tokens == 512
    assert bf16_layout.ssm_tokens == 256
    assert bf16_layout.conv_tokens == fp32_layout.conv_tokens
    assert bf16_layout.required_tokens < fp32_layout.required_tokens
