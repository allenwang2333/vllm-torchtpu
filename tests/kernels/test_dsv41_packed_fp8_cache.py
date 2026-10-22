# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed MQA reader needs feature bytes in rows and tokens in columns."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.packed_fp8_cache import (
    PackedFP8Layout,
    pack_fp8_pages,
    read_fp8_records,
    write_fp8_records,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("pages", [1, 3])
def test_fp8_page_bytes_and_scales_keep_token_identity(pages):
    # All FP8 bit patterns, including signed zero and NaNs, must survive packing.
    rng = np.random.default_rng(17)
    bits = rng.integers(0, 256, (pages * 128, 128), dtype=np.uint8)
    keys = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.float8_e4m3fn)
    scales = jnp.arange(pages * 128, dtype=jnp.float32) + 0.5
    packed = np.asarray(pack_fp8_pages(keys, scales)).reshape(pages, 33, 128)
    expected = bits.reshape(pages, 128, 128)
    # Decode by logical coordinates, independently of the encoder's reshape.
    for feature in range(128):
        actual = (
            packed[:, feature // 4].astype(np.uint32) >> (8 * (feature % 4))
        ) & 255
        np.testing.assert_array_equal(actual, expected[:, :, feature])
    np.testing.assert_array_equal(packed[:, 32].view(np.float32).ravel(), scales)


@pytest.mark.cpu_test
def test_fp8_tail_page_preserves_valid_columns_and_zero_padding():
    keys = jnp.zeros((128, 128), jnp.float8_e4m3fn).at[:17].set(1)
    scales = jnp.ones((128,), jnp.float32).at[:17].set(0.5)
    packed = np.asarray(pack_fp8_pages(keys, scales))
    assert np.all(packed[:32, 17:] == 0)
    assert np.all(packed[:32, :17] != 0)
    np.testing.assert_array_equal(packed[32].view(np.float32), scales)
    with pytest.raises(ValueError, match="complete 128-record pages"):
        pack_fp8_pages(keys[:17], scales[:17])


@pytest.mark.cpu_test
def test_incremental_fp8_write_preserves_neighbor_tokens_and_shared_owners():
    layout = PackedFP8Layout(pages=4, page_stride=40, page_offset=5)
    banks = (jnp.full((80, 128), 731, jnp.int32),) * 2
    records = jnp.array([0, 17, 127, 128, 255, 256, 510, -1, 512], jnp.int32)
    keys = jnp.arange(records.size * 128, dtype=jnp.float32).reshape(-1, 128) / 17
    keys = (keys - 32).astype(jnp.bfloat16)
    updated = write_fp8_records(banks, records, keys, layout)
    touched = np.zeros((160, 128), bool)
    for record in np.asarray(records[:7]):
        base = layout.page_offset + record // 128 * layout.page_stride
        touched[base : base + 33, record % 128] = True
    combined = np.concatenate(updated)
    assert np.all(combined[~touched] == 731)

    # Independent reference follows the model's existing round_fp8_activation.
    import torch

    values = torch.from_numpy(np.asarray(keys, dtype=np.float32).copy())
    scales = 2 ** torch.ceil(torch.log2(values.abs().amax(-1).clamp_min(1e-4) / 448))
    expected = (
        (
            (values / scales[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            * scales[:, None]
        )
        .to(torch.bfloat16)
        .float()
        .numpy()
    )
    actual = read_fp8_records(updated, records, layout).astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(actual[:7]), expected[:7])
    np.testing.assert_array_equal(np.asarray(actual[7:]), 0)

    # Publishing one token twice must preserve every byte in all other columns.
    rewritten = write_fp8_records(updated, records[:1], -keys[:1], layout)
    unchanged = touched.copy()
    unchanged[5:38, 0] = False
    np.testing.assert_array_equal(
        np.concatenate(rewritten)[unchanged], combined[unchanged]
    )


@pytest.mark.cpu_test
def test_native_page_copy_preserves_fp8_keys_and_scales():
    layout = PackedFP8Layout(pages=2, page_stride=40, page_offset=5)
    bank = jnp.zeros((80, 128), jnp.int32)
    records = jnp.arange(128, dtype=jnp.int32)
    keys = jnp.sin(jnp.arange(128 * 128).reshape(128, 128)).astype(jnp.bfloat16)
    (written,) = write_fp8_records((bank,), records, keys, layout)
    copied = written.at[45:78].set(written[5:38])
    np.testing.assert_array_equal(
        read_fp8_records((copied,), records, layout),
        read_fp8_records((copied,), records + 128, layout),
    )


@pytest.mark.cpu_test
def test_native_fp8_layout_rejects_pages_crossing_bank_boundaries():
    with pytest.raises(ValueError, match="bank boundary"):
        PackedFP8Layout(pages=2, page_stride=40, page_offset=5).validate((60, 60))
    with pytest.raises(ValueError, match="exceed their owners"):
        PackedFP8Layout(pages=4, page_stride=40, page_offset=5).validate((80,))
