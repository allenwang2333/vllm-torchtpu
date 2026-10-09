# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TD7.25: 3D broadcast dequant_fp8_rows_jax and HostRowTable caching."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41 import host_row_gather as hrg
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common import host_row_table


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_dequant_fp8_rows_jax_3d_broadcast_matches_repeat_reference(block_size: int):
    rng = np.random.default_rng(block_size)
    row_bytes = 256
    num_blocks = row_bytes // block_size
    raw_bytes = rng.integers(0, 256, (17, row_bytes), dtype=np.uint8)
    raw_bytes[0, :block_size] = 0x7F  # +NaN in FP8
    raw_bytes[1, :block_size] = 0xFF  # -NaN in FP8
    raw = jnp.asarray(raw_bytes.view(np.int32))
    scale_np = rng.integers(1, 250, (17, num_blocks), dtype=np.uint8)
    scale_np[0, 0] = hrg.INVALID_SCALE
    scale_np[1, 0] = hrg.INVALID_SCALE
    scale_np[2, -1] = 0  # subnormal 2**-127 flushes to +0.0
    scale = jnp.asarray(scale_np)

    got = hrg.dequant_fp8_rows_jax(raw, scale, block_size)

    values_ref = (
        jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
        .reshape(raw.shape[0], -1)
        .astype(jnp.float32)
    )
    power_ref = jax.lax.bitcast_convert_type(scale.astype(jnp.int32) << 23, jnp.float32)
    power_ref = jnp.repeat(power_ref, block_size, axis=-1)
    invalid_ref = jnp.repeat(scale == hrg.INVALID_SCALE, block_size, axis=-1)
    expected = jnp.where(invalid_ref, 0.0, values_ref * power_ref).astype(jnp.bfloat16)

    got_bits = np.asarray(jax.lax.bitcast_convert_type(got, jnp.int16))
    exp_bits = np.asarray(jax.lax.bitcast_convert_type(expected, jnp.int16))
    np.testing.assert_array_equal(got_bits, exp_bits)


def test_host_row_table_caches_lookup_op_and_skips_redundant_casts(monkeypatch):
    monkeypatch.setattr(host_row_table, "_install", lambda: None)
    monkeypatch.setattr(
        torch.ops, "ao", SimpleNamespace(offload=lambda t: t.clone()), raising=False
    )

    seen_gather_ids: list[torch.Tensor] = []
    seen_lookup_args: list[tuple[torch.Tensor, torch.Tensor]] = []
    lookup_op_builds: list[int] = []

    def gather_op(num_chunks, row_bytes):
        def op(row_ids, *chunks):
            assert len(chunks) == num_chunks
            seen_gather_ids.append(row_ids)
            out = hrg.gather_rows_jax(
                jnp.asarray(row_ids.numpy()),
                *(jnp.asarray(c.numpy()) for c in chunks),
                row_bytes=row_bytes,
                interpret=True,
            )
            return torch.from_numpy(np.asarray(out).copy())

        return op

    def lookup_and_dequant_op(num_chunks, row_bytes, block_size):
        lookup_op_builds.append(block_size)

        def op(row_ids, valid, scale, *chunks):
            assert len(chunks) == num_chunks
            seen_lookup_args.append((row_ids, valid))
            out = hrg.lookup_and_dequant_rows_jax(
                jnp.asarray(row_ids.numpy()),
                jnp.asarray(valid.numpy()),
                jnp.asarray(scale.numpy()),
                *(jnp.asarray(c.numpy()) for c in chunks),
                row_bytes=row_bytes,
                block_size=block_size,
                interpret=True,
            )
            bits = np.asarray(jax.lax.bitcast_convert_type(out, jnp.int16))
            return torch.from_numpy(bits.copy()).view(torch.bfloat16)

        return op

    monkeypatch.setattr(host_row_table, "_gather_op", gather_op)
    monkeypatch.setattr(host_row_table, "_lookup_and_dequant_op", lookup_and_dequant_op)

    rng = np.random.default_rng(19)
    rows = rng.integers(0, 256, (128, 256), dtype=np.uint8)
    rows[rows == 0x7F] = 0x7E
    rows[rows == 0xFF] = 0xFE
    scales = rng.integers(100, 150, (128, 2), dtype=np.uint8)

    table = host_row_table.HostRowTable(
        torch.from_numpy(rows).view(torch.float8_e4m3fn),
        torch.device("cpu"),
    )

    # 1. gather: int32 input skips .to(torch.int32) allocation; int64 casts.
    ids_i32 = torch.tensor([0, 7, 63, 127], dtype=torch.int32)
    ids_i64 = ids_i32.to(torch.int64)
    got_i32 = table.gather(ids_i32)
    got_i64 = table.gather(ids_i64)
    assert seen_gather_ids[0] is ids_i32
    assert seen_gather_ids[1] is not ids_i64 and seen_gather_ids[1].dtype == torch.int32
    assert torch.equal(got_i32, got_i64)

    # 2. lookup_and_dequant: caches op per block_size and skips matching casts.
    valid_bool = torch.tensor([True, False, True, True], dtype=torch.bool)
    valid_u8 = valid_bool.to(torch.uint8)
    scales_t = torch.from_numpy(scales)

    out1 = table.lookup_and_dequant(ids_i32, valid_bool, scales_t, 128)
    assert lookup_op_builds == [128]
    assert seen_lookup_args[0][0] is ids_i32
    assert seen_lookup_args[0][1] is valid_bool

    # Second call with same block_size hits _lookup_ops cache; int64/uint8 cast.
    out2 = table.lookup_and_dequant(ids_i64, valid_u8, scales_t, 128)
    assert lookup_op_builds == [128]
    assert (
        seen_lookup_args[1][0] is not ids_i64
        and seen_lookup_args[1][0].dtype == torch.int32
    )
    assert (
        seen_lookup_args[1][1] is not valid_u8
        and seen_lookup_args[1][1].dtype == torch.bool
    )
    assert torch.equal(out1.view(torch.int16), out2.view(torch.int16))

    exp_scale = scales_t[ids_i64].clone()
    exp_scale[~valid_bool] = hrg.INVALID_SCALE
    exp_deq = common_engram.dequant_fp8_rows(got_i32, exp_scale, 128)
    assert torch.equal(out1.view(torch.int16), exp_deq.view(torch.int16))
