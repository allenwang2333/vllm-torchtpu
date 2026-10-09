# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for linear_common.py 2D fast paths and full-module coverage (TD7.20)."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    MODEL_MATMUL_FUSION_TRUTH_TABLE,
    WEIGHT_FLIPPED_ATTR,
    _get_quantized_matmul_fp4_op,
    _get_quantized_matmul_op,
    _get_x_q_dtype,
    _quantized_matmul_fp4_jax,
    _quantized_matmul_jax,
    get_model_matmul_fusion_assignment,
    quantized_matmul,
    quantized_matmul_fp4,
    reorder_concatenated_tensor_for_sharding,
    sharded_quantized_matmul,
    slice_sharded_tensor_for_concatenation,
)


def test_constants_and_get_x_q_dtype():
    assert KEEP_VLLM_LAYOUT_ATTR == "_tpu_keep_vllm_weight_layout"
    assert WEIGHT_FLIPPED_ATTR == "_tpu_weight_flipped"

    assert _get_x_q_dtype(jnp.int8) == jnp.int8
    assert _get_x_q_dtype(jnp.int4) == jnp.int8
    assert _get_x_q_dtype(jnp.float8_e4m3fn) == jnp.float8_e4m3fn
    assert _get_x_q_dtype(jnp.bfloat16) == jnp.float8_e4m3fn

    with pytest.raises(ValueError, match="Unsupported quantized dtype"):
        _get_x_q_dtype(jnp.bool_)


def test_quantized_matmul_jax_validation_and_dispatch(monkeypatch):
    x = jnp.ones((4, 16), dtype=jnp.bfloat16)
    w_q_bad = jnp.ones((8, 12), dtype=jnp.float8_e4m3fn)
    w_s_1d = jnp.ones((12,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Input hidden dim 16 must match"):
        _quantized_matmul_jax(x, w_q_bad, w_s_1d)

    w_q = jnp.ones((16, 12), dtype=jnp.float8_e4m3fn)

    # 4D scale validation errors
    with pytest.raises(ValueError, match="Blockwise weight scale axis 2 must be 1"):
        _quantized_matmul_jax(x, w_q, jnp.ones((1, 2, 2, 12), dtype=jnp.float32))

    with pytest.raises(
        ValueError, match="Blockwise weight scale output dim 10 must match"
    ):
        _quantized_matmul_jax(x, w_q, jnp.ones((1, 2, 1, 10), dtype=jnp.float32))

    with pytest.raises(
        ValueError, match="Input hidden dim 16 must be divisible by block scale count 0"
    ):
        _quantized_matmul_jax(x, w_q, jnp.ones((1, 0, 1, 12), dtype=jnp.float32))

    with pytest.raises(
        ValueError, match="Input hidden dim 16 must be divisible by block scale count 3"
    ):
        _quantized_matmul_jax(x, w_q, jnp.ones((1, 3, 1, 12), dtype=jnp.float32))

    # Valid 4D scale -> dispatches to gmm_v2
    captured_gmm = {}

    def fake_gmm_v2(**kwargs):
        captured_gmm.update(kwargs)
        return jnp.zeros((x.shape[0], w_q.shape[1]), dtype=x.dtype)

    monkeypatch.setattr(linear_common, "gmm_v2", fake_gmm_v2)
    w_s_4d = jnp.ones((1, 2, 1, 12), dtype=jnp.float32)
    out_4d = _quantized_matmul_jax(x, w_q, w_s_4d)
    assert out_4d.shape == (4, 12)
    assert captured_gmm["rhs"].shape == (1, 16, 12)
    assert captured_gmm["maybe_quantize_lhs"] is True
    assert captured_gmm["zero_initialize"] is True

    # Valid 1D scale -> dispatches to xla_quantized_matmul
    captured_xla = {}

    def fake_xla_qmm(lhs, rhs, scale):
        captured_xla["args"] = (lhs, rhs, scale)
        return jnp.ones((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", fake_xla_qmm)
    out_1d = _quantized_matmul_jax(x, w_q, w_s_1d)
    assert out_1d.shape == (4, 12)
    assert captured_xla["args"] == (x, w_q, w_s_1d)


def test_quantized_matmul_fp4_jax_dispatch(monkeypatch):
    x = jnp.ones((6, 32), dtype=jnp.bfloat16)
    w_q = jnp.ones((32, 16), dtype=jnp.int8)
    w_s = jnp.ones((1, 2, 1, 16), dtype=jnp.float32)

    captured = {}

    def fake_gmm_v2(**kwargs):
        captured.update(kwargs)
        return jnp.zeros((x.shape[0], w_q.shape[1]), dtype=x.dtype)

    monkeypatch.setattr(linear_common, "gmm_v2", fake_gmm_v2)
    out = _quantized_matmul_fp4_jax(x, w_q, w_s)
    assert out.shape == (6, 16)
    assert captured["rhs"].shape == (1, 32, 16)
    assert int(captured["group_sizes"][0]) == 6
    assert int(captured["group_offset"][0]) == 0
    assert captured["preferred_element_type"] == jnp.bfloat16


def test_get_quantized_matmul_op_caching_and_fake(monkeypatch):
    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op", None)

    registered_fake = {}
    mock_op = MagicMock(
        side_effect=lambda x, w_q, w_s: torch.zeros(
            x.shape[0], w_q.shape[-1], dtype=x.dtype
        )
    )

    def capture_register_fake(fn):
        registered_fake["fn"] = fn

    mock_op.register_fake.side_effect = capture_register_fake
    monkeypatch.setattr(
        linear_common.pallas,
        "jax_op",
        lambda name, fn: mock_op,
    )

    op1 = _get_quantized_matmul_op()
    assert op1 is mock_op
    assert linear_common._quantized_matmul_kernel_op is mock_op

    # Second call hits the outer fast path (line 103-104)
    op2 = _get_quantized_matmul_op()
    assert op2 is mock_op

    # Verify the registered fake implementation
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    w_q = torch.ones(16, 24, dtype=torch.int8)
    w_s = torch.ones(24, dtype=torch.float32)
    fake_out = registered_fake["fn"](x, w_q, w_s)
    assert fake_out.shape == (5, 24)
    assert fake_out.dtype == torch.bfloat16

    # Cover double-checked locking inner branch (lines 107-108)
    sentinel_op = MagicMock()

    class RaceLock:
        def __enter__(self):
            linear_common._quantized_matmul_kernel_op = sentinel_op

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op", None)
    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op_lock", RaceLock())
    assert _get_quantized_matmul_op() is sentinel_op


def test_quantized_matmul_2d_fast_path_and_nd_reshape(monkeypatch):
    calls = []

    def fake_op(x_2d: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor):
        calls.append(x_2d)
        return x_2d.float() @ w_q.float() * w_s.view(1, -1)

    w_q = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    w_s = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)

    # Mismatched hidden dim raises ValueError
    with pytest.raises(ValueError, match="Input hidden dim 5 must match"):
        quantized_matmul(torch.ones(2, 5), w_q, w_s)

    # Path A: _quantized_matmul_kernel_op is None -> calls _get_quantized_matmul_op
    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op", None)
    monkeypatch.setattr(linear_common, "_get_quantized_matmul_op", lambda: fake_op)

    x_2d = torch.ones(6, 4, dtype=torch.float32)
    out_2d = quantized_matmul(x_2d, w_q, w_s)
    assert out_2d.shape == (6, 3)
    # 2D fast path passes the exact tensor object without reshape view creation
    assert calls[-1] is x_2d

    # Path B: _quantized_matmul_kernel_op is already cached -> uses op directly
    monkeypatch.setattr(linear_common, "_quantized_matmul_kernel_op", fake_op)
    monkeypatch.setattr(
        linear_common,
        "_get_quantized_matmul_op",
        lambda: pytest.fail(
            "_get_quantized_matmul_op should not be called when cached"
        ),
    )

    out_2d_cached = quantized_matmul(x_2d, w_q, w_s)
    assert torch.equal(out_2d, out_2d_cached)
    assert calls[-1] is x_2d

    # Path C: 3D input -> reshapes to 2D and back to 3D, matching 2D result
    x_3d = x_2d.view(2, 3, 4)
    out_3d = quantized_matmul(x_3d, w_q, w_s)
    assert out_3d.shape == (2, 3, 3)
    assert torch.equal(out_3d.view(6, 3), out_2d)


def test_get_quantized_matmul_fp4_op_and_dispatch(monkeypatch):
    monkeypatch.setattr(linear_common, "_quantized_matmul_fp4_op", None)

    registered_fake = {}
    mock_op = MagicMock(
        side_effect=lambda x, w_q, w_s: torch.ones(
            x.shape[0], w_s.shape[-1], dtype=x.dtype
        )
    )

    def capture_register_fake(fn):
        registered_fake["fn"] = fn

    mock_op.register_fake.side_effect = capture_register_fake
    monkeypatch.setattr(
        linear_common.pallas,
        "jax_op",
        lambda name, fn: mock_op,
    )

    # First call via quantized_matmul_fp4 when _quantized_matmul_fp4_op is None
    x_2d = torch.zeros(8, 16, dtype=torch.bfloat16)
    w_q = torch.zeros(16, 8, dtype=torch.uint8)
    w_s = torch.ones(1, 1, 1, 16, dtype=torch.float32)

    out_2d = quantized_matmul_fp4(x_2d, w_q, w_s)
    assert out_2d.shape == (8, 16)
    assert mock_op.call_args[0][0] is x_2d

    # Second call uses cached _quantized_matmul_fp4_op and 3D reshape path
    x_3d = torch.zeros(2, 4, 16, dtype=torch.bfloat16)
    out_3d = quantized_matmul_fp4(x_3d, w_q, w_s)
    assert out_3d.shape == (2, 4, 16)

    # Direct _get_quantized_matmul_fp4_op cached fast path
    assert _get_quantized_matmul_fp4_op() is mock_op

    # Verify fake op registration
    fake_out = registered_fake["fn"](x_2d, w_q, w_s)
    assert fake_out.shape == (8, 16)
    assert fake_out.dtype == torch.bfloat16

    # Cover double-checked locking inner branch in _get_quantized_matmul_fp4_op
    sentinel_op = MagicMock()

    class RaceLock:
        def __enter__(self):
            linear_common._quantized_matmul_fp4_op = sentinel_op

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(linear_common, "_quantized_matmul_fp4_op", None)
    monkeypatch.setattr(linear_common, "_quantized_matmul_fp4_op_lock", RaceLock())
    assert _get_quantized_matmul_fp4_op() is sentinel_op


def test_sharded_quantized_matmul_both_in_axis_branches(monkeypatch):
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    x = jnp.ones((4, 8), dtype=jnp.bfloat16)
    w_q = jnp.ones((8, 6), dtype=jnp.float8_e4m3fn)
    w_s = jnp.ones((6,), dtype=jnp.float32)

    psum_calls = []
    monkeypatch.setattr(
        linear_common,
        "_quantized_matmul_jax",
        lambda lhs, rhs, scale: jnp.ones((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype),
    )
    monkeypatch.setattr(
        linear_common.jax.lax,
        "with_sharding_constraint",
        lambda arr, sharding: arr,
    )
    monkeypatch.setattr(
        linear_common.jax.lax,
        "psum",
        lambda arr, axis_name: (psum_calls.append(axis_name), arr)[1],
    )
    monkeypatch.setattr(
        linear_common,
        "shard_map",
        lambda fn, mesh, in_specs, out_specs, check_rep: fn,
    )

    # Branch 1: in_axis is None -> psum not called
    out_no_in_axis = sharded_quantized_matmul(x, w_q, w_s, mesh, P("tp", None))
    assert out_no_in_axis.shape == (4, 6)
    assert psum_calls == []

    # Branch 2: in_axis is "tp" -> psum called with axis_name="tp"
    out_with_in_axis = sharded_quantized_matmul(x, w_q, w_s, mesh, P(None, "tp"))
    assert out_with_in_axis.shape == (4, 6)
    assert psum_calls == ["tp"]


def test_reorder_and_slice_sharded_tensors():
    a = jnp.full((4, 12), 1, dtype=jnp.int32)
    b = jnp.full((4, 8), 2, dtype=jnp.int32)
    c = jnp.full((4, 4), 3, dtype=jnp.int32)
    concat = jnp.concatenate([a, b, c], axis=1)

    reordered = reorder_concatenated_tensor_for_sharding(
        concat, split_sizes=[12, 8, 4], n_shards=4, dim=1
    )
    assert reordered.shape == (4, 24)

    sliced = slice_sharded_tensor_for_concatenation(
        reordered, split_sizes=[12, 8, 4], n_shards=4
    )
    assert len(sliced) == 3
    np.testing.assert_array_equal(sliced[0], a)
    np.testing.assert_array_equal(sliced[1], b)
    np.testing.assert_array_equal(sliced[2], c)

    with pytest.raises(AssertionError):
        slice_sharded_tensor_for_concatenation(reordered, split_sizes=[10], n_shards=4)


def test_get_model_matmul_fusion_assignment():
    for key, expected in MODEL_MATMUL_FUSION_TRUTH_TABLE.items():
        assert get_model_matmul_fusion_assignment(*key) is expected

    # Unknown model/layer defaults to True
    assert (
        get_model_matmul_fusion_assignment(
            "deepseek-ai/DeepSeek-V4.1", 1024, 8, "QKVParallelLinear"
        )
        is True
    )
